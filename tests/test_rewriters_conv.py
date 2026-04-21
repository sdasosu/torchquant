"""Tests for the Conv2d rewriter."""

from __future__ import annotations

import torch

from torchquant.export.rewriters.conv import rewrite_conv2d
from torchquant.export.runtime import QuantizedConv2d

from ._rewriter_fixtures import make_conv_record


def test_conv_rewriter_preserves_quantization_bias_and_structure() -> None:
    """Conv2d rewrite should keep q_int, bias, and convolution attributes."""
    module, record, oracle_int = make_conv_record(bits=4, group_size=4, symmetric=False)
    rewritten = rewrite_conv2d(module, record)

    assert isinstance(rewritten, QuantizedConv2d)
    assert torch.equal(rewritten.int_weight, oracle_int)
    assert rewritten.bias is record.original_bias
    assert rewritten.bias is module.bias
    assert rewritten.bias is not None
    assert module.bias is not None
    assert rewritten.bias.dtype == module.bias.dtype
    assert rewritten.stride == module.stride
    assert rewritten.padding == module.padding
    assert rewritten.dilation == module.dilation
    assert rewritten.groups == module.groups

    sample_input = torch.randn(2, module.in_channels, 6, 6)
    expected = torch.nn.functional.conv2d(
        sample_input,
        record.result.quantized_weight,
        module.bias,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
    )
    atol = rewritten.scales.max().item() * 1e-3
    assert torch.allclose(rewritten(sample_input), expected, rtol=0, atol=atol)
