"""Tests for the Linear rewriter."""

from __future__ import annotations

import pytest
import torch

from torchquant._types import Algorithm
from torchquant.export.rewriters.linear import rewrite_linear
from torchquant.export.runtime import QuantizedLinear

from ._rewriter_fixtures import make_linear_record


@pytest.mark.parametrize("algorithm", [Algorithm.RTN, Algorithm.GPTQ])
def test_linear_rewriter_preserves_quantization_and_bias(algorithm: Algorithm) -> None:
    """Linear rewrite should keep oracle q_int, bias, dtype, and forward parity."""
    module, record, oracle_int = make_linear_record(
        algorithm,
        bits=4,
        group_size=-1,
        symmetric=True,
    )
    rewritten = rewrite_linear(module, record)

    assert isinstance(rewritten, QuantizedLinear)
    assert torch.equal(rewritten.int_weight, oracle_int)
    assert rewritten.bias is record.original_bias
    assert rewritten.bias is module.bias
    assert rewritten.bias is not None
    assert rewritten.bias.dtype == module.bias.dtype

    sample_input = torch.randn(3, module.in_features)
    expected = torch.nn.functional.linear(
        sample_input,
        record.result.quantized_weight,
        module.bias,
    )
    atol = rewritten.scales.max().item() * 1e-3
    assert torch.allclose(rewritten(sample_input), expected, rtol=0, atol=atol)


def test_linear_rewriter_handles_biasless_module() -> None:
    """Linear rewrite should preserve bias absence."""
    module, record, oracle_int = make_linear_record(
        Algorithm.RTN,
        bits=4,
        group_size=4,
        symmetric=False,
        bias=False,
    )
    rewritten = rewrite_linear(module, record)

    assert isinstance(rewritten, QuantizedLinear)
    assert torch.equal(rewritten.int_weight, oracle_int)
    assert rewritten.bias is None
