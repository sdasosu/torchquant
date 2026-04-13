"""Tests for RTN quantization."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torchquant._types import Algorithm
from torchquant.config import QuantScheme
from torchquant.quantizers import rtn
from torchquant.quantizers._fake_quant import fake_quantize_2d


def _linear(
    in_features: int = 8,
    out_features: int = 4,
    *,
    dtype: torch.dtype = torch.float32,
) -> nn.Linear:
    module = nn.Linear(in_features, out_features, bias=False, dtype=dtype)
    weight = torch.linspace(
        -1.0,
        1.0,
        steps=in_features * out_features,
        dtype=torch.float32,
    ).reshape(out_features, in_features)
    with torch.no_grad():
        module.weight.copy_(weight.to(dtype))
    return module


def _scheme(
    bits: int = 8,
    group_size: int = -1,
    symmetric: bool = True,
) -> QuantScheme:
    return QuantScheme(
        weight_bits=bits,
        group_size=group_size,
        symmetric=symmetric,
        algorithm=Algorithm.RTN,
    )


def test_returns_quant_result_with_correct_shapes() -> None:
    """RTN returns tensors with the expected shapes and dtypes."""
    module = _linear()

    result = rtn.quantize_layer(module, _scheme(), stats={})

    assert result.quantized_weight.shape == module.weight.shape
    assert result.quantized_weight.dtype == module.weight.dtype
    assert result.scales.shape == (module.out_features, 1)
    assert result.scales.dtype == torch.float32
    assert result.zero_points is None
    assert torch.equal(result.original_weight, module.weight)


def test_supports_conv2d_and_embedding() -> None:
    """RTN accepts Conv2d and Embedding weights in addition to Linear."""
    conv = nn.Conv2d(3, 4, kernel_size=2, bias=False)
    embedding = nn.Embedding(5, 4)

    conv_result = rtn.quantize_layer(conv, _scheme(), stats={})
    embedding_result = rtn.quantize_layer(embedding, _scheme(), stats={})

    assert conv_result.quantized_weight.shape == conv.weight.shape
    assert conv_result.scales.shape == (conv.out_channels, 1)
    assert embedding_result.quantized_weight.shape == embedding.weight.shape
    assert embedding_result.scales.shape == (embedding.num_embeddings, 1)


def test_rejects_unsupported_module() -> None:
    """RTN rejects modules outside the supported weight-bearing set."""
    with pytest.raises(
        TypeError,
        match=r"nn\.Linear, nn\.Conv2d, and nn\.Embedding",
    ):
        rtn.quantize_layer(nn.ReLU(), _scheme(), stats={})


def test_roundtrip_error_bound_for_eight_bit_symmetric_linear() -> None:
    """Eight-bit symmetric RTN stays within one quantization step."""
    module = _linear()

    result = rtn.quantize_layer(module, _scheme(bits=8, symmetric=True), stats={})

    error = (result.quantized_weight - module.weight).abs().max().item()
    bound = module.weight.detach().abs().max().item() / 127.0
    assert error <= bound + 1e-6


def test_group_quant_scales_shape() -> None:
    """Per-group RTN returns one scale column per input group."""
    module = _linear(in_features=8, out_features=4)

    result = rtn.quantize_layer(module, _scheme(bits=4, group_size=4), stats={})

    assert result.scales.shape == (module.out_features, 2)


def test_four_bit_group_quant_relative_error_bound() -> None:
    """Aligned four-bit asymmetric groups keep relative error very small."""
    module = _linear(in_features=128, out_features=4)
    levels = torch.arange(16, dtype=torch.float32).repeat_interleave(8) / 15.0
    weight = torch.stack(
        [levels, levels * 0.5, levels * 1.5, levels * 2.0],
        dim=0,
    )
    with torch.no_grad():
        module.weight.copy_(weight)

    result = rtn.quantize_layer(
        module,
        _scheme(bits=4, group_size=128, symmetric=False),
        stats={},
    )

    relative_error = (
        result.quantized_weight - module.weight
    ).abs() / module.weight.abs().clamp(
        min=1e-6,
    )
    assert relative_error.max().item() <= 0.02


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp16_and_bf16_weight_roundtrip(dtype: torch.dtype) -> None:
    """RTN preserves low-precision weight dtypes while computing fp32 scales."""
    module = _linear(dtype=dtype)

    result = rtn.quantize_layer(module, _scheme(), stats={})

    assert result.quantized_weight.dtype == dtype
    assert result.scales.dtype == torch.float32
    assert torch.isfinite(result.scales).all()


def test_all_zero_row_does_not_nan() -> None:
    """All-zero rows quantize back to zeros without producing NaNs."""
    weight = torch.linspace(-1.0, 1.0, steps=32, dtype=torch.float32).reshape(4, 8)
    weight[0] = 0.0

    fake_q, scales, zero_points = fake_quantize_2d(
        weight,
        bits=8,
        group_size=-1,
        symmetric=True,
    )

    assert zero_points is None
    assert torch.equal(fake_q[0], torch.zeros_like(fake_q[0]))
    assert torch.isfinite(scales).all()


def test_asymmetric_zero_points_are_int32() -> None:
    """Asymmetric RTN exposes int32 zero points in the valid range."""
    module = _linear()

    result = rtn.quantize_layer(module, _scheme(bits=8, symmetric=False), stats={})

    assert result.zero_points is not None
    assert result.zero_points.dtype == torch.int32
    assert result.zero_points.min().item() >= 0
    assert result.zero_points.max().item() <= 255


def test_embedding_group_misdivision_raises() -> None:
    """Embedding group sizes must divide the embedding dimension exactly."""
    module = nn.Embedding(4, 10)

    with pytest.raises(ValueError, match=r"group_size 64 must divide in_features 10"):
        rtn.quantize_layer(module, _scheme(group_size=64), stats={})
