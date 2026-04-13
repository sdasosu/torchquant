"""Tests for GPTQ quantization."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torchquant._types import Algorithm
from torchquant.config import QuantScheme
from torchquant.quantizers import gptq, rtn


def _linear(in_features: int = 8, out_features: int = 4) -> nn.Linear:
    module = nn.Linear(in_features, out_features, bias=False)
    weight = torch.linspace(
        -1.5,
        1.5,
        steps=in_features * out_features,
        dtype=torch.float32,
    ).reshape(out_features, in_features)
    with torch.no_grad():
        module.weight.copy_(weight)
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
        algorithm=Algorithm.GPTQ,
    )


def _hessian_from_inputs(inputs: torch.Tensor) -> torch.Tensor:
    return 2.0 * (inputs.T @ inputs) / inputs.shape[0]


def test_returns_quant_result_with_correct_shapes() -> None:
    """GPTQ returns tensors with the expected shapes and metadata."""
    module = _linear()
    hessian = torch.eye(module.in_features, dtype=torch.float32)

    result = gptq.quantize_layer(module, _scheme(), {"hessian": hessian})

    assert result.quantized_weight.shape == module.weight.shape
    assert result.quantized_weight.dtype == module.weight.dtype
    assert result.scales.shape == (module.out_features, 1)
    assert result.zero_points is None
    assert torch.equal(result.original_weight, module.weight)


def test_missing_stats_raises() -> None:
    """GPTQ fails fast when Hessian statistics are absent."""
    with pytest.raises(ValueError, match="hessian"):
        gptq.quantize_layer(_linear(), _scheme(), stats={})


def test_rejects_unsupported_module() -> None:
    """GPTQ only accepts linear layers in Phase 4."""
    with pytest.raises(TypeError, match=r"nn\.Linear only"):
        gptq.quantize_layer(
            nn.ReLU(),
            _scheme(),
            stats={"hessian": torch.eye(8, dtype=torch.float32)},
        )


def test_dead_neuron_column_is_zero() -> None:
    """Dead Hessian diagonals zero the matching weight column before quantizing."""
    module = _linear()
    hessian = torch.eye(module.in_features, dtype=torch.float32)
    hessian[3, 3] = 0.0

    result = gptq.quantize_layer(module, _scheme(), {"hessian": hessian})

    assert torch.equal(
        result.quantized_weight[:, 3], torch.zeros_like(result.quantized_weight[:, 3])
    )


def test_non_spd_hessian_raises() -> None:
    """GPTQ surfaces Cholesky failures instead of silently falling back."""
    module = _linear()
    hessian = -torch.eye(module.in_features, dtype=torch.float32)

    with pytest.raises(torch.linalg.LinAlgError):
        gptq.quantize_layer(module, _scheme(), {"hessian": hessian})


def test_better_than_rtn_on_correlated_inputs() -> None:
    """GPTQ improves output MSE over RTN on correlated Gaussian activations."""
    torch.manual_seed(0)
    module = _linear()
    indices = torch.arange(module.in_features, dtype=torch.float32)
    covariance = 0.9 ** (indices[:, None] - indices[None, :]).abs()
    cholesky = torch.linalg.cholesky(covariance)
    inputs = torch.randn(512, module.in_features, dtype=torch.float32) @ cholesky.T
    hessian = _hessian_from_inputs(inputs)

    original = F.linear(inputs, module.weight)
    scheme = _scheme(bits=8, group_size=-1, symmetric=True)
    rtn_result = rtn.quantize_layer(module, scheme, stats={})
    gptq_result = gptq.quantize_layer(module, scheme, {"hessian": hessian})

    rtn_mse = F.mse_loss(F.linear(inputs, rtn_result.quantized_weight), original)
    gptq_mse = F.mse_loss(F.linear(inputs, gptq_result.quantized_weight), original)

    assert gptq_mse.item() <= rtn_mse.item()


def test_group_quant_scales_shape() -> None:
    """Grouped GPTQ reports one scale column per 128-input group."""
    module = _linear(in_features=256, out_features=4)
    hessian = torch.eye(module.in_features, dtype=torch.float32)

    result = gptq.quantize_layer(
        module,
        _scheme(bits=4, group_size=128, symmetric=False),
        {"hessian": hessian},
    )

    assert result.scales.shape == (module.out_features, 2)
    assert result.zero_points is not None
    assert result.zero_points.dtype == torch.int32


def test_group_misalignment_large_group_across_blocks() -> None:
    """GPTQ keeps group scales alive when one group spans multiple blocks."""
    module = _linear(in_features=256, out_features=4)
    hessian = torch.eye(module.in_features, dtype=torch.float32)

    result = gptq.quantize_layer(
        module,
        _scheme(bits=4, group_size=256, symmetric=True),
        {"hessian": hessian},
    )

    assert result.scales.shape == (module.out_features, 1)
    assert torch.isfinite(result.quantized_weight).all()


def test_reverse_group_misalignment_scales_shape() -> None:
    """GPTQ handles blocks that start in the middle of a quantization group."""
    module = _linear(in_features=192, out_features=4)
    hessian = torch.eye(module.in_features, dtype=torch.float32)

    result = gptq.quantize_layer(
        module,
        _scheme(bits=4, group_size=64, symmetric=True),
        {"hessian": hessian},
    )

    assert result.scales.shape == (module.out_features, 3)
