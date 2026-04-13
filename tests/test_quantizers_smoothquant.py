"""Tests for SmoothQuant quantization."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torchquant._types import Algorithm
from torchquant.config import QuantScheme
from torchquant.quantizers import rtn, smoothquant


def _linear(
    in_features: int = 8,
    out_features: int = 4,
    *,
    dtype: torch.dtype = torch.float32,
) -> nn.Linear:
    module = nn.Linear(in_features, out_features, bias=False, dtype=dtype)
    weight = torch.tensor(
        [
            [0.0030, 1.00, -1.00, 0.50, -0.50, 0.25, -0.25, 0.125],
            [-0.0025, 0.75, -0.80, 0.40, -0.30, 0.20, -0.10, 0.050],
            [0.0020, -0.90, 0.85, -0.35, 0.30, -0.15, 0.12, -0.060],
            [-0.0015, 0.60, -0.55, 0.45, -0.20, 0.10, -0.08, 0.040],
        ],
        dtype=torch.float32,
    )
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
        algorithm=Algorithm.SMOOTHQUANT,
    )


def test_returns_quant_result_with_correct_shapes() -> None:
    """SmoothQuant returns weight tensors with stable shapes and dtypes."""
    module = _linear()
    act_max = torch.full((module.in_features,), 2.0)

    result = smoothquant.quantize_layer(module, _scheme(), {"act_max": act_max})

    assert result.quantized_weight.shape == module.weight.shape
    assert result.quantized_weight.dtype == module.weight.dtype
    assert result.scales.shape == (module.out_features, 1)
    assert result.zero_points is None
    assert torch.equal(result.original_weight, module.weight)


def test_missing_stats_raises() -> None:
    """SmoothQuant fails fast when act_max statistics are absent."""
    with pytest.raises(ValueError, match="act_max"):
        smoothquant.quantize_layer(_linear(), _scheme(), stats={})


def test_rejects_unsupported_module() -> None:
    """SmoothQuant is limited to linear layers in Phase 4."""
    with pytest.raises(TypeError, match=r"nn\.Linear only"):
        smoothquant.quantize_layer(
            nn.ReLU(),
            _scheme(),
            stats={"act_max": torch.ones(8)},
        )


def test_group_quant_scales_shape() -> None:
    """Per-group SmoothQuant preserves the shared fake-quant scale layout."""
    module = _linear(in_features=8, out_features=4)
    act_max = torch.arange(1, 9, dtype=torch.float32)

    result = smoothquant.quantize_layer(
        module,
        _scheme(bits=8, group_size=4),
        {"act_max": act_max},
    )

    assert result.scales.shape == (module.out_features, 2)


def test_beats_rtn_on_outlier_input_channels() -> None:
    """SmoothQuant improves output MSE on activation-outlier channels."""
    module = _linear()
    inputs = torch.tensor(
        [
            [100.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.125],
            [80.0, -0.8, 0.9, -0.4, 0.3, -0.2, 0.1, -0.05],
            [120.0, 0.7, -0.6, 0.3, -0.2, 0.1, -0.1, 0.05],
        ],
        dtype=torch.float32,
    )
    act_max = inputs.abs().amax(dim=0)

    original = F.linear(inputs, module.weight)
    rtn_result = rtn.quantize_layer(module, _scheme(bits=8), stats={})
    smooth_result = smoothquant.quantize_layer(
        module,
        _scheme(bits=8),
        {"act_max": act_max},
    )

    rtn_mse = F.mse_loss(F.linear(inputs, rtn_result.quantized_weight), original)
    smooth_mse = F.mse_loss(F.linear(inputs, smooth_result.quantized_weight), original)

    assert smooth_mse.item() <= rtn_mse.item()


def test_rejects_group_misdivision() -> None:
    """SmoothQuant surfaces fake-quant group-size validation errors."""
    with pytest.raises(ValueError, match=r"group_size 3 must divide in_features 8"):
        smoothquant.quantize_layer(
            _linear(),
            _scheme(group_size=3),
            stats={"act_max": torch.ones(8)},
        )
