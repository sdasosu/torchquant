"""Tests for AWQ quantization."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torchquant._types import Algorithm
from torchquant.config import QuantScheme
from torchquant.quantizers import awq, rtn
from torchquant.quantizers._fake_quant import fake_quantize_2d


def _linear(
    in_features: int = 8,
    out_features: int = 4,
    *,
    dtype: torch.dtype = torch.float32,
) -> nn.Linear:
    module = nn.Linear(in_features, out_features, bias=False, dtype=dtype)
    weight = torch.tensor(
        [
            [0.05, -0.04, 0.03, -0.02, 0.80, -0.75, 0.50, -0.45],
            [-0.04, 0.03, -0.02, 0.01, -0.70, 0.65, -0.40, 0.35],
            [0.03, -0.02, 0.01, -0.01, 0.60, -0.55, 0.30, -0.25],
            [-0.02, 0.01, -0.01, 0.01, -0.50, 0.45, -0.20, 0.15],
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
        algorithm=Algorithm.AWQ,
    )


def _expected_awq_weight(
    module: nn.Linear, scheme: QuantScheme, act_mean: torch.Tensor
):
    scale = act_mean.clamp(min=1e-5).pow(0.5)
    normalizer = (scale.max() * scale.min()).clamp(min=1e-5).sqrt()
    scale = (scale / normalizer).clamp(min=1e-5)
    fake_q, scales, zero_points = fake_quantize_2d(
        module.weight.detach() * scale.unsqueeze(0),
        bits=scheme.weight_bits,
        group_size=scheme.group_size,
        symmetric=scheme.symmetric,
    )
    return fake_q / scale.unsqueeze(0), scales, zero_points


@pytest.mark.parametrize("symmetric", [True, False])
def test_returns_quant_result_with_correct_shapes(symmetric: bool) -> None:
    """AWQ preserves the fake-quant result contract for both modes."""
    module = _linear()
    act_mean = torch.arange(1, 9, dtype=torch.float32)

    result = awq.quantize_layer(
        module, _scheme(symmetric=symmetric), {"act_mean": act_mean}
    )

    assert result.quantized_weight.shape == module.weight.shape
    assert result.scales.shape == (module.out_features, 1)
    if symmetric:
        assert result.zero_points is None
    else:
        assert result.zero_points is not None
        assert result.zero_points.dtype == torch.int32


def test_missing_stats_raises() -> None:
    """AWQ fails fast when act_mean statistics are absent."""
    with pytest.raises(ValueError, match="act_mean"):
        awq.quantize_layer(_linear(), _scheme(), stats={})


def test_rejects_unsupported_module() -> None:
    """AWQ only accepts linear layers in Phase 4."""
    with pytest.raises(TypeError, match=r"nn\.Linear only"):
        awq.quantize_layer(
            nn.ReLU(),
            _scheme(),
            stats={"act_mean": torch.ones(8)},
        )


def test_uses_act_mean_and_internal_alpha() -> None:
    """AWQ recomputes its own scale from act_mean instead of caller scale."""
    module = _linear()
    scheme = _scheme(bits=8, symmetric=True)
    act_mean = torch.tensor([16.0, 9.0, 4.0, 1.0, 1.0, 4.0, 9.0, 16.0])

    result = awq.quantize_layer(
        module,
        scheme,
        {"act_mean": act_mean, "scale": torch.full_like(act_mean, 999.0)},
    )
    expected_weight, expected_scales, expected_zero_points = _expected_awq_weight(
        module,
        scheme,
        act_mean,
    )

    torch.testing.assert_close(result.quantized_weight, expected_weight)
    torch.testing.assert_close(result.scales, expected_scales)
    assert result.zero_points is expected_zero_points


def test_ignores_extra_stats_keys() -> None:
    """Passing observer scale metadata does not affect the AWQ result."""
    module = _linear()
    act_mean = torch.arange(1, 9, dtype=torch.float32)

    base_result = awq.quantize_layer(module, _scheme(), {"act_mean": act_mean})
    extra_result = awq.quantize_layer(
        module,
        _scheme(),
        {"act_mean": act_mean, "scale": torch.full_like(act_mean, 123.0)},
    )

    torch.testing.assert_close(
        base_result.quantized_weight, extra_result.quantized_weight
    )
    torch.testing.assert_close(base_result.scales, extra_result.scales)
    assert base_result.zero_points is extra_result.zero_points


def test_group_quant_scales_shape() -> None:
    """Per-group AWQ uses one scale column per group."""
    module = _linear(in_features=8, out_features=4)
    act_mean = torch.arange(1, 9, dtype=torch.float32)

    result = awq.quantize_layer(
        module,
        _scheme(bits=4, group_size=4),
        {"act_mean": act_mean},
    )

    assert result.scales.shape == (module.out_features, 2)


def test_beats_rtn_on_salient_channels() -> None:
    """AWQ reduces salient-channel weight error compared with pure RTN."""
    module = nn.Linear(128, 4, bias=False)
    base = torch.linspace(-1.0, 1.0, steps=128, dtype=torch.float32)
    weight = torch.stack([base, -base, base * 0.5, -base * 0.5], dim=0)
    weight[:, :4] = torch.tensor(
        [
            [0.05, -0.04, 0.03, -0.02],
            [-0.04, 0.03, -0.02, 0.01],
            [0.03, -0.02, 0.01, -0.01],
            [-0.02, 0.01, -0.01, 0.01],
        ],
        dtype=torch.float32,
    )
    with torch.no_grad():
        module.weight.copy_(weight)

    scheme = _scheme(bits=4, group_size=128, symmetric=True)
    act_mean = torch.cat(
        [torch.full((4,), 100.0), torch.full((124,), 0.1)],
    )

    rtn_result = rtn.quantize_layer(module, scheme, stats={})
    awq_result = awq.quantize_layer(module, scheme, {"act_mean": act_mean})

    rtn_error = (rtn_result.quantized_weight[:, :4] - module.weight[:, :4]).abs().max()
    awq_error = (awq_result.quantized_weight[:, :4] - module.weight[:, :4]).abs().max()

    assert awq_error.item() < rtn_error.item()


def test_rejects_group_misdivision() -> None:
    """AWQ surfaces fake-quant group-size validation errors."""
    with pytest.raises(ValueError, match=r"group_size 3 must divide in_features 8"):
        awq.quantize_layer(
            _linear(),
            _scheme(group_size=3),
            stats={"act_mean": torch.ones(8)},
        )
