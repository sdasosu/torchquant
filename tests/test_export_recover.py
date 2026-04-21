"""Tests for integer-weight recovery used by exported runtime modules."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torchquant._types import Algorithm
from torchquant.config import QuantScheme
from torchquant.export._recover import recover_int_weight
from torchquant.export.runtime import UnsupportedExportError
from torchquant.quantizers import QuantResult, _oracle, awq, gptq, rtn, smoothquant
from torchquant.quantizers._fake_quant import _asymmetric_qmax, _symmetric_qmax


def _scheme(
    algorithm: Algorithm,
    *,
    bits: int,
    group_size: int,
    symmetric: bool,
) -> QuantScheme:
    return QuantScheme(
        weight_bits=bits,
        group_size=group_size,
        symmetric=symmetric,
        algorithm=algorithm,
    )


def _linear(
    *, in_features: int = 128, out_features: int = 4, dtype: torch.dtype
) -> nn.Linear:
    module = nn.Linear(in_features, out_features, bias=False, dtype=dtype)
    weight = torch.linspace(
        -1.25,
        1.75,
        steps=in_features * out_features,
        dtype=torch.float32,
    ).reshape(out_features, in_features)
    with torch.no_grad():
        module.weight.copy_(weight.to(dtype))
    return module


def _hessian(in_features: int) -> torch.Tensor:
    diagonal = torch.linspace(1.0, 2.0, steps=in_features, dtype=torch.float32)
    return torch.diag(diagonal)


def _awq_stats(module: nn.Linear) -> dict[str, torch.Tensor]:
    return {
        "act_mean": torch.linspace(
            0.25,
            1.25,
            steps=module.in_features,
            dtype=torch.float32,
        )
    }


def _smoothquant_stats(module: nn.Linear) -> dict[str, torch.Tensor]:
    return {
        "act_max": torch.linspace(
            0.5,
            1.5,
            steps=module.in_features,
            dtype=torch.float32,
        )
    }


def _run_quantizer_with_oracle(
    algorithm: Algorithm,
    *,
    bits: int,
    group_size: int,
    symmetric: bool,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, QuantResult]:
    module = _linear(dtype=dtype)
    scheme = _scheme(
        algorithm,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
    )
    if algorithm is Algorithm.RTN:
        quantizer = rtn.quantize_layer
        stats: dict[str, torch.Tensor] = {}
    else:
        quantizer = gptq.quantize_layer
        stats = {"hessian": _hessian(module.in_features)}

    with _oracle.recording() as records, _oracle.bound_fqn(module, "layer"):
        result = quantizer(module, scheme, stats)

    return records["layer"], result


@pytest.mark.parametrize("algorithm", [Algorithm.RTN, Algorithm.GPTQ])
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("group_size", [-1, 32, 128])
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_recover_int_weight_matches_oracle(
    algorithm: Algorithm,
    bits: int,
    group_size: int,
    symmetric: bool,
    dtype: torch.dtype,
) -> None:
    """recover_int_weight reproduces the quantizer's canonical integer tensor."""
    oracle_int, result = _run_quantizer_with_oracle(
        algorithm,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
        dtype=dtype,
    )
    scheme = _scheme(
        algorithm,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
    )

    recovered_int, scales, zero_points = recover_int_weight(
        result,
        scheme,
        tuple(result.quantized_weight.shape),
        fqn="layer",
    )

    assert torch.equal(recovered_int, oracle_int)
    assert recovered_int.shape == result.quantized_weight.shape
    if symmetric:
        assert zero_points is None
        assert recovered_int.min().item() >= -_symmetric_qmax(bits)
        assert recovered_int.max().item() <= _symmetric_qmax(bits)
    else:
        assert zero_points is not None
        assert zero_points.dtype == torch.int32
        assert recovered_int.min().item() >= 0
        assert recovered_int.max().item() <= _asymmetric_qmax(bits)
    assert scales.dtype == torch.float32


@pytest.mark.parametrize(
    ("algorithm", "quantizer", "stats_factory"),
    [
        (Algorithm.AWQ, awq.quantize_layer, _awq_stats),
        (Algorithm.SMOOTHQUANT, smoothquant.quantize_layer, _smoothquant_stats),
    ],
)
def test_recover_int_weight_rejects_unsupported_algorithms(
    algorithm: Algorithm,
    quantizer,
    stats_factory,
) -> None:
    """AWQ and SmoothQuant fail fast with a Phase 6 export limitation message."""
    module = _linear(dtype=torch.float32)
    scheme = _scheme(
        algorithm,
        bits=4,
        group_size=128,
        symmetric=True,
    )
    result = quantizer(module, scheme, stats_factory(module))

    with pytest.raises(UnsupportedExportError, match=r"layer.*not exportable"):
        recover_int_weight(
            result,
            scheme,
            tuple(result.quantized_weight.shape),
            fqn="layer",
        )


def test_recover_int_weight_manual_gptq_spot_check() -> None:
    """Recovered GPTQ integers have a sensible bounded, non-degenerate distribution."""
    module = nn.Linear(128, 4, bias=False, dtype=torch.float32)
    column_positions = torch.linspace(-3.0, 3.0, steps=128, dtype=torch.float32)
    weight = torch.stack(
        [
            torch.sin(column_positions),
            torch.cos(column_positions * 0.7),
            torch.linspace(-1.5, 1.5, steps=128, dtype=torch.float32),
            ((torch.arange(128, dtype=torch.float32) % 19) - 9.0) / 3.0,
        ],
        dim=0,
    )
    with torch.no_grad():
        module.weight.copy_(weight)

    scheme = _scheme(
        Algorithm.GPTQ,
        bits=4,
        group_size=128,
        symmetric=False,
    )
    with _oracle.recording() as records, _oracle.bound_fqn(module, "spotcheck"):
        result = gptq.quantize_layer(module, scheme, {"hessian": _hessian(128)})

    recovered_int, _, _ = recover_int_weight(
        result,
        scheme,
        tuple(result.quantized_weight.shape),
        fqn="spotcheck",
    )

    assert torch.equal(recovered_int, records["spotcheck"])
    assert recovered_int.min().item() >= 0
    assert recovered_int.max().item() <= _asymmetric_qmax(4)
    assert torch.var(recovered_int.to(torch.float32)).item() > 0
    assert torch.all(recovered_int.to(torch.float32).var(dim=1) > 0)
