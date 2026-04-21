"""Linear module rewriter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from torchquant._types import Algorithm
from torchquant.export._recover import recover_int_weight
from torchquant.export.rewriter import register_quantized_module
from torchquant.export.runtime import QuantizedLinear

if TYPE_CHECKING:
    from torchquant.registry import QuantRecord


def rewrite_linear(module: nn.Module, record: QuantRecord) -> nn.Module:
    """Rewrite a Linear-like module into QuantizedLinear."""
    if not isinstance(module, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(module).__name__}.")
    int_weight, scales, zero_points = recover_int_weight(
        record.result,
        record.scheme,
        tuple(module.weight.shape),
        fqn=record.fqn,
    )
    return QuantizedLinear(
        int_weight=int_weight,
        scales=scales,
        zero_points=zero_points,
        bias=record.original_bias,
        scheme=record.scheme,
        in_features=module.in_features,
        out_features=module.out_features,
        weight_dtype=record.result.original_weight.dtype,
    )


register_quantized_module(
    nn.Linear,
    rewrite_linear,
    frozenset({Algorithm.RTN, Algorithm.GPTQ}),
)
