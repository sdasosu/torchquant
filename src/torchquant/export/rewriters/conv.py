"""Conv2d module rewriter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from torchquant._types import Algorithm
from torchquant.export._recover import recover_int_weight
from torchquant.export.rewriter import register_quantized_module
from torchquant.export.runtime import QuantizedConv2d
from torchquant.export.runtime.conv import Conv2dSpec

if TYPE_CHECKING:
    from torchquant.registry import QuantRecord


def _coerce_pair(value: object) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"Expected an int or length-2 tuple of ints, got {value!r}.")
    first, second = value
    if not isinstance(first, int) or not isinstance(second, int):
        raise TypeError(f"Expected an int or length-2 tuple of ints, got {value!r}.")
    return (first, second)


def rewrite_conv2d(module: nn.Module, record: QuantRecord) -> nn.Module:
    """Rewrite a Conv2d module into QuantizedConv2d."""
    if not isinstance(module, nn.Conv2d):
        raise TypeError(f"Expected nn.Conv2d, got {type(module).__name__}.")
    int_weight, scales, zero_points = recover_int_weight(
        record.result,
        record.scheme,
        tuple(module.weight.shape),
        fqn=record.fqn,
    )
    return QuantizedConv2d(
        int_weight=int_weight,
        scales=scales,
        zero_points=zero_points,
        bias=record.original_bias,
        scheme=record.scheme,
        spec=Conv2dSpec(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=_coerce_pair(module.kernel_size),
            stride=_coerce_pair(module.stride),
            padding=_coerce_pair(module.padding),
            dilation=_coerce_pair(module.dilation),
            groups=module.groups,
        ),
        weight_dtype=record.result.original_weight.dtype,
    )


register_quantized_module(
    nn.Conv2d,
    rewrite_conv2d,
    frozenset({Algorithm.RTN}),
)
