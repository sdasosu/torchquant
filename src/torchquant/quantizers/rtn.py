"""Round-to-nearest (RTN) weight quantization.

The simplest and fastest quantization method. Rounds weights to the
nearest quantization level without any calibration-based optimization.
Supports per-channel and per-group granularity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import QuantResult, _oracle
from ._fake_quant import (
    fake_quantize_2d,
    fake_quantize_2d_with_int,
    is_conv2d_module,
    is_embedding_module,
    is_linear_module,
    resolve_group_layout,
)

if TYPE_CHECKING:
    from torch import Tensor, nn

    from torchquant.config import QuantScheme


def quantize_layer(
    module: nn.Module,
    scheme: QuantScheme,
    stats: dict[str, Tensor],
) -> QuantResult:
    """Quantize a single supported layer using round-to-nearest.

    Args:
        module: The layer to quantize.
        scheme: Quantization scheme (bits, group_size, symmetric).
        stats: Collected statistics (unused for RTN, kept for API consistency).

    Returns:
        QuantResult with quantized weight, scales, and zero points.

    Raises:
        TypeError: If ``module`` is not ``nn.Linear``, ``nn.Conv2d``, or
            ``nn.Embedding``.
    """
    del stats

    oracle_fqn = _oracle.get_bound_fqn(module) if _oracle.is_recording() else None
    exact_quantized_weight: Tensor | None = None

    if is_linear_module(module) or is_embedding_module(module):
        weight = module.get_parameter("weight").detach()
        if oracle_fqn is None:
            fake_q, scales, zero_points = fake_quantize_2d(
                weight,
                bits=scheme.weight_bits,
                group_size=scheme.group_size,
                symmetric=scheme.symmetric,
            )
        else:
            fake_q, scales, zero_points, q_int = fake_quantize_2d_with_int(
                weight,
                bits=scheme.weight_bits,
                group_size=scheme.group_size,
                symmetric=scheme.symmetric,
            )
            _oracle.record(oracle_fqn, q_int.view_as(weight))
            exact_quantized_weight = _dequantize_int_weight(
                q_int, scales, zero_points, scheme
            )
        quantized_weight = fake_q.view_as(weight)
    elif is_conv2d_module(module):
        weight = module.get_parameter("weight").detach()
        flat_weight = weight.reshape(weight.shape[0], -1)
        if oracle_fqn is None:
            fake_q, scales, zero_points = fake_quantize_2d(
                flat_weight,
                bits=scheme.weight_bits,
                group_size=scheme.group_size,
                symmetric=scheme.symmetric,
            )
        else:
            fake_q, scales, zero_points, q_int = fake_quantize_2d_with_int(
                flat_weight,
                bits=scheme.weight_bits,
                group_size=scheme.group_size,
                symmetric=scheme.symmetric,
            )
            _oracle.record(oracle_fqn, q_int.reshape_as(weight))
            exact_quantized_weight = _dequantize_int_weight(
                q_int,
                scales,
                zero_points,
                scheme,
            ).reshape_as(weight)
        quantized_weight = fake_q.reshape_as(weight)
    else:
        supported = "nn.Linear, nn.Conv2d, and nn.Embedding"
        raise TypeError(
            f"RTN quantize_layer supports {supported}, got {type(module).__name__}."
        )

    result = QuantResult(
        quantized_weight=quantized_weight,
        scales=scales,
        zero_points=zero_points,
        original_weight=weight.clone(),
    )
    if exact_quantized_weight is not None:
        object.__setattr__(
            result, "_oracle_exact_quantized_weight", exact_quantized_weight
        )
    return result


def _dequantize_int_weight(
    q_int: Tensor,
    scales: Tensor,
    zero_points: Tensor | None,
    scheme: QuantScheme,
) -> Tensor:
    group_width, n_groups = resolve_group_layout(
        in_features=q_int.shape[1],
        group_size=scheme.group_size,
    )
    grouped_int = q_int.to(torch.float32).reshape(q_int.shape[0], n_groups, group_width)
    grouped_scales = scales.to(torch.float32).unsqueeze(-1)
    if zero_points is None:
        return (grouped_int * grouped_scales).reshape_as(q_int)
    grouped_zero_points = zero_points.to(torch.float32).unsqueeze(-1)
    return ((grouped_int - grouped_zero_points) * grouped_scales).reshape_as(q_int)
