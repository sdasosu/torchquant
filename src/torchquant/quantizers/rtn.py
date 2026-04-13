"""Round-to-nearest (RTN) weight quantization.

The simplest and fastest quantization method. Rounds weights to the
nearest quantization level without any calibration-based optimization.
Supports per-channel and per-group granularity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import QuantResult
from ._fake_quant import (
    fake_quantize_2d,
    is_conv2d_module,
    is_embedding_module,
    is_linear_module,
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

    if is_linear_module(module) or is_embedding_module(module):
        weight = module.get_parameter("weight").detach()
        fake_q, scales, zero_points = fake_quantize_2d(
            weight,
            bits=scheme.weight_bits,
            group_size=scheme.group_size,
            symmetric=scheme.symmetric,
        )
        quantized_weight = fake_q.view_as(weight)
    elif is_conv2d_module(module):
        weight = module.get_parameter("weight").detach()
        flat_weight = weight.reshape(weight.shape[0], -1)
        fake_q, scales, zero_points = fake_quantize_2d(
            flat_weight,
            bits=scheme.weight_bits,
            group_size=scheme.group_size,
            symmetric=scheme.symmetric,
        )
        quantized_weight = fake_q.reshape_as(weight)
    else:
        supported = "nn.Linear, nn.Conv2d, and nn.Embedding"
        raise TypeError(
            f"RTN quantize_layer supports {supported}, got {type(module).__name__}."
        )

    return QuantResult(
        quantized_weight=quantized_weight,
        scales=scales,
        zero_points=zero_points,
        original_weight=weight.clone(),
    )
