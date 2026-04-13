"""Activation-aware weight quantization (AWQ).

Determines per-channel importance from activation statistics and
applies importance-weighted scaling before round-to-nearest quantization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import QuantResult
from ._fake_quant import fake_quantize_2d, is_linear_module

if TYPE_CHECKING:
    from torch import Tensor, nn

    from torchquant.config import QuantScheme

_AWQ_ALPHA: float = 0.5  # The AWQ paper grid-searches alpha from 0.00 to 0.95.


def quantize_layer(
    module: nn.Module,
    scheme: QuantScheme,
    stats: dict[str, Tensor],
) -> QuantResult:
    """Quantize a single linear layer using AWQ.

    Args:
        module: The layer to quantize.
        scheme: Quantization scheme.
        stats: Must contain ``"act_mean"`` with per-channel activation means.

    Returns:
        QuantResult with importance-scaled quantized weights.

    Raises:
        TypeError: If ``module`` is not ``nn.Linear``.
        ValueError: If ``stats`` is missing ``"act_mean"``.
    """
    # TODO: Real AWQ grid-searches alpha and post-scale clipping; see
    # https://github.com/mit-han-lab/llm-awq (awq/quantize/auto_scale.py).
    if not is_linear_module(module):
        raise TypeError(
            f"AWQ quantize_layer supports nn.Linear only, got {type(module).__name__}."
        )
    if "act_mean" not in stats:
        raise ValueError("AWQ requires 'act_mean' statistics.")

    weight = module.get_parameter("weight").detach()
    importance_scale = stats["act_mean"].clamp(min=1e-5).pow(_AWQ_ALPHA)
    normalizer = (
        (importance_scale.max() * importance_scale.min())
        .clamp(
            min=1e-5,
        )
        .sqrt()
    )
    importance_scale = (importance_scale / normalizer).clamp(min=1e-5)

    weight_scaled = weight * importance_scale.unsqueeze(0)
    fake_q, scales, zero_points = fake_quantize_2d(
        weight_scaled,
        bits=scheme.weight_bits,
        group_size=scheme.group_size,
        symmetric=scheme.symmetric,
    )
    quantized_weight = fake_q / importance_scale.unsqueeze(0)

    return QuantResult(
        quantized_weight=quantized_weight.to(weight.dtype),
        scales=scales,
        zero_points=zero_points,
        original_weight=weight.clone(),
    )
