"""SmoothQuant: activation-weight scale migration + RTN.

Migrates quantization difficulty from activations to weights by
applying per-channel scaling before quantization. This enables
layer-local weight-only SmoothQuant in Phase 4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import QuantResult
from ._fake_quant import fake_quantize_2d, is_linear_module

if TYPE_CHECKING:
    from torch import Tensor, nn

    from torchquant.config import QuantScheme

_SMOOTHQUANT_ALPHA: float = 0.5  # Llama-2/3 often tunes this closer to 0.85.


def quantize_layer(
    module: nn.Module,
    scheme: QuantScheme,
    stats: dict[str, Tensor],
) -> QuantResult:
    """Quantize a single linear layer using SmoothQuant.

    Args:
        module: The layer to quantize.
        scheme: Quantization scheme.
        stats: Must contain ``"act_max"`` with per-channel activation maxima.

    Returns:
        QuantResult with smooth-scaled quantized weights.

    Raises:
        TypeError: If ``module`` is not ``nn.Linear``.
        ValueError: If ``stats`` is missing ``"act_max"``.
    """
    if not is_linear_module(module):
        raise TypeError(
            "SmoothQuant quantize_layer supports nn.Linear only, got "
            f"{type(module).__name__}."
        )
    if "act_max" not in stats:
        raise ValueError("SmoothQuant requires 'act_max' statistics.")

    weight = module.get_parameter("weight").detach()
    act_max = stats["act_max"]
    weight_max = weight.abs().amax(dim=0)
    smooth_scale = act_max.clamp(min=1e-5).pow(_SMOOTHQUANT_ALPHA) / weight_max.clamp(
        min=1e-5,
    ).pow(1 - _SMOOTHQUANT_ALPHA)

    weight_scaled = weight * smooth_scale.unsqueeze(0)
    fake_q, scales, zero_points = fake_quantize_2d(
        weight_scaled,
        bits=scheme.weight_bits,
        group_size=scheme.group_size,
        symmetric=scheme.symmetric,
    )
    quantized_weight = fake_q / smooth_scale.unsqueeze(0)

    return QuantResult(
        quantized_weight=quantized_weight.to(weight.dtype),
        scales=scales,
        zero_points=zero_points,
        original_weight=weight.clone(),
    )
