"""SmoothQuant: activation-weight scale migration + RTN.

Migrates quantization difficulty from activations to weights by
applying per-channel scaling before quantization. This enables
effective W8A8 (weight 8-bit, activation 8-bit) quantization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor, nn

    from torchquant.config import QuantScheme
    from torchquant.quantizers import QuantResult


def quantize_layer(
    module: nn.Module,
    scheme: QuantScheme,
    stats: dict[str, Tensor],
) -> QuantResult:
    """Quantize a single layer using SmoothQuant.

    Requires "act_max" key in stats (from SmoothQuantObserver).
    Computes migration scales and applies them before RTN quantization.

    Args:
        module: The layer to quantize.
        scheme: Quantization scheme.
        stats: Must contain "act_max" -- per-channel max absolute activation.

    Returns:
        QuantResult with smooth-scaled quantized weights.
    """
    raise NotImplementedError
