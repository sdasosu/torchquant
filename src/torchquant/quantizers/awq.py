"""Activation-aware weight quantization (AWQ).

Determines per-channel importance from activation statistics and
applies importance-weighted scaling before round-to-nearest quantization.
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
    """Quantize a single layer using AWQ.

    Requires "act_mean" key in stats (from AWQObserver).

    Args:
        module: The layer to quantize.
        scheme: Quantization scheme.
        stats: Must contain "act_mean" -- per-channel activation importance.

    Returns:
        QuantResult with importance-scaled quantized weights.
    """
    raise NotImplementedError
