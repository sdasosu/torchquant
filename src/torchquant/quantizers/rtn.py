"""Round-to-nearest (RTN) weight quantization.

The simplest and fastest quantization method. Rounds weights to the
nearest quantization level without any calibration-based optimization.
Supports per-channel and per-group granularity.
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
    """Quantize a single layer using round-to-nearest.

    Args:
        module: The layer to quantize (nn.Linear or nn.Conv2d).
        scheme: Quantization scheme (bits, group_size, symmetric).
        stats: Collected statistics (unused for RTN, kept for API consistency).

    Returns:
        QuantResult with quantized weight, scales, and zero points.
    """
    raise NotImplementedError
