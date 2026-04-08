"""GPTQ blockwise weight quantization.

Implements the Optimal Brain Quantization (OBQ) approach from the GPTQ
paper, using Hessian information to find optimal weight rounding
decisions that minimize layer-wise reconstruction error.
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
    """Quantize a single layer using GPTQ.

    Requires "hessian" key in stats (from HessianObserver).

    Args:
        module: The layer to quantize.
        scheme: Quantization scheme.
        stats: Must contain "hessian" -- the accumulated H matrix.

    Returns:
        QuantResult with optimally rounded quantized weights.
    """
    raise NotImplementedError
