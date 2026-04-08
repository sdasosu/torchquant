"""Quantization algorithm implementations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor, nn

    from torchquant.config import QuantScheme


@dataclass(frozen=True)
class QuantResult:
    """Output of a single-layer quantization pass.

    Args:
        quantized_weight: The quantized weight tensor.
        scales: Per-channel or per-group scale factors.
        zero_points: Zero-point offsets (None for symmetric quantization).
        original_weight: The original unquantized weight (for reference/debugging).
    """

    quantized_weight: Tensor
    scales: Tensor
    zero_points: Tensor | None
    original_weight: Tensor


type QuantizerFn = Callable[[nn.Module, QuantScheme, dict[str, Tensor]], QuantResult]
