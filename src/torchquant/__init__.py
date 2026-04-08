"""torchquant -- a universal PyTorch model quantization toolkit."""

from __future__ import annotations

from ._types import Algorithm, Granularity, LayerKind
from .config import QuantRecipe, QuantScheme
from .pipeline import quantize_model

__all__ = [
    "Algorithm",
    "Granularity",
    "LayerKind",
    "QuantRecipe",
    "QuantScheme",
    "quantize_model",
]
