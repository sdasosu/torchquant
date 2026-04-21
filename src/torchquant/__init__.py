"""torchquant -- a universal PyTorch model quantization toolkit."""

from __future__ import annotations

from ._types import Algorithm, Granularity, LayerKind
from .config import QuantRecipe, QuantScheme
from .export import (
    QuantizedConv2d,
    QuantizedEmbedding,
    QuantizedLinear,
    UnsupportedExportError,
    export_pt2e,
    register_quantized_module,
)
from .pipeline import build_quantized_model, quantize_model
from .registry import QuantRegistry

__all__ = [
    "Algorithm",
    "Granularity",
    "LayerKind",
    "QuantRecipe",
    "QuantRegistry",
    "QuantScheme",
    "QuantizedConv2d",
    "QuantizedEmbedding",
    "QuantizedLinear",
    "UnsupportedExportError",
    "build_quantized_model",
    "export_pt2e",
    "quantize_model",
    "register_quantized_module",
]
