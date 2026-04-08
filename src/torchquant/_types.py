"""Shared enums and type aliases."""

from __future__ import annotations

from enum import Enum, auto


class LayerKind(Enum):
    """Classification of quantizable layer types."""

    LINEAR = auto()
    CONV2D = auto()
    DEPTHWISE_CONV = auto()
    POINTWISE_CONV = auto()
    CONV_TRANSPOSE = auto()
    EMBEDDING = auto()
    ATTENTION_QKV = auto()
    ATTENTION_OUT = auto()
    KV_CACHE = auto()


class Algorithm(Enum):
    """Supported quantization algorithms."""

    RTN = auto()
    GPTQ = auto()
    AWQ = auto()
    SMOOTHQUANT = auto()


class Granularity(Enum):
    """Quantization granularity."""

    PER_TENSOR = auto()
    PER_CHANNEL = auto()
    PER_GROUP = auto()
