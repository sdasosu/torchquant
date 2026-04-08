"""Adapter for Sequential / MLP-style (SMP) models.

Supports:
    - Plain sequential models (nn.Sequential)
    - MLP-style models with repeated blocks (GPT-2 FFN, LLaMA FFN, BERT FFN, etc.)
    - Any model whose quantizable units are nn.Linear layers

This adapter intentionally skips attention projections — those are
handled by the llm adapter.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn

    from torchquant._types import LayerKind

# Known block-name patterns for transformer FFN sub-blocks
HF_BLOCK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^transformer\.h\.\d+$"),  # GPT-2
    re.compile(r"^model\.layers\.\d+$"),  # LLaMA / Mistral / Qwen
    re.compile(r"^encoder\.layer\.\d+$"),  # BERT / RoBERTa encoder
    re.compile(r"^decoder\.layers\.\d+$"),  # BART / T5 decoder
    re.compile(r"^layers\.\d+$"),  # generic flat stack
    re.compile(r"^blocks\.\d+$"),  # generic flat stack (alt)
    re.compile(r"^\d+$"),  # plain nn.Sequential index
]

# Attention projection leaf names to skip (handled by llm adapter)
ATTENTION_PROJECTION_NAMES: frozenset[str] = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "query",
        "key",
        "value",
        "out_proj",
        "query_key_value",
        "c_attn",
        "c_proj",
    }
)


def classify_module(name: str, module: nn.Module) -> LayerKind | None:
    """Classify an SMP module: only nn.Linear, skip attention projections."""
    raise NotImplementedError


def find_blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Find repeated blocks matching HF_BLOCK_PATTERNS."""
    raise NotImplementedError


def is_skip_target(name: str) -> bool:
    """Return True if name is an attention projection (handled by llm adapter)."""
    raise NotImplementedError


def prepare_model(model: nn.Module) -> nn.Module:
    """Prepare SMP model: eval, freeze, fuse BatchNorm where possible."""
    raise NotImplementedError
