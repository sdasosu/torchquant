"""Adapter for LLM / attention-based models.

Supports:
    - Transformer models with Q/K/V/O attention projections
    - Grouped-query attention (GQA) variants
    - KV-cache quantization targets
    - Models following Hugging Face naming conventions
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn

    from torchquant._types import LayerKind

# Attention projection leaf-name groups
QUERY_NAMES: frozenset[str] = frozenset({"q_proj", "query"})
KEY_NAMES: frozenset[str] = frozenset({"k_proj", "key"})
VALUE_NAMES: frozenset[str] = frozenset({"v_proj", "value"})
OUTPUT_NAMES: frozenset[str] = frozenset({"o_proj", "out_proj", "c_proj"})
MERGED_QKV_NAMES: frozenset[str] = frozenset({"query_key_value", "c_attn"})

ALL_ATTENTION_NAMES: frozenset[str] = (
    QUERY_NAMES | KEY_NAMES | VALUE_NAMES | OUTPUT_NAMES | MERGED_QKV_NAMES
)


def classify_module(name: str, module: nn.Module) -> LayerKind | None:
    """Classify an LLM module: attention projections get ATTENTION_QKV/ATTENTION_OUT.

    All other nn.Linear layers get LINEAR.
    """
    raise NotImplementedError


def find_blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Find transformer layer blocks (e.g. model.layers.0, model.layers.1, ...)."""
    raise NotImplementedError


def is_skip_target(name: str) -> bool:
    """LLM adapter does not skip attention projections — it handles them."""
    raise NotImplementedError


def prepare_model(model: nn.Module) -> nn.Module:
    """Prepare LLM: eval mode, freeze parameters."""
    raise NotImplementedError
