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

ALL_ATTENTION_NAMES: frozenset[str] = ( QUERY_NAMES | KEY_NAMES | VALUE_NAMES | OUTPUT_NAMES | MERGED_QKV_NAMES )


def classify_module(name: str, module: nn.Module) -> LayerKind | None:
    """Classify an LLM module: attention projections get ATTENTION_QKV/ATTENTION_OUT.

    All other nn.Linear layers get LINEAR.
    """
    leaf = name.split(".")[-1].lower()

    if "kv_cache" in name.lower():
        return LayerKind.KV_CACHE

    if isinstance(module, nn.Embedding):
        return LayerKind.EMBEDDING

    if isinstance(module, nn.Linear):
        if leaf in QUERY_NAMES or leaf in KEY_NAMES or leaf in VALUE_NAMES or leaf in MERGED_QKV_NAMES:
            return LayerKind.ATTENTION_QKV
        if leaf in OUTPUT_NAMES:
            return LayerKind.ATTENTION_OUT
        return LayerKind.LINEAR

    return None



def find_blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Find transformer layer blocks (e.g. model.layers.0, model.layers.1, ...)."""
    results: list[tuple[str, nn.Module]] = []

    for name, module in model.named_modules():
        parts = name.split(".")
        if len(parts) >= 3 and parts[-1].isdigit():
            if parts[-2] in {"layers", "h", "layer", "blocks"}:
                results.append((name, module))

    return results


def is_skip_target(name: str) -> bool:
    """LLM adapter does not skip attention projections — it handles them."""
    raise False



def prepare_model(model: nn.Module) -> nn.Module:
    """Prepare LLM: eval mode, freeze parameters."""
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model
