"""Adapter for LLM / attention-based models.

Supports:
    - Transformer models with Q/K/V/O attention projections
    - Grouped-query attention (GQA) variants
    - Models following Hugging Face naming conventions
"""

from __future__ import annotations

from torch import nn

from torchquant._types import LayerKind

# Attention projection leaf-name groups
QUERY_NAMES: frozenset[str] = frozenset({"q_proj", "query"})
KEY_NAMES: frozenset[str] = frozenset({"k_proj", "key"})
VALUE_NAMES: frozenset[str] = frozenset({"v_proj", "value"})
OUTPUT_NAMES: frozenset[str] = frozenset({"o_proj", "out_proj"})
MERGED_QKV_NAMES: frozenset[str] = frozenset({"query_key_value", "c_attn"})

# Segments that indicate an attention context in the FQN path
_ATTENTION_CONTEXT: frozenset[str] = frozenset(
    {"attn", "attention", "self_attn", "self_attention"},
)

ALL_ATTENTION_NAMES: frozenset[str] = (
    QUERY_NAMES | KEY_NAMES | VALUE_NAMES | OUTPUT_NAMES | MERGED_QKV_NAMES
)


def _in_attention_context(name: str) -> bool:
    """Return True if the FQN path contains an attention-related segment."""
    return bool(_ATTENTION_CONTEXT & frozenset(name.lower().split(".")))


def classify_module(
    name: str,
    module: nn.Module,
) -> LayerKind | None:
    """Classify an LLM module.

    Attention projections get ATTENTION_QKV/ATTENTION_OUT.
    All other nn.Linear layers get LINEAR.
    """
    if isinstance(module, nn.Embedding):
        return LayerKind.EMBEDDING

    if isinstance(module, nn.Linear):
        leaf = name.rsplit(".", maxsplit=1)[-1].lower()
        if (
            leaf in QUERY_NAMES
            or leaf in KEY_NAMES
            or leaf in VALUE_NAMES
            or leaf in MERGED_QKV_NAMES
        ):
            return LayerKind.ATTENTION_QKV
        if leaf in OUTPUT_NAMES:
            return LayerKind.ATTENTION_OUT
        # c_proj is ambiguous: attn.c_proj (attention out) vs mlp.c_proj (FFN)
        if leaf == "c_proj" and _in_attention_context(name):
            return LayerKind.ATTENTION_OUT
        return LayerKind.LINEAR

    return None


def find_blocks(
    model: nn.Module,
) -> list[tuple[str, nn.Module]]:
    """Find transformer layer blocks.

    Matches patterns like model.layers.0, model.h.1, etc.
    """
    results: list[tuple[str, nn.Module]] = []

    for name, module in model.named_modules():
        parts = name.split(".")
        if (
            len(parts) >= 3
            and parts[-1].isdigit()
            and parts[-2] in {"layers", "h", "layer", "blocks"}
        ):
            results.append((name, module))

    return results


def is_skip_target(_name: str) -> bool:
    """LLM adapter does not skip attention projections -- it handles them."""
    return False


def prepare_model(model: nn.Module) -> nn.Module:
    """Prepare LLM: eval mode, freeze parameters."""
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model
