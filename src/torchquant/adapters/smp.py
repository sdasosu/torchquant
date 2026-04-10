"""Adapter for Sequential / MLP-style (SMP) models.

Supports:
    - Plain sequential models (nn.Sequential)
    - MLP-style models with repeated blocks (GPT-2 FFN, LLaMA FFN, BERT FFN, etc.)
    - Any model whose quantizable units are nn.Linear layers

This adapter intentionally skips attention projections -- those are
handled by the llm adapter.
"""

from __future__ import annotations

import re

from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

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

# Unambiguous attention projection leaf names to skip
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
    },
)

# Segments that indicate an attention context in the FQN path
_ATTENTION_CONTEXT: frozenset[str] = frozenset(
    {"attn", "attention", "self_attn", "self_attention"},
)


def classify_module(
    _name: str,
    module: nn.Module,
) -> LayerKind | None:
    """Classify a module by isinstance checks only.

    The SMP adapter does not infer attention sub-types from names;
    attention projections are skipped via ``is_skip_target`` instead.
    """
    if isinstance(module, nn.ConvTranspose2d):
        return LayerKind.CONV_TRANSPOSE

    if isinstance(module, nn.Conv2d):
        if module.groups == module.in_channels and module.in_channels > 1:
            return LayerKind.DEPTHWISE_CONV
        if module.kernel_size == (1, 1):
            return LayerKind.POINTWISE_CONV
        return LayerKind.CONV2D

    if isinstance(module, nn.Linear):
        return LayerKind.LINEAR

    if isinstance(module, nn.Embedding):
        return LayerKind.EMBEDDING

    return None


def find_blocks(
    model: nn.Module,
) -> list[tuple[str, nn.Module]]:
    """Find repeated blocks matching HF_BLOCK_PATTERNS."""
    results: list[tuple[str, nn.Module]] = []

    for name, module in model.named_modules():
        if any(pattern.match(name) for pattern in HF_BLOCK_PATTERNS):
            results.append((name, module))

    return results


def is_skip_target(name: str) -> bool:
    """Return True if the layer is an attention projection to skip.

    ``c_proj`` is ambiguous (``attn.c_proj`` vs ``mlp.c_proj`` in GPT-2),
    so it is only skipped when the FQN path contains an attention segment.
    """
    leaf = name.rsplit(".", maxsplit=1)[-1]
    if leaf in ATTENTION_PROJECTION_NAMES:
        return True
    if leaf == "c_proj":
        return bool(_ATTENTION_CONTEXT & frozenset(name.lower().split(".")))
    return False


def _fuse_batchnorm(model: nn.Module) -> None:
    """Recursively fuse Conv2d + BatchNorm2d pairs in-place (bottom-up)."""
    for _name, child in list(model.named_children()):
        _fuse_batchnorm(child)

    children = list(model.named_children())
    for i in range(len(children) - 1):
        name_a, mod_a = children[i]
        name_b, mod_b = children[i + 1]
        if isinstance(mod_a, nn.Conv2d) and isinstance(mod_b, nn.BatchNorm2d):
            fused = fuse_conv_bn_eval(mod_a, mod_b)
            setattr(model, name_a, fused)
            setattr(model, name_b, nn.Identity())


def prepare_model(model: nn.Module) -> nn.Module:
    """Prepare SMP model: eval, freeze, fuse BatchNorm where possible."""
    model.eval()
    _fuse_batchnorm(model)
    for param in model.parameters():
        param.requires_grad = False
    return model
