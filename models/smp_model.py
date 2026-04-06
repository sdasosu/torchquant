"""
smp_model.py — ModelWrapper for Sequential / MLP-style (SMP) models.

Supports:
  - Plain sequential models (nn.Sequential)
  - MLP-style models with repeated blocks (GPT-2, LLaMA FFN, BERT FFN, etc.)
  - Hugging Face transformer FFN sub-modules
  - Any model whose quantizable units are nn.Linear layers

Architecture assumptions:
  - Quantizable layers are nn.Linear (or subclasses)
  - Blocks are identified by repeated module patterns (e.g. h.0, h.1, ...)
  - SMP = no cross-layer attention dependency needed at quantization time
"""

import re
from typing import Iterator, Tuple, Dict, List, Optional, Any
import torch
import torch.nn as nn

from .base_model import BaseModelWrapper
from ..config import QuantConfig


# ---------------------------------------------------------------------------
# Layer-type tags (consumed by pipeline.py to pick the right quantizer)
# ---------------------------------------------------------------------------

LAYER_TYPE_WEIGHT_ONLY   = "weight_only"
LAYER_TYPE_ACTIVATION    = "activation"
LAYER_TYPE_EMBEDDING     = "embedding"
LAYER_TYPE_SKIP          = "skip"


# ---------------------------------------------------------------------------
# Known block-name patterns in popular SMP / FFN-based architectures
# ---------------------------------------------------------------------------

# Hugging Face naming conventions for transformer FFN sub-blocks
_HF_BLOCK_PATTERNS = [
    r"^transformer\.h\.\d+$",          # GPT-2
    r"^model\.layers\.\d+$",           # LLaMA / Mistral / Qwen
    r"^encoder\.layer\.\d+$",          # BERT / RoBERTa encoder
    r"^decoder\.layers\.\d+$",         # BART / T5 decoder
    r"^layers\.\d+$",                  # generic flat stack
    r"^blocks\.\d+$",                  # generic flat stack (alt)
    r"^\d+$",                          # plain nn.Sequential index
]

# Linear sub-layers to skip inside blocks (projection layers handled
# by attention_model.py, not here)
_ATTENTION_PROJECTION_NAMES = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "query", "key", "value", "out_proj",
    "query_key_value",                  # Falcon / merged QKV
    "c_attn", "c_proj",                # GPT-2 style
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SMPModelWrapper(BaseModelWrapper):
    """
    Wraps Sequential / MLP-style PyTorch models for targeted quantization.

    Usage:
        wrapper = SMPModelWrapper(model, config)
        for name, layer in wrapper.iter_quantizable_layers():
            quantizer = pipeline.get_quantizer(wrapper.get_layer_type(layer))
            quantizer.quantize(layer)
    """

    def __init__(self, model: nn.Module, config: QuantConfig):
        super().__init__(model, config)
        self._layer_cache: Optional[List[Tuple[str, nn.Module]]] = None

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def iter_quantizable_layers(self) -> Iterator[Tuple[str, nn.Module]]:
        """
        Yields (dotted_name, layer) for every quantizable nn.Linear.

        Skips:
          - Attention projection layers (handled by attention_model.py)
          - Layers whose names are in config.skip_layers
          - Layers smaller than config.min_layer_size (if set)
        """
        if self._layer_cache is not None:
            yield from self._layer_cache
            return

        cache = []
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if self._should_skip(name, module):
                continue
            cache.append((name, module))

        self._layer_cache = cache
        yield from cache

    def get_layer_type(self, layer: nn.Module) -> str:
        """
        Returns quantization strategy tag for a given layer.

        Rules (in priority order):
          1. Embedding → skip (handled separately or not at all)
          2. Config says weight_only → weight_only
          3. Config says dynamic activation → activation
          4. Default → weight_only
        """
        if isinstance(layer, nn.Embedding):
            return LAYER_TYPE_EMBEDDING

        if self.config.weight_only:
            return LAYER_TYPE_WEIGHT_ONLY

        if self.config.quantize_activations:
            return LAYER_TYPE_ACTIVATION

        return LAYER_TYPE_WEIGHT_ONLY

    def pre_quantize(self):
        """Prepare model: eval mode, freeze params, fuse batchnorm if present."""
        self.model.eval()
        self.freeze()
        self._fuse_batchnorm()
        self._layer_cache = None  # invalidate cache after any structural changes

    def post_quantize(self):
        """Post-quantization cleanup."""
        # Nothing model-specific needed for SMP; subclasses can extend.
        pass

    # ------------------------------------------------------------------
    # SMP-specific public helpers
    # ------------------------------------------------------------------

    def get_blocks(self) -> List[Tuple[str, nn.Module]]:
        """
        Returns top-level repeated blocks (e.g. transformer layers).

        Useful for block-wise quantization (GPTQ-style) where you process
        one block at a time with calibration data.

        Returns list of (name, module) sorted by block index.
        """
        blocks = []
        for name, module in self.model.named_modules():
            if self._is_block(name):
                blocks.append((name, module))
        return blocks

    def iter_layers_in_block(
        self, block: nn.Module, block_name: str = ""
    ) -> Iterator[Tuple[str, nn.Module]]:
        """
        Yields quantizable nn.Linear layers within a single block.

        Args:
            block:      the block module (e.g. a single transformer layer)
            block_name: dotted prefix for constructing full layer names
        """
        for rel_name, module in block.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            full_name = f"{block_name}.{rel_name}" if block_name else rel_name
            if self._should_skip(full_name, module):
                continue
            yield full_name, module

    def get_ffn_layers(self, block: nn.Module, block_name: str = "") -> List[Tuple[str, nn.Module]]:
        """
        Returns only the FFN (MLP) linear layers inside a block.
        Excludes attention projections even if present.
        """
        return [
            (n, m) for n, m in self.iter_layers_in_block(block, block_name)
            if not self._is_attention_projection(n)
        ]

    def summary(self) -> Dict[str, Any]:
        """
        Returns a dict describing what will be quantized.

        Handy for debugging before running the full pipeline.
        """
        layers = list(self.iter_quantizable_layers())
        total_params = sum(
            m.weight.numel() for _, m in layers if hasattr(m, "weight")
        )
        return {
            "num_quantizable_layers": len(layers),
            "total_quantizable_params": total_params,
            "layer_names": [n for n, _ in layers],
            "blocks": [n for n, _ in self.get_blocks()],
            "config": {
                "bits": self.config.bits,
                "weight_only": self.config.weight_only,
                "quantize_activations": self.config.quantize_activations,
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _should_skip(self, name: str, module: nn.Linear) -> bool:
        """True if this layer should not be quantized."""
        # User-specified skip list
        if name in (self.config.skip_layers or []):
            return True

        # Attention projections: let attention_model.py handle those
        if self._is_attention_projection(name):
            return True

        # Too small to bother (e.g. tiny classifier heads)
        min_size = getattr(self.config, "min_layer_size", 0)
        if min_size and module.weight.numel() < min_size:
            return True

        return False

    def _is_attention_projection(self, name: str) -> bool:
        """True if the layer name matches a known attention projection."""
        leaf = name.split(".")[-1]
        return leaf in _ATTENTION_PROJECTION_NAMES

    def _is_block(self, name: str) -> bool:
        """True if the module name looks like a top-level repeated block."""
        for pattern in _HF_BLOCK_PATTERNS:
            if re.fullmatch(pattern, name):
                return True
        return False

    def _fuse_batchnorm(self):
        """
        Fuse Conv+BN and Linear+BN pairs in-place where possible.
        No-op if no BN layers are found (common in transformers).
        """
        try:
            torch.quantization.fuse_modules(
                self.model,
                self._find_fuseable_pairs(),
                inplace=True,
            )
        except Exception:
            # fuse_modules is best-effort; not all models support it
            pass

    def _find_fuseable_pairs(self) -> List[List[str]]:
        """
        Finds [Linear, BatchNorm] or [Conv, BatchNorm] sequential pairs
        for fusion. Returns list of name-lists suitable for fuse_modules.
        """
        pairs = []
        children = list(self.model.named_children())
        for i in range(len(children) - 1):
            name_a, mod_a = children[i]
            name_b, mod_b = children[i + 1]
            if isinstance(mod_a, (nn.Linear, nn.Conv2d)) and isinstance(mod_b, nn.BatchNorm1d):
                pairs.append([name_a, name_b])
        return pairs