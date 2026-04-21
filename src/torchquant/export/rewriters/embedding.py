"""Embedding module rewriter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from torchquant._types import Algorithm
from torchquant.export._recover import recover_int_weight
from torchquant.export.rewriter import register_quantized_module
from torchquant.export.runtime import QuantizedEmbedding
from torchquant.export.runtime.embedding import EmbeddingSpec

if TYPE_CHECKING:
    from torchquant.registry import QuantRecord


def rewrite_embedding(module: nn.Module, record: QuantRecord) -> nn.Module:
    """Rewrite an Embedding module into QuantizedEmbedding."""
    if not isinstance(module, nn.Embedding):
        raise TypeError(f"Expected nn.Embedding, got {type(module).__name__}.")
    int_weight, scales, zero_points = recover_int_weight(
        record.result,
        record.scheme,
        tuple(module.weight.shape),
        fqn=record.fqn,
    )
    return QuantizedEmbedding(
        int_weight=int_weight,
        scales=scales,
        zero_points=zero_points,
        scheme=record.scheme,
        spec=EmbeddingSpec(
            num_embeddings=module.num_embeddings,
            embedding_dim=module.embedding_dim,
            padding_idx=module.padding_idx,
            max_norm=module.max_norm,
            norm_type=module.norm_type,
            scale_grad_by_freq=module.scale_grad_by_freq,
            sparse=module.sparse,
        ),
        weight_dtype=record.result.original_weight.dtype,
    )


register_quantized_module(
    nn.Embedding,
    rewrite_embedding,
    frozenset({Algorithm.RTN}),
)
