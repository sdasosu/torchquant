"""Tests for the Embedding rewriter."""

from __future__ import annotations

import torch

from torchquant.export.rewriters.embedding import rewrite_embedding
from torchquant.export.runtime import QuantizedEmbedding

from ._rewriter_fixtures import make_embedding_record


def test_embedding_rewriter_preserves_quantization_and_structure() -> None:
    """Embedding rewrite should keep q_int and embedding metadata."""
    module, record, oracle_int = make_embedding_record(
        bits=4,
        group_size=4,
        symmetric=False,
    )
    rewritten = rewrite_embedding(module, record)

    assert isinstance(rewritten, QuantizedEmbedding)
    assert torch.equal(rewritten.int_weight, oracle_int)
    assert rewritten.padding_idx == module.padding_idx
    assert rewritten.max_norm == module.max_norm
    assert rewritten.norm_type == module.norm_type
    assert rewritten.scale_grad_by_freq == module.scale_grad_by_freq
    assert rewritten.sparse == module.sparse

    sample_input = torch.tensor([[0, 1, 5], [2, 4, 3]], dtype=torch.int64)
    expected = torch.nn.functional.embedding(
        sample_input,
        record.result.quantized_weight,
        padding_idx=module.padding_idx,
        max_norm=module.max_norm,
        norm_type=module.norm_type,
        scale_grad_by_freq=module.scale_grad_by_freq,
        sparse=module.sparse,
    )
    atol = rewritten.scales.max().item() * 1e-3
    assert torch.allclose(rewritten(sample_input), expected, rtol=0, atol=atol)
