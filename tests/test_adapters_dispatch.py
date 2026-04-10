"""Tests for adapter auto-dispatch."""

from __future__ import annotations

from torch import nn

from torchquant.adapters import generic, get_adapter, llm, smp


class _SelfAttention(nn.Module):
    """Tiny attention block with canonical projection names."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)
        self.o_proj = nn.Linear(4, 4)


class _FeedForwardBlock(nn.Module):
    """Tiny feed-forward block that matches SMP block layouts."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 4)


class _TransformerLayer(nn.Module):
    """Transformer-like layer with attention and MLP submodules."""

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _SelfAttention()
        self.mlp = _FeedForwardBlock()


class _LlmLikeModel(nn.Module):
    """Model whose structure should select the LLM adapter."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TransformerLayer()])


class _SmpLikeModel(nn.Module):
    """Model whose repeated blocks should select the SMP adapter."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_FeedForwardBlock()])


class _GenericModel(nn.Module):
    """Model without block or attention patterns."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.decoder = nn.Linear(4, 4)


def test_get_adapter_prefers_llm_over_smp() -> None:
    """LLM adapter wins when attention projections are present."""
    adapter = get_adapter(_LlmLikeModel())

    assert adapter.classify_module is llm.classify_module
    assert adapter.find_blocks is llm.find_blocks
    assert adapter.is_skip_target is llm.is_skip_target
    assert adapter.prepare_model is llm.prepare_model


def test_get_adapter_uses_smp_when_block_patterns_match() -> None:
    """SMP adapter is selected when transformer block patterns match."""
    adapter = get_adapter(_SmpLikeModel())

    assert adapter.classify_module is smp.classify_module
    assert adapter.find_blocks is smp.find_blocks
    assert adapter.is_skip_target is smp.is_skip_target
    assert adapter.prepare_model is smp.prepare_model


def test_get_adapter_falls_back_to_generic() -> None:
    """Generic adapter is the fallback when no heuristics match."""
    adapter = get_adapter(_GenericModel())

    assert adapter.classify_module is generic.classify_module
    assert adapter.find_blocks is generic.find_blocks
    assert adapter.is_skip_target is generic.is_skip_target
    assert adapter.prepare_model is generic.prepare_model
