"""Tests for graph discovery and module resolution."""

from __future__ import annotations

import pytest
from torch import nn

from torchquant._types import LayerKind
from torchquant.adapters import AdapterFns, smp
from torchquant.graph import QuantNode, find_quantizable_nodes, resolve_modules


class _GenericGraphModel(nn.Module):
    """Small model with quantizable and non-quantizable modules."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(5, 4)
        self.activation = nn.ReLU()
        self.projection = nn.Linear(4, 2)


class _SmpSkipBlock(nn.Module):
    """Block that exposes SMP skip-target behavior."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.c_proj = nn.Linear(4, 4)


class _SmpSkipModel(nn.Module):
    """Model with repeated blocks that can be traversed by the SMP adapter."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_SmpSkipBlock()])


def _smp_adapter() -> AdapterFns:
    return AdapterFns(
        classify_module=smp.classify_module,
        find_blocks=smp.find_blocks,
        is_skip_target=smp.is_skip_target,
        prepare_model=smp.prepare_model,
    )


def test_find_quantizable_nodes_auto_detects_generic_layers() -> None:
    """Auto-detection returns quantizable nodes with stable FQNs and counts."""
    nodes = find_quantizable_nodes(_GenericGraphModel())

    assert nodes == [
        QuantNode(fqn="embedding", kind=LayerKind.EMBEDDING, param_count=20),
        QuantNode(fqn="projection", kind=LayerKind.LINEAR, param_count=10),
    ]


def test_find_quantizable_nodes_respects_smp_skip_targets() -> None:
    """Explicit SMP discovery skips attention projections but keeps other linears."""
    nodes = find_quantizable_nodes(_SmpSkipModel(), adapter=_smp_adapter())

    assert nodes == [
        QuantNode(fqn="layers.0.c_proj", kind=LayerKind.LINEAR, param_count=20),
    ]


def test_resolve_modules_returns_requested_subset() -> None:
    """Resolution maps discovered FQNs back to the live module objects."""
    model = _GenericGraphModel()
    nodes = find_quantizable_nodes(model)

    modules = resolve_modules(model, [nodes[1]])

    assert modules == {"projection": model.projection}


def test_resolve_modules_raises_for_unknown_fqn() -> None:
    """Resolution fails fast when a node points at a missing module."""
    model = _GenericGraphModel()
    missing_node = QuantNode(fqn="missing", kind=LayerKind.LINEAR, param_count=0)

    with pytest.raises(KeyError, match="missing"):
        resolve_modules(model, [missing_node])
