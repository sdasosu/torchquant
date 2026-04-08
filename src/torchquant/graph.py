"""Model introspection: discover quantizable nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn

    from ._types import LayerKind
    from .adapters import AdapterFns


@dataclass(frozen=True)
class QuantNode:
    """A discovered quantizable layer."""

    fqn: str
    kind: LayerKind
    param_count: int


def find_quantizable_nodes(
    model: nn.Module,
    adapter: AdapterFns | None = None,
) -> list[QuantNode]:
    """Walk the model graph and return all quantizable nodes.

    Args:
        model: The PyTorch model to introspect.
        adapter: Model-family-specific functions. Auto-detected if None.

    Returns:
        List of quantizable nodes found in the model.
    """
    raise NotImplementedError


def resolve_modules(
    model: nn.Module,
    nodes: list[QuantNode],
) -> dict[str, nn.Module]:
    """Map discovered nodes back to their actual nn.Module instances.

    QuantNode intentionally does not store a module reference to stay
    lightweight and serializable. This function bridges the gap between
    the discovery stage (graph) and the quantization stage (quantizers)
    by resolving FQN strings to live module objects.

    Args:
        model: The original model that was introspected.
        nodes: Nodes returned by find_quantizable_nodes().

    Returns:
        Dict mapping FQN to the corresponding nn.Module.

    Raises:
        KeyError: If a node's FQN does not exist in the model.
    """
    raise NotImplementedError
