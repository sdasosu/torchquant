"""Model-family adapter dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn

    from torchquant._types import LayerKind


@dataclass(frozen=True)
class AdapterFns:
    """Bundle of model-family-specific functions.

    Each adapter module (generic, smp, llm) exports these four functions.
    The pipeline calls get_adapter() to auto-detect and bundle them.
    """

    classify_module: Callable[[str, nn.Module], LayerKind | None]
    find_blocks: Callable[[nn.Module], list[tuple[str, nn.Module]]]
    is_skip_target: Callable[[str], bool]
    prepare_model: Callable[[nn.Module], nn.Module]


def get_adapter(model: nn.Module) -> AdapterFns:
    """Auto-detect model family and return the matching function bundle.

    Detection heuristics (in priority order):
        1. If model has attention projections (q_proj, k_proj, etc.) → llm adapter
        2. If model has named sequential blocks → smp adapter
        3. Otherwise → generic adapter

    Args:
        model: The PyTorch model to inspect.

    Returns:
        An AdapterFns bundle for the detected model family.
    """
    raise NotImplementedError
