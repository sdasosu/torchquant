"""Model-family adapter dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchquant._types import LayerKind

from . import generic, llm, smp

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn


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


def _llm_detected(model: nn.Module) -> bool:
    """Return True if the model contains LLM-style attention projections."""
    attention_kinds = {LayerKind.ATTENTION_QKV, LayerKind.ATTENTION_OUT}
    return any(
        llm.classify_module(name, module) in attention_kinds
        for name, module in model.named_modules()
    )


def _bundle_adapter(
    classify_module: Callable[[str, nn.Module], LayerKind | None],
    find_blocks: Callable[[nn.Module], list[tuple[str, nn.Module]]],
    is_skip_target: Callable[[str], bool],
    prepare_model: Callable[[nn.Module], nn.Module],
) -> AdapterFns:
    """Construct an AdapterFns bundle from adapter module functions."""
    return AdapterFns(
        classify_module=classify_module,
        find_blocks=find_blocks,
        is_skip_target=is_skip_target,
        prepare_model=prepare_model,
    )


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
    if _llm_detected(model):
        return _bundle_adapter(
            classify_module=llm.classify_module,
            find_blocks=llm.find_blocks,
            is_skip_target=llm.is_skip_target,
            prepare_model=llm.prepare_model,
        )

    if smp.find_blocks(model):
        return _bundle_adapter(
            classify_module=smp.classify_module,
            find_blocks=smp.find_blocks,
            is_skip_target=smp.is_skip_target,
            prepare_model=smp.prepare_model,
        )

    return _bundle_adapter(
        classify_module=generic.classify_module,
        find_blocks=generic.find_blocks,
        is_skip_target=generic.is_skip_target,
        prepare_model=generic.prepare_model,
    )
