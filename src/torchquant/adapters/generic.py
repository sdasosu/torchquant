"""Fallback adapter for plain nn.Module / nn.Sequential."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn

    from torchquant._types import LayerKind


def classify_module(name: str, module: nn.Module) -> LayerKind | None:
    """Classify a module by isinstance checks.

    Args:
        name: Fully-qualified module name.
        module: The module to classify.

    Returns:
        LayerKind if quantizable, None otherwise.
    """
    raise NotImplementedError


def find_blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return top-level child modules as blocks.

    Args:
        model: The model to inspect.

    Returns:
        List of (name, module) pairs for each top-level block.
    """
    raise NotImplementedError


def is_skip_target(name: str) -> bool:
    """Return True if the layer should be skipped.

    The generic adapter does not skip anything by default.

    Args:
        name: Fully-qualified module name.
    """
    raise NotImplementedError


def prepare_model(model: nn.Module) -> nn.Module:
    """Prepare model for quantization: eval mode, freeze parameters.

    Args:
        model: The model to prepare.

    Returns:
        The prepared model (same object, mutated in-place).
    """
    raise NotImplementedError
