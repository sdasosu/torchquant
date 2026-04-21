"""Module-level quantized rewriter registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import torch

from torchquant.export.runtime import UnsupportedExportError
from torchquant.registry import QuantRecord

if TYPE_CHECKING:
    from torchquant._types import Algorithm


type RewriteFn = Callable[[torch.nn.Module, QuantRecord], torch.nn.Module]

_REWRITERS: dict[type[torch.nn.Module], RewriteFn] = {}
_ALLOWED_ALGORITHMS: dict[type[torch.nn.Module], frozenset[Algorithm]] = {}


def register_quantized_module(
    module_type: type[torch.nn.Module],
    rewriter: RewriteFn,
    allowed_algorithms: frozenset[Algorithm],
) -> None:
    """Register a rewriter for a module type and its supported algorithms."""
    _REWRITERS[module_type] = rewriter
    _ALLOWED_ALGORITHMS[module_type] = allowed_algorithms


def _get_registered_type(module: torch.nn.Module) -> type[torch.nn.Module]:
    for candidate in type(module).__mro__:
        if not issubclass(candidate, torch.nn.Module):
            continue
        module_type = cast("type[torch.nn.Module]", candidate)
        if module_type in _REWRITERS:
            return module_type
    raise UnsupportedExportError(
        "Unsupported export module type "
        f"{type(module).__name__}; register a rewriter with "
        "register_quantized_module()."
    )


def get_rewriter(module: torch.nn.Module) -> RewriteFn:
    """Return the nearest registered rewriter for a module instance."""
    return _REWRITERS[_get_registered_type(module)]


def get_allowed_algorithms(module: torch.nn.Module) -> frozenset[Algorithm]:
    """Return the supported algorithms for the nearest registered module type."""
    return _ALLOWED_ALGORITHMS[_get_registered_type(module)]
