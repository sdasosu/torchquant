"""Model export via module-level rewriting."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from .rewriter import get_allowed_algorithms, get_rewriter
from .runtime import UnsupportedExportError

if TYPE_CHECKING:
    from torch import nn

    from torchquant.registry import QuantRecord, QuantRegistry


def export_pt2e(model: nn.Module, registry: QuantRegistry) -> nn.Module:
    """Rewrite quantized layers into runtime modules and return a plain nn.Module."""
    working = copy.deepcopy(model)
    errors = _collect_preflight_errors(working, registry)
    if errors:
        raise UnsupportedExportError(
            "Unsupported export targets:\n- " + "\n- ".join(errors)
        )

    rewritten_root = working
    for fqn, record in registry:
        target = _get_target_module(rewritten_root, fqn)
        rewritten = get_rewriter(target)(target, record)
        if not fqn:
            rewritten_root = rewritten
            continue
        parent, attribute = _get_parent_and_attribute(rewritten_root, fqn)
        setattr(parent, attribute, rewritten)
    return rewritten_root


def _collect_preflight_errors(model: nn.Module, registry: QuantRegistry) -> list[str]:
    errors: list[str] = []
    for fqn, record in registry:
        target_name = fqn or "<root>"
        try:
            module = _get_target_module(model, fqn)
            get_rewriter(module)
            allowed_algorithms = get_allowed_algorithms(module)
        except UnsupportedExportError as error:
            errors.append(f"{target_name}: {error}")
            continue
        if record.scheme.algorithm not in allowed_algorithms:
            errors.append(
                _format_algorithm_error(target_name, module, record, allowed_algorithms)
            )
    return errors


def _get_target_module(model: nn.Module, fqn: str) -> nn.Module:
    if not fqn:
        return model
    return model.get_submodule(fqn)


def _get_parent_and_attribute(model: nn.Module, fqn: str) -> tuple[nn.Module, str]:
    if "." not in fqn:
        return model, fqn
    parent_path, attribute = fqn.rsplit(".", 1)
    return model.get_submodule(parent_path), attribute


def _format_algorithm_error(
    fqn: str,
    module: nn.Module,
    record: QuantRecord,
    allowed_algorithms: frozenset,
) -> str:
    algorithm_name = record.scheme.algorithm.name
    module_name = type(module).__name__
    if algorithm_name in {"AWQ", "SMOOTHQUANT"}:
        return (
            f"{fqn}: {module_name} with {algorithm_name} is not exportable via the "
            "Phase 6 module-level rewriter because it requires migration-scale "
            "metadata and cross-layer folding."
        )
    allowed = ", ".join(sorted(algorithm.name for algorithm in allowed_algorithms))
    return (
        f"{fqn}: {module_name} does not support {algorithm_name} export; "
        f"allowed algorithms: {allowed}."
    )
