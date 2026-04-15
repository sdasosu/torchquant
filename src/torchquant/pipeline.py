"""End-to-end quantization pipeline."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from ._types import Algorithm
from .adapters import get_adapter
from .calibration import run_calibration
from .graph import find_quantizable_nodes, resolve_modules
from .observers import get_observer_specs
from .registry import QuantRecord, QuantRegistry, apply_records_inplace
from .rules import RuleEngine, default_rule

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import Tensor, nn

    from .config import QuantRecipe
    from .observers import ObserverSpec
    from .quantizers import QuantizerFn

QUANTIZER_MAP: dict[Algorithm, QuantizerFn] = {}


def _init_quantizer_map() -> None:
    """Lazily populate QUANTIZER_MAP with algorithm implementations."""
    if QUANTIZER_MAP:
        return
    from .quantizers import awq, gptq, rtn, smoothquant

    QUANTIZER_MAP.update(
        {
            Algorithm.RTN: rtn.quantize_layer,
            Algorithm.GPTQ: gptq.quantize_layer,
            Algorithm.AWQ: awq.quantize_layer,
            Algorithm.SMOOTHQUANT: smoothquant.quantize_layer,
        }
    )


def quantize_model(
    model: nn.Module,
    recipe: QuantRecipe,
    calibration_data: Iterable[Any] | None = None,
) -> nn.Module:
    """Quantize a model end-to-end.

    Pipeline stages:
        1. Auto-detect model adapter
        2. Prepare model (eval, freeze, fuse BN)
        3. Discover quantizable nodes
        4. Assign schemes via rule engine
        5. Run calibration if needed
        6. Apply quantizers
        7. Rebuild model with quantized weights

    Args:
        model: The PyTorch model to quantize.
        recipe: Quantization recipe describing what and how to quantize.
        calibration_data: Optional calibration dataset for methods that need it.

    Returns:
        A new model with quantized weights.
    """
    _init_quantizer_map()
    working = copy.deepcopy(model)
    adapter = get_adapter(working)
    adapter.prepare_model(working)
    nodes = find_quantizable_nodes(working, adapter)
    decisions = RuleEngine([default_rule]).decide(nodes, recipe)

    if not decisions:
        return working

    algorithm_targets: dict[Algorithm, set[str]] = {}
    all_specs: list[ObserverSpec] = []
    required_calibration_algorithms: set[Algorithm] = set()

    for decision in decisions:
        algorithm_targets.setdefault(decision.scheme.algorithm, set()).add(
            decision.node.fqn,
        )

    for algorithm, targets in algorithm_targets.items():
        specs = get_observer_specs(algorithm, frozenset(targets))
        if specs:
            required_calibration_algorithms.add(algorithm)
        all_specs.extend(specs)

    if all_specs:
        if calibration_data is None:
            algorithm_names = ", ".join(
                sorted(algorithm.name for algorithm in required_calibration_algorithms)
            )
            raise ValueError(
                "calibration_data is required for calibration-based algorithms: "
                f"{algorithm_names}.",
            )
        stats: dict[str, dict[str, Tensor]] = run_calibration(
            working,
            calibration_data,
            all_specs,
            max_samples=recipe.calibration_samples,
        )
    else:
        stats = {}

    modules = resolve_modules(working, [decision.node for decision in decisions])
    registry = QuantRegistry()

    for decision in decisions:
        module = modules[decision.node.fqn]
        layer_stats = stats.get(decision.node.fqn, {})
        result = QUANTIZER_MAP[decision.scheme.algorithm](
            module,
            decision.scheme,
            layer_stats,
        )
        registry.add(
            QuantRecord(
                fqn=decision.node.fqn,
                kind=decision.node.kind,
                scheme=decision.scheme,
                result=result,
            )
        )

    apply_records_inplace(working, registry)
    return working
