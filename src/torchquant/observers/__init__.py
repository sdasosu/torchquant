"""Hook-based statistics collectors for quantization calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from torchquant._types import Algorithm


@dataclass(frozen=True)
class ObserverSpec:
    """Declares which observer to attach and where.

    Args:
        factory: Callable that creates a new observer instance.
        targets: Set of FQN patterns specifying where to attach.
    """

    factory: Callable[[], Any]
    targets: frozenset[str]


def get_observer_specs(
    algorithm: Algorithm,
    targets: frozenset[str],
) -> list[ObserverSpec]:
    """Return the observer specs required by a quantization algorithm.

    Bridges the gap between the recipe layer (which declares an Algorithm)
    and the calibration layer (which needs concrete ObserverSpecs).

    Mapping:
        RTN         → [] (no calibration needed)
        GPTQ        → [HessianObserver]
        AWQ         → [AWQObserver]
        SMOOTHQUANT → [SmoothQuantObserver]

    Args:
        algorithm: The quantization algorithm from QuantScheme.
        targets: FQN patterns for layers that need observation.

    Returns:
        List of ObserverSpecs to pass to run_calibration().
    """
    raise NotImplementedError
