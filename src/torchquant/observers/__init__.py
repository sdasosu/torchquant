"""Hook-based statistics collectors for quantization calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from torchquant._types import Algorithm

from . import awq, hessian, smoothquant

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class ObserverSpec:
    """Declares which observer to attach and where.

    Args:
        factory: Callable that creates a new observer instance.
        targets: Set of FQN patterns specifying where to attach.
    """

    factory: Callable[[], Any]
    targets: frozenset[str]


_FACTORIES: dict[Algorithm, Callable[[], Any] | None] = {
    Algorithm.RTN: None,
    Algorithm.GPTQ: hessian.create,
    Algorithm.AWQ: awq.create,
    Algorithm.SMOOTHQUANT: smoothquant.create,
}


def get_observer_specs(
    algorithm: Algorithm,
    targets: frozenset[str],
) -> list[ObserverSpec]:
    """Return the observer specs required by a quantization algorithm.

    Bridges the gap between the recipe layer (which declares an Algorithm)
    and the calibration layer (which needs concrete ObserverSpecs).

    Mapping:
        RTN         -> [] (no calibration needed)
        GPTQ        -> [HessianObserver]
        AWQ         -> [AWQObserver]
        SMOOTHQUANT -> [SmoothQuantObserver]

    The factories are called with zero arguments, so the resulting
    observers use their default ``channel_dim``.  This default
    (``-1``) matches the LLM linear-layer convention
    ``(batch, seq_len, in_features)``.  CNN layers (``(N, C, H, W)``)
    require ``channel_dim=1`` instead; callers that need to mix
    layouts should construct ``ObserverSpec`` instances directly
    rather than relying on this helper.

    Args:
        algorithm: The quantization algorithm from QuantScheme.
        targets: FQN patterns for layers that need observation.

    Returns:
        List of ObserverSpecs to pass to run_calibration().  Returns an
        empty list for algorithms (such as RTN) that need no calibration.

    Raises:
        KeyError: If ``algorithm`` is not a recognised member of
            ``_FACTORIES`` (i.e. a new ``Algorithm`` value was added
            without a corresponding entry here).
    """
    factory = _FACTORIES[algorithm]
    if factory is None:
        return []
    return [ObserverSpec(factory=factory, targets=targets)]
