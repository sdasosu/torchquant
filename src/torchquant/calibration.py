"""Calibration loop: feed data through model to collect observer statistics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import Tensor, nn

    from .observers import ObserverSpec


def run_calibration(
    model: nn.Module,
    dataset: Iterable[Any],
    observers: list[ObserverSpec],
    *,
    max_samples: int = 128,
) -> dict[str, dict[str, Tensor]]:
    """Run calibration data through the model and collect observer statistics.

    Args:
        model: The model to calibrate.
        dataset: Iterable of calibration samples.
        observers: Observer specifications to attach.
        max_samples: Maximum number of samples to process.

    Returns:
        Mapping of {fqn: {stat_name: tensor}} collected from observers.
    """
    raise NotImplementedError
