"""Calibration loop: feed data through model to collect observer statistics."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import islice
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import Tensor, nn
    from torch.utils.hooks import RemovableHandle

    from .observers import ObserverSpec


def run_calibration(
    model: nn.Module,
    dataset: Iterable[Any],
    observers: list[ObserverSpec],
    *,
    max_samples: int = 128,
) -> dict[str, dict[str, Tensor]]:
    """Run calibration data through the model and collect observer statistics.

    For each ObserverSpec, a fresh observer instance is created per
    target FQN and attached as a forward hook on that module.  The
    model is then run over the dataset under ``torch.no_grad`` and
    inference (``train(False)``) mode until ``max_samples`` samples
    have been processed, after which every hook is removed and each
    observer's ``get_stats()`` is collected into the returned mapping.

    When ``observers`` is empty (e.g. for the RTN algorithm which needs
    no calibration), the function short-circuits and returns ``{}``
    without consuming any samples from ``dataset``.

    Args:
        model: The model to calibrate.  It is temporarily switched to
            inference mode and its original ``training`` flag is
            restored on exit, even if a forward pass raises.
        dataset: Iterable of calibration samples.  Each sample is
            dispatched to ``model`` as ``model(**sample)`` for any
            ``Mapping`` (covers ``dict``, ``UserDict``, and HuggingFace
            ``BatchEncoding``), ``model(*sample)`` for tuples / lists,
            or ``model(sample)`` otherwise.
        observers: Observer specifications declaring which observer
            factory to attach at which target FQNs.  An empty list
            short-circuits the function.
        max_samples: Maximum number of samples to consume from
            ``dataset``.

    Returns:
        Mapping of ``{fqn: {stat_name: tensor}}``.  Multiple observers
        attached to the same FQN merge their stats into the same nested
        dict; later specs win on key collisions.

    Raises:
        KeyError: If a target FQN in any ObserverSpec is absent from
            the model's ``named_modules()``.
    """
    if not observers:
        return {}

    named_modules = dict(model.named_modules())
    attached: list[tuple[str, Any, RemovableHandle]] = []

    was_training = model.training
    model.train(mode=False)

    try:
        for spec in observers:
            for fqn in spec.targets:
                if fqn not in named_modules:
                    raise KeyError(
                        f"Observer target FQN {fqn!r} not found in model.",
                    )
                instance = spec.factory()
                handle = named_modules[fqn].register_forward_hook(instance)
                attached.append((fqn, instance, handle))

        with torch.no_grad():
            for sample in islice(dataset, max_samples):
                _forward(model, sample)

        result: dict[str, dict[str, Tensor]] = {}
        for fqn, instance, _handle in attached:
            result.setdefault(fqn, {}).update(instance.get_stats())
        return result
    finally:
        for _fqn, _instance, handle in attached:
            handle.remove()
        model.train(mode=was_training)


def _forward(model: nn.Module, sample: Any) -> None:
    """Dispatch one calibration sample to the model based on its type."""
    if isinstance(sample, Mapping):
        model(**sample)
    elif isinstance(sample, (list, tuple)):
        model(*sample)
    else:
        model(sample)
