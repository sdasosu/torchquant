"""Test-only oracle recorder for quantizer integer weights."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch import Tensor, nn

_RECORDS: ContextVar[dict[str, torch.Tensor] | None] = ContextVar(
    "torchquant_oracle_records",
    default=None,
)
_FQN_ATTR = "_torchquant_oracle_fqn"
_MISSING = object()


@contextmanager
def recording() -> Iterator[dict[str, Tensor]]:
    """Capture quantizer integer weights for the duration of a test."""
    records: dict[str, Tensor] = {}
    token = _RECORDS.set(records)
    try:
        yield records
    finally:
        _RECORDS.reset(token)


def is_recording() -> bool:
    """Return whether oracle recording is active in the current context."""
    return _RECORDS.get() is not None


def record(fqn: str, q_int: Tensor) -> None:
    """Store a quantizer-produced integer weight tensor when recording is active."""
    records = _RECORDS.get()
    if records is None:
        return
    records[fqn] = q_int.detach().to(torch.int32).clone()


@contextmanager
def bound_fqn(module: nn.Module, fqn: str) -> Iterator[None]:
    """Temporarily bind a module FQN so quantizers can tag oracle records."""
    previous = getattr(module, _FQN_ATTR, _MISSING)
    setattr(module, _FQN_ATTR, fqn)
    try:
        yield
    finally:
        if previous is _MISSING:
            delattr(module, _FQN_ATTR)
        else:
            setattr(module, _FQN_ATTR, previous)


def get_bound_fqn(module: nn.Module) -> str | None:
    """Return the currently bound FQN for a quantizer-owned module."""
    fqn = getattr(module, _FQN_ATTR, None)
    return fqn if isinstance(fqn, str) else None
