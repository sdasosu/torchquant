"""Tests for rewriter registry lookup and custom registration."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torchquant._types import Algorithm
from torchquant.export import rewriter
from torchquant.export.runtime import UnsupportedExportError


class _CustomLinear(nn.Linear):
    """Linear subclass used to verify MRO lookup."""


class _CustomModule(nn.Module):
    """Fresh module type used to verify user registration."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones((2, 2)))


def _custom_rewriter(module: nn.Module, record) -> nn.Module:
    del record
    return module


def test_builtin_rewriters_are_registered() -> None:
    """Importing torchquant.export registers the three built-in rewriters."""
    assert rewriter.get_rewriter(nn.Linear(8, 4))
    assert rewriter.get_rewriter(nn.Conv2d(2, 2, 1))
    assert rewriter.get_rewriter(nn.Embedding(5, 4))


def test_get_rewriter_walks_mro() -> None:
    """A Linear subclass should inherit the built-in Linear rewriter."""
    assert rewriter.get_rewriter(_CustomLinear(8, 4)) is rewriter.get_rewriter(
        nn.Linear(8, 4)
    )


def test_unsupported_module_raises() -> None:
    """Unsupported modules should mention the extension registration API."""
    with pytest.raises(
        UnsupportedExportError, match=r"LSTM.*register_quantized_module"
    ):
        rewriter.get_rewriter(nn.LSTM(4, 4))


def test_conv_transpose_not_supported() -> None:
    """ConvTranspose2d should not match the Conv2d rewriter."""
    with pytest.raises(UnsupportedExportError, match=r"ConvTranspose2d"):
        rewriter.get_rewriter(nn.ConvTranspose2d(2, 2, 1))


def test_allowed_algorithms_match_builtin_types() -> None:
    """Built-in module types expose the planned support matrix."""
    assert rewriter.get_allowed_algorithms(nn.Linear(8, 4)) == frozenset(
        {Algorithm.RTN, Algorithm.GPTQ}
    )
    assert rewriter.get_allowed_algorithms(nn.Conv2d(2, 2, 1)) == frozenset(
        {Algorithm.RTN}
    )
    assert rewriter.get_allowed_algorithms(nn.Embedding(5, 4)) == frozenset(
        {Algorithm.RTN}
    )


def test_custom_registration_is_discoverable() -> None:
    """User-registered rewriters should be returned by registry lookups."""
    rewriter.register_quantized_module(
        _CustomModule,
        _custom_rewriter,
        frozenset({Algorithm.RTN}),
    )
    module = _CustomModule()
    assert rewriter.get_rewriter(module) is _custom_rewriter
    assert rewriter.get_allowed_algorithms(module) == frozenset({Algorithm.RTN})
