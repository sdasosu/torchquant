"""Quantization record store and model rebuilding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch import Tensor, nn

    from ._types import LayerKind
    from .config import QuantScheme
    from .quantizers import QuantResult


@dataclass
class QuantRecord:
    """Stores quantization results for a single layer."""

    fqn: str
    kind: LayerKind
    scheme: QuantScheme
    result: QuantResult
    original_bias: Tensor | None = None


class QuantRegistry:
    """Central store for quantization records."""

    def __init__(self) -> None:
        self._records: dict[str, QuantRecord] = {}

    def add(self, record: QuantRecord) -> None:
        """Add or overwrite a quantization record."""
        self._records[record.fqn] = record

    def __iter__(self) -> Iterator[tuple[str, QuantRecord]]:
        """Iterate over (fqn, record) pairs."""
        yield from self._records.items()

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self._records)


def rebuild_model(model: nn.Module, registry: QuantRegistry) -> nn.Module:
    """Create a new model with quantized weights from the registry.

    Deep-copies the original model and writes quantized weights back
    from registry records. Never mutates the original.

    Args:
        model: The original (unquantized) model.
        registry: Registry containing quantization results.

    Returns:
        A new model with quantized weights applied.
    """
    raise NotImplementedError
