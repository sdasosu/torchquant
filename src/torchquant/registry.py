"""Quantization record store and model rebuilding."""

from __future__ import annotations

import copy
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


def apply_records_inplace(model: nn.Module, registry: QuantRegistry) -> None:
    """Write quantized weights into a caller-owned model in-place.

    This is an internal helper shared by ``rebuild_model`` and the pipeline.
    User code should prefer ``rebuild_model`` unless it already owns a
    throwaway deep copy of ``model``.

    Args:
        model: Model whose modules will receive quantized weights.
        registry: Registry containing quantization results.

    Raises:
        KeyError: If a record FQN does not exist in ``model``.
    """
    for fqn, record in registry:
        try:
            target = model.get_submodule(fqn)
        except AttributeError as error:
            raise KeyError(
                f"Quantized module FQN {fqn!r} not found in model."
            ) from error
        target.get_parameter("weight").data.copy_(record.result.quantized_weight)


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
    new_model = copy.deepcopy(model)
    apply_records_inplace(new_model, registry)
    return new_model
