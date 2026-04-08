"""Min/max activation range observer."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


class MinMaxObserver:
    """Tracks per-channel min/max activation ranges.

    Collects running min/max statistics from forward pass activations.
    Used as the default observer for static activation quantization.
    """

    def __init__(self) -> None:
        self._min: Tensor | None = None
        self._max: Tensor | None = None

    def __call__(
        self,
        module: object,
        inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        """Forward hook: update min/max from output tensor."""
        raise NotImplementedError

    def get_stats(self) -> dict[str, Tensor]:
        """Return collected statistics.

        Returns:
            Dict with keys "min" and "max", each a per-channel tensor.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Clear all collected statistics."""
        raise NotImplementedError


def create() -> MinMaxObserver:
    """Factory function for MinMaxObserver."""
    return MinMaxObserver()
