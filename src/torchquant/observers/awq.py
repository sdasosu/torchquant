"""AWQ activation-aware weight scale observer.

Collects activation statistics used to determine importance-weighted
scaling factors for weight quantization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


class AWQObserver:
    """Collects per-channel activation scales for AWQ.

    Tracks mean absolute activation values per channel to identify
    salient weight channels that need larger quantization scales.
    """

    def __init__(self) -> None:
        self._act_mean: Tensor | None = None
        self._sample_count: int = 0

    def __call__(
        self,
        module: object,
        inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        """Forward hook: accumulate activation magnitude statistics."""
        raise NotImplementedError

    def get_stats(self) -> dict[str, Tensor]:
        """Return activation importance statistics.

        Returns:
            Dict with key "act_mean" (per-channel mean absolute activation).
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Clear all collected statistics."""
        raise NotImplementedError


def create() -> AWQObserver:
    """Factory function for AWQObserver."""
    return AWQObserver()
