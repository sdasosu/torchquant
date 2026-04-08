"""SmoothQuant migration scale observer.

Collects per-channel activation magnitude statistics to compute
migration scales that balance quantization difficulty between
weights and activations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


class SmoothQuantObserver:
    """Collects activation channel-wise max-abs for SmoothQuant scale migration.

    Records running max of absolute activation values per channel.
    These are used to compute the migration factor alpha that shifts
    quantization difficulty from activations to weights.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        self._alpha = alpha
        self._act_max: Tensor | None = None

    def __call__(
        self,
        module: object,
        inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        """Forward hook: update activation max-abs statistics."""
        raise NotImplementedError

    def get_stats(self) -> dict[str, Tensor]:
        """Return migration scale statistics.

        Returns:
            Dict with key "act_max" (per-channel max absolute activation).
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Clear all collected statistics."""
        raise NotImplementedError


def create(*, alpha: float = 0.5) -> SmoothQuantObserver:
    """Factory function for SmoothQuantObserver."""
    return SmoothQuantObserver(alpha=alpha)
