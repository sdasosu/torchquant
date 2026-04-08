"""Hessian / Fisher information observer for GPTQ.

Collects second-order statistics (Hessian diagonal approximation)
used by GPTQ for optimal weight rounding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


class HessianObserver:
    """Collects Hessian-diagonal approximation for GPTQ.

    Accumulates H = 2 * X^T @ X (input outer products) which
    approximates the Hessian of the layer-wise reconstruction loss.
    """

    def __init__(self) -> None:
        self._hessian: Tensor | None = None
        self._sample_count: int = 0

    def __call__(
        self,
        module: object,
        inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        """Forward hook: accumulate input outer-product Hessian estimate."""
        raise NotImplementedError

    def get_stats(self) -> dict[str, Tensor]:
        """Return Hessian statistics.

        Returns:
            Dict with key "hessian" (the accumulated H matrix).
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Clear the accumulated Hessian."""
        raise NotImplementedError


def create() -> HessianObserver:
    """Factory function for HessianObserver."""
    return HessianObserver()
