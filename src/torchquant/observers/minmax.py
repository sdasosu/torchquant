"""Min/max activation range observer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


class MinMaxObserver:
    """Tracks per-channel min/max activation ranges.

    Collects running min/max statistics from the *output* activations of
    the hooked layer (``output``), not its inputs.  We observe the output
    because static activation quantization cares about the post-activation
    range that downstream consumers actually see; the input range of a
    layer is irrelevant here and is instead the job of input-side
    observers such as ``AWQObserver`` / ``HessianObserver``.

    Used as the default observer for static activation quantization.

    Args:
        channel_dim: Dimension index to treat as the channel axis.
            Use ``-1`` (default) for LLM linear layers
            ``(batch, seq_len, features)``; use ``1`` for CNNs
            ``(batch, C, H, W)``.
    """

    def __init__(self, channel_dim: int = -1) -> None:
        self._channel_dim = channel_dim
        self._min: Tensor | None = None
        self._max: Tensor | None = None

    def __call__(
        self,
        module: object,
        inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        """Forward hook: update min/max from the *output* tensor.

        Observes ``output`` rather than ``inputs[0]`` because static
        activation quantization targets the post-activation range that
        downstream consumers see.

        Args:
            module: Hooked layer (unused).
            inputs: Layer inputs (unused).
            output: Layer output; the tensor whose per-channel range we
                accumulate.
        """
        x: Tensor = output.detach().float()

        ch = self._channel_dim % x.dim()
        n_channels = x.shape[ch]

        flat = x.movedim(ch, 0).reshape(n_channels, -1)
        batch_min = flat.min(dim=1).values
        batch_max = flat.max(dim=1).values
        if self._min is None or self._max is None:
            self._min = batch_min
            self._max = batch_max
        else:
            self._min = torch.min(self._min, batch_min)
            self._max = torch.max(self._max, batch_max)

    def get_stats(self) -> dict[str, Tensor]:
        """Return collected statistics.

        Returns:
            Dict with keys ``"min"`` and ``"max"``, each a per-channel
            ``float32`` tensor cloned from the internal buffers so the
            caller cannot mutate observer state.

        Raises:
            RuntimeError: If called before any forward pass has been
                observed.
        """
        if self._min is None or self._max is None:
            raise RuntimeError(
                "No activations have been observed yet. "
                "Run at least one forward pass through the hooked module."
            )
        return {"min": self._min.clone(), "max": self._max.clone()}

    def reset(self) -> None:
        """Clear all collected statistics."""
        self._min = None
        self._max = None


def create(channel_dim: int = -1) -> MinMaxObserver:
    """Factory function for MinMaxObserver."""
    return MinMaxObserver(channel_dim=channel_dim)
