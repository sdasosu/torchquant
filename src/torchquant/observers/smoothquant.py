"""SmoothQuant migration scale observer.

Collects per-channel activation magnitude statistics used to compute
migration scales that balance quantization difficulty between weights
and activations.

Reference:
    Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training
    Quantization for Large Language Models", ICML 2023.
    https://arxiv.org/abs/2211.10438
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


class SmoothQuantObserver:
    """Collects per-channel activation max-abs for SmoothQuant.

    SmoothQuant migrates quantization difficulty from activations
    (which are hard to quantize because of outlier channels) to
    weights (which quantize easily) by absorbing a per-channel
    diagonal scale s into the layer:

        Y = (X / s) @ (diag(s) @ W)

    The migration scale is computed offline from per-channel
    activation magnitudes and per-channel weight magnitudes:

        s_j = max(|X_:,j|)^alpha / max(|W_:,j|)^(1 - alpha)

    where alpha in [0, 1] balances how much difficulty is moved
    from activations to weights.  This observer is responsible only
    for the *activation* half of that formula -- it accumulates
    max(|X|) per input channel across all calibration batches.  The
    quantizer is responsible for combining it with the weight half
    and applying the alpha trade-off.

    Args:
        channel_dim: Dimension index to treat as the *input-channel*
            axis.  Use ``-1`` (default) for LLM linear layers
            ``(batch, seq_len, in_features)``; use ``1`` for CNNs
            ``(batch, C, H, W)``.
    """

    def __init__(self, *, channel_dim: int = -1) -> None:
        self._channel_dim = channel_dim
        self._act_max: Tensor | None = None

    def __call__(
        self,
        module: object,
        inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        """Forward hook: update running per-channel max-abs activation.

        Uses the *input* activation ``inputs[0]``, not the output,
        because SmoothQuant migration scales live in the input-channel
        space of W.

        Args:
            module: Hooked layer (unused).
            inputs: Layer inputs; ``inputs[0]`` is the activation tensor.
            output: Layer output (unused).
        """
        x: Tensor = inputs[0].detach().float().abs()

        ch = self._channel_dim % x.dim()
        n_channels = x.shape[ch]

        flat = x.movedim(ch, 0).reshape(n_channels, -1)
        batch_max = flat.amax(dim=1)

        if self._act_max is None:
            self._act_max = batch_max
        else:
            self._act_max = torch.max(self._act_max, batch_max)

    def get_stats(self) -> dict[str, Tensor]:
        """Return migration scale statistics.

        Returns:
            Dict with key ``"act_max"``: per-channel max absolute
            activation, shape ``(in_features,)``, ``float32``, cloned
            from internal state so callers cannot mutate the observer
            through the returned dict.

        Raises:
            RuntimeError: If called before any forward pass has been
                observed.
        """
        if self._act_max is None:
            raise RuntimeError(
                "No activations have been observed yet. "
                "Run at least one forward pass through the hooked module."
            )
        return {"act_max": self._act_max.clone()}

    def reset(self) -> None:
        """Clear all collected statistics."""
        self._act_max = None


def create(*, channel_dim: int = -1) -> SmoothQuantObserver:
    """Factory function for SmoothQuantObserver.

    Args:
        channel_dim: Input channel axis.  Use ``-1`` for LLMs, ``1``
            for CNNs.

    Returns:
        A freshly initialised :class:`SmoothQuantObserver`.
    """
    return SmoothQuantObserver(channel_dim=channel_dim)
