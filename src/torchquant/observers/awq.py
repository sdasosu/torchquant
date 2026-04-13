"""AWQ activation-aware weight scale observer.

Collects activation statistics used to determine importance-weighted
scaling factors for weight quantization.

Reference:
    Lin et al., "AWQ: Activation-aware Weight Quantization for LLM
    Compression and Acceleration", MLSys 2024.
    https://arxiv.org/abs/2306.00978
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


class AWQObserver:
    """Collects per-channel activation scales for AWQ.

    Tracks mean absolute activation values per channel to identify
    salient weight channels that need larger quantization scales.

    AWQ's core insight: 1% of channels with the largest activation
    magnitudes contribute disproportionately to quantization error.
    Rather than quantizing weights uniformly, these salient channels
    are protected by a per-channel scale s:

        W_q = round( (W / s) * 2^(b-1) ) * s / 2^(b-1)
        X'  = X * s       (scale applied to activation instead)

    so the product W @ X is preserved exactly while W/s has a
    narrower effective range that quantizes with lower error.

    The scale s is derived from the activation magnitude:

        s_j = mean(|X_j|)^α,   α ∈ [0, 1]  (default α = 0.5)

    This observer accumulates mean(|X|) per input channel across
    all calibration batches so the GPTQ / AWQ solver can compute
    optimal per-channel scales offline.

    Args:
        alpha: Exponent applied to activation magnitudes when computing
            scales in ``get_stats``.  α=0 → uniform scaling (disabled),
            α=1 → full activation-proportional scaling.  The AWQ paper
            uses α=0.5 as the default after grid search.
        channel_dim: Dimension index to treat as the *input-channel* axis.
            Use ``-1`` (default) for LLM linear layers
            ``(batch, seq_len, in_features)``; use ``1`` for CNNs
            ``(batch, C, H, W)``.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        channel_dim: int = -1,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
        self._act_mean: Tensor | None = None
        self._sample_count: int = 0
        self._alpha = alpha
        self._channel_dim = channel_dim

    # ------------------------------------------------------------------
    # Forward hook
    # ------------------------------------------------------------------

    def __call__(
        self,
        module: object,
        inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        """Forward hook: accumulate activation magnitude statistics.

        Uses the *input* activation ``inputs[0]`` (X), not the output,
        because AWQ scales are defined in the input-channel space of W.

        Per-channel mean absolute value is accumulated as a running sum::

            _act_mean += |X|.mean(over all dims except channel_dim)

        Normalisation by ``_sample_count`` is deferred to ``get_stats``
        so the hook itself is as cheap as possible.

        Args:
            module: Hooked layer (unused).
            inputs: Layer inputs; ``inputs[0]`` is the activation tensor.
            output: Layer output (unused).
        """
        x: Tensor = inputs[0].detach().float().abs()

        # Normalise channel_dim to a positive index.
        ch = self._channel_dim % x.dim()
        n_channels = x.shape[ch]

        # Move channel axis to front, flatten everything else →
        # shape (n_channels, N) where N = product of all other dims.
        flat = x.movedim(ch, 0).reshape(n_channels, -1)  # (C, N)

        # Sum of absolute values across the non-channel tokens/pixels.
        # Dividing by flat.shape[1] converts to a per-batch mean;
        # we accumulate the mean rather than the raw sum so that batches
        # of different sizes are weighted equally (matches AWQ paper).
        batch_mean = flat.mean(dim=1)   # (C,)
        n_new = flat.shape[1]           # tokens / spatial positions

        if self._act_mean is None:
            self._act_mean = torch.zeros(
                n_channels,
                dtype=torch.float32,
                device=x.device,
            )

        # Welford-style incremental mean over *tokens* (not batches) so
        # that long sequences are not down-weighted vs short ones.
        #   mean_new = mean_old + (batch_mean * n_new - mean_old * n_new)
        #                         / (sample_count + n_new)
        # Simplified to a weighted accumulation:
        self._act_mean = (
            self._act_mean * self._sample_count + batch_mean * n_new
        ) / (self._sample_count + n_new)
        self._sample_count += n_new

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Tensor]:
        """Return activation importance statistics.

        Returns:
            A dict with the following keys:

            ``"act_mean"``
                Per-channel mean absolute activation,
                shape ``(in_features,)``, ``float32``.

            ``"scale"``
                Per-channel AWQ importance scale ``s = act_mean^alpha``,
                shape ``(in_features,)``, ``float32``.
                Multiply weight columns by ``1/s`` before quantization
                and activation channels by ``s`` to compensate.

        Raises:
            RuntimeError: If called before any forward pass.
        """
        if self._act_mean is None or self._sample_count == 0:
            raise RuntimeError(
                "No activations have been observed yet. "
                "Run at least one forward pass through the hooked module."
            )

        act_mean = self._act_mean.clone()

        # Clamp to avoid zero/negative values before pow().
        act_mean_clamped = act_mean.clamp(min=1e-8)
        scale = act_mean_clamped.pow(self._alpha)

        return {
            "act_mean": act_mean,
            "scale": scale,
        }

    def reset(self) -> None:
        """Clear all collected statistics."""
        self._act_mean = None
        self._sample_count = 0


def create(alpha: float = 0.5, channel_dim: int = -1) -> AWQObserver:
    """Factory function for AWQObserver.

    Args:
        alpha: Scale exponent (see :class:`AWQObserver`).
            Grid-searched in [0, 1]; paper default is 0.5.
        channel_dim: Input channel axis.  Use ``-1`` for LLMs, ``1`` for
            CNNs.

    Returns:
        A freshly initialised :class:`AWQObserver`.
    """
    return AWQObserver(alpha=alpha, channel_dim=channel_dim)