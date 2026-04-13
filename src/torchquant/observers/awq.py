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

        s_j = mean(|X_j|)^alpha,   alpha in [0, 1]  (default alpha = 0.5)

    This observer accumulates mean(|X|) per input channel across
    all calibration batches so the GPTQ / AWQ solver can compute
    optimal per-channel scales offline.

    Args:
        alpha: Exponent applied to activation magnitudes when computing
            scales in ``get_stats``.  alpha=0 -> uniform scaling (disabled),
            alpha=1 -> full activation-proportional scaling.  The AWQ paper
            uses alpha=0.5 as the default after grid search.
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

        ch = self._channel_dim % x.dim()
        n_channels = x.shape[ch]

        # Move channel axis to front, flatten everything else ->
        # shape (n_channels, N) where N = product of all other dims.
        flat = x.movedim(ch, 0).reshape(n_channels, -1)
        batch_mean = flat.mean(dim=1)
        n_new = flat.shape[1]

        # Token-weighted running mean: combine the prior accumulated mean
        # weighted by sample_count with the new batch mean weighted by
        # n_new tokens.  We weight per-token (not per-batch) so a long
        # sequence contributes proportionally more than a short one;
        # the original AWQ paper uses per-batch means and we deliberately
        # diverge here for stability against varying sequence lengths.
        if self._act_mean is None:
            self._act_mean = batch_mean
        else:
            self._act_mean = (
                self._act_mean * self._sample_count + batch_mean * n_new
            ) / (self._sample_count + n_new)
        self._sample_count += n_new

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

        # clamp() already returns a fresh tensor, so deriving scale from
        # it is allocation-free; act_mean is cloned separately so callers
        # cannot mutate observer state through the returned dict.
        scale = self._act_mean.clamp(min=1e-8).pow(self._alpha)

        return {
            "act_mean": self._act_mean.clone(),
            "scale": scale,
        }

    def reset(self) -> None:
        """Clear all collected statistics."""
        self._act_mean = None
        self._sample_count = 0


def create(*, alpha: float = 0.5, channel_dim: int = -1) -> AWQObserver:
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
