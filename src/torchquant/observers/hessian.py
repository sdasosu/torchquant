"""Hessian / Fisher information observer for GPTQ.

Collects second-order statistics (Hessian diagonal approximation)
used by GPTQ for optimal weight rounding.

Reference:
    Frantar et al., "GPTQ: Accurate Post-Training Quantization for
    Generative Pre-trained Transformers", ICLR 2023.
    https://arxiv.org/abs/2210.17323
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


class HessianObserver:
    """Collects Hessian-diagonal approximation for GPTQ.

    Accumulates H = 2 * X^T @ X (input outer products) which
    approximates the Hessian of the layer-wise reconstruction loss.

    For a linear layer with weight W ∈ R^{out × in}, GPTQ minimises:

        ||WX - W_q X||_F²

    whose Hessian w.r.t. W is H = 2 * X @ X^T (in GPTQ notation where
    X is in × N).  This observer accumulates that matrix incrementally
    across calibration batches so memory never exceeds ``in_features²``.

    Args:
        damping: Tikhonov regularisation added to the diagonal of H
            before it is returned, expressed as a fraction of the mean
            diagonal value.  Prevents singularity for dead neurons.
            GPTQ paper uses 0.01.  Set to 0.0 to disable.
    """

    def __init__(self, damping: float = 0.01) -> None:
        self._hessian: Tensor | None = None
        self._sample_count: int = 0
        self._damping = damping

    # ------------------------------------------------------------------
    # Forward hook
    # ------------------------------------------------------------------

    def __call__(
        self,
        module: object,
        inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        """Forward hook: accumulate input outer-product Hessian estimate.

        Uses the *input* to the layer (``inputs[0]``), not the output,
        because H depends on the activations seen by W, not on Wx.

        Supports inputs of shape:
            - ``(batch, in_features)``          — fully-connected / MLP
            - ``(batch, seq_len, in_features)`` — transformer linear layers

        The update rule (matching the original GPTQ codebase) is::

            X  ← reshape to (in_features, N)   where N = batch * seq
            H  += 2 / N_total * X @ X^T

        Dividing by ``N_total`` (total samples seen so far) keeps H as a
        proper average across calibration batches.

        Args:
            module: The hooked layer (unused — we only need the input).
            inputs: Tuple of layer inputs; ``inputs[0]`` is the activation.
            output: Layer output (unused).
        """
        x: Tensor = inputs[0].detach().float()

        # Flatten to 2-D: (n_tokens, in_features)
        if x.dim() == 3:
            # (batch, seq_len, in_features) → (batch * seq_len, in_features)
            x = x.reshape(-1, x.shape[-1])
        elif x.dim() != 2:
            raise ValueError(
                f"HessianObserver expects 2-D or 3-D input tensors, "
                f"got shape {tuple(x.shape)}."
            )

        n_new = x.shape[0]           # number of new token / sample rows
        in_features = x.shape[1]

        # X transposed to (in_features, n_new) for the outer product
        # H_batch = 2 * X^T @ X  —  shape (in_features, in_features)
        x_t = x.T                    # (in_features, n_new)
        h_batch = 2.0 * (x_t @ x)   # (in_features, in_features)

        if self._hessian is None:
            # First batch: initialise with zeros on the correct device/dtype
            self._hessian = torch.zeros(
                in_features, in_features,
                dtype=torch.float32,
                device=x.device,
            )

        # Online update: keep a running *sum* weighted by sample count so
        # that get_stats() can return the properly scaled average at any
        # time without storing all batches.
        #
        # Running mean update:
        #   H_new = (n_old * H_old + n_new * H_batch / 2) * 2 / n_total
        # Simplified to an additive accumulation of the raw sum, divided
        # once in get_stats():
        #
        #   _hessian += h_batch   (unnormalised running sum)
        #   divide by total_n in get_stats() to recover the mean.
        self._hessian += h_batch
        self._sample_count += n_new

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Tensor]:
        """Return Hessian statistics.

        Returns the *averaged* Hessian H = (2 / N) * Σ X_i^T X_i with
        optional Tikhonov damping added to the diagonal.

        Returns:
            Dict with key ``"hessian"``: a ``float32`` tensor of shape
            ``(in_features, in_features)``.

        Raises:
            RuntimeError: If called before any forward pass has been
                observed.
        """
        if self._hessian is None or self._sample_count == 0:
            raise RuntimeError(
                "No activations have been observed yet. "
                "Run at least one forward pass through the hooked module."
            )

        # Normalise the running sum by total sample count.
        h = self._hessian / self._sample_count

        # Tikhonov / diagonal damping — stabilises the Cholesky factorisation
        # performed later by GPTQ.  λ = damping * mean(diag(H)).
        if self._damping > 0.0:
            lam = self._damping * h.diagonal().mean()
            h = h + lam * torch.eye(h.shape[0], dtype=h.dtype, device=h.device)

        return {"hessian": h}

    def reset(self) -> None:
        """Clear the accumulated Hessian and sample counter."""
        self._hessian = None
        self._sample_count = 0


def create(damping: float = 0.01) -> HessianObserver:
    """Factory function for HessianObserver.

    Args:
        damping: Diagonal regularisation fraction (see :class:`HessianObserver`).

    Returns:
        A freshly initialised :class:`HessianObserver`.
    """
    return HessianObserver(damping=damping)