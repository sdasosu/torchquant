"""Min/max activation range observer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


class MinMaxObserver:
    """Tracks per-channel min/max activation ranges.

    Collects running min/max statistics from forward pass activations.
    Used as the default observer for static activation quantization.
    """

    def __init__(self, channel_dim: int = -1) -> None:
        self._channel_dim = channel_dim
        self._min: Tensor | None = None
        self._max: Tensor | None = None
        self._channel_dim =channel_dim

    def __call__( self, module: object, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        """Forward hook: update min/max from output tensor."""
        
        x:Tensor = output.detach().float()

        ch=self._channel_dim % x.dim()
        n_channels = x.shape[ch]

        flat=x.movedim(ch,0).reshape(n_channels,-1)
        batch_min=flat.min(dim=1).values
        batch_max=flat.max(dim=1).values
        if self._min is None:
            self._min = batch_min
            self._max = batch_max
        else:
            self._min = torch.min(self._min, batch_min)
            self._max = torch.max(self._max, batch_max)

    #-------------------------------
    # Public API    
    # ------------------------------

    def get_stats(self) -> dict[str, Tensor]:
        """Return collected statistics.

        Returns:
            Dict with keys "min" and "max", each a per-channel tensor.
        """
        if self._min is None or self._max is None:
             raise ValueError("No statistics collected yet.")
        return {"min": self._min, "max": self._max}     
        
    #-------------------------------

    def reset(self) -> None:
        """Clear all collected statistics."""
        self._min = None
        self._max = None    
    #-------------------------------
    # Factory function
    # -------------------------------            
def create(channel_dim: int = -1) -> MinMaxObserver:
    """Factory function for MinMaxObserver."""

    return MinMaxObserver(channel_dim=channel_dim)

    
