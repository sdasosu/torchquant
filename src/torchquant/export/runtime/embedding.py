"""Runtime module for quantized embedding weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from .base import (
    QuantizedRuntimeModule,
    _deserialize_dtype,
    _deserialize_scheme,
    _prepare_weight_metadata,
    _validate_module_class,
    _validate_payload_version,
    _validate_weight_dtype,
    dequantize_weight_2d,
)

if TYPE_CHECKING:
    from torchquant.config import QuantScheme


@dataclass(frozen=True)
class EmbeddingSpec:
    """Structural attributes needed to rebuild an embedding runtime module."""

    num_embeddings: int
    embedding_dim: int
    padding_idx: int | None
    max_norm: float | None
    norm_type: float
    scale_grad_by_freq: bool
    sparse: bool


class QuantizedEmbedding(QuantizedRuntimeModule):
    """Embedding layer backed by integer weights plus quantization metadata."""

    int_weight: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor | None
    scheme: QuantScheme
    weight_dtype: torch.dtype
    spec: EmbeddingSpec
    num_embeddings: int
    embedding_dim: int
    padding_idx: int | None
    max_norm: float | None
    norm_type: float
    scale_grad_by_freq: bool
    sparse: bool

    def __init__(
        self,
        *,
        int_weight: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor | None,
        scheme: QuantScheme,
        spec: EmbeddingSpec,
        weight_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        rows, columns = _prepare_weight_metadata(
            int_weight_2d=int_weight,
            scales=scales,
            zero_points=zero_points,
            scheme=scheme,
        )
        if rows != spec.num_embeddings or columns != spec.embedding_dim:
            raise ValueError(
                "Embedding weight shape mismatch: expected "
                f"({spec.num_embeddings}, {spec.embedding_dim}), "
                f"got {tuple(int_weight.shape)}."
            )
        _validate_weight_dtype(weight_dtype)

        self.scheme = scheme
        self.weight_dtype = weight_dtype
        self.spec = spec
        self.num_embeddings = spec.num_embeddings
        self.embedding_dim = spec.embedding_dim
        self.padding_idx = spec.padding_idx
        self.max_norm = spec.max_norm
        self.norm_type = spec.norm_type
        self.scale_grad_by_freq = spec.scale_grad_by_freq
        self.sparse = spec.sparse
        self.register_buffer("int_weight", int_weight.detach().to(torch.int32).clone())
        self.register_buffer("scales", scales.detach().to(torch.float32).clone())
        zero_points_buffer = None
        if zero_points is not None:
            zero_points_buffer = zero_points.detach().to(torch.int32).clone()
        self.register_buffer("zero_points", zero_points_buffer)

    def dequantize_weight(self) -> torch.Tensor:
        """Return the dequantized embedding table."""
        return dequantize_weight_2d(
            int_weight_2d=self.int_weight,
            scales=self.scales,
            zero_points=self.zero_points,
            scheme=self.scheme,
            weight_dtype=self.weight_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run an embedding lookup with dequantized weights."""
        return F.embedding(
            x,
            self.dequantize_weight(),
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

    def extra_repr(self) -> str:
        """Return a concise summary for ``repr(module)``."""
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"padding_idx={self.padding_idx}, "
            f"max_norm={self.max_norm}, "
            f"norm_type={self.norm_type}, "
            f"scale_grad_by_freq={self.scale_grad_by_freq}, "
            f"sparse={self.sparse}, "
            f"{self._quantization_repr()}"
        )

    def export_state_dict(self) -> dict[str, Any]:
        """Export tensors plus metadata for version-tolerant persistence."""
        payload = self._base_payload()
        payload.update(
            {
                "int_weight": self.int_weight.detach().clone(),
                "scales": self.scales.detach().clone(),
                "zero_points": None
                if self.zero_points is None
                else self.zero_points.detach().clone(),
                "num_embeddings": self.num_embeddings,
                "embedding_dim": self.embedding_dim,
                "padding_idx": self.padding_idx,
                "max_norm": self.max_norm,
                "norm_type": self.norm_type,
                "scale_grad_by_freq": self.scale_grad_by_freq,
                "sparse": self.sparse,
            }
        )
        return payload

    @classmethod
    def rebuild_from_state_dict(cls, payload: dict[str, Any]) -> QuantizedEmbedding:
        """Rebuild a ``QuantizedEmbedding`` from ``export_state_dict()`` data."""
        _validate_module_class(payload, cls.__name__)
        _validate_payload_version(payload, cls.version)
        spec = EmbeddingSpec(
            num_embeddings=payload["num_embeddings"],
            embedding_dim=payload["embedding_dim"],
            padding_idx=payload["padding_idx"],
            max_norm=payload["max_norm"],
            norm_type=payload["norm_type"],
            scale_grad_by_freq=payload["scale_grad_by_freq"],
            sparse=payload["sparse"],
        )
        return cls(
            int_weight=payload["int_weight"],
            scales=payload["scales"],
            zero_points=payload["zero_points"],
            scheme=_deserialize_scheme(payload["scheme"]),
            spec=spec,
            weight_dtype=_deserialize_dtype(payload["weight_dtype"]),
        )
