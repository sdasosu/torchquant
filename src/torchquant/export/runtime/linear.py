"""Runtime module for quantized linear weights."""

from __future__ import annotations

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


class QuantizedLinear(QuantizedRuntimeModule):
    """Linear layer backed by integer weights plus quantization metadata."""

    int_weight: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor | None
    bias: torch.Tensor | None
    scheme: QuantScheme
    weight_dtype: torch.dtype
    in_features: int
    out_features: int

    def __init__(
        self,
        *,
        int_weight: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor | None,
        bias: torch.Tensor | None,
        scheme: QuantScheme,
        in_features: int,
        out_features: int,
        weight_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        rows, columns = _prepare_weight_metadata(
            int_weight_2d=int_weight,
            scales=scales,
            zero_points=zero_points,
            scheme=scheme,
        )
        if rows != out_features or columns != in_features:
            raise ValueError(
                "Linear weight shape mismatch: expected "
                f"({out_features}, {in_features}), got {tuple(int_weight.shape)}."
            )
        if bias is not None and tuple(bias.shape) != (out_features,):
            raise ValueError(
                "bias shape mismatch: expected "
                f"{(out_features,)}, got {tuple(bias.shape)}."
            )
        _validate_weight_dtype(weight_dtype)

        self.scheme = scheme
        self.weight_dtype = weight_dtype
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("int_weight", int_weight.detach().to(torch.int32).clone())
        self.register_buffer("scales", scales.detach().to(torch.float32).clone())
        zero_points_buffer = None
        if zero_points is not None:
            zero_points_buffer = zero_points.detach().to(torch.int32).clone()
        self.register_buffer("zero_points", zero_points_buffer)
        bias_buffer = bias
        self.register_buffer("bias", bias_buffer)

    def dequantize_weight(self) -> torch.Tensor:
        """Return the dequantized weight matrix."""
        return dequantize_weight_2d(
            int_weight_2d=self.int_weight,
            scales=self.scales,
            zero_points=self.zero_points,
            scheme=self.scheme,
            weight_dtype=self.weight_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a linear layer with dequantized weights."""
        return F.linear(x, self.dequantize_weight(), self.bias)

    def extra_repr(self) -> str:
        """Return a concise summary for ``repr(module)``."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
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
                "bias": None if self.bias is None else self.bias.detach().clone(),
                "in_features": self.in_features,
                "out_features": self.out_features,
            }
        )
        return payload

    @classmethod
    def rebuild_from_state_dict(cls, payload: dict[str, Any]) -> QuantizedLinear:
        """Rebuild a ``QuantizedLinear`` from ``export_state_dict()`` data."""
        _validate_module_class(payload, cls.__name__)
        _validate_payload_version(payload, cls.version)
        return cls(
            int_weight=payload["int_weight"],
            scales=payload["scales"],
            zero_points=payload["zero_points"],
            bias=payload["bias"],
            scheme=_deserialize_scheme(payload["scheme"]),
            in_features=payload["in_features"],
            out_features=payload["out_features"],
            weight_dtype=_deserialize_dtype(payload["weight_dtype"]),
        )
