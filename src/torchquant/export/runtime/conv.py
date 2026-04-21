"""Runtime module for quantized Conv2d weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from .base import (
    QuantizedRuntimeModule,
    _deserialize_dtype,
    _deserialize_scheme,
    _normalize_2tuple,
    _prepare_weight_metadata,
    _validate_module_class,
    _validate_payload_version,
    _validate_weight_dtype,
    dequantize_weight_2d,
)

if TYPE_CHECKING:
    from torchquant.config import QuantScheme


@dataclass(frozen=True)
class Conv2dSpec:
    """Structural attributes needed to rebuild a Conv2d runtime module."""

    in_channels: int
    out_channels: int
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    groups: int


class QuantizedConv2d(QuantizedRuntimeModule):
    """Conv2d layer backed by integer weights plus quantization metadata."""

    int_weight: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor | None
    bias: torch.Tensor | None
    scheme: QuantScheme
    weight_dtype: torch.dtype
    spec: Conv2dSpec
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    groups: int

    def __init__(
        self,
        *,
        int_weight: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor | None,
        bias: torch.Tensor | None,
        scheme: QuantScheme,
        spec: Conv2dSpec,
        weight_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if int_weight.dim() != 4:
            raise ValueError(
                f"Conv2d int_weight must be 4-D, got shape {tuple(int_weight.shape)}."
            )
        if spec.groups <= 0:
            raise ValueError(f"groups must be positive, got {spec.groups}.")
        if spec.out_channels != int_weight.shape[0]:
            raise ValueError(
                "Conv2d out_channels mismatch: expected "
                f"{spec.out_channels}, got {int_weight.shape[0]}."
            )
        if int_weight.shape[1] * spec.groups != spec.in_channels:
            raise ValueError(
                "Conv2d in_channels mismatch: expected "
                f"{spec.in_channels}, got {int_weight.shape[1] * spec.groups}."
            )
        if tuple(int_weight.shape[2:]) != spec.kernel_size:
            raise ValueError(
                "Conv2d kernel_size mismatch: expected "
                f"{spec.kernel_size}, got {tuple(int_weight.shape[2:])}."
            )
        if bias is not None and tuple(bias.shape) != (spec.out_channels,):
            raise ValueError(
                "bias shape mismatch: expected "
                f"{(spec.out_channels,)}, got {tuple(bias.shape)}."
            )
        _validate_weight_dtype(weight_dtype)

        flat_weight = int_weight.reshape(int_weight.shape[0], -1)
        _prepare_weight_metadata(
            int_weight_2d=flat_weight,
            scales=scales,
            zero_points=zero_points,
            scheme=scheme,
        )

        self.scheme = scheme
        self.weight_dtype = weight_dtype
        self.spec = spec
        self.in_channels = spec.in_channels
        self.out_channels = spec.out_channels
        self.kernel_size = spec.kernel_size
        self.stride = spec.stride
        self.padding = spec.padding
        self.dilation = spec.dilation
        self.groups = spec.groups
        self.register_buffer("int_weight", int_weight.detach().to(torch.int32).clone())
        self.register_buffer("scales", scales.detach().to(torch.float32).clone())
        zero_points_buffer = None
        if zero_points is not None:
            zero_points_buffer = zero_points.detach().to(torch.int32).clone()
        self.register_buffer("zero_points", zero_points_buffer)
        bias_buffer = bias
        self.register_buffer("bias", bias_buffer)

    def dequantize_weight(self) -> torch.Tensor:
        """Return the dequantized convolution kernel."""
        flat_weight = self.int_weight.reshape(self.out_channels, -1)
        dequantized = dequantize_weight_2d(
            int_weight_2d=flat_weight,
            scales=self.scales,
            zero_points=self.zero_points,
            scheme=self.scheme,
            weight_dtype=self.weight_dtype,
        )
        return dequantized.reshape_as(self.int_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a conv2d layer with dequantized weights."""
        return F.conv2d(
            x,
            self.dequantize_weight(),
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def extra_repr(self) -> str:
        """Return a concise summary for ``repr(module)``."""
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"dilation={self.dilation}, "
            f"groups={self.groups}, "
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
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": self.padding,
                "dilation": self.dilation,
                "groups": self.groups,
            }
        )
        return payload

    @classmethod
    def rebuild_from_state_dict(cls, payload: dict[str, Any]) -> QuantizedConv2d:
        """Rebuild a ``QuantizedConv2d`` from ``export_state_dict()`` data."""
        _validate_module_class(payload, cls.__name__)
        _validate_payload_version(payload, cls.version)
        spec = Conv2dSpec(
            in_channels=payload["in_channels"],
            out_channels=payload["out_channels"],
            kernel_size=_normalize_2tuple(tuple(payload["kernel_size"])),
            stride=_normalize_2tuple(tuple(payload["stride"])),
            padding=_normalize_2tuple(tuple(payload["padding"])),
            dilation=_normalize_2tuple(tuple(payload["dilation"])),
            groups=payload["groups"],
        )
        return cls(
            int_weight=payload["int_weight"],
            scales=payload["scales"],
            zero_points=payload["zero_points"],
            bias=payload["bias"],
            scheme=_deserialize_scheme(payload["scheme"]),
            spec=spec,
            weight_dtype=_deserialize_dtype(payload["weight_dtype"]),
        )
