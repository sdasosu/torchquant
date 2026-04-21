"""Shared runtime helpers for exported quantized modules."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import nn

from torchquant._types import Algorithm
from torchquant.config import QuantScheme
from torchquant.quantizers._fake_quant import resolve_group_layout


class UnsupportedExportError(RuntimeError):
    """Raised when a quantized record cannot be rewritten for export."""


class QuantizedRuntimeModule(nn.Module):
    """Common state shared by quantized runtime modules."""

    version: ClassVar[int] = 1

    scheme: QuantScheme
    weight_dtype: torch.dtype

    def _base_payload(self) -> dict[str, Any]:
        return {
            "module_class": type(self).__name__,
            "scheme": _serialize_scheme(self.scheme),
            "version": self.version,
            "weight_dtype": _serialize_dtype(self.weight_dtype),
        }

    def _quantization_repr(self) -> str:
        return (
            f"bits={self.scheme.weight_bits}, "
            f"group_size={self.scheme.group_size}, "
            f"algorithm={self.scheme.algorithm.name}"
        )


def _serialize_scheme(scheme: QuantScheme) -> dict[str, Any]:
    return {
        "weight_bits": scheme.weight_bits,
        "activation_bits": scheme.activation_bits,
        "group_size": scheme.group_size,
        "symmetric": scheme.symmetric,
        "algorithm": scheme.algorithm.name,
    }


def _deserialize_scheme(payload: dict[str, Any]) -> QuantScheme:
    return QuantScheme.model_validate(
        {
            **payload,
            "algorithm": Algorithm[payload["algorithm"]],
        }
    )


def _serialize_dtype(dtype: torch.dtype) -> str:
    return str(dtype)


def _deserialize_dtype(name: str) -> torch.dtype:
    dtype_name = name.removeprefix("torch.")
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"Unsupported dtype metadata: {name}.")
    return dtype


def _validate_module_class(payload: dict[str, Any], expected: str) -> None:
    module_class = payload.get("module_class")
    if module_class != expected:
        raise ValueError(f"Expected payload for {expected}, got {module_class!r}.")


def _validate_payload_version(payload: dict[str, Any], expected: int) -> None:
    version = payload.get("version")
    if version != expected:
        raise ValueError(f"Unsupported payload version: {version!r}.")


def _normalize_2tuple(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if len(value) != 2:
        raise ValueError(f"Expected a 2-tuple, got {value!r}.")
    return tuple(value)


def _validate_weight_dtype(weight_dtype: torch.dtype) -> None:
    if not weight_dtype.is_floating_point:
        raise ValueError(f"weight_dtype must be floating, got {weight_dtype}.")


def _prepare_weight_metadata(
    *,
    int_weight_2d: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor | None,
    scheme: QuantScheme,
) -> tuple[int, int]:
    if int_weight_2d.dim() != 2:
        raise ValueError(
            f"Expected a 2-D weight matrix, got shape {tuple(int_weight_2d.shape)}."
        )
    if scales.dim() != 2:
        raise ValueError(f"scales must be 2-D, got shape {tuple(scales.shape)}.")
    if scheme.symmetric and zero_points is not None:
        raise ValueError("zero_points must be None for symmetric quantization.")
    if not scheme.symmetric and zero_points is None:
        raise ValueError("zero_points must be provided for asymmetric quantization.")

    rows, in_features = int_weight_2d.shape
    _, n_groups = resolve_group_layout(
        in_features=in_features,
        group_size=scheme.group_size,
    )
    expected_shape = (rows, n_groups)
    if tuple(scales.shape) != expected_shape:
        raise ValueError(
            "scales shape mismatch: expected "
            f"{expected_shape}, got {tuple(scales.shape)}."
        )
    if zero_points is not None and tuple(zero_points.shape) != expected_shape:
        raise ValueError(
            "zero_points shape mismatch: expected "
            f"{expected_shape}, got {tuple(zero_points.shape)}."
        )
    return rows, in_features


def dequantize_weight_2d(
    *,
    int_weight_2d: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor | None,
    scheme: QuantScheme,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize an integer weight matrix using grouped scale metadata."""
    _, in_features = _prepare_weight_metadata(
        int_weight_2d=int_weight_2d,
        scales=scales,
        zero_points=zero_points,
        scheme=scheme,
    )
    _validate_weight_dtype(weight_dtype)

    group_width, n_groups = resolve_group_layout(
        in_features=in_features,
        group_size=scheme.group_size,
    )
    grouped_int = int_weight_2d.to(torch.float32).reshape(
        int_weight_2d.shape[0],
        n_groups,
        group_width,
    )
    grouped_scales = scales.to(torch.float32).unsqueeze(-1)

    if zero_points is None:
        dequantized = grouped_int * grouped_scales
    else:
        dequantized = (
            grouped_int - zero_points.to(torch.float32).unsqueeze(-1)
        ) * grouped_scales

    return dequantized.reshape_as(int_weight_2d).to(dtype=weight_dtype)
