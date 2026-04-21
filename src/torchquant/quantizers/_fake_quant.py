"""Shared fake-quantization helpers for weight-only quantizers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

_EPS: float = 1e-8
_SUPPORTED_BITS = frozenset({2, 3, 4, 8})


def is_linear_module(module: object) -> bool:
    """Return whether the module is a torch.nn.Linear instance."""
    return isinstance(module, torch.nn.Linear)


def is_conv2d_module(module: object) -> bool:
    """Return whether the module is a torch.nn.Conv2d instance."""
    return isinstance(module, torch.nn.Conv2d)


def is_embedding_module(module: object) -> bool:
    """Return whether the module is a torch.nn.Embedding instance."""
    return isinstance(module, torch.nn.Embedding)


def fake_quantize_2d(
    weight_2d: Tensor,
    *,
    bits: int,
    group_size: int,
    symmetric: bool,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Return fake-quantized weights plus scale metadata.

    Args:
        weight_2d: Weight tensor shaped ``(out_features, in_features)``.
        bits: Quantization bit width.
        group_size: Group width along the input axis. ``-1`` means one
            full-width group per output channel.
        symmetric: Whether to use signed symmetric quantization.

    Returns:
        A tuple of ``(fake_quantized_weight, scales, zero_points)``.
        ``scales`` always has shape ``(out_features, n_groups)`` and dtype
        ``float32``. ``zero_points`` is ``None`` for symmetric quantization,
        otherwise an ``int32`` tensor with the same shape as ``scales``.
    """
    dequant, scales, zero_points, _ = _fake_quantize_2d_impl(
        weight_2d,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
    )
    return dequant, scales, zero_points


def fake_quantize_2d_with_int(
    weight_2d: Tensor,
    *,
    bits: int,
    group_size: int,
    symmetric: bool,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor]:
    """Return fake-quantized weights plus the integer tensor before dequantize."""
    return _fake_quantize_2d_impl(
        weight_2d,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
    )


def _fake_quantize_2d_impl(
    weight_2d: Tensor,
    *,
    bits: int,
    group_size: int,
    symmetric: bool,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor]:
    if weight_2d.dim() != 2:
        raise ValueError(f"weight_2d must be 2-D, got shape {tuple(weight_2d.shape)}.")

    group_width, n_groups = resolve_group_layout(
        in_features=weight_2d.shape[1],
        group_size=group_size,
    )
    weight_fp32 = weight_2d.float()
    grouped = weight_fp32.reshape(weight_2d.shape[0], n_groups, group_width)

    if symmetric:
        q_max = _symmetric_qmax(bits)
        scales = grouped.abs().amax(dim=2).clamp(min=_EPS) / q_max
        q_weight = torch.round(grouped / scales.unsqueeze(-1)).clamp(-q_max, q_max)
        dequant = q_weight * scales.unsqueeze(-1)
        zero_points = None
    else:
        q_max = _asymmetric_qmax(bits)
        group_min = grouped.amin(dim=2)
        group_max = grouped.amax(dim=2)
        scales = (group_max - group_min).clamp(min=_EPS) / q_max
        zero_points = torch.round(-group_min / scales).clamp(0, q_max).to(torch.int32)
        zero_points_fp32 = zero_points.to(torch.float32)
        q_weight = (
            torch.round(grouped / scales.unsqueeze(-1)) + zero_points_fp32.unsqueeze(-1)
        ).clamp(0, q_max)
        dequant = (q_weight - zero_points_fp32.unsqueeze(-1)) * scales.unsqueeze(-1)

    return (
        dequant.reshape_as(weight_2d).to(weight_2d.dtype),
        scales,
        zero_points,
        q_weight.reshape_as(weight_2d).to(torch.int32),
    )


def compute_scale_zero(
    weight_group: Tensor,
    *,
    bits: int,
    symmetric: bool,
) -> tuple[Tensor, Tensor | None]:
    """Return scale metadata for one input-group slice.

    Args:
        weight_group: Slice shaped ``(out_features, group_width)``.
        bits: Quantization bit width.
        symmetric: Whether to use signed symmetric quantization.

    Returns:
        A tuple of ``(scale, zero_point)`` where ``scale`` has shape
        ``(out_features,)`` and dtype ``float32``. ``zero_point`` is
        ``None`` for symmetric quantization, otherwise ``int32`` with the
        same shape.
    """
    if weight_group.dim() != 2:
        raise ValueError(
            f"weight_group must be 2-D, got shape {tuple(weight_group.shape)}.",
        )

    _validate_bits(bits)
    weight_fp32 = weight_group.float()

    if symmetric:
        q_max = _symmetric_qmax(bits)
        scale = weight_fp32.abs().amax(dim=1).clamp(min=_EPS) / q_max
        return scale, None

    q_max = _asymmetric_qmax(bits)
    group_min = weight_fp32.amin(dim=1)
    group_max = weight_fp32.amax(dim=1)
    scale = (group_max - group_min).clamp(min=_EPS) / q_max
    zero_point = torch.round(-group_min / scale).clamp(0, q_max).to(torch.int32)
    return scale, zero_point


def quantize_column(
    weight_col: Tensor,
    scale_col: Tensor,
    zero_col: Tensor | None,
    *,
    bits: int,
    symmetric: bool,
) -> Tensor:
    """Round one weight column and dequantize it back to float32."""
    dequantized, _ = quantize_column_with_int(
        weight_col,
        scale_col,
        zero_col,
        bits=bits,
        symmetric=symmetric,
    )
    return dequantized


def quantize_column_with_int(
    weight_col: Tensor,
    scale_col: Tensor,
    zero_col: Tensor | None,
    *,
    bits: int,
    symmetric: bool,
) -> tuple[Tensor, Tensor]:
    """Round one weight column, returning dequantized and integer forms."""
    if weight_col.dim() != 1:
        raise ValueError(
            f"weight_col must be 1-D, got shape {tuple(weight_col.shape)}."
        )

    _validate_bits(bits)
    weight_fp32 = weight_col.float()
    scale_fp32 = scale_col.float()

    if symmetric:
        q_max = _symmetric_qmax(bits)
        q_weight = torch.round(weight_fp32 / scale_fp32).clamp(-q_max, q_max)
        return q_weight * scale_fp32, q_weight.to(torch.int32)

    if zero_col is None:
        raise ValueError("zero_col must be provided for asymmetric quantization.")

    q_max = _asymmetric_qmax(bits)
    zero_fp32 = zero_col.to(torch.float32)
    q_weight = (torch.round(weight_fp32 / scale_fp32) + zero_fp32).clamp(0, q_max)
    return (q_weight - zero_fp32) * scale_fp32, q_weight.to(torch.int32)


def _validate_bits(bits: int) -> None:
    if bits not in _SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {sorted(_SUPPORTED_BITS)}, got {bits}.")


def resolve_group_layout(*, in_features: int, group_size: int) -> tuple[int, int]:
    _validate_group_size(group_size)

    if group_size == -1:
        return in_features, 1

    if in_features % group_size != 0:
        raise ValueError(
            f"group_size {group_size} must divide in_features {in_features}.",
        )

    return group_size, in_features // group_size


def _validate_group_size(group_size: int) -> None:
    if group_size == 0:
        raise ValueError("group_size must not be 0.")
    if group_size < -1:
        raise ValueError(
            f"group_size must be -1 or a positive integer, got {group_size}."
        )


def _symmetric_qmax(bits: int) -> int:
    _validate_bits(bits)
    return (2 ** (bits - 1)) - 1


def _asymmetric_qmax(bits: int) -> int:
    _validate_bits(bits)
    return (2**bits) - 1
