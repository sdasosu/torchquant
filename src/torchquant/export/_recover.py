"""Integer-weight recovery helpers for Phase 6 runtime export."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

import torch

from torchquant._types import Algorithm
from torchquant.quantizers._fake_quant import (
    _asymmetric_qmax,
    _symmetric_qmax,
    resolve_group_layout,
)

from .runtime.base import UnsupportedExportError

if TYPE_CHECKING:
    from torch import Tensor

    from torchquant.config import QuantScheme
    from torchquant.quantizers import QuantResult


def recover_int_weight(
    result: QuantResult,
    scheme: QuantScheme,
    weight_shape: tuple[int, ...],
    *,
    fqn: str | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Recover integer weights plus quantization metadata from a QuantResult."""
    if scheme.algorithm not in {Algorithm.RTN, Algorithm.GPTQ}:
        target = "<unknown>" if fqn is None else fqn
        raise UnsupportedExportError(
            "Cannot recover integer weight for "
            f"{target}: {scheme.algorithm.name} is not exportable via the Phase 6 "
            "module-level rewriter because it requires migration-scale metadata "
            "and cross-layer folding."
        )
    if not weight_shape:
        raise ValueError("weight_shape must not be empty.")
    if prod(weight_shape) != result.quantized_weight.numel():
        raise ValueError(
            "weight_shape element count mismatch: expected "
            f"{prod(weight_shape)}, got {result.quantized_weight.numel()}."
        )

    outer_dim = weight_shape[0]
    stored_weight = getattr(
        result, "_oracle_exact_quantized_weight", result.quantized_weight
    )
    weight_2d = stored_weight.reshape(outer_dim, -1)
    in_features = weight_2d.shape[1]
    group_width, n_groups = resolve_group_layout(
        in_features=in_features,
        group_size=scheme.group_size,
    )
    expected_meta_shape = (outer_dim, n_groups)
    scales = result.scales.to(torch.float32).clone()
    if tuple(scales.shape) != expected_meta_shape:
        raise ValueError(
            "scales shape mismatch: expected "
            f"{expected_meta_shape}, got {tuple(scales.shape)}."
        )

    zero_points = None
    if scheme.symmetric:
        if result.zero_points is not None:
            raise ValueError("zero_points must be None for symmetric quantization.")
        qmax = _symmetric_qmax(scheme.weight_bits)
    else:
        if result.zero_points is None:
            raise ValueError(
                "zero_points must be provided for asymmetric quantization."
            )
        zero_points = result.zero_points.to(torch.int32).clone()
        if tuple(zero_points.shape) != expected_meta_shape:
            raise ValueError(
                "zero_points shape mismatch: expected "
                f"{expected_meta_shape}, got {tuple(zero_points.shape)}."
            )
        qmax = _asymmetric_qmax(scheme.weight_bits)

    grouped_scales = scales.unsqueeze(-1).expand(outer_dim, n_groups, group_width)
    scales_broadcast = grouped_scales.reshape_as(weight_2d)
    weight_fp32 = weight_2d.to(torch.float32)

    if zero_points is None:
        q_int = torch.round(weight_fp32 / scales_broadcast).clamp(-qmax, qmax)
    else:
        grouped_zero_points = (
            zero_points.to(torch.float32)
            .unsqueeze(-1)
            .expand(
                outer_dim,
                n_groups,
                group_width,
            )
        )
        zero_points_broadcast = grouped_zero_points.reshape_as(weight_2d)
        q_int = torch.round(weight_fp32 / scales_broadcast) + zero_points_broadcast
        q_int = q_int.clamp(0, qmax)

    return q_int.to(torch.int32).reshape(weight_shape), scales, zero_points
