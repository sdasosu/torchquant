"""Shared fixtures for quantized runtime module tests."""

from __future__ import annotations

import torch
from torch import nn

from torchquant._types import Algorithm
from torchquant.config import QuantScheme
from torchquant.export.runtime import (
    QuantizedConv2d,
    QuantizedEmbedding,
    QuantizedLinear,
)
from torchquant.export.runtime.conv import Conv2dSpec
from torchquant.export.runtime.embedding import EmbeddingSpec


def make_scheme(*, group_size: int, symmetric: bool) -> QuantScheme:
    """Build a small RTN scheme for runtime module tests."""
    return QuantScheme(
        weight_bits=4,
        group_size=group_size,
        symmetric=symmetric,
        algorithm=Algorithm.RTN,
    )


def manual_dequantize(
    int_weight: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor | None,
    *,
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize grouped integer weights for reference-module construction."""
    group_width = int_weight.shape[1] if group_size == -1 else group_size
    n_groups = int_weight.shape[1] // group_width
    grouped = int_weight.to(torch.float32).reshape(
        int_weight.shape[0],
        n_groups,
        group_width,
    )
    scales_fp32 = scales.to(torch.float32).unsqueeze(-1)
    if zero_points is None:
        dequantized = grouped * scales_fp32
    else:
        dequantized = (
            grouped - zero_points.to(torch.float32).unsqueeze(-1)
        ) * scales_fp32
    return dequantized.reshape_as(int_weight).to(dtype)


def make_linear_case() -> tuple[QuantizedLinear, nn.Linear, torch.Tensor]:
    """Return a quantized linear module, float reference, and sample input."""
    scheme = make_scheme(group_size=2, symmetric=False)
    int_weight = torch.tensor([[3, 7, 5, 9], [4, 2, 11, 6]], dtype=torch.int32)
    scales = torch.tensor([[0.25, 0.5], [0.125, 0.25]], dtype=torch.float32)
    zero_points = torch.tensor([[4, 4], [3, 5]], dtype=torch.int32)
    bias = torch.tensor([0.25, -0.125], dtype=torch.float32)
    module = QuantizedLinear(
        int_weight=int_weight,
        scales=scales,
        zero_points=zero_points,
        bias=bias,
        scheme=scheme,
        in_features=4,
        out_features=2,
        weight_dtype=torch.float32,
    )
    reference = nn.Linear(4, 2, bias=True)
    expected_weight = manual_dequantize(
        int_weight,
        scales,
        zero_points,
        group_size=scheme.group_size,
        dtype=torch.float32,
    )
    assert reference.bias is not None
    with torch.no_grad():
        reference.weight.copy_(expected_weight)
        reference.bias.copy_(bias)
    sample_input = torch.tensor(
        [[0.25, -1.0, 0.75, 0.5], [1.0, 0.0, -0.5, 1.5]],
        dtype=torch.float32,
    )
    return module, reference, sample_input


def make_conv_case() -> tuple[QuantizedConv2d, nn.Conv2d, torch.Tensor]:
    """Return a quantized conv module, float reference, and sample input."""
    scheme = make_scheme(group_size=4, symmetric=False)
    int_weight = torch.tensor(
        [
            [[[3, 6], [5, 8]], [[4, 2], [7, 9]]],
            [[[10, 12], [6, 4]], [[8, 14], [5, 7]]],
        ],
        dtype=torch.int32,
    )
    scales = torch.tensor([[0.25, 0.5], [0.125, 0.25]], dtype=torch.float32)
    zero_points = torch.tensor([[4, 6], [8, 5]], dtype=torch.int32)
    bias = torch.tensor([0.5, -0.25], dtype=torch.float32)
    spec = Conv2dSpec(
        in_channels=2,
        out_channels=2,
        kernel_size=(2, 2),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
    )
    module = QuantizedConv2d(
        int_weight=int_weight,
        scales=scales,
        zero_points=zero_points,
        bias=bias,
        scheme=scheme,
        spec=spec,
        weight_dtype=torch.float32,
    )
    reference = nn.Conv2d(2, 2, kernel_size=2, bias=True)
    expected_weight = manual_dequantize(
        int_weight.reshape(int_weight.shape[0], -1),
        scales,
        zero_points,
        group_size=scheme.group_size,
        dtype=torch.float32,
    ).reshape_as(int_weight)
    assert reference.bias is not None
    with torch.no_grad():
        reference.weight.copy_(expected_weight)
        reference.bias.copy_(bias)
    sample_input = torch.arange(50, dtype=torch.float32).reshape(1, 2, 5, 5) / 10
    return module, reference, sample_input


def make_embedding_case() -> tuple[QuantizedEmbedding, nn.Embedding, torch.Tensor]:
    """Return a quantized embedding module, float reference, and sample input."""
    scheme = make_scheme(group_size=2, symmetric=False)
    int_weight = torch.tensor(
        [
            [3, 6, 5, 7],
            [8, 4, 9, 3],
            [2, 10, 6, 12],
            [5, 5, 5, 5],
            [11, 7, 4, 8],
        ],
        dtype=torch.int32,
    )
    scales = torch.tensor(
        [[0.25, 0.5], [0.125, 0.25], [0.5, 0.125], [0.25, 0.25], [0.5, 0.5]],
        dtype=torch.float32,
    )
    zero_points = torch.tensor(
        [[4, 6], [3, 5], [6, 10], [5, 5], [8, 7]],
        dtype=torch.int32,
    )
    spec = EmbeddingSpec(
        num_embeddings=5,
        embedding_dim=4,
        padding_idx=1,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
    )
    module = QuantizedEmbedding(
        int_weight=int_weight,
        scales=scales,
        zero_points=zero_points,
        scheme=scheme,
        spec=spec,
        weight_dtype=torch.float32,
    )
    reference = nn.Embedding(5, 4, padding_idx=1)
    expected_weight = manual_dequantize(
        int_weight,
        scales,
        zero_points,
        group_size=scheme.group_size,
        dtype=torch.float32,
    )
    with torch.no_grad():
        reference.weight.copy_(expected_weight)
    sample_input = torch.tensor([[0, 1, 4], [3, 2, 1]], dtype=torch.int64)
    return module, reference, sample_input
