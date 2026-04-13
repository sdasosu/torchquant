"""GPTQ blockwise weight quantization.

Implements the Optimal Brain Quantization (OBQ) approach from the GPTQ
paper, using Hessian information to find optimal weight rounding
decisions that minimize layer-wise reconstruction error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import QuantResult
from ._fake_quant import (
    compute_scale_zero,
    is_linear_module,
    quantize_column,
    resolve_group_layout,
)

if TYPE_CHECKING:
    from torch import Tensor, nn

    from torchquant.config import QuantScheme

_BLOCKSIZE: int = 128
_PERCDAMP: float = 0.01


def quantize_layer(
    module: nn.Module,
    scheme: QuantScheme,
    stats: dict[str, Tensor],
) -> QuantResult:
    """Quantize a single linear layer using GPTQ.

    Args:
        module: The layer to quantize.
        scheme: Quantization scheme.
        stats: Must contain ``"hessian"`` with averaged second-order stats.

    Returns:
        QuantResult with GPTQ-optimized fake-quantized weights.

    Raises:
        TypeError: If ``module`` is not ``nn.Linear``.
        ValueError: If ``stats`` is missing ``"hessian"``.
        torch.linalg.LinAlgError: If the damped Hessian is not SPD.
    """
    if not is_linear_module(module):
        raise TypeError(
            f"GPTQ quantize_layer supports nn.Linear only, got {type(module).__name__}."
        )
    if "hessian" not in stats:
        raise ValueError("GPTQ requires 'hessian' statistics.")

    original_weight = module.get_parameter("weight").detach().clone()
    quantized_weight = original_weight.to(dtype=torch.float32, copy=True)
    hessian = stats["hessian"].clone()
    group_width, n_groups = resolve_group_layout(
        in_features=quantized_weight.shape[1],
        group_size=scheme.group_size,
    )

    diagonal = hessian.diagonal()
    dead = diagonal == 0
    diagonal.masked_fill_(dead, 1.0)
    quantized_weight[:, dead] = 0.0

    damp = _PERCDAMP * hessian.diagonal().mean()
    hessian.diagonal().add_(damp)

    cholesky = torch.linalg.cholesky(hessian)
    hessian_inverse = torch.cholesky_inverse(cholesky)
    error_propagation = torch.linalg.cholesky(hessian_inverse, upper=True)

    scales = torch.empty(
        quantized_weight.shape[0],
        n_groups,
        dtype=torch.float32,
        device=quantized_weight.device,
    )
    zero_points = (
        None
        if scheme.symmetric
        else torch.empty(
            quantized_weight.shape[0],
            n_groups,
            dtype=torch.int32,
            device=quantized_weight.device,
        )
    )

    current_group = -1
    current_scale: Tensor | None = None
    current_zero: Tensor | None = None

    for start in range(0, quantized_weight.shape[1], _BLOCKSIZE):
        stop = min(start + _BLOCKSIZE, quantized_weight.shape[1])
        block_weight = quantized_weight[:, start:stop].clone()
        block_error = torch.empty_like(block_weight)
        block_quantized = torch.empty_like(block_weight)
        block_hessian_inverse = error_propagation[start:stop, start:stop]

        for block_column in range(stop - start):
            column_index = start + block_column
            group_index = 0 if scheme.group_size == -1 else column_index // group_width

            if group_index != current_group:
                group_start = (
                    0 if scheme.group_size == -1 else group_index * group_width
                )
                group_stop = group_start + group_width
                group_slice = quantized_weight[:, group_start:group_stop]
                current_scale, current_zero = compute_scale_zero(
                    group_slice,
                    bits=scheme.weight_bits,
                    symmetric=scheme.symmetric,
                )
                scales[:, group_index] = current_scale
                if zero_points is not None:
                    assert current_zero is not None
                    zero_points[:, group_index] = current_zero
                current_group = group_index

            assert current_scale is not None
            diagonal_entry = block_hessian_inverse[block_column, block_column]
            quantized_column = quantize_column(
                block_weight[:, block_column],
                current_scale,
                current_zero,
                bits=scheme.weight_bits,
                symmetric=scheme.symmetric,
            )
            column_error = (
                block_weight[:, block_column] - quantized_column
            ) / diagonal_entry
            if block_column + 1 < stop - start:
                block_weight[:, block_column + 1 :] -= column_error.unsqueeze(
                    1,
                ) * block_hessian_inverse[block_column, block_column + 1 :].unsqueeze(0)
            block_quantized[:, block_column] = quantized_column
            block_error[:, block_column] = column_error

        quantized_weight[:, start:stop] = block_quantized
        if stop < quantized_weight.shape[1]:
            quantized_weight[:, stop:] -= (
                block_error @ error_propagation[start:stop, stop:]
            )

    return QuantResult(
        quantized_weight=quantized_weight.to(original_weight.dtype),
        scales=scales,
        zero_points=zero_points,
        original_weight=original_weight,
    )
