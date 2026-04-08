"""End-to-end quantization pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._types import Algorithm

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import nn

    from .config import QuantRecipe
    from .quantizers import QuantizerFn

QUANTIZER_MAP: dict[Algorithm, QuantizerFn] = {}


def _init_quantizer_map() -> None:
    """Lazily populate QUANTIZER_MAP with algorithm implementations."""
    if QUANTIZER_MAP:
        return
    from .quantizers import awq, gptq, rtn, smoothquant

    QUANTIZER_MAP.update(
        {
            Algorithm.RTN: rtn.quantize_layer,
            Algorithm.GPTQ: gptq.quantize_layer,
            Algorithm.AWQ: awq.quantize_layer,
            Algorithm.SMOOTHQUANT: smoothquant.quantize_layer,
        }
    )


def quantize_model(
    model: nn.Module,
    recipe: QuantRecipe,
    calibration_data: Iterable[Any] | None = None,
) -> nn.Module:
    """Quantize a model end-to-end.

    Pipeline stages:
        1. Auto-detect model adapter
        2. Prepare model (eval, freeze, fuse BN)
        3. Discover quantizable nodes
        4. Assign schemes via rule engine
        5. Run calibration if needed
        6. Apply quantizers
        7. Rebuild model with quantized weights

    Args:
        model: The PyTorch model to quantize.
        recipe: Quantization recipe describing what and how to quantize.
        calibration_data: Optional calibration dataset for methods that need it.

    Returns:
        A new model with quantized weights.
    """
    raise NotImplementedError
