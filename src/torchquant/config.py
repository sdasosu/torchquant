"""Quantization configuration: schemes and recipes."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ._types import Algorithm


class QuantScheme(BaseModel):
    """Describes HOW to quantize a set of targets."""

    model_config = ConfigDict(frozen=True)

    weight_bits: int = 8
    activation_bits: int | None = None
    group_size: int = -1
    symmetric: bool = True
    algorithm: Algorithm = Algorithm.RTN


class QuantRecipe(BaseModel):
    """Declares WHAT to quantize and HOW."""

    model_config = ConfigDict(frozen=True)

    default_scheme: QuantScheme = QuantScheme()
    overrides: dict[str, QuantScheme] = {}
    ignore: frozenset[str] = frozenset()
    calibration_samples: int = 128
