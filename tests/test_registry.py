"""Tests for registry helpers and model rebuilding."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torchquant._types import Algorithm, LayerKind
from torchquant.config import QuantScheme
from torchquant.quantizers import rtn
from torchquant.registry import (
    QuantRecord,
    QuantRegistry,
    apply_records_inplace,
    rebuild_model,
)


def _linear(
    in_features: int = 8,
    out_features: int = 4,
    *,
    bias: bool = False,
    dtype: torch.dtype = torch.float32,
) -> nn.Linear:
    module = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
    weight = torch.linspace(
        -1.0,
        1.0,
        steps=in_features * out_features,
        dtype=torch.float32,
    ).reshape(out_features, in_features)
    with torch.no_grad():
        module.weight.copy_(weight.to(dtype))
        if module.bias is not None:
            module.bias.copy_(
                torch.linspace(-0.5, 0.5, steps=out_features, dtype=dtype),
            )
    return module


def _scheme() -> QuantScheme:
    return QuantScheme(weight_bits=8, algorithm=Algorithm.RTN)


def _record_for(
    fqn: str,
    module: nn.Module,
    *,
    kind: LayerKind = LayerKind.LINEAR,
) -> QuantRecord:
    result = rtn.quantize_layer(module, _scheme(), stats={})
    return QuantRecord(fqn=fqn, kind=kind, scheme=_scheme(), result=result)


def _registry(*records: QuantRecord) -> QuantRegistry:
    registry = QuantRegistry()
    for record in records:
        registry.add(record)
    return registry


def test_apply_records_inplace_writes_weight() -> None:
    """apply_records_inplace writes the quantized weight into the target."""
    module = _linear()
    record = _record_for("", module)

    with torch.no_grad():
        module.weight.zero_()

    apply_records_inplace(module, _registry(record))

    assert torch.equal(module.weight, record.result.quantized_weight)


def test_apply_records_inplace_mutates_argument() -> None:
    """apply_records_inplace mutates the passed-in model and returns None."""
    module = _linear()
    original_weight = module.weight.detach().clone()
    record = _record_for("", module)

    result = apply_records_inplace(module, _registry(record))

    assert result is None
    assert not torch.equal(module.weight, original_weight)
    assert torch.equal(module.weight, record.result.quantized_weight)


def test_apply_records_inplace_missing_fqn_raises() -> None:
    """Missing record targets raise KeyError naming the missing FQN."""
    module = _linear()

    with pytest.raises(KeyError, match="does_not_exist"):
        apply_records_inplace(module, _registry(_record_for("does_not_exist", module)))


def test_apply_records_inplace_empty_registry_is_noop() -> None:
    """An empty registry leaves the model unchanged."""
    module = _linear()
    original_weight = module.weight.detach().clone()

    apply_records_inplace(module, QuantRegistry())

    assert torch.equal(module.weight, original_weight)


def test_rebuild_returns_new_object() -> None:
    """rebuild_model returns a fresh module object."""
    module = _linear()

    rebuilt = rebuild_model(module, QuantRegistry())

    assert rebuilt is not module


def test_rebuild_does_not_mutate_input_model() -> None:
    """rebuild_model leaves the input model weights untouched."""
    module = _linear()
    original_weight = module.weight.detach().clone()

    rebuild_model(module, _registry(_record_for("", module)))

    assert torch.equal(module.weight, original_weight)


def test_rebuild_writes_quantized_weight() -> None:
    """rebuild_model writes the quantized weight into the returned model."""
    module = _linear()
    record = _record_for("", module)

    rebuilt = rebuild_model(module, _registry(record))

    assert isinstance(rebuilt, nn.Linear)
    assert torch.equal(rebuilt.weight, record.result.quantized_weight)


def test_rebuild_empty_registry_returns_clean_copy() -> None:
    """An empty registry still returns a deep copy of the input model."""
    module = _linear()

    rebuilt = rebuild_model(module, QuantRegistry())

    assert isinstance(rebuilt, nn.Linear)
    assert rebuilt is not module
    assert torch.equal(rebuilt.weight, module.weight)


def test_rebuild_preserves_bias() -> None:
    """Bias tensors are preserved by the deepcopy-based rebuild."""
    module = _linear(bias=True)

    rebuilt = rebuild_model(module, _registry(_record_for("", module)))

    assert isinstance(rebuilt, nn.Linear)
    assert rebuilt.bias is not None
    assert module.bias is not None
    assert torch.equal(rebuilt.bias, module.bias)


def test_rebuild_preserves_dtype() -> None:
    """Rebuilt weights keep the original parameter dtype."""
    module = _linear(dtype=torch.float16)

    rebuilt = rebuild_model(module, _registry(_record_for("", module)))

    assert isinstance(rebuilt, nn.Linear)
    assert rebuilt.weight.dtype == torch.float16


def test_rebuild_handles_conv2d_shape() -> None:
    """Conv2d rebuild preserves the original 4-D weight shape."""
    module = nn.Conv2d(3, 4, 3, bias=False)
    record = _record_for("", module, kind=LayerKind.CONV2D)

    rebuilt = rebuild_model(module, _registry(record))

    assert isinstance(rebuilt, nn.Conv2d)
    assert rebuilt.weight.shape == module.weight.shape
    assert torch.equal(rebuilt.weight, record.result.quantized_weight)


def test_rebuild_multi_record_writes_all() -> None:
    """Multiple registry records are applied to the matching submodules."""
    model = nn.Sequential(_linear(), nn.ReLU(), _linear(in_features=4, out_features=2))
    first_record = _record_for("0", model[0])
    second_record = _record_for("2", model[2])

    rebuilt = rebuild_model(model, _registry(first_record, second_record))

    assert isinstance(rebuilt, nn.Sequential)
    assert torch.equal(rebuilt[0].weight, first_record.result.quantized_weight)
    assert torch.equal(rebuilt[2].weight, second_record.result.quantized_weight)
