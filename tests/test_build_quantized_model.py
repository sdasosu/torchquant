"""Tests for build_quantized_model and registry exposure."""

from __future__ import annotations

import torch
from torch import nn

from torchquant import build_quantized_model, quantize_model
from torchquant._types import Algorithm
from torchquant.config import QuantRecipe, QuantScheme
from torchquant.registry import QuantRegistry


def _set_linear_params(module: nn.Linear, *, start: float, stop: float) -> None:
    weight = torch.linspace(
        start,
        stop,
        steps=module.in_features * module.out_features,
        dtype=torch.float32,
    ).reshape(module.out_features, module.in_features)
    with torch.no_grad():
        module.weight.copy_(weight.to(module.weight.dtype))
        if module.bias is not None:
            module.bias.copy_(
                torch.linspace(
                    -0.5, 0.5, steps=module.out_features, dtype=torch.float32
                ),
            )


def _three_linear_model() -> nn.Sequential:
    model = nn.Sequential(
        nn.Linear(8, 6, bias=True),
        nn.ReLU(),
        nn.Linear(6, 4, bias=False),
        nn.ReLU(),
        nn.Linear(4, 2, bias=True),
    )
    _set_linear_params(model[0], start=-1.2, stop=0.8)
    _set_linear_params(model[2], start=-0.7, stop=1.1)
    _set_linear_params(model[4], start=-1.5, stop=1.5)
    return model


def _recipe() -> QuantRecipe:
    return QuantRecipe(
        default_scheme=QuantScheme(weight_bits=8, algorithm=Algorithm.RTN)
    )


def test_build_quantized_model_returns_model_and_registry() -> None:
    """build_quantized_model returns a quantized module plus QuantRegistry."""
    model = _three_linear_model()

    quantized_model, registry = build_quantized_model(model, _recipe())

    assert isinstance(quantized_model, nn.Module)
    assert isinstance(registry, QuantRegistry)
    assert len(registry) == 3


def test_build_quantized_model_captures_original_bias() -> None:
    """Registry records preserve bias tensors from the original modules."""
    model = _three_linear_model()

    _, registry = build_quantized_model(model, _recipe())
    records = dict(registry)

    assert set(records) == {"0", "2", "4"}
    for fqn, record in records.items():
        original_module = model.get_submodule(fqn)
        assert isinstance(original_module, nn.Linear)
        bias = original_module.bias
        if bias is None:
            assert record.original_bias is None
            continue
        assert record.original_bias is not None
        assert torch.equal(record.original_bias, bias)
        assert record.original_bias is not bias


def test_quantize_model_matches_build_quantized_model_output() -> None:
    """Legacy wrapper returns the same quantized weights as the core function."""
    model = _three_linear_model()

    wrapper_output = quantize_model(model, _recipe())
    core_output, _ = build_quantized_model(model, _recipe())

    wrapper_state = wrapper_output.state_dict()
    core_state = core_output.state_dict()

    assert wrapper_state.keys() == core_state.keys()
    for name, tensor in wrapper_state.items():
        assert torch.equal(tensor, core_state[name])
