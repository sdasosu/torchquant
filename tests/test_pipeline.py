"""Tests for the end-to-end quantization pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch
from torch import nn

from torchquant import pipeline, quantize_model
from torchquant._types import Algorithm
from torchquant.config import QuantRecipe, QuantScheme

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torchquant.observers import ObserverSpec


def _set_linear_weight(module: nn.Linear, *, start: float, stop: float) -> None:
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
                torch.linspace(-0.25, 0.25, steps=module.out_features),
            )


def _toy_sequential() -> nn.Sequential:
    model = nn.Sequential(
        nn.Linear(8, 4, bias=False),
        nn.ReLU(),
        nn.Linear(4, 2, bias=False),
    )
    _set_linear_weight(model[0], start=-1.3, stop=1.7)
    _set_linear_weight(model[2], start=-0.8, stop=0.9)
    return model


def _single_linear_sequential() -> nn.Sequential:
    model = nn.Sequential(nn.Linear(8, 4, bias=False))
    _set_linear_weight(model[0], start=-1.1, stop=1.4)
    return model


def _toy_calibration_data(samples: int = 4, in_f: int = 8) -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(2, in_f) for _ in range(samples)]


def _recipe(
    algorithm: Algorithm,
    *,
    ignore: frozenset[str] = frozenset(),
    overrides: dict[str, QuantScheme] | None = None,
) -> QuantRecipe:
    return QuantRecipe(
        default_scheme=QuantScheme(weight_bits=8, algorithm=algorithm),
        overrides={} if overrides is None else overrides,
        ignore=ignore,
    )


def _snapshot(model: nn.Module) -> tuple[dict[str, torch.Tensor], bool, list[bool]]:
    return (
        {name: tensor.detach().clone() for name, tensor in model.state_dict().items()},
        model.training,
        [parameter.requires_grad for parameter in model.parameters()],
    )


def _assert_input_untouched(
    before: tuple[dict[str, torch.Tensor], bool, list[bool]],
    model: nn.Module,
) -> None:
    state_dict_before, training_before, requires_grad_before = before
    state_dict_after = model.state_dict()

    assert state_dict_after.keys() == state_dict_before.keys()
    for name, tensor_before in state_dict_before.items():
        assert torch.equal(state_dict_after[name], tensor_before)
    assert model.training is training_before
    assert [
        parameter.requires_grad for parameter in model.parameters()
    ] == requires_grad_before


def test_rtn_end_to_end_returns_new_object() -> None:
    """RTN returns a fresh model object."""
    model = _toy_sequential()

    returned = quantize_model(model, _recipe(Algorithm.RTN))

    assert returned is not model


def test_rtn_end_to_end_does_not_mutate_input() -> None:
    """RTN leaves the input model unchanged."""
    model = _toy_sequential()
    before = _snapshot(model)

    quantize_model(model, _recipe(Algorithm.RTN))

    _assert_input_untouched(before, model)


def test_rtn_end_to_end_weights_change() -> None:
    """RTN changes the returned linear weights."""
    model = _toy_sequential()

    returned = quantize_model(model, _recipe(Algorithm.RTN))

    assert isinstance(returned, nn.Sequential)
    assert not torch.equal(returned[0].weight, model[0].weight)
    assert not torch.equal(returned[2].weight, model[2].weight)


def test_returned_model_is_inference_ready() -> None:
    """Returned models are always eval-only and parameter-frozen."""
    model = _toy_sequential()

    returned = quantize_model(model, _recipe(Algorithm.RTN))

    assert returned.training is False
    assert all(not parameter.requires_grad for parameter in returned.parameters())


def test_gptq_requires_calibration_data() -> None:
    """GPTQ rejects missing calibration data."""
    model = _single_linear_sequential()

    with pytest.raises(ValueError, match=r"calibration_data.*GPTQ"):
        quantize_model(model, _recipe(Algorithm.GPTQ))


@pytest.mark.parametrize(
    "algorithm",
    [Algorithm.GPTQ, Algorithm.AWQ, Algorithm.SMOOTHQUANT],
)
def test_calibration_algorithm_end_to_end(algorithm: Algorithm) -> None:
    """Calibration-based algorithms quantize a single linear layer end-to-end."""
    model = _single_linear_sequential()

    returned = quantize_model(
        model,
        _recipe(algorithm),
        _toy_calibration_data(),
    )

    assert isinstance(returned, nn.Sequential)
    assert not torch.equal(returned[0].weight, model[0].weight)
    assert returned[0].weight.shape == model[0].weight.shape
    assert returned[0].weight.dtype == model[0].weight.dtype


def test_ignore_all_fqns_returns_prepared_copy() -> None:
    """Ignoring every quantizable FQN still returns a prepared fresh copy."""
    model = _toy_sequential()
    before = _snapshot(model)

    returned = quantize_model(
        model,
        _recipe(Algorithm.RTN, ignore=frozenset({"0", "2"})),
    )

    assert isinstance(returned, nn.Sequential)
    assert returned is not model
    assert torch.equal(returned[0].weight, model[0].weight)
    assert torch.equal(returned[2].weight, model[2].weight)
    assert returned.training is False
    assert all(not parameter.requires_grad for parameter in returned.parameters())
    _assert_input_untouched(before, model)


def test_overrides_per_layer_take_effect() -> None:
    """Per-layer overrides are honored during end-to-end quantization."""
    model = _toy_sequential()
    recipe = _recipe(
        Algorithm.RTN,
        overrides={
            "0": QuantScheme(weight_bits=8, algorithm=Algorithm.GPTQ),
        },
    )

    returned = quantize_model(model, recipe, _toy_calibration_data())

    assert isinstance(returned, nn.Sequential)
    assert not torch.equal(returned[0].weight, model[0].weight)


def test_mixed_algorithms_share_single_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mixed recipes run calibration once with only the calibration-needed targets."""
    model = _toy_sequential()
    recipe = _recipe(
        Algorithm.RTN,
        overrides={
            "0": QuantScheme(weight_bits=8, algorithm=Algorithm.GPTQ),
        },
    )
    captured_calls: list[
        tuple[list[ObserverSpec], dict[str, dict[str, torch.Tensor]]]
    ] = []
    original_run_calibration = pipeline.run_calibration

    def wrapped(
        model: nn.Module,
        dataset: Iterable[Any],
        observers: list[ObserverSpec],
        *,
        max_samples: int = 128,
    ) -> dict[str, dict[str, torch.Tensor]]:
        result = original_run_calibration(
            model,
            dataset,
            observers,
            max_samples=max_samples,
        )
        captured_calls.append((observers, result))
        return result

    monkeypatch.setattr(pipeline, "run_calibration", wrapped)

    quantize_model(model, recipe, _toy_calibration_data())

    assert len(captured_calls) == 1
    observers, stats = captured_calls[0]
    assert len(observers) == 1
    assert observers[0].targets == frozenset({"0"})
    assert set(stats) == {"0"}
    assert set(stats["0"]) == {"hessian"}


def test_quantizer_map_populated_after_first_call() -> None:
    """The lazy quantizer dispatch table is populated after first use."""
    model = _toy_sequential()
    pipeline.QUANTIZER_MAP.clear()

    quantize_model(model, _recipe(Algorithm.RTN))

    assert set(pipeline.QUANTIZER_MAP) == set(Algorithm)


def test_bn_fusion_only_in_returned_model() -> None:
    """BatchNorm fusion happens only on the returned deep-copied model."""
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3, bias=False),
        nn.BatchNorm2d(4),
        nn.ReLU(),
    )

    returned = quantize_model(model, _recipe(Algorithm.RTN))

    assert isinstance(returned, nn.Sequential)
    assert isinstance(model[1], nn.BatchNorm2d)
    assert isinstance(returned[1], nn.Identity)


def test_typeerror_propagates_for_rtn_on_conv_transpose() -> None:
    """RTN errors on ConvTranspose2d propagate unchanged through the pipeline."""
    model = nn.Sequential(nn.ConvTranspose2d(3, 4, 3))

    with pytest.raises(TypeError, match=r"RTN quantize_layer supports"):
        quantize_model(model, _recipe(Algorithm.RTN))


def test_typeerror_propagates_for_smoothquant_on_conv2d() -> None:
    """SmoothQuant type errors after calibration propagate unchanged."""
    model = nn.Sequential(nn.Conv2d(3, 4, 3))
    calibration_data = [torch.randn(2, 3, 8, 8) for _ in range(4)]

    with pytest.raises(TypeError, match=r"SmoothQuant quantize_layer supports"):
        quantize_model(model, _recipe(Algorithm.SMOOTHQUANT), calibration_data)
