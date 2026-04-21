"""Tests for the export_pt2e module-level rewriter."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from torchquant import build_quantized_model
from torchquant._types import Algorithm, LayerKind
from torchquant.config import QuantRecipe, QuantScheme
from torchquant.export import (
    QuantizedConv2d,
    QuantizedEmbedding,
    QuantizedLinear,
    UnsupportedExportError,
    export_pt2e,
)
from torchquant.quantizers import QuantResult, _oracle, awq, rtn
from torchquant.registry import QuantRecord, QuantRegistry


class _MixedModel(nn.Module):
    """Small mixed topology used for RTN export integration tests."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(6, 8, padding_idx=1)
        self.conv = nn.Conv2d(2, 3, kernel_size=2, stride=1, padding=0, bias=True)
        self.proj = nn.Linear(8, 4, bias=True)

    def forward(self, token_ids: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids).mean(dim=1)
        projected = self.proj(embedded)
        convolved = self.conv(image).flatten(1)
        return projected + convolved[:, : projected.shape[1]]


def _mixed_model() -> _MixedModel:
    model = _MixedModel()
    with torch.no_grad():
        model.embedding.weight.copy_(
            torch.linspace(-1.0, 1.0, steps=48, dtype=torch.float32).reshape(6, 8)
        )
        model.conv.weight.copy_(
            torch.linspace(
                -1.5,
                1.5,
                steps=model.conv.weight.numel(),
                dtype=torch.float32,
            ).reshape_as(model.conv.weight)
        )
        assert model.conv.bias is not None
        model.conv.bias.copy_(torch.linspace(-0.25, 0.25, steps=3, dtype=torch.float32))
        model.proj.weight.copy_(
            torch.linspace(-1.2, 1.2, steps=32, dtype=torch.float32).reshape(4, 8)
        )
        assert model.proj.bias is not None
        model.proj.bias.copy_(torch.linspace(-0.5, 0.5, steps=4, dtype=torch.float32))
    return model


def _linear_model() -> nn.Sequential:
    model = nn.Sequential(
        nn.Linear(8, 4, bias=True),
        nn.ReLU(),
        nn.Linear(4, 2, bias=True),
    )
    with torch.no_grad():
        first = model[0]
        second = model[2]
        assert isinstance(first, nn.Linear)
        assert isinstance(second, nn.Linear)
        first.weight.copy_(
            torch.linspace(-1.0, 1.0, steps=32, dtype=torch.float32).reshape(4, 8)
        )
        second.weight.copy_(
            torch.linspace(-0.5, 0.5, steps=8, dtype=torch.float32).reshape(2, 4)
        )
        assert first.bias is not None
        assert second.bias is not None
        first.bias.copy_(torch.linspace(-0.2, 0.2, steps=4, dtype=torch.float32))
        second.bias.copy_(torch.linspace(-0.1, 0.1, steps=2, dtype=torch.float32))
    return model


def _rtn_recipe() -> QuantRecipe:
    return QuantRecipe(
        default_scheme=QuantScheme(weight_bits=4, group_size=4, algorithm=Algorithm.RTN)
    )


def _gptq_recipe() -> QuantRecipe:
    return QuantRecipe(
        default_scheme=QuantScheme(
            weight_bits=4,
            group_size=4,
            algorithm=Algorithm.GPTQ,
        )
    )


def _sample_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64),
        torch.arange(72, dtype=torch.float32).reshape(2, 2, 6, 3) / 10,
    )


def _calibration_data() -> list[tuple[torch.Tensor, torch.Tensor]]:
    token_ids, image = _sample_inputs()
    return [
        (token_ids, image),
        (token_ids.flip(1), image + 0.1),
        (token_ids.roll(1, dims=1), image - 0.2),
    ]


def _linear_calibration_data() -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(3, 8) for _ in range(4)]


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }


def _make_awq_result() -> tuple[QuantScheme, QuantResult]:
    module = nn.Linear(8, 4, bias=True)
    with torch.no_grad():
        module.weight.copy_(
            torch.linspace(-1.0, 1.0, steps=32, dtype=torch.float32).reshape(4, 8)
        )
    scheme = QuantScheme(weight_bits=4, group_size=4, algorithm=Algorithm.AWQ)
    stats = {
        "act_mean": torch.linspace(0.25, 1.25, steps=8, dtype=torch.float32),
    }
    return scheme, awq.quantize_layer(module, scheme, stats)


def _make_rtn_result() -> tuple[QuantScheme, QuantResult]:
    module = nn.Linear(8, 4, bias=True)
    with torch.no_grad():
        module.weight.copy_(
            torch.linspace(-1.0, 1.0, steps=32, dtype=torch.float32).reshape(4, 8)
        )
    scheme = QuantScheme(weight_bits=4, group_size=4, algorithm=Algorithm.RTN)
    return scheme, rtn.quantize_layer(module, scheme, {})


def test_export_pt2e_rewrites_rtn_model_end_to_end() -> None:
    """RTN export should rewrite supported submodules into runtime modules."""
    model = _mixed_model()
    quantized_model, registry = build_quantized_model(
        model, _rtn_recipe(), _calibration_data()
    )

    exported = export_pt2e(quantized_model, registry)

    assert isinstance(exported, _MixedModel)
    assert isinstance(exported.embedding, QuantizedEmbedding)
    assert isinstance(exported.conv, QuantizedConv2d)
    assert isinstance(exported.proj, QuantizedLinear)


def test_export_pt2e_rewrites_gptq_linear_model() -> None:
    """GPTQ export should rewrite Linear layers into QuantizedLinear."""
    model = _linear_model()
    quantized_model, registry = build_quantized_model(
        model,
        _gptq_recipe(),
        _linear_calibration_data(),
    )

    exported = export_pt2e(quantized_model, registry)

    assert isinstance(exported, nn.Sequential)
    first = exported[0]
    second = exported[2]
    assert isinstance(first, QuantizedLinear)
    assert isinstance(second, QuantizedLinear)


def test_export_pt2e_does_not_mutate_inputs() -> None:
    """Export should deep-copy the quantized model before rewriting."""
    model = _mixed_model()
    quantized_model, registry = build_quantized_model(
        model, _rtn_recipe(), _calibration_data()
    )
    before = _snapshot(quantized_model)

    exported = export_pt2e(quantized_model, registry)

    assert exported is not quantized_model
    for name, tensor in quantized_model.state_dict().items():
        assert torch.equal(tensor, before[name])


def test_export_pt2e_preserves_oracle_int_weight_for_gptq() -> None:
    """Exported GPTQ Linear modules should keep the quantizer's oracle q_int."""
    model = _linear_model()
    recipe = _gptq_recipe()
    with _oracle.recording() as records:
        quantized_model, registry = build_quantized_model(
            model,
            recipe,
            _linear_calibration_data(),
        )
    exported = export_pt2e(quantized_model, registry)

    assert isinstance(exported, nn.Sequential)
    first = exported[0]
    second = exported[2]
    assert isinstance(first, QuantizedLinear)
    assert isinstance(second, QuantizedLinear)
    assert torch.equal(first.int_weight, records["0"])
    assert torch.equal(second.int_weight, records["2"])


def test_export_pt2e_forward_matches_quantized_model() -> None:
    """Exported model should match the Phase 5 fake-quant model numerically."""
    model = _mixed_model()
    quantized_model, registry = build_quantized_model(
        model, _rtn_recipe(), _calibration_data()
    )
    exported = export_pt2e(quantized_model, registry)
    sample_input = _sample_inputs()

    expected = quantized_model(*sample_input)
    actual = exported(*sample_input)
    max_scale = max(record.result.scales.max().item() for _, record in registry)

    assert torch.allclose(actual, expected, rtol=0, atol=max_scale * 1e-3)


def test_exported_runtime_modules_support_state_dict_roundtrip() -> None:
    """Each rewritten submodule should support export_state_dict roundtrips."""
    model = _mixed_model()
    quantized_model, registry = build_quantized_model(
        model, _rtn_recipe(), _calibration_data()
    )
    exported = export_pt2e(quantized_model, registry)
    token_ids, image = _sample_inputs()

    assert isinstance(exported, _MixedModel)
    embedding = exported.embedding
    conv = exported.conv
    proj = exported.proj
    assert isinstance(embedding, QuantizedEmbedding)
    assert isinstance(conv, QuantizedConv2d)
    assert isinstance(proj, QuantizedLinear)

    rebuilt_embedding = QuantizedEmbedding.rebuild_from_state_dict(
        embedding.export_state_dict()
    )
    rebuilt_conv = QuantizedConv2d.rebuild_from_state_dict(conv.export_state_dict())
    rebuilt_linear = QuantizedLinear.rebuild_from_state_dict(proj.export_state_dict())

    assert torch.equal(rebuilt_embedding(token_ids), embedding(token_ids))
    assert torch.equal(rebuilt_conv(image), conv(image))
    embedded = embedding(token_ids).mean(dim=1)
    assert torch.equal(rebuilt_linear(embedded), proj(embedded))


def test_exported_model_is_torch_export_compatible() -> None:
    """The exported model should remain compatible with torch.export."""
    model = _mixed_model()
    quantized_model, registry = build_quantized_model(
        model, _rtn_recipe(), _calibration_data()
    )
    exported = export_pt2e(quantized_model, registry)
    sample_input = _sample_inputs()

    exported_program = torch.export.export(exported, sample_input)

    assert torch.allclose(
        exported_program.module()(*sample_input),
        exported(*sample_input),
        rtol=0,
        atol=0,
    )


def test_export_pt2e_batches_preflight_errors() -> None:
    """Preflight should report unsupported algorithms and module types together."""
    model = nn.Sequential(nn.Linear(8, 4, bias=True), nn.LSTM(4, 4))
    awq_scheme, awq_result = _make_awq_result()
    rtn_scheme, rtn_result = _make_rtn_result()
    first = model[0]
    assert isinstance(first, nn.Linear)
    registry = QuantRegistry()
    registry.add(
        QuantRecord(
            fqn="0",
            kind=LayerKind.LINEAR,
            scheme=awq_scheme,
            result=copy.deepcopy(awq_result),
            original_bias=first.bias,
        )
    )
    registry.add(
        QuantRecord(
            fqn="1",
            kind=LayerKind.LINEAR,
            scheme=rtn_scheme,
            result=copy.deepcopy(rtn_result),
            original_bias=None,
        )
    )

    with pytest.raises(UnsupportedExportError, match=r"(?s).*0: .*AWQ.*1: .*LSTM"):
        export_pt2e(model, registry)


def test_export_pt2e_rejects_conv_transpose() -> None:
    """ConvTranspose2d should fail preflight because no rewriter is registered."""
    model = nn.Sequential(nn.ConvTranspose2d(2, 2, 1))
    scheme, result = _make_rtn_result()
    registry = QuantRegistry()
    registry.add(
        QuantRecord(
            fqn="0",
            kind=LayerKind.CONV_TRANSPOSE,
            scheme=scheme,
            result=copy.deepcopy(result),
            original_bias=None,
        )
    )

    with pytest.raises(UnsupportedExportError, match=r"ConvTranspose2d"):
        export_pt2e(model, registry)


def test_export_pt2e_rejects_awq_record() -> None:
    """AWQ records should fail preflight with the Phase 6 limitation message."""
    model = _linear_model()
    recipe = QuantRecipe(
        default_scheme=QuantScheme(
            weight_bits=4, group_size=4, algorithm=Algorithm.RTN
        ),
        overrides={
            "0": QuantScheme(weight_bits=4, group_size=4, algorithm=Algorithm.AWQ)
        },
    )
    quantized_model, registry = build_quantized_model(
        model,
        recipe,
        _linear_calibration_data(),
    )

    with pytest.raises(UnsupportedExportError, match=r"0: .*AWQ.*not exportable"):
        export_pt2e(quantized_model, registry)
