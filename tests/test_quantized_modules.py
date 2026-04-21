"""Tests for exported quantized runtime modules."""

from __future__ import annotations

import pytest
import torch

from torchquant.export.runtime import (
    QuantizedConv2d,
    QuantizedEmbedding,
    QuantizedLinear,
)
from torchquant.export.runtime.conv import Conv2dSpec
from torchquant.export.runtime.embedding import EmbeddingSpec

from ._quantized_module_fixtures import (
    make_conv_case,
    make_embedding_case,
    make_linear_case,
    make_scheme,
)


def test_quantized_linear_registers_buffers() -> None:
    """Linear runtime module registers metadata as buffers."""
    module, _, _ = make_linear_case()

    assert {name for name, _ in module.named_buffers()} == {
        "int_weight",
        "scales",
        "zero_points",
        "bias",
    }


def test_quantized_conv2d_registers_buffers() -> None:
    """Conv runtime module registers metadata as buffers."""
    module, _, _ = make_conv_case()

    assert {name for name, _ in module.named_buffers()} == {
        "int_weight",
        "scales",
        "zero_points",
        "bias",
    }


def test_quantized_embedding_registers_buffers() -> None:
    """Embedding runtime module registers metadata as buffers."""
    module, _, _ = make_embedding_case()

    assert {name for name, _ in module.named_buffers()} == {
        "int_weight",
        "scales",
        "zero_points",
    }


def test_quantized_linear_forward_matches_reference() -> None:
    """QuantizedLinear forward matches a float nn.Linear reference."""
    module, reference, sample_input = make_linear_case()

    output = module(sample_input)
    expected = reference(sample_input)

    assert torch.allclose(
        output, expected, rtol=0, atol=module.scales.max().item() * 1e-3
    )
    assert torch.equal(module.dequantize_weight(), reference.weight.detach())


def test_quantized_conv2d_forward_matches_reference() -> None:
    """QuantizedConv2d forward matches a float nn.Conv2d reference."""
    module, reference, sample_input = make_conv_case()

    output = module(sample_input)
    expected = reference(sample_input)

    assert torch.allclose(
        output, expected, rtol=0, atol=module.scales.max().item() * 1e-3
    )
    assert torch.equal(module.dequantize_weight(), reference.weight.detach())


def test_quantized_embedding_forward_matches_reference() -> None:
    """QuantizedEmbedding forward matches a float nn.Embedding reference."""
    module, reference, sample_input = make_embedding_case()

    output = module(sample_input)
    expected = reference(sample_input)

    assert torch.allclose(
        output, expected, rtol=0, atol=module.scales.max().item() * 1e-3
    )
    assert torch.equal(module.dequantize_weight(), reference.weight.detach())


@torch.no_grad()
def test_extra_repr_includes_quantization_metadata() -> None:
    """All runtime modules include bits, group size, and algorithm in repr."""
    modules = [make_linear_case()[0], make_conv_case()[0], make_embedding_case()[0]]

    for module in modules:
        summary = module.extra_repr()
        assert "bits=4" in summary
        assert "group_size=" in summary
        assert "algorithm=RTN" in summary


def test_export_state_dict_roundtrip_preserves_linear_output() -> None:
    """QuantizedLinear rebuilds losslessly from export_state_dict data."""
    module, _, sample_input = make_linear_case()

    rebuilt = QuantizedLinear.rebuild_from_state_dict(module.export_state_dict())

    assert torch.equal(module(sample_input), rebuilt(sample_input))


def test_export_state_dict_roundtrip_preserves_conv_output() -> None:
    """QuantizedConv2d rebuilds losslessly from export_state_dict data."""
    module, _, sample_input = make_conv_case()

    rebuilt = QuantizedConv2d.rebuild_from_state_dict(module.export_state_dict())

    assert torch.equal(module(sample_input), rebuilt(sample_input))


def test_export_state_dict_roundtrip_preserves_embedding_output() -> None:
    """QuantizedEmbedding rebuilds losslessly from export_state_dict data."""
    module, _, sample_input = make_embedding_case()

    rebuilt = QuantizedEmbedding.rebuild_from_state_dict(module.export_state_dict())

    assert torch.equal(module(sample_input), rebuilt(sample_input))


def test_torch_save_roundtrip_preserves_linear_output(tmp_path) -> None:
    """QuantizedLinear survives full-module torch.save / torch.load."""
    module, _, sample_input = make_linear_case()
    path = tmp_path / "linear.pt"

    torch.save(module, path)
    loaded = torch.load(path, weights_only=False)

    assert isinstance(loaded, QuantizedLinear)
    assert torch.equal(module(sample_input), loaded(sample_input))


def test_torch_save_roundtrip_preserves_conv_output(tmp_path) -> None:
    """QuantizedConv2d survives full-module torch.save / torch.load."""
    module, _, sample_input = make_conv_case()
    path = tmp_path / "conv.pt"

    torch.save(module, path)
    loaded = torch.load(path, weights_only=False)

    assert isinstance(loaded, QuantizedConv2d)
    assert torch.equal(module(sample_input), loaded(sample_input))


def test_torch_save_roundtrip_preserves_embedding_output(tmp_path) -> None:
    """QuantizedEmbedding survives full-module torch.save / torch.load."""
    module, _, sample_input = make_embedding_case()
    path = tmp_path / "embedding.pt"

    torch.save(module, path)
    loaded = torch.load(path, weights_only=False)

    assert isinstance(loaded, QuantizedEmbedding)
    assert torch.equal(module(sample_input), loaded(sample_input))


def test_linear_shape_validation_raises() -> None:
    """Linear constructor rejects incompatible metadata shapes."""
    scheme = make_scheme(group_size=2, symmetric=False)

    with pytest.raises(ValueError, match="scales shape mismatch"):
        QuantizedLinear(
            int_weight=torch.ones((2, 4), dtype=torch.int32),
            scales=torch.ones((3, 2), dtype=torch.float32),
            zero_points=torch.ones((2, 2), dtype=torch.int32),
            bias=torch.zeros(2, dtype=torch.float32),
            scheme=scheme,
            in_features=4,
            out_features=2,
            weight_dtype=torch.float32,
        )


def test_conv_shape_validation_raises() -> None:
    """Conv constructor rejects incompatible metadata shapes."""
    scheme = make_scheme(group_size=4, symmetric=False)
    spec = Conv2dSpec(
        in_channels=2,
        out_channels=2,
        kernel_size=(2, 2),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
    )

    with pytest.raises(ValueError, match="scales shape mismatch"):
        QuantizedConv2d(
            int_weight=torch.ones((2, 2, 2, 2), dtype=torch.int32),
            scales=torch.ones((1, 2), dtype=torch.float32),
            zero_points=torch.ones((2, 2), dtype=torch.int32),
            bias=torch.zeros(2, dtype=torch.float32),
            scheme=scheme,
            spec=spec,
            weight_dtype=torch.float32,
        )


def test_embedding_shape_validation_raises() -> None:
    """Embedding constructor rejects incompatible metadata shapes."""
    scheme = make_scheme(group_size=2, symmetric=False)
    spec = EmbeddingSpec(
        num_embeddings=5,
        embedding_dim=4,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
    )

    with pytest.raises(ValueError, match="scales shape mismatch"):
        QuantizedEmbedding(
            int_weight=torch.ones((5, 4), dtype=torch.int32),
            scales=torch.ones((4, 2), dtype=torch.float32),
            zero_points=torch.ones((5, 2), dtype=torch.int32),
            scheme=scheme,
            spec=spec,
            weight_dtype=torch.float32,
        )


def test_bias_dtype_is_preserved() -> None:
    """Bias buffers keep the caller-provided dtype."""
    module = QuantizedLinear(
        int_weight=torch.ones((2, 4), dtype=torch.int32),
        scales=torch.ones((2, 2), dtype=torch.float32),
        zero_points=torch.ones((2, 2), dtype=torch.int32),
        bias=torch.ones(2, dtype=torch.float16),
        scheme=make_scheme(group_size=2, symmetric=False),
        in_features=4,
        out_features=2,
        weight_dtype=torch.float32,
    )

    assert module.bias is not None
    assert module.bias.dtype == torch.float16
