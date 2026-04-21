"""Shared fixtures for module rewriter tests."""

from __future__ import annotations

import torch
from torch import nn

from torchquant._types import Algorithm, LayerKind
from torchquant.config import QuantScheme
from torchquant.quantizers import _oracle, gptq, rtn
from torchquant.registry import QuantRecord


def make_linear_module(*, bias: bool = True) -> nn.Linear:
    """Create a deterministic Linear layer for rewrite tests."""
    module = nn.Linear(8, 4, bias=bias, dtype=torch.float32)
    weight = torch.linspace(-1.5, 1.5, steps=32, dtype=torch.float32).reshape(4, 8)
    with torch.no_grad():
        module.weight.copy_(weight)
        if module.bias is not None:
            module.bias.copy_(torch.linspace(-0.25, 0.25, steps=4, dtype=torch.float32))
    return module


def make_conv_module() -> nn.Conv2d:
    """Create a deterministic Conv2d layer for rewrite tests."""
    module = nn.Conv2d(2, 3, kernel_size=2, stride=2, padding=1, dilation=1, bias=True)
    weight = torch.linspace(
        -2.0,
        2.0,
        steps=module.weight.numel(),
        dtype=torch.float32,
    ).reshape_as(module.weight)
    with torch.no_grad():
        module.weight.copy_(weight)
        assert module.bias is not None
        module.bias.copy_(torch.linspace(-0.5, 0.5, steps=3, dtype=torch.float32))
    return module


def make_embedding_module() -> nn.Embedding:
    """Create a deterministic Embedding layer for rewrite tests."""
    module = nn.Embedding(6, 8, padding_idx=1)
    weight = torch.linspace(-1.0, 1.0, steps=48, dtype=torch.float32).reshape(6, 8)
    with torch.no_grad():
        module.weight.copy_(weight)
    return module


def make_scheme(
    algorithm: Algorithm,
    *,
    bits: int = 4,
    group_size: int = -1,
    symmetric: bool = True,
) -> QuantScheme:
    """Construct a quantization scheme for rewrite tests."""
    return QuantScheme(
        weight_bits=bits,
        group_size=group_size,
        symmetric=symmetric,
        algorithm=algorithm,
    )


def make_linear_record(
    algorithm: Algorithm,
    *,
    bits: int = 4,
    group_size: int = -1,
    symmetric: bool = True,
    bias: bool = True,
    fqn: str = "linear",
) -> tuple[nn.Linear, QuantRecord, torch.Tensor]:
    """Quantize a Linear layer and return its record plus oracle q_int."""
    module = make_linear_module(bias=bias)
    scheme = make_scheme(
        algorithm,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
    )
    quantizer = (
        rtn.quantize_layer if algorithm is Algorithm.RTN else gptq.quantize_layer
    )
    stats = {} if algorithm is Algorithm.RTN else {"hessian": torch.eye(8)}
    with _oracle.recording() as records, _oracle.bound_fqn(module, fqn):
        result = quantizer(module, scheme, stats)
    return (
        module,
        QuantRecord(
            fqn=fqn,
            kind=LayerKind.LINEAR,
            scheme=scheme,
            result=result,
            original_bias=module.bias,
        ),
        records[fqn],
    )


def make_conv_record(
    *,
    bits: int = 4,
    group_size: int = 4,
    symmetric: bool = True,
    fqn: str = "conv",
) -> tuple[nn.Conv2d, QuantRecord, torch.Tensor]:
    """Quantize a Conv2d layer and return its record plus oracle q_int."""
    module = make_conv_module()
    scheme = make_scheme(
        Algorithm.RTN,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
    )
    with _oracle.recording() as records, _oracle.bound_fqn(module, fqn):
        result = rtn.quantize_layer(module, scheme, {})
    return (
        module,
        QuantRecord(
            fqn=fqn,
            kind=LayerKind.CONV2D,
            scheme=scheme,
            result=result,
            original_bias=module.bias,
        ),
        records[fqn],
    )


def make_embedding_record(
    *,
    bits: int = 4,
    group_size: int = 4,
    symmetric: bool = True,
    fqn: str = "embedding",
) -> tuple[nn.Embedding, QuantRecord, torch.Tensor]:
    """Quantize an Embedding layer and return its record plus oracle q_int."""
    module = make_embedding_module()
    scheme = make_scheme(
        Algorithm.RTN,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
    )
    with _oracle.recording() as records, _oracle.bound_fqn(module, fqn):
        result = rtn.quantize_layer(module, scheme, {})
    return (
        module,
        QuantRecord(
            fqn=fqn,
            kind=LayerKind.EMBEDDING,
            scheme=scheme,
            result=result,
            original_bias=None,
        ),
        records[fqn],
    )
