"""Fallback adapter for plain nn.Module / nn.Sequential."""

from __future__ import annotations

from torch import nn

from torchquant._types import LayerKind


def classify_module(_name: str, module: nn.Module) -> LayerKind | None:
    """Classify a module by isinstance checks only.

    Attention sub-typing is handled by the llm adapter; the generic
    adapter maps every ``nn.Linear`` to ``LINEAR`` unconditionally.

    Args:
        _name: Fully-qualified module name (unused by this adapter).
        module: The module to classify.

    Returns:
        LayerKind if quantizable, None otherwise.
    """
    if isinstance(module, nn.ConvTranspose2d):
        return LayerKind.CONV_TRANSPOSE
    if isinstance(module, nn.Conv2d):
        if module.groups == module.in_channels and module.in_channels > 1:
            return LayerKind.DEPTHWISE_CONV
        if module.kernel_size == (1, 1):
            return LayerKind.POINTWISE_CONV
        return LayerKind.CONV2D
    if isinstance(module, nn.Linear):
        return LayerKind.LINEAR
    if isinstance(module, nn.Embedding):
        return LayerKind.EMBEDDING
    return None


def find_blocks(
    model: nn.Module,
) -> list[tuple[str, nn.Module]]:
    """Return top-level child modules as structural blocks.

    Args:
        model: The model to inspect.

    Returns:
        List of (name, module) pairs for each top-level child.
    """
    return list(model.named_children())


def is_skip_target(_name: str) -> bool:
    """Return True if the layer should be skipped.

    The generic adapter does not skip anything by default.
    Skip logic is deferred to the rules engine via ``QuantRecipe.ignore``.
    """
    return False


def prepare_model(model: nn.Module) -> nn.Module:
    """Prepare model for quantization: eval mode, freeze parameters.

    Args:
        model: The model to prepare.

    Returns:
        The prepared model (same object, mutated in-place).
    """
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model
