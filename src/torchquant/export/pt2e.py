"""PT2E / XNNPACK / ExecuTorch export path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn

    from torchquant.registry import QuantRegistry


def export_pt2e(
    model: nn.Module,
    registry: QuantRegistry,
    *,
    sample_input: Any = None,
) -> Any:
    """Export a quantized model to PT2E format.

    Traces the model using torch.export, applies quantization annotations
    from the registry, and lowers to ExecuTorch-compatible format.

    Args:
        model: The quantized model.
        registry: Registry containing quantization metadata.
        sample_input: Example input for tracing.

    Returns:
        The exported program (ExportedProgram or ExecuTorch artifact).
    """
    raise NotImplementedError
