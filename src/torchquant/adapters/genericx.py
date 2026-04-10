"""Fallback adapter for plain nn.Module / nn.Sequential."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch 
from torch import nn
from torchquant._types import LayerKind
import segmentation_models_pytorch as smp
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from torch import nn
    from torchquant._types import LayerKind

#------------------------------------------------------
# Generic functions. 
#------------------------------------------------------
def classify_module(name: str, module: nn.Module) -> LayerKind | None:
    """Classify a module by isinstance checks.
    Args:
        name: Fully-qualified module name.
        module: The module to classify.
    Returns:
        LayerKind if quantizable, None otherwise.
    """
    #-------------------------------
    if isinstance(module, nn.ConvTranspose2d):
        return LayerKind.CONV_TRANSPOSE
    if isinstance(module, nn.Conv2d):
        if module.groups == module.in_channels and module.in_channels > 1:
            return LayerKind.DEPTHWISE_CONV
        elif module.kernel_size == (1, 1):
            return LayerKind.POINTWISE_CONV
        return LayerKind.CONV2D
    
    if isinstance(module, nn.Linear):
        if any(kw in name for kw in ["q_proj", "k_proj", "v_proj", "qkv", "in_proj"]):
            return LayerKind.ATTENTION_QKV
        if any(kw in name for kw in ["out_proj", "o_proj", "attn.c_proj"]):
            return LayerKind.ATTENTION_OUT
        return LayerKind.LINEAR
    
    if isinstance(module, nn.Embedding):
        return LayerKind.EMBEDDING
#    
    return None
#
#    raise NotImplementedError
#------------------------------------------------------



#------------------------------------------------------
def find_blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return top-level child modules as blocks.

    Args:
        model: The model to inspect.

    Returns:
        List of (name, module) pairs for each top-level block.
    """
    # Get top-level blocks
    results = []
    for name, layer in model.named_modules():
        if not list(layer.children()):
            if is_skip_target(name):
                continue
            kind = classify_module(name, layer)
            if kind:
                #print(f"{name}    :    {kind}")
                results.append((name, layer))
    return results

#    raise NotImplementedError
#------------------------------------------------------

#------------------------------------------------------
def is_skip_target(name: str) -> bool:
    """Return True if the layer should be skipped.

    The generic adapter does not skip anything by default.

    Args:
        name: Fully-qualified module name.
    """
    skip_keywords=["lm_head", "embed", "norm", "head", "dropout", "act", "pooler"]
    return any(kw in name for kw in skip_keywords)
#------------------------------------------------------

#------------------------------------------------------
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
#------------------------------------------------------



#=============================================================
# main function is to test the coverage of this modules 
# on different types of models (Eg: LLM, SMP nad others) 
#=============================================================
if __name__ == "__main__":
    modelx = smp.Unet( encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    results=find_blocks(modelx)
    #tokenizer = AutoTokenizer.from_pretrained("gpt2")
    #model2 = AutoModelForCausalLM.from_pretrained("gpt2")
    #results=find_blocks(model2)

    for name, layer in results:
        print(name, ":", layer.__class__.__name__)
