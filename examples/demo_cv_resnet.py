"""torchquant CV demo: ResNet18 weight-only quantization (marimo notebook)."""

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # torchquant · CV Demo (ResNet18)

    Weight-only quantization end-to-end on torchvision ResNet18 with ImageNet
    weights. Flow: `import` → load → adapter → nodes → recipe → quantize →
    compare outputs → export → footprint → round-trip.
    """)
    return


@app.cell
def _():
    import os
    import tempfile
    from collections import Counter

    import torch
    import torch.nn.functional as F
    import torchvision
    from torchvision.models import ResNet18_Weights

    from torchquant import (
        Algorithm,
        QuantizedConv2d,
        QuantizedLinear,
        QuantRecipe,
        QuantScheme,
        build_quantized_model,
        export_pt2e,
    )
    from torchquant.adapters import get_adapter
    from torchquant.graph import find_quantizable_nodes

    torch.manual_seed(0)
    return (
        Algorithm,
        Counter,
        F,
        QuantRecipe,
        QuantScheme,
        QuantizedConv2d,
        QuantizedLinear,
        ResNet18_Weights,
        build_quantized_model,
        export_pt2e,
        find_quantizable_nodes,
        get_adapter,
        os,
        tempfile,
        torch,
        torchvision,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load pretrained ResNet18
    """)
    return


@app.cell
def _(ResNet18_Weights, torchvision):
    fp32_model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    fp32_model.train(False)
    return (fp32_model,)


@app.cell
def _(fp32_model, mo):
    num_params = sum(p.numel() for p in fp32_model.parameters())
    mo.md(
        f"- Architecture: **{type(fp32_model).__name__}**  \n"
        f"- Parameters: **{num_params / 1e6:.2f} M**  \n"
        f"- Weights: `ResNet18_Weights.DEFAULT` (ImageNet-1k)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Auto-detect adapter · discover quantizable nodes
    """)
    return


@app.cell
def _(find_quantizable_nodes, fp32_model, get_adapter):
    adapter = get_adapter(fp32_model)
    nodes = find_quantizable_nodes(fp32_model, adapter)
    adapter_name = adapter.classify_module.__module__.rsplit(".", maxsplit=1)[-1]
    return adapter_name, nodes


@app.cell
def _(Counter, adapter_name, mo, nodes):
    kind_counts = Counter(node.kind.name for node in nodes)
    breakdown = "\n".join(f"- `{k}` × **{v}**" for k, v in sorted(kind_counts.items()))
    mo.md(
        f"**Adapter**: `{adapter_name}` · discovered **{len(nodes)}** quantizable layers\n\n{breakdown}"
    )
    return


@app.cell
def _(mo, nodes):
    top_nodes = sorted(nodes, key=lambda n: -n.param_count)[:10]
    rows = [
        {"fqn": n.fqn, "kind": n.kind.name, "params": f"{n.param_count:,}"}
        for n in top_nodes
    ]
    mo.vstack(
        [
            mo.md("**Top 10 layers by parameter count**"),
            mo.ui.table(rows, selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Build a `QuantRecipe`

    Default INT8 RTN per-channel; override `fc` (ImageNet classifier) with INT4 RTN `group_size=128` to demonstrate per-layer policy.
    """)
    return


@app.cell
def _(Algorithm, QuantRecipe, QuantScheme):
    recipe = QuantRecipe(
        default_scheme=QuantScheme(
            weight_bits=8, group_size=-1, symmetric=True, algorithm=Algorithm.RTN
        ),
        overrides={
            "fc": QuantScheme(
                weight_bits=4, group_size=128, symmetric=True, algorithm=Algorithm.RTN
            ),
        },
    )
    return (recipe,)


@app.cell
def _(mo, recipe):
    mo.md(f"""
    ```python\n{recipe.model_dump()}\n```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Quantize the model
    """)
    return


@app.cell
def _(build_quantized_model, fp32_model, recipe):
    quantized_model, registry = build_quantized_model(fp32_model, recipe)
    return quantized_model, registry


@app.cell
def _(mo, registry):
    records = list(registry)
    fqn, record = records[0]
    mo.md(
        f"- Registry holds **{len(records)}** `QuantRecord` entries.\n"
        f"- sample `{fqn}`: `{record.kind.name}` · bits={record.scheme.weight_bits} · group_size={record.scheme.group_size}\n"
        f"- scales shape: `{tuple(record.result.scales.shape)}`"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Compare outputs: FP32 vs fake-quantized
    """)
    return


@app.cell
def _(fp32_model, quantized_model, torch):
    torch.manual_seed(0)
    batch = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        y_fp32 = fp32_model(batch)
        y_quant = quantized_model(batch)
    return batch, y_fp32, y_quant


@app.cell
def _(F, mo, y_fp32, y_quant):
    mse_q = F.mse_loss(y_quant, y_fp32).item()
    cos_q = F.cosine_similarity(y_quant.flatten(1), y_fp32.flatten(1)).mean().item()
    top1_q = (y_quant.argmax(-1) == y_fp32.argmax(-1)).float().mean().item()
    mo.md(
        f"| Metric | Value |\n|---|---|\n"
        f"| MSE (logits) | `{mse_q:.4e}` |\n"
        f"| Cosine similarity | `{cos_q:.6f}` |\n"
        f"| Top-1 label agreement | `{top1_q:.2%}` |"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Export to module-level runtime
    """)
    return


@app.cell
def _(export_pt2e, quantized_model, registry):
    exported = export_pt2e(quantized_model, registry)
    return (exported,)


@app.cell
def _(QuantizedConv2d, QuantizedLinear, exported, mo):
    counts = {}
    for _, m in exported.named_modules():
        if isinstance(m, (QuantizedConv2d, QuantizedLinear)):
            name = type(m).__name__
            counts[name] = counts.get(name, 0) + 1
    export_breakdown = "\n".join(
        f"- `{k}` × **{v}**" for k, v in sorted(counts.items())
    )
    mo.md(
        f"Rewrote **{sum(counts.values())}** layers into runtime modules:\n\n{export_breakdown}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Footprint: measured on-disk vs. logical packed size

    - **Measured on-disk** — what `torch.save(state_dict)` writes today (every `int_weight` lives in an `int32` slot).
    - **Logical packed** — same payload with `int_weight` stored at its declared bit-width (int4 → 0.5 B, int8 → 1 B) plus scales / zero-points.

    The gap is the MVP scope boundary: the current export layer stores one `int32` per weight for correctness and API simplicity; bit-packed on-disk storage is a future optimization.
    """)
    return


@app.cell
def _(exported, fp32_model, os, tempfile, torch):
    with tempfile.TemporaryDirectory() as tmp:
        p1 = os.path.join(tmp, "fp32.pt")
        p2 = os.path.join(tmp, "exp.pt")
        torch.save(fp32_model.state_dict(), p1)
        torch.save(exported.state_dict(), p2)
        fp32_mb = os.path.getsize(p1) / 1e6
        exp_mb = os.path.getsize(p2) / 1e6
    return exp_mb, fp32_mb


@app.cell
def _(QuantizedConv2d, QuantizedLinear, exported):
    def _packed_bytes(module):
        total = 0.0
        seen: set[int] = set()

        def _add(tensor, bytes_per_elem):
            nonlocal total
            ptr = tensor.data_ptr()
            if ptr in seen:
                return
            seen.add(ptr)
            total += tensor.numel() * bytes_per_elem

        for leaf in module.modules():
            if isinstance(leaf, (QuantizedConv2d, QuantizedLinear)):
                _add(leaf.int_weight, leaf.scheme.weight_bits / 8.0)
                _add(leaf.scales, 4)
                if leaf.zero_points is not None:
                    _add(leaf.zero_points, 1)
                if leaf.bias is not None:
                    _add(leaf.bias, leaf.bias.element_size())
            elif not list(leaf.children()):
                for p in leaf.parameters(recurse=False):
                    _add(p, p.element_size())
                for b in leaf.buffers(recurse=False):
                    _add(b, b.element_size())
        return total

    logical_mb = _packed_bytes(exported) / 1e6
    return (logical_mb,)


@app.cell
def _(exp_mb, fp32_mb, logical_mb, mo):
    mo.md(f"""
    | Artifact | Size | vs FP32 |\n|---|---|---|\n"
        f"| FP32 `state_dict` | **{fp32_mb:.2f} MB** | 100.0% |\n"
        f"| Exported `state_dict` (measured) | **{exp_mb:.2f} MB** | {exp_mb / fp32_mb * 100:.1f}% |\n"
        f"| Exported — **logical packed** (future target) | **{logical_mb:.2f} MB** | {logical_mb / fp32_mb * 100:.1f}% |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Round-trip: save → load → forward
    """)
    return


@app.cell
def _(batch, exported, os, tempfile, torch):
    import copy as _copy

    with tempfile.TemporaryDirectory() as tmp2:
        ckpt = os.path.join(tmp2, "exported.pt")
        torch.save(exported.state_dict(), ckpt)
        reloaded = _copy.deepcopy(exported)
        reloaded.load_state_dict(torch.load(ckpt, weights_only=True))
        reloaded.train(False)
        with torch.no_grad():
            y_reloaded = reloaded(batch)
            y_exp = exported(batch)
    return y_exp, y_reloaded


@app.cell
def _(mo, torch, y_exp, y_reloaded):
    max_abs = (y_reloaded - y_exp).abs().max().item()
    bit_exact = torch.allclose(y_reloaded, y_exp, atol=0.0, rtol=0.0)
    mo.md(
        f"- Max abs difference after reload: `{max_abs:.2e}`\n"
        f"- Bit-exact match: **{bit_exact}**"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Takeaways

    - Public API is three symbols: `QuantRecipe`, `build_quantized_model`, `export_pt2e`.
    - Per-layer overrides compose inside a `QuantRecipe`.
    - The exported artifact is a plain `nn.Module`; downstream deployment (torch.export, ExecuTorch, torch.compile, …) is the caller's choice.
    - MVP ships algorithmic quantization. Storage packing (`int_weight` → packed int8/int4) is future optimization.
    """)
    return


if __name__ == "__main__":
    app.run()
