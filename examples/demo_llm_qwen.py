"""torchquant LLM demo: Qwen3.5-0.8B GPTQ 4bit quantization (marimo notebook)."""

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
    # torchquant · LLM Demo (Qwen3.5-0.8B · GPTQ 4bit)

    End-to-end weight-only GPTQ on a HuggingFace causal LM. Flow: `import`
    → load tokenizer & model → `llm` adapter auto-detects attention
    projections → build calibration data → `build_quantized_model` runs
    GPTQ with Hessian observer → generate text before/after → export →
    footprint comparison → round-trip.
    """)
    return


@app.cell
def _():
    import os
    import tempfile

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from torchquant import (
        Algorithm,
        QuantizedEmbedding,
        QuantizedLinear,
        QuantRecipe,
        QuantScheme,
        build_quantized_model,
        export_pt2e,
    )
    from torchquant.adapters import get_adapter
    from torchquant.graph import find_quantizable_nodes

    torch.manual_seed(0)
    model_id = "Qwen/Qwen3.5-0.8B"
    return (
        Algorithm,
        AutoModelForCausalLM,
        AutoTokenizer,
        QuantRecipe,
        QuantScheme,
        QuantizedEmbedding,
        QuantizedLinear,
        build_quantized_model,
        export_pt2e,
        find_quantizable_nodes,
        get_adapter,
        model_id,
        os,
        tempfile,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load tokenizer & model
    """)
    return


@app.cell
def _(AutoModelForCausalLM, AutoTokenizer, model_id, torch):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    fp_model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cpu"
    )
    fp_model.train(False)
    return fp_model, tokenizer


@app.cell
def _(fp_model, mo):
    total_params = sum(p.numel() for p in fp_model.parameters())
    mo.md(
        f"- Architecture: **{type(fp_model).__name__}**  \n"
        f"- Parameters: **{total_params / 1e9:.2f} B**  \n"
        f"- dtype: `{next(fp_model.parameters()).dtype}`"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Adapter auto-dispatch · node discovery
    """)
    return


@app.cell
def _(find_quantizable_nodes, fp_model, get_adapter):
    adapter = get_adapter(fp_model)
    nodes = find_quantizable_nodes(fp_model, adapter)
    adapter_name = adapter.classify_module.__module__.rsplit(".", maxsplit=1)[-1]
    return adapter_name, nodes


@app.cell
def _(adapter_name, mo, nodes):
    from collections import Counter

    kind_counter = Counter(n.kind.name for n in nodes)
    node_breakdown = "\n".join(
        f"- `{k}` × **{v}**" for k, v in sorted(kind_counter.items())
    )
    mo.md(
        f"**Adapter**: `{adapter_name}` · **{len(nodes)}** quantizable layers\n\n{node_breakdown}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Calibration data

    GPTQ needs a small set of representative inputs to estimate the Hessian.
    We tokenize a handful of prompts; each `BatchEncoding` becomes one
    calibration sample (dispatched via `model(**sample)`).
    """)
    return


@app.cell
def _(tokenizer):
    calib_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "In 2026, neural network quantization has become standard practice for",
        "The principle of least surprise suggests that software interfaces should",
        "Albert Einstein is best known for his theory of relativity, which states",
        "To compile a kernel in Linux, the first step is to clone the source tree",
        "Transformer architectures rely on self-attention, which computes",
        "def fibonacci(n: int) -> int:\n    if n < 2:\n        return n\n",
        "The capital of France is",
    ]
    calib_samples = [
        tokenizer(p, return_tensors="pt", max_length=64, truncation=True)
        for p in calib_prompts
    ]
    return calib_prompts, calib_samples


@app.cell
def _(calib_prompts, calib_samples, mo):
    mo.md(f"""
    - **{len(calib_prompts)}** calibration prompts tokenized.\n"
        f"- Sample 0 `input_ids` shape: `{tuple(calib_samples[0]["input_ids"].shape)}`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Build the recipe

    GPTQ 4bit, `group_size=128`, symmetric. We ignore `lm_head` (precision-
    sensitive output projection) and `model.embed_tokens` (often tied to
    `lm_head`, keeping them float avoids tie-break headaches for this demo).
    """)
    return


@app.cell
def _(Algorithm, QuantRecipe, QuantScheme):
    recipe = QuantRecipe(
        default_scheme=QuantScheme(
            weight_bits=4,
            group_size=128,
            symmetric=True,
            algorithm=Algorithm.GPTQ,
        ),
        ignore=frozenset({"lm_head", "model.embed_tokens"}),
        calibration_samples=8,
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
    ## 5. Quantize the model

    `build_quantized_model` runs the 7-stage pipeline: graph → rules →
    Hessian observer → GPTQ blockwise update → registry. This is the slow
    step on CPU (~1-2 min); grab a coffee.
    """)
    return


@app.cell
def _(build_quantized_model, calib_samples, fp_model, recipe):
    quant_model, registry = build_quantized_model(fp_model, recipe, calib_samples)
    return quant_model, registry


@app.cell
def _(mo, registry):
    records = list(registry)
    fqn0, rec0 = records[0]
    mo.md(
        f"- Registry holds **{len(records)}** `QuantRecord` entries.\n"
        f"- sample `{fqn0}`: `{rec0.kind.name}` · bits={rec0.scheme.weight_bits} · group_size={rec0.scheme.group_size}\n"
        f"- scales shape: `{tuple(rec0.result.scales.shape)}`"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Generate text: FP16 vs GPTQ-quantized
    """)
    return


@app.cell
def _(fp_model, quant_model, tokenizer, torch):
    gen_prompt = "The three fundamental laws of software engineering are"
    inputs = tokenizer(gen_prompt, return_tensors="pt")

    with torch.no_grad():
        out_fp = fp_model.generate(**inputs, max_new_tokens=40, do_sample=False)
        out_q = quant_model.generate(**inputs, max_new_tokens=40, do_sample=False)

    text_fp = tokenizer.decode(out_fp[0], skip_special_tokens=True)
    text_q = tokenizer.decode(out_q[0], skip_special_tokens=True)
    return inputs, text_fp, text_q


@app.cell
def _(mo, text_fp, text_q):
    mo.vstack(
        [
            mo.md("**FP baseline**"),
            mo.md(f"> {text_fp}"),
            mo.md("**GPTQ 4bit**"),
            mo.md(f"> {text_q}"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Export to module-level runtime
    """)
    return


@app.cell
def _(export_pt2e, quant_model, registry):
    exported = export_pt2e(quant_model, registry)
    return (exported,)


@app.cell
def _(QuantizedEmbedding, QuantizedLinear, exported, mo):
    rt_counts = {}
    for _, m in exported.named_modules():
        if isinstance(m, (QuantizedLinear, QuantizedEmbedding)):
            n = type(m).__name__
            rt_counts[n] = rt_counts.get(n, 0) + 1
    rt_breakdown = "\n".join(f"- `{k}` × **{v}**" for k, v in sorted(rt_counts.items()))
    mo.md(
        f"Rewrote **{sum(rt_counts.values())}** layers into runtime modules:\n\n{rt_breakdown}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Footprint: measured on-disk vs. logical packed

    Same MVP scope boundary as the CV demo: `int_weight` currently lives in
    `int32` slots for API simplicity. The **logical packed** column shows
    what bit-packed on-disk storage would achieve on this exact payload.
    """)
    return


@app.cell
def _(exported, fp_model, os, tempfile, torch):
    with tempfile.TemporaryDirectory() as tmp:
        p1 = os.path.join(tmp, "fp.pt")
        p2 = os.path.join(tmp, "exp.pt")
        torch.save(fp_model.state_dict(), p1)
        torch.save(exported.state_dict(), p2)
        fp_mb = os.path.getsize(p1) / 1e6
        exp_mb = os.path.getsize(p2) / 1e6
    return exp_mb, fp_mb


@app.cell
def _(QuantizedEmbedding, QuantizedLinear, exported):
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
            if isinstance(leaf, (QuantizedLinear, QuantizedEmbedding)):
                _add(leaf.int_weight, leaf.scheme.weight_bits / 8.0)
                _add(leaf.scales, 4)
                if leaf.zero_points is not None:
                    _add(leaf.zero_points, 1)
                if getattr(leaf, "bias", None) is not None:
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
def _(exp_mb, fp_mb, logical_mb, mo):
    mo.md(f"""
    | Artifact | Size | vs FP baseline |\n|---|---|---|\n"
        f"| FP bf16 `state_dict` | **{fp_mb:.1f} MB** | 100.0% |\n"
        f"| Exported `state_dict` (measured) | **{exp_mb:.1f} MB** | {exp_mb / fp_mb * 100:.1f}% |\n"
        f"| Exported — **logical packed** (future target) | **{logical_mb:.1f} MB** | {logical_mb / fp_mb * 100:.1f}% |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Round-trip: save → load → same logits
    """)
    return


@app.cell
def _(exported, inputs, os, tempfile, torch):
    import copy as _copy

    with tempfile.TemporaryDirectory() as tmp2:
        ckpt = os.path.join(tmp2, "qwen_exported.pt")
        torch.save(exported.state_dict(), ckpt)
        reloaded = _copy.deepcopy(exported)
        reloaded.load_state_dict(torch.load(ckpt, weights_only=True))
        reloaded.train(False)
        with torch.no_grad():
            logits_fresh = exported(**inputs).logits
            logits_reload = reloaded(**inputs).logits
    return logits_fresh, logits_reload


@app.cell
def _(logits_fresh, logits_reload, mo, torch):
    max_abs = (logits_reload - logits_fresh).abs().max().item()
    bit_exact = torch.allclose(logits_reload, logits_fresh, atol=0.0, rtol=0.0)
    mo.md(
        f"- Max abs logits difference after reload: `{max_abs:.2e}`\n"
        f"- Bit-exact match: **{bit_exact}**"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Takeaways

    - Same three public symbols drive LLM and CV workflows: `QuantRecipe`, `build_quantized_model`, `export_pt2e`.
    - `llm` adapter classifies attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) without any manual annotation.
    - GPTQ calibration slots in as one list of HF `BatchEncoding` samples — no custom DataLoader required.
    - AWQ / SmoothQuant exist as fake-quant evaluators; only RTN + GPTQ currently round-trip through `export_pt2e`.
    - Bit-packed storage is future target; the exported module is already shape-correct for it.
    """)
    return


if __name__ == "__main__":
    app.run()
