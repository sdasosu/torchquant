# torchquant

A modular quantization toolkit for applying universal quantization across CNNs, segmentation models, and LLMs.

## Quick Start

```bash
# Install (editable)
uv sync

# Verify imports
uv run - <<'PY'
from torchquant import (
    Algorithm,
    QuantRecipe,
    QuantScheme,
    build_quantized_model,
    export_pt2e,
)

print("ok")
PY

# Test
uv run pytest tests/

# Lint
ruff check src/ tests/
ruff format src/ tests/
```

## Architecture Overview

```
QuantRecipe        ─┐
                    ├─→ pipeline.build_quantized_model()
model + data       ─┘         │
                              ├── 1. adapters.get_adapter()         → AdapterFns
                              ├── 2. graph.find_quantizable_nodes() → list[QuantNode]
                              ├── 3. rules.RuleEngine.decide()      → list[QuantDecision]
                              ├── 4. observers.get_observer_specs() → list[ObserverSpec]
                              │   calibration.run_calibration()     → stats
                              │   graph.resolve_modules()           → modules
                              ├── 5. quantizers.QUANTIZER_MAP[alg]() → QuantResult
                              ├── 6. registry.apply_records_inplace() → (nn.Module, QuantRegistry)
                              └── 7. export.export_pt2e()            → nn.Module with Quantized* submodules
```

`quantize_model()` remains the convenience wrapper that returns only the quantized model.
`build_quantized_model()` returns both the quantized model and the `QuantRegistry` needed by `export_pt2e()`.

## Project Structure

```
src/torchquant/
├── _types.py           # Enums: LayerKind, Algorithm, Granularity
├── config.py           # QuantScheme (how) + QuantRecipe (what)
│
├── graph.py            # find_quantizable_nodes(), resolve_modules()
├── adapters/           # Model-family dispatch (functions, not classes)
│   ├── __init__.py     #   AdapterFns dataclass + get_adapter()
│   ├── generic.py      #   Fallback: any nn.Module
│   ├── smp.py          #   Sequential / MLP-style models
│   └── llm.py          #   LLM / attention-based models
│
├── rules.py            # RuleEngine: nodes x recipe → decisions
│
├── observers/          # Hook-based statistics collectors
│   ├── __init__.py     #   ObserverSpec + get_observer_specs()
│   ├── minmax.py       #   Min/max activation range
│   ├── smoothquant.py  #   SmoothQuant migration scales
│   ├── awq.py          #   AWQ activation importance
│   └── hessian.py      #   GPTQ Hessian approximation
│
├── calibration.py      # run_calibration(model, dataset, observers)
│
├── quantizers/         # Algorithm implementations
│   ├── __init__.py     #   QuantResult + QuantizerFn type
│   ├── rtn.py          #   Round-to-nearest (baseline)
│   ├── gptq.py         #   GPTQ blockwise
│   ├── awq.py          #   Activation-aware weight quant
│   ├── smoothquant.py  #   Scale migration + RTN
│   └── _oracle.py      #   Test-only q_int recorder
│
├── registry.py         # QuantRecord store + rebuild helpers
├── pipeline.py         # build_quantized_model() + quantize_model()
│
└── export/
    ├── pt2e.py         # Module-level export orchestrator
    ├── _recover.py     # Integer-weight recovery from QuantResult
    ├── rewriter.py     # Rewriter registry and lookup
    ├── rewriters/      # Built-in Linear / Conv2d / Embedding rewriters
    └── runtime/        # QuantizedLinear / QuantizedConv2d / QuantizedEmbedding
```

## Key Types

```python
from torchquant import Algorithm, QuantRecipe, QuantScheme

# Describe HOW to quantize
scheme = QuantScheme(weight_bits=4, group_size=128, algorithm=Algorithm.GPTQ)

# Describe WHAT to quantize
recipe = QuantRecipe(
    default_scheme=scheme,
    overrides={"model.lm_head": QuantScheme(weight_bits=8)},
    ignore=frozenset({"model.embed_tokens"}),
)
```

### Export

```python
from torchquant import (
    Algorithm,
    QuantRecipe,
    QuantScheme,
    build_quantized_model,
    export_pt2e,
)

recipe = QuantRecipe(
    default_scheme=QuantScheme(
        weight_bits=4,
        group_size=128,
        algorithm=Algorithm.GPTQ,
    )
)

quantized_model, registry = build_quantized_model(model, recipe, calibration_data)
exported_model = export_pt2e(quantized_model, registry)

# exported_model is still a plain nn.Module
# with QuantizedLinear / QuantizedConv2d / QuantizedEmbedding submodules
```

`export_pt2e()` performs module-level rewriting only. It does not call `torch.export`,
ExecuTorch lowering, XNNPACK partitioning, or backend-specific code generation.
If you want a downstream export artifact, run your own deployment stack on the
returned `nn.Module`.

## Export Runtime Model

The rewritten runtime modules store integer weights plus scale / zero-point metadata
as registered buffers. Their `forward()` path dequantizes to float and dispatches to
`torch.nn.functional.linear`, `conv2d`, or `embedding`.

This keeps the exported object as a normal `nn.Module`, which means you can:

- save it with `torch.save()`
- persist a versioned payload with `export_state_dict()`
- pass it to `torch.export.export()`
- register your own rewriter with `register_quantized_module()`

## Limitations

- AWQ and SmoothQuant are not exportable through module-level rewriting.
- `nn.ConvTranspose2d` is not supported.
- Tied weights are broken by rewrite; re-tie them on the exported model if needed.
- Distributed wrappers such as DDP and FSDP must be unwrapped before quantization or export.
- Bias dtype is preserved as-is; export does not coerce mixed-precision bias tensors.

## Tooling

| Tool | Purpose |
|------|---------|
| `uv` | Package management, virtualenv, running scripts |
| `ruff` | Linting (ALL rules enabled) + formatting |
| `pytest` | Test suite |
| `hatchling` | Build backend |

Python >= 3.12 required. Run `uv sync` after cloning.
