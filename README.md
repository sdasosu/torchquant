# torchquant

A modular quantization toolkit for applying **universal quantization** across CNNs, segmentation models, and LLMs.

## Quick Start

```bash
# Install (editable)
uv sync

# Verify
echo "from torchquant import QuantRecipe, quantize_model; print('ok')" | uv run -

# Lint
ruff check src/
ruff format src/
```

## Architecture Overview

```
QuantRecipe        ─┐
                    ├─→ pipeline.quantize_model()
model + data       ─┘         │
                              ├── 1. adapters.get_adapter()        → AdapterFns
                              ├── 2. graph.find_quantizable_nodes() → list[QuantNode]
                              ├── 3. rules.RuleEngine.decide()      → list[QuantDecision]
                              ├── 4. observers.get_observer_specs()  → list[ObserverSpec]
                              │   calibration.run_calibration()      → stats
                              │   graph.resolve_modules()            → modules
                              ├── 5. quantizers.QUANTIZER_MAP[alg]() → QuantResult
                              ├── 6. registry.rebuild_model()        → nn.Module
                              └── 7. (optional) export.pt2e
```

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
│   └── smoothquant.py  #   Scale migration + RTN
│
├── registry.py         # QuantRecord store + rebuild_model()
├── pipeline.py         # quantize_model() orchestrator
│
└── export/
    └── pt2e.py         # PT2E / ExecuTorch export
```

## Key Types

```python
from torchquant import QuantScheme, QuantRecipe, Algorithm

# Describe HOW to quantize
scheme = QuantScheme(weight_bits=4, group_size=128, algorithm=Algorithm.GPTQ)

# Describe WHAT to quantize
recipe = QuantRecipe(
    default_scheme=scheme,
    overrides={"model.lm_head": QuantScheme(weight_bits=8)},  # keep head at 8-bit
    ignore=frozenset({"model.embed_tokens"}),                  # skip embeddings
)
```

## Implementation Guide

All function bodies are currently `raise NotImplementedError`. Fill them in this order -- each phase only depends on the ones above it.

### Phase 1: Foundation (no dependencies)

These modules are self-contained. Start here.

| File | What to implement |
|------|-------------------|
| `_types.py` | **Done** -- enums are complete |
| `config.py` | **Done** -- pydantic models are complete |
| `adapters/generic.py` | `classify_module()`: isinstance checks for Linear/Conv2d/Embedding -> LayerKind. `prepare_model()`: model.eval() + freeze params. `find_blocks()`: return list(model.named_children()). `is_skip_target()`: return False |
| `adapters/smp.py` | Port from old `smp_model.py` (in git history) -- block detection via `HF_BLOCK_PATTERNS`, skip attention projections via `ATTENTION_PROJECTION_NAMES`, BN fusion in `prepare_model()` |
| `adapters/llm.py` | `classify_module()`: check leaf name against QUERY/KEY/VALUE/OUTPUT name sets -> ATTENTION_QKV or ATTENTION_OUT, else LINEAR. `is_skip_target()`: return False (LLM adapter handles everything) |

### Phase 2: Graph + Rules (depends on Phase 1)

| File | What to implement |
|------|-------------------|
| `adapters/__init__.py` | `get_adapter()`: walk model.named_modules(), check for attention projection names -> llm, check for block patterns -> smp, else -> generic. Construct `AdapterFns` from the chosen module’s functions |
| `graph.py` | `find_quantizable_nodes()`: call adapter.classify_module on each named_module, skip Nones, build QuantNode list. `resolve_modules()`: dict(model.named_modules()) filtered by node FQNs |
| `rules.py` | `RuleEngine.decide()`: for each node, try rules in order, first non-None wins. `default_rule()`: check recipe.ignore, check recipe.overrides, fall back to recipe.default_scheme |

### Phase 3: Observers + Calibration (depends on Phase 2)

| File | What to implement |
|------|-------------------|
| `observers/minmax.py` | `__call__`: track running min/max of output tensor per channel. `get_stats`: return `{"min": ..., "max": ...}` |
| `observers/hessian.py` | `__call__`: accumulate H += 2 * X^T @ X from input. `get_stats`: return `{"hessian": H / n}` |
| `observers/awq.py` | `__call__`: accumulate mean abs activation per channel. `get_stats`: return `{"act_mean": ...}` |
| `observers/smoothquant.py` | `__call__`: track running max abs per channel. `get_stats`: return `{"act_max": ...}` |
| `observers/__init__.py` | `get_observer_specs()`: map Algorithm -> factory + targets. RTN -> [], GPTQ -> hessian, AWQ -> awq, SMOOTHQUANT -> smoothquant |
| `calibration.py` | `run_calibration()`: attach observer hooks via register_forward_hook, iterate dataset up to max_samples, collect stats, remove hooks |

### Phase 4: Quantizers (depends on Phase 3)

| File | What to implement |
|------|-------------------|
| `quantizers/rtn.py` | Per-channel/per-group round-to-nearest. Compute scale = max_abs / (2^(bits-1) - 1), quantize, return QuantResult |
| `quantizers/gptq.py` | GPTQ: use Hessian from stats to do blockwise optimal rounding (Cholesky on H, iterate columns) |
| `quantizers/awq.py` | AWQ: compute importance-weighted scale from act_mean, apply scale to weight, then RTN |
| `quantizers/smoothquant.py` | SmoothQuant: compute migration scale s = act_max^a / weight_max^(1-a), apply to weight, then RTN |

### Phase 5: Pipeline + Registry (depends on all above)

| File | What to implement |
|------|-------------------|
| `registry.py` | `rebuild_model()`: deep-copy model, write quantized weights from registry records back into the copy |
| `pipeline.py` | `quantize_model()`: wire the 7 stages together -- get_adapter, find_nodes, decide, get_observer_specs, calibrate, quantize, rebuild |

### Phase 6: Export (optional, depends on Phase 5)

| File | What to implement |
|------|-------------------|
| `export/pt2e.py` | `export_pt2e()`: torch.export.export() + quantization annotation from registry + optional ExecuTorch lowering |

## Design Conventions

- **No class hierarchies** -- adapters are function bundles (`AdapterFns`), not base classes
- **Frozen value objects** -- `QuantNode`, `QuantDecision`, `QuantResult`, `ObserverSpec` are `@dataclass(frozen=True)`
- **Pydantic for config** -- `QuantScheme` and `QuantRecipe` are frozen pydantic models
- **TYPE_CHECKING guards** -- heavy imports (torch, nn) go under `if TYPE_CHECKING:`
- **Explicit calibration** -- calibration is a named pipeline stage, never hidden
- **Dispatch tables** -- `QUANTIZER_MAP[Algorithm]` and `get_adapter()` replace inheritance

## Tooling

| Tool | Purpose |
|------|---------|
| `uv` | Package management, virtualenv, running scripts |
| `ruff` | Linting (ALL rules enabled) + formatting |
| `hatchling` | Build backend |

Python >= 3.12 required. Run `uv sync` after cloning.
