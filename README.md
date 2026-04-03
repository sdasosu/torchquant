# Pyquant

A modular quantization toolkit for applying **universal quantization** across a wide variety of models, including **CNNs**, **segmentation models**, and **LLMs**.

Pyquant is designed around a clean, extensible pipeline that separates:
- **Tracing and statistics collection**
- **Model-specific adaptation**
- **Quantization logic**

This makes it easier to experiment with different quantization strategies while keeping the codebase organized and reusable.

---

## Why Pyquant?

Quantization workflows often become tightly coupled to a single model family or backend. Pyquant takes a more modular approach by introducing clear interfaces for:

- **Tracer modules** for collecting activation and weight statistics
- **Model wrappers** for adapting different architectures
- **Quantizers** for implementing reusable quantization logic
- **A central pipeline** that wires everything together

The goal is to provide a flexible foundation for building and testing quantization methods across diverse model types.

---

## Features

- Supports **multiple model families** such as CNNs, SMP-style models, and LLM/attention-based architectures
- Built with **pluggable tracers** for outlier detection and statistics collection
- Includes adapters for **model-specific module discovery and surgery**
- Provides reusable quantization components for:
  - **weights**
  - **activations**
  - **KV cache**
- Centralized configuration through a `QuantConfig` dataclass
- Organized project structure for easy extension and experimentation

---

## High-Level Pipeline

Pyquant follows a simple modular flow:

```text
Tracer -> Model Wrapper -> Quantizer -> Quantized Model
```

### Pipeline stages
1. **Tracer**
   Collects statistics from activations, weights, or Hessian-like signals.

2. **Model Wrapper**
   Understands the target architecture and exposes the correct layers or projections.

3. **Quantizer**
   Applies the actual quantization logic based on the collected information.

4. **Pipeline**
   Coordinates the full process from calibration to quantized model generation.

---

## Project Structure

```text
pyquant/
├── tracers/
│   ├── base_tracer.py
│   ├── smoothquant_tracer.py
│   ├── awq_tracer.py
│   └── gptq_tracer.py
│
├── models/
│   ├── base_model.py
│   ├── smp_model.py
│   ├── attention_model.py
│   └── layer_registry.py
│
├── quantizers/
│   ├── base_quantizer.py
│   ├── weight_quantizer.py
│   ├── activation_quantizer.py
│   └── kv_cache_quantizer.py
│
├── pipeline.py
├── calibrate.py
├── config.py
├── utils.py
├── pyproject.toml
├── setup.py
├── tests/
└── examples/
```

---

## Module Overview

### `tracers/`
Tracing components for collecting quantization-relevant statistics.

- **`base_tracer.py`**  
  Defines the abstract `BaseTracer` interface with lifecycle hooks such as:
  - `register_hooks`
  - `collect_stats`
  - `remove_hooks`

- **`smoothquant_tracer.py`**  
  Supports activation/weight scale migration for SmoothQuant-style workflows.

- **`awq_tracer.py`**  
  Implements activation-aware tracing for weight scaling.

- **`gptq_tracer.py`**  
  Provides Hessian-based or blockwise statistics collection for GPTQ-like methods.

### `models/`
Model-type adapters that understand architecture-specific structure.

- **`base_model.py`**  
  Defines the abstract `BaseModelWrapper` for:
  - iterating over layers
  - discovering quantization targets
  - performing module surgery

- **`smp_model.py`**  
  Handles SMP-style sequential or MLP-oriented blocks.

- **`attention_model.py`**  
  Identifies attention projections such as **Q/K/V/O** and supports grouped-query variants.

- **`layer_registry.py`**  
  Maps layer class names to quantization targets — effectively the lookup table for deciding **what to quantize**.

### `quantizers/`
Core quantization logic.

- **`base_quantizer.py`**  
  Defines the abstract quantizer interface.

- **`weight_quantizer.py`**  
  Implements per-channel and per-tensor weight quantization.

- **`activation_quantizer.py`**  
  Implements static and dynamic activation quantization.

- **`kv_cache_quantizer.py`**  
  Applies targeted quantization for attention KV cache optimization.

### Root files

- **`pipeline.py`**  
  Main entry point that connects tracer, model wrapper, and quantizer.

- **`calibrate.py`**  
  Calibration utilities for collecting statistics before quantization.

- **`config.py`**  
  Contains the `QuantConfig` dataclass with settings such as:
  - bit width
  - granularity
  - quantization scheme
  - tracer selection
  - target layers

- **`utils.py`**  
  Shared helper functions for:
  - scale clipping
  - rounding
  - dtype casting

---

## Supported Quantization Styles

Pyquant is structured to support workflows inspired by:

- **SmoothQuant**
- **AWQ**
- **GPTQ**
- **Static activation quantization**
- **Dynamic activation quantization**
- **KV-cache quantization for attention models**

---

## Intended Model Coverage

Pyquant is designed for a broad range of model families, including:

- **CNNs**
- **SMP-style models**
- **Attention-based models**
- **LLMs**

This architecture-first abstraction makes it easier to extend the framework to new model types over time.

---

## Getting Started

### 1. Install

If the project uses `pyproject.toml` or `setup.py`, a typical local install would be:

```bash
pip install -e .
```

### 2. Configure quantization

Define a `QuantConfig` with your preferred settings such as bit width, tracer type, granularity, and target layers.

### 3. Calibrate

Run calibration to collect the required statistics.

### 4. Launch the pipeline

Use `pipeline.py` to connect:
- the selected tracer
- the target model wrapper
- the quantizer

> You can expand this section with concrete commands once the CLI or example scripts are finalized.

---

## Design Philosophy

Pyquant is built around a few core ideas:

- **Modularity** — each responsibility lives in its own layer
- **Extensibility** — new tracers, wrappers, and quantizers can be added without rewriting the full system
- **Model-awareness** — architecture-specific logic belongs in model wrappers, not in generic quantizers
- **Reusability** — shared utilities and registries reduce duplication across methods

---

## Example Use Cases

- Compare different quantization strategies on the same model family
- Add support for a new attention architecture by extending the model wrapper
- Experiment with alternate tracing strategies without changing the pipeline
- Apply targeted KV-cache quantization for LLM inference optimization

---

## Roadmap

Potential future additions may include:

- richer example scripts
- benchmark results
- backend-specific deployment paths
- CLI entry points
- expanded unit tests
- more model adapters and quantization recipes

---

## Contributing

Contributions are welcome. A good way to contribute is by adding:

- a new tracer
- a new model wrapper
- a new quantizer
- example notebooks or scripts
- tests and validation utilities

---

## License

Add your project license here.

---

## Summary

Pyquant provides a clean and extensible foundation for building quantization workflows across different model families. By separating **tracing**, **model adaptation**, and **quantization**, it becomes easier to scale experimentation while keeping the framework maintainable.

