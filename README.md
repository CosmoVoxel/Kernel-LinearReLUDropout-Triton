# Fused Linear + ReLU + Dropout — Triton GPU Kernel

A high-performance, **single-pass GPU kernel** that fuses three common deep-learning operations — linear projection, ReLU activation, and inverted dropout — into one JIT-compiled Triton kernel. Running everything in a single pass eliminates two intermediate global-memory round-trips compared to chaining `nn.Linear → nn.ReLU → nn.Dropout` in PyTorch, which translates directly into lower latency on large activation tensors.

---

## Architecture

```
Input (FP16)
    │
    ▼
┌─────────────────────────────────────────────────┐
│           Tiled Matrix Multiplication            │
│  (FP16 dot products, FP32 accumulation)          │
│  ┌─────────┐   ┌──────────┐                     │
│  │ Input   │ × │ Weight^T │  + Bias  →  acc      │
│  │ tile    │   │  tile    │  (FP32)              │
│  └─────────┘   └──────────┘                     │
│                                                 │
│  ReLU:     acc = max(acc, 0.0)                  │
│                                                 │
│  Dropout:  keep = rand() > p                    │
│            out  = keep ? acc / (1−p) : 0.0      │
└─────────────────────────────────────────────────┘
    │
    ▼
Output (FP16)
```

All three stages execute **in-register** inside a single kernel launch — no intermediate tensor is written back to global memory.

---

## Autotune Configurations

Triton's `@triton.autotune` benchmarks all six configurations at runtime and selects the fastest one for each unique `(batch_size, input_features, output_features)` shape. The search is keyed on those three dimensions, so the best tile size is found once and cached.

| # | BLOCK\_M | BLOCK\_N | BLOCK\_K | EPT\_M | EPT\_N | Warps | Stages |
|---|----------|----------|----------|--------|--------|-------|--------|
| 1 | 16       | 16       | 64       | 4      | 4      | 4     | 2      |
| 2 | 32       | 32       | 32       | 2      | 2      | 4     | 3      |
| 3 | 64       | 64       | 16       | 1      | 1      | 8     | 4      |
| 4 | 128      | 32       | 32       | 1      | 1      | 4     | 4      |
| 5 | 32       | 128      | 32       | 1      | 1      | 4     | 4      |
| 6 | 64       | 16       | 64       | 1      | 1      | 2     | 3      |

**EPT** (Elements-Per-Thread) controls how many output elements each thread computes, allowing the tile to cover a larger region of the output matrix without increasing shared-memory usage proportionally.

---

## Quick Start

### Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/) ≥ 2.0 (with CUDA)
- [OpenAI Triton](https://github.com/openai/triton) ≥ 2.0

```bash
pip install triton torch
```

### Run the benchmark

```bash
python kernel.py
```

Expected output (numbers will vary by GPU):

```
Triton: 0.0412s, PyTorch: 0.0731s, Speedup: 1.77x
✅ Triton output matches PyTorch (1-layer, w/o dropout)
```

---

## Benchmark

The `__main__` block runs a head-to-head comparison between the fused Triton kernel and a `@torch.compile`-optimised PyTorch `Sequential(Linear → ReLU → Dropout)` model.

| Setting         | Value                     |
|-----------------|---------------------------|
| Batch size      | 8,192                     |
| Input features  | 1,024                     |
| Output features | 2,048                     |
| Iterations      | 100 (+ 10 warm-up)        |
| Precision       | FP16                      |
| Synchronisation | `torch.cuda.synchronize()` |

Correctness is verified with `torch.testing.assert_close(rtol=1e-2, atol=1e-2)` against a plain PyTorch reference (without dropout, so the stochastic path is excluded from the numerical check).

---

## Tech Stack

| Component | Details |
|-----------|---------|
| **Kernel language** | [OpenAI Triton](https://github.com/openai/triton) — Python-embedded GPU DSL |
| **Compute** | FP16 `tl.dot` with FP32 accumulation |
| **Activation** | Fused ReLU via `tl.maximum(acc, 0.0)` |
| **Dropout** | Inverted dropout using `tl.rand()` PRNG |
| **Autotuning** | `@triton.autotune` across 6 tile configurations |
| **Framework** | PyTorch (for tensor allocation, reference baseline, and `@torch.compile` comparison) |
| **Target GPU** | Tuned on an NVIDIA RTX 3060 Ti |

---

## License

See [LICENSE](LICENSE).
