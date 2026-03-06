# Fused Linear-ReLU-Dropout Triton Kernel

A high-performance GPU kernel written in [OpenAI Triton](https://github.com/openai/triton) that fuses three operations — **Linear (matmul + bias)**, **ReLU**, and **Dropout** — into a single kernel launch.

---

## Why Kernel Fusion?

In standard PyTorch, a `Linear → ReLU → Dropout` stack requires **three separate CUDA kernel launches**. Each launch reads from and writes to global GPU memory, introducing costly memory round-trips:

```
Standard PyTorch:
  Input → [Matmul+Bias] → (write) → [ReLU] → (write) → [Dropout] → Output
              kernel 1                kernel 2              kernel 3
```

This fused kernel eliminates the two intermediate memory writes by keeping intermediate results **in registers** across all three operations:

```
Fused Triton Kernel:
  Input → [ Matmul + Bias → ReLU → Dropout ] → Output
                   single kernel launch
```

The result is reduced memory bandwidth pressure and lower kernel-launch overhead — especially beneficial for large batch sizes or latency-sensitive inference.

---

## Technical Highlights

- **Triton JIT compilation** via `@triton.jit` — compiles to PTX/SASS at runtime
- **Autotuning** via `@triton.autotune` across 6 configurations, keyed on `(batch_size, input_features, output_features)`
- **FP16 compute with FP32 accumulation** — dot products in `tl.float16`, accumulator in `tl.float32` to maintain numerical stability
- **Tiled matrix multiplication** with configurable `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`
- **In-register ReLU** using `tl.maximum(acc, 0.0)` — no memory write between matmul and activation
- **Inverted dropout** using Triton's built-in PRNG (`tl.rand`) — scales kept values by `1 / (1 - p)` inline
- **Elements-per-thread (EPT)** tuning parameter for register pressure control
- Tuned for **NVIDIA RTX 3060 Ti**

---

## Autotune Configurations

| # | BLOCK_SIZE_M | BLOCK_SIZE_N | BLOCK_SIZE_K | EPT_M | EPT_N | Warps | Stages |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 16  | 16  | 64 | 4 | 4 | 4 | 2 |
| 2 | 32  | 32  | 32 | 2 | 2 | 4 | 3 |
| 3 | 64  | 64  | 16 | 1 | 1 | 8 | 4 |
| 4 | 128 | 32  | 32 | 1 | 1 | 4 | 4 |
| 5 | 32  | 128 | 32 | 1 | 1 | 4 | 4 |
| 6 | 64  | 16  | 64 | 1 | 1 | 2 | 3 |

Triton selects the fastest configuration automatically the first time each `(batch_size, input_features, output_features)` shape is encountered and caches the result.

---

## Quick Start

### Requirements

- Python 3.8+
- CUDA-capable GPU
- [PyTorch](https://pytorch.org/) with CUDA support
- [OpenAI Triton](https://github.com/openai/triton)

### Install dependencies

```bash
pip install triton torch
```

### Run the benchmark

```bash
python kernel.py
```

The benchmark runs with the default configuration:

| Parameter | Value |
|-----------|-------|
| Batch size | 8192 |
| Input features | 1024 |
| Output features | 2048 |
| Iterations | 100 (+ 10 warm-up) |

It compares the fused Triton kernel against a `@torch.compile`-optimized `nn.Sequential(Linear, ReLU, Dropout)` baseline and prints:

```
Triton: Xs, PyTorch: Ys, Speedup: Zx
✅ Triton output matches PyTorch (1-layer, w/o dropout)
```

Correctness is verified with `torch.testing.assert_close(rtol=1e-2, atol=1e-2)`.

---

## Tech Stack

| Technology | Role |
|---|---|
| Python | Host code, benchmarking |
| [OpenAI Triton](https://github.com/openai/triton) | GPU kernel authoring & JIT compilation |
| [PyTorch](https://pytorch.org/) | Tensor management, reference baseline |
| CUDA | Execution backend |
| FP16 Mixed Precision | Fast compute with FP32 accumulation |

---

## License

See [LICENSE](LICENSE).
