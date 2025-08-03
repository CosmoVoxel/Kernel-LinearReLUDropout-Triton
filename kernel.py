import time

import torch
import triton
import triton.language as tl

#3060 ti
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 16,
                "BLOCK_SIZE_K": 64,
                "EPT_M": 4,
                "EPT_N": 4,
            },
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "EPT_M": 2,
                "EPT_N": 2,
            },
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 16,
                "EPT_M": 1,
                "EPT_N": 1,
            },
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "EPT_M": 1,
                "EPT_N": 1,
            },
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "EPT_M": 1,
                "EPT_N": 1,
            },
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 16,
                "BLOCK_SIZE_K": 64,
                "EPT_M": 1,
                "EPT_N": 1,
            },
            num_warps=2,
            num_stages=3,
        ),
    ],
    key=["batch_size", "input_features", "output_features"],
)
@triton.jit
def fused_linear_relu_dropout(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    dropout_seed: tl.constexpr,
    batch_size: tl.constexpr,
    input_features: tl.constexpr,
    output_features: tl.constexpr,
    dropout_prob: tl.constexpr,
    input_stride_bs,
    input_stride_fea,
    weight_stride_ofea,
    weight_stride_ifea,
    output_stride_bs,
    output_stride_fea,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    EPT_M: tl.constexpr,
    EPT_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    off_m = pid_m * BLOCK_SIZE_M * EPT_M + tl.arange(0, BLOCK_SIZE_M * EPT_M)
    off_n = pid_n * BLOCK_SIZE_N * EPT_N + tl.arange(0, BLOCK_SIZE_N * EPT_N)
    mask_m = off_m < batch_size
    mask_n = off_n < output_features

    acc = tl.zeros((BLOCK_SIZE_M * EPT_M, BLOCK_SIZE_N * EPT_N), dtype=tl.float32)

    for k0 in range(0, input_features, BLOCK_SIZE_K):
        inp = tl.load(
            input_ptr
            + off_m[:, None] * input_stride_bs
            + (k0 + tl.arange(0, BLOCK_SIZE_K))[None, :] * input_stride_fea,
            mask=mask_m[:, None]
            & ((k0 + tl.arange(0, BLOCK_SIZE_K)) < input_features)[None, :],
            other=0.0,
        )
        w = tl.load(
            weight_ptr
            + off_n[:, None] * weight_stride_ofea
            + (k0 + tl.arange(0, BLOCK_SIZE_K))[None, :] * weight_stride_ifea,
            mask=mask_n[:, None]
            & ((k0 + tl.arange(0, BLOCK_SIZE_K)) < input_features)[None, :],
            other=0.0,
        )
        acc += tl.dot(inp.to(tl.float16), w.T.to(tl.float16), out_dtype=tl.float32)

    if bias_ptr is not None:
        b = tl.load(bias_ptr + off_n, mask=mask_n, other=0.0)
        acc += b[None, :]
    relu = tl.maximum(acc, 0.0)
    if dropout_prob > 0:
        lin = off_m[:, None] * output_features + off_n[None, :]
        rnd = tl.rand(dropout_seed, lin)
        keep = rnd > dropout_prob
        valid = mask_m[:, None] & mask_n[None, :]
        out = tl.where(keep & valid, relu / (1.0 - dropout_prob), 0.0)
    else:
        out = relu
    tl.store(
        output_ptr
        + off_m[:, None] * output_stride_bs
        + off_n[None, :] * output_stride_fea,
        out.to(tl.float16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


# Runner and benchmark for single layer
if __name__ == "__main__":
    batch, in_fea, out_fea = 8192, 1024, 2048
    repeats = 100

    x16 = torch.randn((batch, in_fea), device="cuda", dtype=torch.float16)
    w = torch.randn((out_fea, in_fea), device="cuda", dtype=torch.float16)
    b = torch.randn((out_fea,), device="cuda", dtype=torch.float16)

    def run_triton(batch, in_fea, out_fea):
        out = torch.empty((batch, out_fea), device="cuda", dtype=torch.float16)
        grid = lambda meta: (
            triton.cdiv(batch, meta["BLOCK_SIZE_M"] * meta["EPT_M"]),
            triton.cdiv(out_fea, meta["BLOCK_SIZE_N"] * meta["EPT_N"]),
        )
        fused_linear_relu_dropout[grid](
            x16,
            w,
            b,
            out,
            0,
            batch,
            in_fea,
            out_fea,
            0,
            x16.stride(0),
            x16.stride(1),
            w.stride(0),
            w.stride(1),
            out.stride(0),
            out.stride(1),
        )
        return out

    @torch.compile
    def run_torch(batch, in_fea, out_fea):
        model = torch.nn.Sequential(
            torch.nn.Linear(in_fea, out_fea).half().cuda(),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.0),
        )
        with torch.no_grad():
            model[0].weight.data.copy_(w)
            model[0].bias.data.copy_(b)
            return model(x16)

    # Warm-up
    for _ in range(10):
        run_triton(batch, in_fea, out_fea)
        run_torch(batch, in_fea, out_fea)
    torch.cuda.synchronize()

    # Benchmark Triton
    t0 = time.time()
    for _ in range(repeats):
        run_triton(batch, in_fea, out_fea)
    torch.cuda.synchronize()
    t_triton = time.time() - t0

    # Benchmark PyTorch
    t0 = time.time()
    for _ in range(repeats):
        run_torch(batch, in_fea, out_fea)
    torch.cuda.synchronize()
    t_torch = time.time() - t0

    print(
        f"Triton: {t_triton:.4f}s, PyTorch: {t_torch:.4f}s, Speedup: {t_torch / t_triton:.2f}x"
    )

    # Verify correctness
    with torch.no_grad():
        out_ref = torch.nn.functional.relu(torch.nn.functional.linear(x16, w, b))
        out_triton = run_triton(batch, in_fea, out_fea)
        torch.testing.assert_close(out_triton, out_ref, rtol=1e-2, atol=1e-2)
        print("✅ Triton output matches PyTorch (1-layer, w/o dropout)")
