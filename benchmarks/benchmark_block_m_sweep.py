import math

import torch
import torch.nn.functional as F

from benchmarks.common.result import BenchmarkResults

from flash_infer.attention.forward_pass import flash_attention
from flash_infer.config import FlashAttentionConfig


DEVICE = "cuda"
DTYPE = torch.float32

WARMUP = 30
ITERATIONS = 100


def naive_attention(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[-1])

    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores * scale

    probs = torch.softmax(scores, dim=-1)

    return torch.matmul(probs, v)


def benchmark_fn(fn, *args):
    # Warmup
    for _ in range(WARMUP):
        fn(*args)

    # Make sure warmup work is finished
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(ITERATIONS):
        fn(*args)

    end.record()

    # Wait for GPU work to finish
    end.synchronize()

    elapsed_ms = start.elapsed_time(end)

    return elapsed_ms / ITERATIONS


def run_case(N, D, config, results):
    print(f"\nN={N}, D={D}")

    q = torch.randn(
        (N, D),
        device=DEVICE,
        dtype=DTYPE,
    )

    k = torch.randn(
        (N, D),
        device=DEVICE,
        dtype=DTYPE,
    )

    v = torch.randn(
        (N, D),
        device=DEVICE,
        dtype=DTYPE,
    )

    naive_ms = benchmark_fn(
        naive_attention,
        q,
        k,
        v,
    )

    # PyTorch SDPA
    # SDPA expects:
    # [batch, heads, seq, dim]
    # Our kernel uses:
    # [seq, dim]
    q_4d = q[None, None, :, :]
    k_4d = k[None, None, :, :]
    v_4d = v[None, None, :, :]

    def sdpa(q_, k_, v_):
        return F.scaled_dot_product_attention(
            q_,
            k_,
            v_,
            is_causal=False,
        )

    sdpa_ms = benchmark_fn(
        sdpa,
        q_4d,
        k_4d,
        v_4d,
    )

    triton_ms = benchmark_fn(
        flash_attention,
        q,
        k,
        v,
        config,
    )

    triton_vs_naive = naive_ms / triton_ms
    triton_vs_sdpa = sdpa_ms / triton_ms

    print(f"Naive PyTorch : {naive_ms:.2f} ms")
    print(f"PyTorch SDPA  : {sdpa_ms:.2f} ms")
    print(f"Triton : {triton_ms:.4f} ms")

    print(
        f"Triton vs Naive: "
        f"{triton_vs_naive:.4f}x"
    )

    print(
        f"Triton vs SDPA : "
        f"{triton_vs_sdpa:.4f}x"
    )

    results.append_csv([
        torch.cuda.get_device_name(0),
        N,
        D,
        config.block_m,
        config.block_n,
        naive_ms,
        sdpa_ms,
        triton_ms,
        triton_vs_naive,
        triton_vs_sdpa,
    ])

    return triton_ms


def main():
    print("FlashInfer attention benchmark")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    results = BenchmarkResults("block_m_sweep")

    results.create_csv([
        "gpu",
        "N",
        "D",
        "block_m",
        "block_n",
        "naive_ms",
        "sdpa_ms",
        "triton_ms",
        "triton_vs_naive",
        "triton_vs_sdpa",
    ])

    #D = 16

    block_m_values = [2,4,8,16,32,64,128,]
    sequence_lengths = [64,128,256, 512,1024,]
    D_values = [16, 32, 64,]

    for D in D_values:
        print(f"\n{'=' * 60}")
        print(f"D = {D}")
        print(f"{'=' * 60}")

        for block_m in block_m_values:

            config = FlashAttentionConfig(
                block_m=block_m,
                block_n=16,
            )
            
            print(f"\n========== BLOCK_M={block_m} ==========")

            for N in sequence_lengths:

                run_case(
                    N,
                    D,
                    config=config,
                    results=results,
                )


if __name__ == "__main__":
    main()
