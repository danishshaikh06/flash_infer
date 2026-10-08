import math

import torch
import torch.nn.functional as F

from common.result import BenchmarkResults

from flash_infer.attention.forward_pass import flash_attention
from flash_infer.config import FlashAttentionConfig


DEVICE = "cuda"
DTYPE = torch.float32

WARMUP = 50
ITERATIONS = 200
REPEATS = 5


def naive_attention(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[-1])

    scores = torch.matmul(
        q,
        k.transpose(-2, -1),
    )

    scores = scores * scale

    probs = torch.softmax(
        scores,
        dim=-1,
    )

    return torch.matmul(
        probs,
        v,
    )


def sdpa_attention(q, k, v, causal):
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=causal,
    )


def benchmark_fn(fn, *args):
    for _ in range(WARMUP):
        fn(*args)

    torch.cuda.synchronize()

    measurements = []

    for _ in range(REPEATS):
        start = torch.cuda.Event(
            enable_timing=True,
        )

        end = torch.cuda.Event(
            enable_timing=True,
        )

        start.record()

        for _ in range(ITERATIONS):
            fn(*args)

        end.record()
        end.synchronize()

        elapsed_ms = start.elapsed_time(end)

        measurements.append(
            elapsed_ms / ITERATIONS
        )

    measurements.sort()

    return measurements[len(measurements) // 2]


def run_case(
    B,
    H,
    N,
    D,
    causal,
    config,
    results,
):
    print(
        f"\n"
        f"B={B}, "
        f"H={H}, "
        f"N={N}, "
        f"D={D}, "
        f"causal={causal}"
    )

    q = torch.randn(
        B,
        H,
        N,
        D,
        device=DEVICE,
        dtype=DTYPE,
    )

    k = torch.randn_like(q)
    v = torch.randn_like(q)

    naive_ms = benchmark_fn(
        naive_attention,
        q,
        k,
        v,
    )

    sdpa_ms = benchmark_fn(
        sdpa_attention,
        q,
        k,
        v,
        causal,
    )

    triton_ms = benchmark_fn(
        flash_attention,
        q,
        k,
        v,
        config,
        causal,
    )

    triton_vs_naive = (
        naive_ms / triton_ms
    )

    triton_vs_sdpa = (
        sdpa_ms / triton_ms
    )

    print(
        f"Naive PyTorch : "
        f"{naive_ms:.4f} ms"
    )

    print(
        f"PyTorch SDPA  : "
        f"{sdpa_ms:.4f} ms"
    )

    print(
        f"Triton        : "
        f"{triton_ms:.4f} ms"
    )

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
        B,
        H,
        N,
        D,
        causal,
        config.block_m,
        config.block_n,
        naive_ms,
        sdpa_ms,
        triton_ms,
        triton_vs_naive,
        triton_vs_sdpa,
    ])


def main():
    print(
        "FlashInfer forward attention benchmark"
    )

    print(
        f"Device: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"PyTorch: "
        f"{torch.__version__}"
    )

    config = FlashAttentionConfig(
        block_m=16,
        block_n=128,
    )

    results = BenchmarkResults(
        "forward_benchmark4"
    )

    results.create_csv([
        "gpu",
        "B",
        "H",
        "N",
        "D",
        "causal",
        "block_m",
        "block_n",
        "naive_ms",
        "sdpa_ms",
        "triton_ms",
        "triton_vs_naive",
        "triton_vs_sdpa",
    ])

    cases = [
        (1, 1, 128, 64),
        (1, 8, 512, 64),
        (1, 16, 1024, 64),
        (2, 8, 1024, 64),
    ]

    for causal in [False, True]:
        print(
            f"\n{'=' * 60}"
        )

        print(
            f"CAUSAL = {causal}"
        )

        print(
            f"{'=' * 60}"
        )

        for B, H, N, D in cases:
            run_case(
                B,
                H,
                N,
                D,
                causal,
                config,
                results,
            )


if __name__ == "__main__":
    main()