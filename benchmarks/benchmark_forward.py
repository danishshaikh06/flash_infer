import math
from pathlib import Path
import csv 

import torch
import torch.nn.functional as F

from flash_infer.attention.forward_pass import flash_attention
from flash_infer.config import FlashAttentionConfig

DEVICE = "cuda"
DTYPE = torch.float32

WARMUP = 30
ITERATIONS = 100

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = RESULTS_DIR / "block_m_sweep.csv"

def naive_attention(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[-1])

    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores * scale

    probs = torch.softmax(scores, dim=-1)

    return torch.matmul(probs, v)


def benchmark_fn(fn,*args):
    # Warmup
    for _ in range(WARMUP):
        fn(*args)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(ITERATIONS):
        fn(*args)

    end.record()

    end.synchronize()

    elapsed_ms = start.elapsed_time(end)

    return elapsed_ms / ITERATIONS


def run_case(N, D,config):
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

    # naive attention
    naive_ms = benchmark_fn(
        naive_attention,
        q,
        k,
        v,
    )

    # Baseline 2: PyTorch SDPA
    # SDPA expects:
    # [batch, heads, seq, dim]
    # Our current kernel works on:
    # [seq, dim]
    # So we temporarily add batch/head dimensions here.
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

    # Custom Triton kernel
    triton_ms = benchmark_fn(
        flash_attention,
        q,
        k,
        v,
        config,
    )

    print(f" Naive PyTorch : {naive_ms:.4f} ms")
    print(f" PyTorch SDPA  : {sdpa_ms:.4f} ms")
    print(f" Triton : {triton_ms:.4f} ms")

    triton_vs_naive = naive_ms / triton_ms
    triton_vs_sdpa = sdpa_ms / triton_ms

    print(
        f"Triton vs Naive: "
        f"{naive_ms / triton_ms:.2f}x"
    )

    print(
        f"Triton vs SDPA : "
        f"{sdpa_ms / triton_ms:.2f}x"
    )

    # Save result
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
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

    # Create CSV header
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
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

    D = 16
    for block_m in [2, 4, 8, 16, 32, 64, 128]:

        config = FlashAttentionConfig(
            block_m=block_m,
            block_n=16,
        )

        print(f"\n========== BLOCK_M={block_m} ==========")
        
        for N in [64, 128, 256, 512, 1024]:
            run_case(N, D,config=config)

if __name__ == "__main__":
    main()