import torch

from common.result import BenchmarkResults

from flash_infer.attention.forward_pass import flash_attention
from flash_infer.config import FlashAttentionConfig
from triton.runtime.errors import OutOfResources


DEVICE = "cuda"
DTYPE = torch.float32

WARMUP = 50
ITERATIONS = 200
REPEATS = 5


def benchmark_fn(fn, *args):
    for _ in range(WARMUP):
        fn(*args)

    torch.cuda.synchronize()

    measurements = []

    for _ in range(REPEATS):
        start = torch.cuda.Event(
            enable_timing=True
        )

        end = torch.cuda.Event(
            enable_timing=True
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

    try:
        triton_ms = benchmark_fn(
            flash_attention,
            q,
            k,
            v,
            config,
            causal,
        )

    except OutOfResources as exc:
        print(
        f"SKIPPED: BM={config.block_m}, "
        f"BN={config.block_n}, "
        f"warps={config.num_warps}, "
        f"stages={config.num_stages} "
        f"exceeded GPU resources: {exc}"
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
            config.num_warps,
            config.num_stages,
            None,
            "skipped",
        ])
        return 

    print(
        f"B={B}, "
        f"H={H}, "
        f"N={N}, "
        f"D={D}, "
        f"causal={causal}, "
        f"BM={config.block_m}, "
        f"BN={config.block_n}, "
        f"warps={config.num_warps}, "
        f"stages={config.num_stages}, "
        f"triton={triton_ms:.4f} ms"
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
        config.num_warps,
        config.num_stages,
        triton_ms,
        'ok',
    ])


def main():
    print(
        "FlashInfer kernel configuration sweep"
    )

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    results = BenchmarkResults(
        "kernel_config_sweep"
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
        "num_warps",
        "num_stages",
        "triton_ms",
        'status',
    ])

    B = 1
    H = 16
    N = 1024
    D = 64

    block_m_values = [16, 32]
    block_n_values = [32, 64]
    num_warps_values = [2, 4, 8]
    num_stages_values = [2, 3, 4]

    for causal in [False, True]:

        print(
            f"\n{'=' * 70}"
        )

        print(
            f"CAUSAL = {causal}"
        )

        print(
            f"{'=' * 70}"
        )

        for block_m in block_m_values:
            for block_n in block_n_values:
                for num_warps in num_warps_values:
                    for num_stages in num_stages_values:

                        config = FlashAttentionConfig(
                            block_m=block_m,
                            block_n=block_n,
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )

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