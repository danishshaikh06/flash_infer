# FlashInfer

**An experimental, memory-efficient FlashAttention forward kernel built with Triton for NVIDIA GPUs.**

FlashInfer implements exact scaled dot-product self-attention with tiled query/key/value processing, numerically stable online softmax, and fused output accumulation. The engineering focus is on correctness, systematic parameter tuning, and GPU profiling rather than unverified peak-performance claims.

**Status:** Batched, multi-head, causal and non-causal forward attention implemented and tested on FP32 workloads. Forward optimization is in progress. Backward support is planned.

> This is an independent educational and performance-engineering project, not a production-ready replacement for PyTorch SDPA or an affiliation with the separate FlashInfer library.

## Overview

| Capability | Current status |
| --- | --- |
| Inputs and outputs | `[B, H, N, D]` self-attention |
| Causal and bidirectional attention | Implemented and validated on FP32 test cases |
| Algorithm | Tiled `QKᵀ`, online softmax, fused `PV`, final normalization |
| Numerical method | FP32 intermediate accumulation |
| Tensor indexing | Explicit per-tensor strides and boundary masking |
| Tuning parameters | `BLOCK_M`, `BLOCK_N`, `num_warps`, `num_stages` |
| Testing | PyTorch `scaled_dot_product_attention` as reference |
| Profiling | Nsight Compute: resource usage, occupancy, throughput |
| Backward / autograd | Not implemented |

## Why tiled attention?

Scaled dot-product attention is

$$
O = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{D}}\right)V.
$$

Conventional implementations can materialize the intermediate score matrix with shape `[B, H, N, N]`. At `(B,H,N,D)=(1,16,1024,64)`, a single FP32 score matrix occupies **64 MiB**:

$$
1 \times 16 \times 1024^2 \times 4\ \mathrm{bytes} = 64\ \mathrm{MiB}.
$$

This implementation avoids materializing the complete score tensor by processing query tiles of `BLOCK_M` rows against successive key/value tiles of `BLOCK_N` rows. Dense exact attention still has quadratic arithmetic complexity in sequence length; the design primarily targets intermediate storage and memory traffic.

### Online-softmax update

For each query row, the kernel maintains a running maximum $m$, exponential sum $\ell$, and unnormalized output accumulator $a$. For a new score tile $S$:

$$
\begin{aligned}
m' &= \max(m,\operatorname{rowmax}(S)), \\
\alpha &= \exp(m-m'), \\
P &= \exp(S-m'), \\
\ell' &= \alpha\ell + \operatorname{rowsum}(P), \\
a' &= \alpha a + PV.
\end{aligned}
$$

The final result is $O=a/\ell$. Boundary and causal masks exclude invalid scores. Each program handles one query tile for one batch/head pair; the grid is `(ceil(N/BLOCK_M), B*H)`.

**Small example:** With `B=1, H=2, N=128, BLOCK_M=32`, each head requires four query-tile programs, giving eight programs in total. Each program iterates over key/value tiles without sharing softmax state across heads.

## Installation and usage

Prerequisites: Python 3.11+, a CUDA-capable NVIDIA GPU, a working CUDA-enabled PyTorch installation, and Triton. Exact package constraints are defined in [`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock).

```bash
git clone https://github.com/danishshaikh06/flash_infer.git
cd flash_infer
uv sync
uv run pytest tests/test2_forward.py -s
```

```python
import torch
from flash_infer.attention.forward_pass import flash_attention
from flash_infer.config import FlashAttentionConfig

q = torch.randn(1, 16, 1024, 64, device="cuda", dtype=torch.float32)
k = torch.randn_like(q)
v = torch.randn_like(q)

config = FlashAttentionConfig(
    block_m=32,
    block_n=64,
    num_warps=4,
    num_stages=2,
)

output = flash_attention(q, k, v, config, causal=True)
assert output.shape == q.shape
```

The sample configuration is the best observed *among a specific measured sweep*; it is not a universal optimum. The current interface requires Q, K, V to have identical four-dimensional shapes and matching dtype/device. FP16 and BF16 are accepted by the wrapper, but their numerical/performance behavior is not yet validated by the documented test suite.

## Correctness results

The generalized forward kernel was evaluated against PyTorch `torch.nn.functional.scaled_dot_product_attention` using FP32 tensors, `atol=2e-3`, and `rtol=2e-3`. All eight listed cases passed.

| Shape `(B,H,N,D)` | Non-causal max error | Causal max error |
| --- | ---: | ---: |
| `(1,1,4,16)` | 3.9e-7 | 3.9e-7 |
| `(1,2,17,16)` | 4.8e-7 | 7.2e-7 |
| `(2,2,33,64)` | 1.31e-6 | 1.07e-6 |
| `(2,4,64,64)` | **1.67e-6** | 1.55e-6 |

The largest observed absolute error was **1.67e-6**. Non-multiple lengths (`N=17`, `N=33`) exercise partial-tile masking. These results validate the listed FP32 cases, **not** arbitrary strides, all supported dtypes, every sequence length, or backward gradients. See [`tests/test2_forward.py`](tests/test2_forward.py).

## Benchmark methodology

**Environment:** NVIDIA GeForce RTX 4070 Laptop GPU (compute capability 8.9), WSL/Linux, Python 3.11, PyTorch `2.13.0+cu130`, FP32. The generalized benchmarking scripts use CUDA events after warmup, with 200 timed iterations per repetition and the median of five repetitions. Timings include Python wrapper/launch activity within the timed loop, not just device-kernel execution.

**Interpretation:** `SDPA / Triton > 1` means the custom implementation is faster; a ratio below 1 means SDPA is faster. Values depend on input shape, dtype, backend selection, power/clock state, and measurement conditions. No blanket SDPA speedup is claimed.

### Generalized forward versus PyTorch SDPA

Recorded results for `BLOCK_M=32`, `BLOCK_N=64` (milliseconds), from [`forward_benchmark_with_naive_sdpa.csv`](benchmarks/results/forward_benchmark_with_naive_sdpa.csv):

**Non-causal**

| Shape `(B,H,N,D)` | Naive PyTorch | SDPA | Triton | SDPA / Triton |
| --- | ---: | ---: | ---: | ---: |
| `(1,1,128,64)` | 0.1428 | 0.0268 | 0.1364 | 0.20× |
| `(1,8,512,64)` | 0.1263 | 0.1281 | 0.1529 | 0.84× |
| `(1,16,1024,64)` | 1.8763 | 0.8578 | **1.1115** | 0.77× |
| `(2,8,1024,64)` | 1.8758 | 0.8558 | 1.1115 | 0.77× |

**Causal**

| Shape `(B,H,N,D)` | SDPA | Triton | SDPA / Triton |
| --- | ---: | ---: | ---: |
| `(1,1,128,64)` | 0.0244 | 0.0479 | 0.51× |
| `(1,8,512,64)` | 0.0939 | 0.1573 | 0.60× |
| `(1,16,1024,64)` | 0.4880 | **1.1432** | 0.43× |
| `(2,8,1024,64)` | 0.4880 | 1.1434 | 0.43× |

**Important:** The historical naive PyTorch implementation did **not** apply a causal mask. Naive values from causal-labeled runs are consequently not valid causal baselines and are not shown in the causal table.

**Current outcome:** the forward kernel is numerically validated but remains slower than optimized PyTorch SDPA on these representative batched/multi-head measurements, especially for causal attention.

### Prototype query and key/value tile sweeps

The earlier two-dimensional `[N,D]` prototype was used to isolate tile tradeoffs. Those findings are historical and should **not** be mixed with generalized `[B,H,N,D]` results.

| Experiment | Tile change | Recorded latency | Relative improvement |
| --- | --- | --- | ---: |
| `N=1024, D=32, BLOCK_N=16` | `BLOCK_M: 2 → 16` | 0.5015 → 0.0793 ms | ~6.3× |
| `N=1024, D=64, BLOCK_M=16` | `BLOCK_N: 16 → 64` | 0.1374 → 0.0863 ms | ~1.59× |

Larger tiles reduced loop/program overhead but increased working-set and hardware-resource requirements. With `BLOCK_M=16, BLOCK_N=128, D=64`, compilation failed because **143,424 bytes of shared memory were required**, exceeding the **101,376-byte** reported limit. Detailed historical notes: [`block_tuning.md`](benchmarks/results/block_tuning.md).

### Launch-configuration sweep

The generalized kernel was tuned for `(B,H,N,D)=(1,16,1024,64)` across **72** combinations:

- `BLOCK_M ∈ {16, 32}` and `BLOCK_N ∈ {32, 64}`
- `num_warps ∈ {2, 4, 8}` and `num_stages ∈ {2, 3, 4}`
- both causal modes

**60 configurations succeeded and 12 were skipped** due to resource limits. The fastest observed configuration in *both* modes was:

| Mode | `BLOCK_M` | `BLOCK_N` | Warps | Stages | Median latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Non-causal | **32** | **64** | **4** | **2** | **1.1111 ms** |
| Causal | **32** | **64** | **4** | **2** | **1.1434 ms** |

Notable observations: higher pipeline stage counts generally did not improve latency; eight warps were often slower; and the two-warp version of `BLOCK_M=32, BLOCK_N=64, num_stages=2` measured **8.2309 ms**, over 7× slower than the four-warp variant. The cause of that anomaly is **not yet established**. Several four-stage configurations exceeded the per-block shared-memory limit (one required **106,496 bytes** against a **101,376-byte** limit).

Full results: [`kernel_config_sweep.csv`](benchmarks/results/kernel_config_sweep.csv). The winning configuration is workload-specific, not globally optimal.

## Nsight Compute: initial kernel profile

Non-causal attention, `B=1, H=16, N=1024, D=64`, `BLOCK_M=32`, `BLOCK_N=64`, `num_warps=4`, `num_stages=2`:

| Hardware metric | Measured value |
| --- | ---: |
| Grid / CUDA block | `(32,16,1)` / 128 threads |
| Register allocation | **255 registers/thread** |
| Dynamic shared memory | **49.15 KiB/block** |
| Driver shared memory | 1.02 KiB/block |
| Register-limited resident blocks/SM | 2 |
| Shared-memory-limited resident blocks/SM | 2 |
| Theoretical active warps/SM | 8 |
| Theoretical occupancy | **16.67%** |
| Achieved occupancy | **16.35%** |
| L1/TEX throughput | **91.42%** |
| Memory throughput (SOL) | 86.69% |
| Compute (SM) throughput | 22.80% |
| L2 throughput | 10.43% |
| DRAM throughput | 2.82% |
| Nsight-reported duration | 1.74 ms |

Both register and shared-memory requirements independently constrain residency to **two blocks per SM**. Eight resident warps against a maximum of 48 explain the **16.67% theoretical occupancy**. High L1/TEX throughput and low external DRAM throughput motivate a more detailed on-chip memory-path investigation.

These figures **do not prove register spilling** or identify a single dominant bottleneck. The Nsight capture duration is also not directly comparable to unprofiled CUDA-event timings. Planned follow-up: local-memory traffic, scheduler/warp stalls, instruction mix, and causal tile-skipping opportunities.

## Running the experiments

```bash
uv run pytest tests/test2_forward.py -s
uv run python benchmarks/benchmark_forward.py
```

The current `benchmarks/benchmark_forward.py` performs the **72-configuration sweep** and writes `benchmarks/results/kernel_config_sweep.csv`. Re-running it can overwrite existing CSV output. The older `benchmark_block_m_sweep.py` and `benchmark_block_n_sweep.py` target the previous two-dimensional prototype interface and require adaptation before use with the generalized kernel.

## Repository structure

```text
flash_infer/
├── src/flash_infer/
│   ├── attention/forward_pass.py      # Triton kernel + public wrapper
│   └── config.py                      # Tile and launch configuration
├── tests/
│   ├── test1_forward.py               # Earlier test cases
│   └── test2_forward.py               # Generalized forward correctness
├── benchmarks/
│   ├── benchmark_block_m_sweep.py     # Historical prototype experiment
│   ├── benchmark_block_n_sweep.py     # Historical prototype experiment
│   ├── benchmark_forward.py           # Current launch-config sweep
│   └── results/                       # Benchmark CSVs and tuning notes
├── docs/flash_attention/
│   └── flash_attention.md             # Mathematical implementation notes
├── pyproject.toml
├── uv.lock
└── README.md
```

## Roadmap

1. **Diagnose the current hardware bottleneck** using detailed Nsight Compute counters, including potential register spilling and warp stalls.
2. **Optimize forward** through resource-aware tiling, execution layouts, improved causal processing, and suitable precision paths.
3. **Extend correctness coverage** to FP16, BF16, additional head dimensions, non-contiguous layouts, and numerical edge cases.
4. **Publish controlled benchmarks** with explicitly identified SDPA backend and consistent GPU operating conditions.
5. **Implement backward** and validate gradients with respect to `Q`, `K`, and `V` against PyTorch autograd.

## References

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [Triton documentation and tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [NVIDIA Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/)

---

**Development status:** Forward v1 has passed the documented FP32 correctness tests. Profiling-guided optimization and backward support are ongoing.
