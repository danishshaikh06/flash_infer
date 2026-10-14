# FlashInfer

**A Triton implementation of exact, memory-efficient scaled dot-product attention on NVIDIA GPUs.**

FlashInfer is an experimental GPU kernel-engineering project focused on implementing, validating, and optimizing FlashAttention-style computation from first principles. It combines **tiled attention**, **numerically stable online softmax**, **multi-head execution**, **controlled CUDA benchmarks**, and **Nsight Compute profiling**.

> **Project status — Forward v1 implemented.** Batched, multi-head, causal and non-causal self-attention has passed the documented FP32 correctness tests. Hardware-guided forward optimization is in progress; the backward pass is not yet implemented.
>
> This is an independent research and learning project, not a production-ready replacement for PyTorch SDPA or an affiliation with the separately maintained FlashInfer library.

## At a glance

| Area | Implementation / result |
| --- | --- |
| Operation | Exact scaled dot-product **self-attention** |
| Tensor interface | `Q, K, V, O: [batch, heads, sequence, head_dim]` |
| Attention modes | Causal and non-causal |
| GPU kernel | Python + [Triton](https://triton-lang.org/) |
| Numerical method | Online softmax; FP32 intermediate accumulation |
| Memory strategy | Tiled Q/K/V processing; no materialized full `N × N` score matrix |
| Launch configuration | `BLOCK_M`, `BLOCK_N`, `num_warps`, `num_stages` |
| Correctness | Eight reported FP32 cases; **maximum absolute error: 1.67 × 10⁻⁶** |
| Benchmark hardware | NVIDIA GeForce RTX 4070 Laptop GPU (SM 8.9) |
| Best measured launch candidate* | `BLOCK_M=32`, `BLOCK_N=64`, `num_warps=4`, `num_stages=2` |
| Profiling | CUDA events and NVIDIA Nsight Compute |

*Best among the tested configurations for `(B,H,N,D)=(1,16,1024,64)` in the recorded sweep—not a universal optimum.

## Motivation

Scaled dot-product attention computes

$$
O = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{D}}\right)V.
$$

A straightforward implementation may materialize the intermediate score tensor with shape `[B,H,N,N]`. With `B=1`, `H=16`, `N=1024`, and FP32 scores, that tensor alone requires

$$
1 \times 16 \times 1024^2 \times 4\;\text{bytes} = 64\;\text{MiB}.
$$

FlashInfer processes attention in tiles and maintains the sufficient statistics needed to calculate the same softmax output without storing the full attention matrix in GPU global memory. This reduces intermediate storage and memory traffic. **Dense exact attention still performs quadratic work with respect to sequence length**; the objective is a more efficient implementation of that computation, not a change in its mathematical complexity.

## Kernel architecture

### Tiled execution

Each Triton program handles `BLOCK_M` query rows for one `(batch, head)` pair. It walks through the sequence in key/value tiles of `BLOCK_N` rows:

```text
                    One batch/head pair
Q [N,D] → [BLOCK_M,D] ─────┬──── K/V tile 0 [BLOCK_N,D]
                          ├──── K/V tile 1 [BLOCK_N,D]
                          ├──── ...
                          └──── K/V tile J [BLOCK_N,D]
                                   ↓
                         Online-softmax update
                                   ↓
                          Output [BLOCK_M,D]
```

The GPU launch grid is:

```python
grid = (triton.cdiv(N, config.block_m), B * H)
```

For `B=1`, `H=16`, `N=1024`, and `BLOCK_M=32`, the launch contains `32 × 16 = 512` Triton programs. The programs independently process their assigned query tiles and heads.

### Numerically stable online softmax

For each query row, the kernel maintains three quantities:

- **`m`**: the largest attention score encountered so far;
- **`l`**: the running exponential sum relative to that maximum;
- **`acc`**: the unnormalized, weighted sum of value vectors.

Given a new score tile `S = Q_tile @ K_tile.T / sqrt(D)`, the recurrence is

$$
\begin{aligned}
m' &= \max\left(m,\operatorname{rowmax}(S)\right),\\
\alpha &= \exp(m-m'),\\
P &= \exp(S-m'),\\
l' &= \alpha l + \operatorname{rowsum}(P),\\
acc' &= \alpha\,acc + PV.
\end{aligned}
$$

After all K/V tiles have been processed:

$$
\boxed{O=\frac{acc}{l}}
$$

Rescaling by `α` preserves the contributions of previously processed tiles when a later tile contains a larger score. Normalization happens **once, after the tile loop**. Invalid key positions and future-token positions are masked before contributing to the softmax.

**Illustrative example.** Suppose the first tile has scores `[1,2]`, so `m=2` and `l=exp(-1)+1`. If the next tile contains `[5,3]`, the new maximum is `5`; the old exponential sum must be rescaled by `exp(2−5)=exp(-3)` before adding the new contributions. This is the essential reason the full score matrix is unnecessary.

### Tensor layout and supported behavior

The forward wrapper accepts identically shaped CUDA tensors `Q`, `K`, and `V` in `[B,H,N,D]` format. It supports:

- Causal masking (`key_position ≤ query_position`) and non-causal attention.
- Batch/head indexing through a two-dimensional Triton launch grid.
- Explicit tensor strides for Q/K/V and output addressing.
- Partial query/key tiles and padded feature lanes, using masked loads and stores.
- FP32, FP16, and BF16 inputs at the API level; the kernel converts operands to FP32 for its dot-product computation and uses FP32 softmax/accumulation state.

**Validated scope:** the published numerical tests cover **FP32** with head dimensions `16` and `64`. FP16/BF16 accuracy, arbitrary non-contiguous layouts, large head dimensions, and additional numerical edge cases still require dedicated validation. This implementation currently targets same-length self-attention; it does not implement GQA/MQA, KV-cache decoding, paged attention, dropout, variable-length packed sequences, or backward gradients.

## Installation

**Requirements:** Linux or WSL with an NVIDIA CUDA-capable GPU and compatible drivers, Python 3.11+, [uv](https://docs.astral.sh/uv/), and the dependencies declared in [`pyproject.toml`](pyproject.toml). The measurements below were obtained using PyTorch `2.13.0+cu130` and the RTX 4070 Laptop GPU.

```bash
git clone https://github.com/danishshaikh06/flash_infer.git
cd flash_infer
uv sync
uv run pytest tests/test2_forward.py -s
```

### Quick start

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

output = flash_attention(q, k, v, config=config, causal=True)
assert output.shape == q.shape
```

The configuration shown is a measured candidate for one FP32 workload. Different input shapes, dtypes, and GPU architectures may favor other settings or exceed hardware resource limits.

## Numerical correctness

The generalized forward kernel was compared against PyTorch's [`scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), using random FP32 inputs and `torch.testing.assert_close(atol=2e-3, rtol=2e-3)`.

| Tensor shape `(B,H,N,D)` | Non-causal max. abs. error | Causal max. abs. error |
| --- | ---: | ---: |
| `(1,1,4,16)` | 3.9 × 10⁻⁷ | 3.9 × 10⁻⁷ |
| `(1,2,17,16)` | 4.8 × 10⁻⁷ | 7.2 × 10⁻⁷ |
| `(2,2,33,64)` | 1.31 × 10⁻⁶ | 1.07 × 10⁻⁶ |
| `(2,4,64,64)` | **1.67 × 10⁻⁶** | 1.55 × 10⁻⁶ |

All eight reported cases passed. Irregular lengths such as `N=17` and `N=33` exercise boundary masks. **These results demonstrate correctness for the documented FP32 inputs only**; the currently committed generalized tests do not cover all supported precision modes, non-contiguous strides, or long multi-tile sequences.

Test source: [`tests/test2_forward.py`](tests/test2_forward.py).

## Benchmark methodology

| Property | Recorded setup |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| Compute capability | 8.9 |
| Software environment | WSL/Linux, Python 3.11, PyTorch `2.13.0+cu130` |
| Input dtype | FP32 |
| Reference implementation | PyTorch scaled dot-product attention (SDPA) |
| Timing | CUDA events; 50 warmup calls; 200 timed operations per repetition; median of five repetitions |
| Configuration-sweep workload | `B=1, H=16, N=1024, D=64` |

CUDA events measure elapsed time on the **GPU timeline**, not direct Python CPU execution time. Repeated host-side dispatch can nevertheless create gaps between GPU operations, affecting the measured interval. Future benchmark work will use CUDA Graph replay and explicitly document the selected SDPA backend to better isolate device execution latency.

All latency values below are **milliseconds**. The comparison ratio is **`SDPA latency / Triton latency`**; a value below `1.0×` means **PyTorch SDPA is faster**. Performance varies with workload, dtype, backend, compilation settings, and hardware operating conditions.

### Generalized forward: Triton versus PyTorch

These numbers come from [`benchmarks/results/forward_benchmark_with_naive_sdpa.csv`](benchmarks/results/forward_benchmark_with_naive_sdpa.csv), using `BLOCK_M=32`, `BLOCK_N=64`, four warps, and two stages.

**Non-causal attention**

| `(B,H,N,D)` | Naive PyTorch | SDPA | Triton | SDPA / Triton |
| --- | ---: | ---: | ---: | ---: |
| `(1,1,128,64)` | 0.1428 | 0.0268 | 0.1364 | 0.20× |
| `(1,8,512,64)` | 0.1263 | 0.1281 | 0.1529 | 0.84× |
| `(1,16,1024,64)` | 1.8763 | 0.8578 | **1.1115** | 0.77× |
| `(2,8,1024,64)` | 1.8758 | 0.8558 | 1.1115 | 0.77× |

**Causal attention**

| `(B,H,N,D)` | SDPA | Triton | SDPA / Triton |
| --- | ---: | ---: | ---: |
| `(1,1,128,64)` | 0.0244 | 0.0479 | 0.51× |
| `(1,8,512,64)` | 0.0939 | 0.1573 | 0.60× |
| `(1,16,1024,64)` | 0.4880 | **1.1432** | 0.43× |
| `(2,8,1024,64)` | 0.4880 | 1.1434 | 0.43× |

The historical naive-attention reference did not implement causal masking, so it is intentionally **not** presented as a causal baseline.

**Result:** The generalized forward kernel is functionally validated on the reported cases but remains **slower than optimized PyTorch SDPA** on these representative FP32 workloads. No universal SDPA speedup is claimed.

### Historical tile sweeps

An earlier two-dimensional `[N,D]` prototype was used to study how tile geometry affects performance. These experiments preceded the generalized `[B,H,N,D]` API and must **not** be compared directly with its results.

| Experiment | Change | Observed latency | Within-experiment improvement |
| --- | --- | --- | ---: |
| `N=1024, D=32, BLOCK_N=16` | `BLOCK_M: 2 → 16` | `0.5015 → 0.0793 ms` | ~6.3× |
| `N=1024, D=64, BLOCK_M=16` | `BLOCK_N: 16 → 64` | `0.1374 → 0.0863 ms` | ~1.59× |

Increasing tile dimensions can reduce loop iterations but raises resource demands. In the prototype, `BLOCK_M=16, BLOCK_N=128, D=64` exceeded the shared-memory limit: **143,424 bytes requested versus 101,376 bytes available**.

Detailed experimental record: [`benchmarks/results/block_tuning.md`](benchmarks/results/block_tuning.md).

### Generalized launch-configuration sweep

The next experiment evaluated **72 combinations** of:

- `BLOCK_M ∈ {16,32}`, `BLOCK_N ∈ {32,64}`;
- `num_warps ∈ {2,4,8}`, `num_stages ∈ {2,3,4}`;
- `causal ∈ {False,True}`.

Of those configurations, **60 completed** and **12 were skipped** because their shared-memory requirements exceeded the GPU's limit. The fastest observed settings for the tested shape `(1,16,1024,64)` were identical in both modes:

| Attention mode | `BLOCK_M` | `BLOCK_N` | Warps | Stages | Median latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Non-causal | **32** | **64** | **4** | **2** | **1.1111 ms** |
| Causal | **32** | **64** | **4** | **2** | **1.1434 ms** |

Important observations: larger stage counts generally did not help; eight warps were frequently slower; and a two-warp version of `BLOCK_M=32, BLOCK_N=64, num_stages=2` measured **8.2309 ms**, compared with **1.1111 ms** for four warps in non-causal mode. The root cause of this regression has **not** yet been demonstrated. Several four-stage configurations were resource-infeasible, including one requiring **106,496 bytes** against a **101,376-byte** reported launch limit.

Full experimental dataset: [`benchmarks/results/kernel_config_sweep.csv`](benchmarks/results/kernel_config_sweep.csv).

## GPU profiling: Nsight Compute

A first non-causal hardware profile was collected for `(B,H,N,D)=(1,16,1024,64)` with `BLOCK_M=32`, `BLOCK_N=64`, four warps and two stages. The recorded report is available as [`profile_forward_noncausal.ncu-rep`](profile_forward_noncausal.ncu-rep).

| Metric | Observed value |
| --- | ---: |
| Grid / CUDA block | `(32,16,1)` / 128 threads |
| Registers per thread | **255** |
| Dynamic shared memory per block | **49.15 KiB** |
| Driver shared memory per block | 1.02 KiB |
| Register-limited resident blocks per SM | **2** |
| Shared-memory-limited resident blocks per SM | **2** |
| Theoretical active warps per SM | 8 |
| Theoretical occupancy | **16.67%** |
| Achieved occupancy | **16.35%** |
| L1/TEX throughput | **91.42%** |
| Overall memory throughput (SOL) | 86.69% |
| Compute (SM) throughput | 22.80% |
| L2 throughput | 10.43% |
| DRAM throughput | 2.82% |
| Profiler-reported kernel duration | 1.74 ms |

Four warps per program and two resource-limited resident blocks imply **eight active warps per SM**. Relative to the 48-warp maximum, this explains the measured **16.67% theoretical occupancy**. Both register usage and shared-memory allocation constrain residency. Meanwhile, the high L1/TEX throughput and low DRAM throughput justify a closer investigation of on-chip memory-path pressure.

**Interpretation limits:** Low occupancy does not automatically identify the dominant stall reason; 255 registers per thread does **not** prove register spilling. Detailed local-memory, warp-stall, and instruction metrics are the next profiling targets. Nsight's instrumented duration should not be used interchangeably with unprofiled CUDA-event latency.

## Reproducing tests, benchmarks and profiling

```bash
# Current generalized FP32 correctness tests
uv run pytest tests/test2_forward.py -s

# Compare Triton with naive PyTorch and PyTorch SDPA
uv run python benchmarks/benchmark_forward0.py

# Sweep tile sizes, warp counts and pipeline stages
uv run python benchmarks/benchmark_kernel_config.py

# Single-workload profiling launcher (non-causal / causal)
uv run python benchmarks/profile_forward.py
uv run python benchmarks/profile_forward.py --causal
```

For Nsight Compute, a representative non-causal capture command is:

```bash
ncu --set full \
  --kernel-name-base demangled \
  --kernel-name 'regex:flash_attention_forward_kernel' \
  --launch-skip 10 \
  --launch-count 1 \
  -o profile_forward_noncausal \
  uv run python benchmarks/profile_forward.py
```

**Repository maintenance note:** The historical `benchmark_block_m_sweep.py`, `benchmark_block_n_sweep.py`, `tests/test1_forward.py`, and `scripts/attention/forward_pass_demo.py` target the former two-dimensional API. They are retained as experimental history but require adaptation before running against the current four-dimensional kernel. Benchmark scripts may overwrite CSV files with matching output names; preserve published measurements before re-running experiments.

## Repository layout

```text
flash_infer/
├── src/flash_infer/
│   ├── attention/forward_pass.py          # Triton kernel and public API
│   └── config.py                          # Tile and launch parameters
├── tests/
│   ├── test1_forward.py                   # Legacy prototype tests
│   └── test2_forward.py                   # Batched/multi-head correctness
├── benchmarks/
│   ├── benchmark_forward0.py              # PyTorch / SDPA / Triton comparison
│   ├── benchmark_kernel_config.py         # Launch-configuration sweep
│   ├── benchmark_block_m_sweep.py         # Historical query-tile sweep
│   ├── benchmark_block_n_sweep.py         # Historical key/value-tile sweep
│   ├── profile_forward.py                 # Nsight profiling workload
│   ├── common/result.py                   # CSV results utility
│   └── results/                           # Recorded data and tuning notes
├── docs/flash_attention/
│   └── flash_attention.md                 # Detailed online-softmax notes
├── profile_forward_noncausal.ncu-rep      # First saved Nsight report
├── pyproject.toml
├── uv.lock
└── README.md
```

## Roadmap

1. **Characterize the forward bottleneck:** inspect local-memory transactions, register spills, warp stalls, instruction mix, and L1/TEX activity.
2. **Optimize causal execution:** avoid visiting wholly future K/V tiles while preserving numerical behavior.
3. **Improve computational efficiency:** investigate compiler-generated layouts, register/shared-memory pressure, and suitable lower-precision or Tensor Core execution paths.
4. **Extend correctness and reproducibility:** validate FP16/BF16, non-contiguous inputs, multi-tile and adversarial cases; standardize SDPA backend selection and CUDA Graph benchmarking.
5. **Implement backward:** derive and test gradients with respect to `Q`, `K`, and `V` against PyTorch autograd after the forward implementation is characterized.

## References

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [NVIDIA Nsight Compute](https://docs.nvidia.com/nsight-compute/)

---

**Current milestone:** Forward attention implemented and validated for the documented FP32 cases. Profiling-guided optimization and backward support remain in development.
