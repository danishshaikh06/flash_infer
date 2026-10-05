\# FlashAttention Kernel Tuning: \`BLOCK\_M\` and \`BLOCK\_N\` Sweeps

\## Overview

This document records the first performance-engineering experiments for the custom Triton FlashAttention forward kernel.

The purpose of the experiments was not just to find a fast configuration, but to understand how tile shape affects:

\- kernel latency

\- number of Triton programs and K/V iterations

\- working-set size

\- compiler resource usage

\- behavior across sequence lengths and head dimensions

\- the trade-off between larger tiles and GPU resource pressure

\### Test environment

\- GPU: \*\*NVIDIA GeForce RTX 4070 Laptop GPU\*\*

\- PyTorch: \*\*2.13.0+cu130\*\*

\- dtype: \*\*FP32\*\*

\- attention: non-causal self-attention

\- tensor shape: \`Q, K, V = \[N, D\]\`

\- Triton kernel: tiled FlashAttention-style forward pass with online softmax and fused \`P @ V\` accumulation

The experiments were performed after the kernel had passed the correctness tests described below.

\---

\## 1. Kernel being tuned

The mathematical operation is:

\\\[

O = \\operatorname{softmax}\\left(\\frac{QK^T}{\\sqrt{D}}\\right)V

\\\]

where:

\\\[

Q,K,V \\in \\mathbb{R}^{N\\times D}

\\\]

A naive implementation creates the complete score matrix:

\\\[

S = QK^T \\in \\mathbb{R}^{N\\times N}

\\\]

The custom Triton implementation instead processes the computation in blocks.

For a query block:

\\\[

Q\_i \\in \\mathbb{R}^{BLOCK\_M\\times D}

\\\]

and a K/V block:

\\\[

K\_j,V\_j \\in \\mathbb{R}^{BLOCK\_N\\times D}

\\\]

it computes:

\\\[

S\_{ij}=Q\_iK\_j^T

\\\]

The kernel maintains online softmax state for each query row:

\\\[

m\_i = \\text{running maximum}

\\\]

\\\[

l\_i = \\text{running exponential sum}

\\\]

and an unnormalized output accumulator:

\\\[

acc\_i

\\\]

For every K/V block:

\\\[

m\_{new}=\\max(m\_{old},m\_{block})

\\\]

\\\[

\\alpha=\\exp(m\_{old}-m\_{new})

\\\]

\\\[

P=\\exp(S-m\_{new})

\\\]

\\\[

l\_{new}=\\alpha l\_{old}+\\sum P

\\\]

\\\[

acc\_{new}=\\alpha acc\_{old}+PV

\\\]

After all K/V blocks have been processed, the output is normalized once:

\\\[

O = \\frac{acc}{l}

\\\]

This final normalization placement is important. Normalizing \`acc\` inside the K/V loop was an earlier correctness bug; the working kernel normalizes only after the loop.

\---

\## 2. Correctness status before benchmarking

The kernel was tested over sequence lengths that deliberately exercise tile boundaries and multiple K/V blocks:

\`\`\`text

N = 4, 5, 16, 17, 31, 32, 33

D = 16

\`\`\`

The test was repeated across five random seeds.

All cases passed.

Observed maximum absolute errors were approximately in the range:

\\\[

4\\times10^{-4} \\text{ to } 1.1\\times10^{-3}

\\\]

with mean errors generally around:

\\\[

10^{-4} \\text{ to } 3\\times10^{-4}

\\\]

The boundary cases were useful because they tested more than simple full-tile execution:

\- \`N=5\`: partial K/V tile

\- \`N=17\`: one full K/V tile plus a partial second tile

\- \`N=31,32,33\`: additional boundary behavior around the tile size

This gave enough confidence to begin performance tuning.

\---

\# 3. Benchmark methodology

Three implementations were compared.

\## Naive PyTorch

\`\`\`text

scores = Q @ K.T

probs = softmax(scores / sqrt(D))

output = probs @ V

\`\`\`

This explicitly materializes the \`N x N\` score matrix.

\## PyTorch SDPA

\`\`\`text

torch.nn.functional.scaled\_dot\_product\_attention

\`\`\`

This was treated as the stronger production-oriented baseline.

\## Custom Triton FlashAttention

The custom kernel uses tiled Q/K/V processing, online softmax, and fused output accumulation without materializing the full attention matrix.

GPU latency was measured with CUDA events.

The initial \`BLOCK\_M\` benchmark used approximately:

\`\`\`text

WARMUP = 30

ITERATIONS = 100

\`\`\`

The later \`BLOCK\_N\` experiment used a more stable setup:

\`\`\`text

WARMUP = 50

ITERATIONS = 200

REPEATS = 5

\`\`\`

and used the median measured latency.

For future performance claims, the repeated-measurement methodology should be preferred.

\---

\# 4. \`BLOCK\_M\` sweep

\## What \`BLOCK\_M\` controls

\`BLOCK\_M\` is the number of query rows processed by one Triton program.

For:

\\\[

N=1024

\\\]

an approximate number of query programs is:

| \`BLOCK\_M\` | Query programs |

|---:|---:|

| 2 | 512 |

| 4 | 256 |

| 8 | 128 |

| 16 | 64 |

| 32 | 32 |

| 64 | 16 |

| 128 | 8 |

A small tile gives many small programs.

A large tile gives fewer programs, but each program carries more state.

The accumulator has shape:

\\\[

BLOCK\_M \\times D

\\\]

so increasing \`BLOCK\_M\` increases per-program working state.

The basic trade-off is:

\\\[

\\boxed{\\text{larger BLOCK\\\_M} \\rightarrow \\text{fewer programs and less overhead}}

\\\]

versus:

\\\[

\\boxed{\\text{larger BLOCK\\\_M} \\rightarrow \\text{larger per-program resource usage}}

\\\]

\---

\# 5. \`BLOCK\_M\` results at \`D=16\`

Best observed \`BLOCK\_M\` for each tested sequence length:

| N | Best \`BLOCK\_M\` | Triton ms |

|---:|---:|---:|

| 64 | 8 | 0.0377 |

| 128 | 128 | 0.0356 |

| 256 | 8 | 0.0352 |

| 512 | 64 | 0.0358 |

| 1024 | 16 | 0.0398 |

There was no universal winner.

The clearest case was \`N=1024\`:

\`\`\`text

BLOCK\_M=2 -> 0.2424 ms

BLOCK\_M=4 -> 0.1067 ms

BLOCK\_M=8 -> 0.0560 ms

BLOCK\_M=16 -> 0.0398 ms

BLOCK\_M=32 -> 0.0484 ms

BLOCK\_M=64 -> 0.0554 ms

BLOCK\_M=128 -> 0.0682 ms

\`\`\`

The ratio between the slowest and the best configuration was approximately:

\\\[

0.2424/0.0398 \\approx 6.1\\times

\\\]

This demonstrated that tile size can have a large impact even though the underlying attention algorithm is unchanged.

\---

\# 6. \`BLOCK\_M\` results at \`D=32\`

Best observed configurations:

| N | Best \`BLOCK\_M\` | Triton ms |

|---:|---:|---:|

| 64 | 4 | 0.0352 |

| 128 | 4 | 0.0337 |

| 256 | 16 | 0.0350 |

| 512 | 16 | 0.0395 |

| 1024 | 16 | 0.0793 |

For \`N=1024\` the sweep was:

\`\`\`text

BLOCK\_M=2 -> 0.5015 ms

BLOCK\_M=4 -> 0.2712 ms

BLOCK\_M=8 -> 0.1398 ms

BLOCK\_M=16 -> 0.0793 ms

BLOCK\_M=32 -> 0.0826 ms

BLOCK\_M=64 -> 0.1448 ms

BLOCK\_M=128 -> 0.1528 ms

\`\`\`

Again, \`BLOCK\_M=16\` landed in the best region, while both very small and very large query tiles performed worse.

\---

\# 7. \`BLOCK\_M\` results at \`D=64\`

With \`BLOCK\_N=16\`, the best observed configurations were:

| N | Best \`BLOCK\_M\` | Triton ms |

|---:|---:|---:|

| 64 | 128 | 0.0360 |

| 128 | 8 | 0.0388 |

| 256 | 32 | 0.0398 |

| 512 | 8 | 0.0680 |

| 1024 | 16 | 0.1377 |

This confirmed that the best \`BLOCK\_M\` depends on workload shape.

The important large-sequence candidate was:

\`\`\`text

N = 1024

D = 64

BLOCK\_M = 16

BLOCK\_N = 16

Triton = 0.1377 ms

\`\`\`

That configuration became the starting point for the \`BLOCK\_N\` sweep.

\---

\# 8. What the \`BLOCK\_M\` sweep taught us

The experiment showed three regimes.

\### \`BLOCK\_M\` too small

For example, at \`N=1024\` and \`D=32\`:

\\\[

BLOCK\_M=2 \\Rightarrow 512

\\\]

query programs.

The large number of programs creates substantial overhead relative to the useful work per program.

\### \`BLOCK\_M\` in a useful middle range

Values such as \`8\`, \`16\`, or \`32\` often performed best for larger workloads.

\### \`BLOCK\_M\` too large

Large values reduce the number of programs but increase the per-program state.

For \`D=64\`, the conceptual accumulator at \`BLOCK\_M=128\` contains:

\\\[

128\\times64=8192

\\\]

FP32 values.

At four bytes per value:

\\\[

8192\\times4=32768\\text{ bytes}

\\\]

of raw accumulator data.

The compiler's actual register/shared-memory allocation is more complicated than this simple calculation, but the example illustrates why an extremely large query tile can become expensive.

\---

\# 9. \`BLOCK\_N\` sweep

\## What \`BLOCK\_N\` controls

\`BLOCK\_N\` is the number of keys/values processed in one K/V iteration.

For:

\\\[

N=1024

\\\]

the approximate number of K/V iterations is:

| \`BLOCK\_N\` | K/V iterations |

|---:|---:|

| 16 | 64 |

| 32 | 32 |

| 64 | 16 |

| 128 | 8 |

Increasing \`BLOCK\_N\` therefore reduces the number of loop iterations.

But the score tile becomes wider.

With \`BLOCK\_M=16\`:

| \`BLOCK\_N\` | Score tile | Elements |

|---:|---:|---:|

| 16 | \`16 x 16\` | 256 |

| 32 | \`16 x 32\` | 512 |

| 64 | \`16 x 64\` | 1024 |

| 128 | \`16 x 128\` | 2048 |

The trade-off is:

\\\[

\\boxed{\\text{larger BLOCK\\\_N} \\rightarrow \\text{fewer K/V loop iterations}}

\\\]

versus:

\\\[

\\boxed{\\text{larger BLOCK\\\_N} \\rightarrow \\text{larger working set and resource usage}}

\\\]

\---

\# 10. \`BLOCK\_N\` results at \`D=64\`

The \`BLOCK\_N\` experiment fixed:

\`\`\`text

D = 64

BLOCK\_M = 16

\`\`\`

and tested:

\`\`\`text

BLOCK\_N = 16, 32, 64, 128

\`\`\`

for:

\`\`\`text

N = 256, 512, 1024

\`\`\`

Measured results before the \`BN=128\` resource failure were:

| N | BN=16 | BN=32 | BN=64 |

|---:|---:|---:|---:|

| 256 | 0.0770 ms | 0.0824 ms | \*\*0.0507 ms\*\* |

| 512 | 0.0418 ms | \*\*0.0387 ms\*\* | 0.0408 ms |

| 1024 | 0.1374 ms | 0.1289 ms | \*\*0.0863 ms\*\* |

The clearest improvement occurred at \`N=1024\`.

Changing:

\`\`\`text

BLOCK\_N = 16

\`\`\`

to:

\`\`\`text

BLOCK\_N = 64

\`\`\`

reduced Triton latency from:

\\\[

0.1374\\text{ ms} \\rightarrow 0.0863\\text{ ms}

\\\]

which is:

\\\[

0.1374/0.0863 \\approx 1.59\\times

\\\]

faster.

This was the strongest tile-size improvement observed in the \`D=64\` experiments.

\---

\# 11. The \`BLOCK\_N=128\` failure

The \`BLOCK\_M=16, BLOCK\_N=128, D=64\` configuration failed during Triton compilation.

The exact error was:

\`\`\`text

triton.runtime.errors.OutOfResources:

out of resource: shared memory,

Required: 143424,

Hardware limit: 101376.

\`\`\`

So:

\\\[

143424 > 101376

\\\]

and the configuration required:

\\\[

143424-101376=42048

\\\]

more bytes than the reported hardware limit.

The requested amount was approximately:

\\\[

143424/101376 \\approx 1.415

\\\]

or about \*\*41.5% above the available limit\*\*.

This is not a numerical correctness failure. It is a \*\*compile-time resource-limit failure\*\*.

It is also a useful optimization result because it demonstrates that increasing the tile size eventually becomes infeasible under the current kernel structure and compilation parameters.

The tuning problem is therefore constrained by both:

1\. performance

2\. resource feasibility

\---

\# 12. Current best observed configurations

There is no single global optimum across all tested \`N\` and \`D\` values.

The best observed configurations at \`D=64\` across the experiments were:

| N | Best observed configuration | Triton ms | Notes |

|---:|---|---:|---|

| 64 | BM=128, BN=16 | 0.0360 | Best observed in the BM sweep |

| 128 | BM=8, BN=16 | 0.0388 | Best observed in the BM sweep |

| 256 | BM=32, BN=16 | 0.0398 | Faster than SDPA in that benchmark run |

| 512 | BM=16, BN=32 | 0.0387 | Strongest measured configuration at this N |

| 1024 | BM=16, BN=64 | 0.0863 | Strongest large-N configuration measured |

These values come from different experiments and benchmark runs, so they should not be treated as one perfectly apples-to-apples leaderboard.

The best configuration to carry into profiling is:

\`\`\`text

N = 1024

D = 64

BLOCK\_M = 16

BLOCK\_N = 64

\`\`\`

It measured approximately:

\`\`\`text

Triton = 0.0863 ms

SDPA = 0.1366 ms

\`\`\`

for the same \`BLOCK\_N\` benchmark run.

The ratio was approximately:

\\\[

0.1366/0.0863 \\approx 1.58\\times

\\\]

relative to SDPA for that particular run.

This number is useful as an experimental result, but should not yet be treated as a final project-wide performance claim.

\---

\# 13. Important observations about the benchmark itself

\## Naive attention did not show a clean quadratic timing curve

Some of the naive PyTorch measurements remained around \`0.08–0.10 ms\` even as \`N\` increased.

From the algorithmic perspective, the score computation has complexity:

\\\[

O(N^2D)

\\\]

For example, going from \`N=64\` to \`N=1024\` increases the number of score elements by:

\\\[

(1024/64)^2 = 256

\\\]

However, the measured wall-clock timings did not increase by anything close to 256x.

The practical explanation is that these are extremely short GPU workloads. Fixed launch overhead, GPU scheduling, clocks, cache effects, warmup behavior, and measurement variance can dominate a large portion of the observed latency.

Therefore the current results are primarily useful for \*\*relative tuning between kernel configurations\*\*. They should not be used as evidence for asymptotic scaling behavior.

Larger workloads and more rigorous repeated benchmarking are appropriate before publishing final speedup claims.

\## SDPA is the important baseline

The custom kernel beats the naive implementation in many cases, but SDPA is the stronger reference point.

For example:

\`\`\`text

D=64, N=512, BM=16, BN=32

Triton = 0.0387 ms

SDPA = 0.0695 ms

\`\`\`

and:

\`\`\`text

D=64, N=1024, BM=16, BN=64

Triton = 0.0863 ms

SDPA = 0.1366 ms

\`\`\`

These are promising results, but other tile configurations were slower than SDPA.

Therefore the project should not claim a universal SDPA speedup based on the current sweep.

\---

\# 14. Failures and what they taught us

\## Failure 1: benchmark argument mismatch

The first benchmark attempt failed with:

\`\`\`text

TypeError: benchmark\_fn() takes 4 positional arguments but 5 were given

\`\`\`

The benchmark helper originally accepted a fixed number of positional arguments, but the Triton call also needed the configuration object.

The helper was changed to accept variadic arguments:

\`\`\`python

def benchmark\_fn(fn, \*args):

\`\`\`

This allowed the same timing mechanism to work with different kernel configurations.

\## Failure 2: \`BLOCK\_N=128\` resource exhaustion

The \`BLOCK\_N=128\` configuration failed compilation because the kernel required more shared memory than the hardware limit.

This was treated as an engineering constraint, not a reason to discard the experiment.

It tells us that the next stage should investigate exactly how the current kernel uses resources before trying to increase tile size further.

\## Earlier correctness edge case

An initial correctness run had one \`N=5\` case with:

\`\`\`text

max\_error = 0.00100160

\`\`\`

which was only slightly above a \`1e-3\` threshold.

After running the test across five seeds, all cases passed and the observed errors remained small.

This was therefore not treated as evidence of a functional attention error.

\---

\# 15. What we have actually achieved

The kernel is now beyond a basic toy implementation.

It has:

\- numerically stable online softmax

\- tiled Q/K/V processing

\- fused \`P @ V\` accumulation

\- deferred normalization

\- boundary masking

\- multiple K/V block support

\- automated correctness tests

\- configurable \`BLOCK\_M\` and \`BLOCK\_N\`

\- GPU timing with CUDA events

\- comparison against naive PyTorch attention

\- comparison against PyTorch SDPA

\- empirical tile-size tuning

\- a demonstrated shared-memory resource limit

The tuning work also produced concrete evidence that configuration matters substantially.

For example:

\\\[

N=1024,D=32:

\\\]

changing \`BLOCK\_M\` from \`2\` to \`16\` changed one measured Triton latency from roughly:

\\\[

0.5015\\text{ ms}\\rightarrow0.0793\\text{ ms}

\\\]

which is about:

\\\[

6.3\\times

\\\]

faster.

Likewise, at:

\\\[

N=1024,D=64,BLOCK\_M=16

\\\]

changing:

\\\[

BLOCK\_N=16\\rightarrow64

\\\]

changed:

\\\[

0.1374\\text{ ms}\\rightarrow0.0863\\text{ ms}

\\\]

or about:

\\\[

1.59\\times

\\\]

faster.

These observations establish that performance is now being driven by real kernel-implementation decisions rather than only the high-level attention algorithm.

\---

\# 16. Current engineering hypothesis

The current evidence suggests that the next performance bottlenecks are likely to involve the interaction between:

\\\[

BLOCK\_M

\\\]

\\\[

BLOCK\_N

\\\]

\\\[

D

\\\]

and GPU resource usage such as:

\- register pressure

\- shared-memory allocation

\- occupancy

\- instruction throughput

\- memory traffic

A useful mental model is:

\\\[

\\text{performance}

\=f(\\text{tile size},\\text{work per program},\\text{resource usage},\\text{GPU utilization})

\\\]

The benchmark alone tells us \*\*which configuration wins\*\*.

Profiling should tell us \*\*why it wins\*\*.

\---

\# 17. Next step: profiler

The current profiling target should be:

\`\`\`text

N = 1024

D = 64

BLOCK\_M = 16

BLOCK\_N = 64

\`\`\`

The profiler should be used to inspect:

\- registers per thread

\- shared-memory usage

\- occupancy

\- SM utilization

\- achieved memory bandwidth

\- achieved FLOP throughput

\- instruction efficiency

\- kernel duration

The purpose is to determine whether the kernel is primarily limited by:

\`\`\`text

register/resource pressure

memory traffic

compute throughput

low occupancy

or launch/loop overhead

\`\`\`

Only after identifying the bottleneck should the next kernel change be selected.

\---

\# 18. Current conclusion

The first tuning phase established the following:

\\\[

\\boxed{\\text{There is no universal best BLOCK\\\_M}}

\\\]

\\\[

\\boxed{\\text{BLOCK\\\_N=64 is a strong candidate at D=64, especially for larger N}}

\\\]

\\\[

\\boxed{\\text{BLOCK\\\_N=128 is currently resource-infeasible}}

\\\]

and the current profiling candidate is:

\\\[

\\boxed{N=1024,\\ D=64,\\ BLOCK\_M=16,\\ BLOCK\_N=64}

\\\]

The engineering process has therefore moved from:

\`\`\`text

correctness

\-> tile sweep

\-> observe performance differences

\-> hit real hardware resource limits

\-> select a promising configuration

\-> profile the kernel

\`\`\`

The next phase should focus on \*\*explaining and reducing the measured bottleneck\*\*, not changing the attention mathematics again.