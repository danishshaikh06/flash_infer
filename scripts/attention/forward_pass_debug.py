import triton 
import triton.language as tl 
import torch 
import math 

@triton.jit
def debug_dot_kernel(
    Q,
    K,
    OUT,
    N,
    stride_qm,
    stride_qd,
    stride_kn,
    stride_kd,
    D:tl.constexpr,
    BLOCK_M:tl.constexpr,
    BLOCK_N:tl.constexpr,
):
    pid = tl.program_id(0)

    offs_m = pid * BLOCK_M + tl.arange(0,BLOCK_M)
    offs_n = tl.arange(0,BLOCK_N)
    offs_d = tl.arange(0,D)

    q_ptrs = (
        Q
        + offs_m[:,None] * stride_qm
        + offs_d[None,:] * stride_qd
    )

    k_ptrs = (
        K
        + offs_n[:, None] * stride_kn
        + offs_d[None, :] * stride_kd
    )

    q = tl.load(
        q_ptrs,
        mask = (offs_m < N)[:,None],
        other = 0.0,
    ).to(tl.float32)

    k = tl.load(
        k_ptrs,
        mask = (offs_n< N)[:,None],
        other = 0.0,
    ).to(tl.float32)

    scores = tl.dot(
        q,
        tl.trans(k),
        input_precision = 'ieee',
    )

    out_ptrs = (
        OUT
        + (offs_m - pid * BLOCK_M)[:, None] * BLOCK_N
        + offs_n[None, :]
    )

    tl.store(
        out_ptrs,
        scores,
        mask=(offs_m < N)[:, None]
        & (offs_n < N)[None, :],
    )


def demo():

    torch.manual_seed(0)

    N = 4
    D = 16

    Q = torch.randn(
        N,
        D,
        device="cuda",
        dtype=torch.float32,
    )

    K = torch.randn(
        N,
        D,
        device="cuda",
        dtype=torch.float32,
    )

    OUT = torch.empty(
        (2, 2),
        device="cuda",
        dtype=torch.float32,
    )

    BLOCK_M = 2
    BLOCK_N = 2

    debug_dot_kernel[(1,)](
        Q,
        K,
        OUT,
        N,
        Q.stride(0),
        Q.stride(1),
        K.stride(0),
        K.stride(1),
        D=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    reference = (
        Q[:2] @ K[:2].T
    )

    print("Triton:")
    print(OUT)

    print("\nPyTorch:")
    print(reference)

    print(
        "\nMax error:",
        (OUT - reference).abs().max().item(),
    )


if __name__ == "__main__":
    demo()

