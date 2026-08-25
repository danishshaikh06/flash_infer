import torch
import triton
import triton.language as tl


@triton.jit
def score_tile_kernel(
    Q,
    K,
    OUT_MAX,
    stride_qm,
    stride_qd,
    stride_kn,
    stride_kd,
    N, 
    D: tl.constexpr,  
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
):
    # Which query block?
    pid_m = tl.program_id(0) 

    # Query rows handled by this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) 

    # Feature dimension
    offs_d = tl.arange(0, D)

    q_mask = offs_m < N

    # Q: [BLOCK_M, D]
    q_ptrs = (
        Q 
        + offs_m[:, None] * stride_qm 
        + offs_d[None, :] * stride_qd
    )

    q = tl.load(
        q_ptrs,
        mask=q_mask[:, None],
        other=0.0, 
    )

    # Intialize the running maximum as -inf 
    m_i = tl.full(
        [BLOCK_M],
        -float("inf"),
        dtype=tl.float32,
    )

    for start_n in range(0, N, BLOCK_N):

        offs_n = start_n + tl.arange(0,BLOCK_N)

        k_mask = offs_n < N

        # K: [BLOCK_N, D]
        k_ptrs = (
            K
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )

        k = tl.load(
            k_ptrs,
            mask=k_mask[:, None],
            other=0.0,
        )

        # QK^T
        scores = tl.dot(
            q,
            tl.trans(k),
        )

        # tl.where(condition, A, B)
        # For every element, if condition is True, choose A; otherwise choose B.
        scores = tl.where( 
            k_mask[None, :], # broadcasting [[True,False] [True, False]]
            scores,
            -float("inf"),
        )

        #one max per query row 
        m_ij = tl.max(
            scores,
            axis=1,
        )

        # compare the previou and new 
        m_i = tl.maximum(
            m_i,
            m_ij,
        )

        # for 1-d tensors 
        out_ptrs = (
            OUT_MAX + offs_m
        )

        '''
        # Store score tile fo N - Dimensions
        out_ptrs = (
            OUT
            + offs_m[:, None] * BLOCK_N
            + offs_n[None, :]
        )'''

        
        tl.store(
            out_ptrs,
            m_i,
            mask=q_mask,
            #scores,
            #mask=q_mask[:, None] & k_mask[None, :],
        )


def demo():
    D = 16
    Q = torch.rand(
    (4,D),
    device="cuda",
    dtype=torch.float32,
    )

    K = torch.rand(
        (4,D),
        device="cuda",
        dtype=torch.float32,
    )

    BLOCK_M = 2
    BLOCK_N = 2

    OUT_MAX = torch.empty(
        (4,),
        device="cuda",
        dtype=torch.float32,
    )

    '''# For now, one output tile
    OUT = torch.zeros(
        (16, 16),
        device="cuda",
        dtype=torch.float32,
    )'''

    N = Q.shape[0]
    #grid = (triton.cdiv(N, BLOCK_M),)
    grid = (1,)

    score_tile_kernel[grid](
        Q,
        K,
        OUT_MAX,
        Q.stride(0),
        Q.stride(1),
        K.stride(0),
        K.stride(1),
        Q.shape[0],
        D=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    print("Final row-wise maximum:")
    print(OUT_MAX[:2])
    #print(OUT)

if __name__ == "__main__":
    demo()