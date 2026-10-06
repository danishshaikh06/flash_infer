import torch
import triton
import triton.language as tl
import math 
from flash_infer.config import FlashAttentionConfig

@triton.jit
def flash_attention_forward_kernel(
    Q,
    K,
    V,
    O,
    stride_qm,
    stride_qd,
    stride_kn,
    stride_kd,
    stride_vn,
    stride_vd,
    stride_om,
    stride_od,
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

    # Q: [BLOCK_M, D] -> Get the contiguous memory location 
    '''
    Imagine Q starts at memory address:
    1000
    
    And Q contains:
    Q =
    [10 20 30 40
     50 60 70 80]
    
    Assume each element takes one memory unit.
    
    Then:
    
    address: 1000 1001 1002 1003 1004 1005 1006 1007
    value:      10   20   30   40   50   60   70   80
    
    Your offsets are:
    
    [0 1 2 3
     4 5 6 7]
    
    So:
    
    q_ptrs = Q + offsets
    
    conceptually becomes:
    
    [1000 1001 1002 1003
     1004 1005 1006 1007]
    
    These are addresses, not values.
    
    Then:
    
    tl.load(q_ptrs)
    
    means:
    
    Go to those addresses and load the values.
    '''

    q_ptrs = (
        Q # ex-> Q[0,0], [0,1] Here is the starting memory address where the Q tensor is stored.
        + offs_m[:, None] * stride_qm 
        + offs_d[None, :] * stride_qd
    )

    #After getting the contiguous load it into fast storages like shared memory/registers
    q = tl.load(
        q_ptrs,
        mask=q_mask[:, None],
        other=0.0, 
    )

    q = q.to(tl.float32)

    # Intialize online softmax state -> one m and l for every row query  
    m_i = tl.full(
        [BLOCK_M],
        -float("inf"),
        dtype=tl.float32,
    )

    #sum of the all the exponetional for each row 
    l_i = tl.zeros(
        [BLOCK_M],
        dtype = tl.float32,
    )

    # Output accumulator shape O[BLOCK_M, D] -> stores the unnormalized attention outputs 
    acc = tl.zeros(
        [BLOCK_M, D],
        dtype = tl.float32,
    )

    scale = 1.0 / math.sqrt(D)

    for start_n in range(0, N, BLOCK_N):

        offs_n = (
            start_n + tl.arange(0,BLOCK_N)
            )

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

        k = k.to(tl.float32)

        # QK^T
        scores = tl.dot(
            q,
            tl.trans(k),
            input_precision = "ieee",
        )
        scores = scores*scale

        # tl.where(condition, A, B)
        # For every element, if condition is True, choose A; otherwise choose B.
        # Invalid K position must not participate in softmax 
        scores = tl.where( 
            k_mask[None, :], # broadcasting [[True,False] [True, False]]
            scores,
            -float("inf"),
        )

        # Online softmax

        #one max per query row 
        m_ij = tl.max(
            scores,
            axis=1,
        )

        # compare the previou and new 
        m_new= tl.maximum(
            m_i,
            m_ij,
        )

        #correction factor
        alpha = tl.exp(m_i - m_new)

        # p = exp(scores-m_new)
        p = tl.exp(scores - m_new[: , None])

        #new sum of exponential for each row after fininding new maximum for that row and applyig the correction factor 
        l_new = (alpha * l_i + tl.sum(p,axis=1))

        # Load V block shape [Block_N, D]

        v_ptrs = (
            V
            + offs_n[:, None] * stride_vn
            + offs_d[None,:] * stride_vd
        )

        v = tl.load(
            v_ptrs,
            mask = k_mask[:,None],
            other = 0.0,
        )

        v = v.to(tl.float32)

        # update ouptpu accumulator
        # formula: acc_new = alpha * acc_old + P@v
        acc_new = alpha[:,None] * acc   

        acc_new += tl.dot(p, v)

        m_i = m_new
        l_i = l_new
        acc = acc_new

    # normalize it finally 
    # O = acc/l
    acc = (
        acc
        /l_i[:,None]
    )

    # Store the output for N-Dimensions
    o_ptrs = (
        O
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )

    tl.store(
        o_ptrs,
        acc,
        mask=q_mask[:,None],
        #scores,
        #mask=q_mask[:, None] & k_mask[None, :],
    )

def flash_attention(q,k,v,config: FlashAttentionConfig):
    assert q.is_cuda
    assert k.is_cuda
    assert v.is_cuda

    assert q.shape == k.shape
    assert q.shape == v.shape 

    assert q.ndim == 2

    assert q.is_contiguous() 
    assert k.is_contiguous() 
    assert v.is_contiguous() 

    N, D = q.shape 

    BLOCK_M = config.block_m
    BLOCK_N = config.block_n

    output = torch.empty_like(q)

    grid = (
        triton.cdiv(N, BLOCK_M),
    )

    flash_attention_forward_kernel[grid](
        q,
        k,
        v,
        output,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        output.stride(0),
        output.stride(1),
        N,
        D=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return output

