import torch
import triton
import triton.language as tl
import math 
from flash_infer.config import FlashAttentionConfig

@triton.jit
def flash_attention_forward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    B,
    H,
    N,
    D,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_on,
    stride_od,
    sm_scale,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Which query block?
    pid_m = tl.program_id(0) 

    # which batch and which head 
    pid_bh = tl.program_id(1)

    # Currently in which batch 
    batch_id = pid_bh // H
    # currently in which head 
    head_id = pid_bh % H
    
    # Query rows handled by this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) 
    # Number of key positions processed in one block 
    offs_n = tl.arange(0,BLOCK_N)
    # Feature dimension
    offs_d = tl.arange(0, D)

    q_valid = offs_m < N
    d_valid = offs_d < N 

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

    q_base = (
        q_ptr
        + batch_id * stride_qb
        + head_id * stride_qh
    )

    k_base = (
        k_ptr
        + batch_id * stride_kb
        + head_id * stride_kh
    )

    v_base = (
            v_ptr
            + batch_id * stride_vb
            + head_id * stride_vh
        )

    o_base = (
            o_ptr
            + batch_id * stride_ob
            + head_id * stride_oh
        )

    q_ptrs = (
        q_base # ex-> Q[0,0], [0,1] Here is the starting memory address where the Q tensor is stored.
        + offs_m[:, None] * stride_qn 
        + offs_d[None, :] * stride_qd
    )

    #After getting the contiguous load it into fast storages like shared memory/registers
    q = tl.load(
        q_ptrs,
        mask=q_valid[:, None] & d_valid[None,:],
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
        [BLOCK_M, BLOCK_D],
        dtype = tl.float32,
    )

    scale = 1.0 / math.sqrt(D)

    for start_n in tl.range(0, N, BLOCK_N):

        current_n = start_n + offs_n

        k_valid = current_n < N

        # K: [BLOCK_N, D]
        k_ptrs = (
            k_base
            + current_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )

        v_ptrs = (
            v_base
            + current_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )

        kv_mask = (
            k_valid[:,None] & d_valid[None,:]
        )

        k = tl.load(
            k_ptrs,
            mask=kv_mask,
            other=0.0,
        )

        v = tl.load(
            v_ptrs,
            mask=kv_mask,
            other=0.0,
        )

        k = k.to(tl.float32)
        v = v.to(tl.float32)

        # QK^T
        scores = tl.dot(
            q,
            tl.trans(k),
            input_precision = "ieee",
        )
        scores = scores * sm_scale

        #causal mask 
        score_mask = (
            q_valid[:,None] & k_valid[None,:]
        )

        if CAUSAL:
            query_idx = offs_m[:,None]
            key_idx = current_n[None,:]

            score_mask = (
                score_mask & (key_idx<=query_idx)
            )

        # tl.where(condition, A, B)
        # For every element, if condition is True, choose A; otherwise choose B.
        # Invalid K position must not participate in softmax 
        scores = tl.where( 
            score_mask,
            scores,
            -float("inf"),
        )

        # Online softmax

        #one max per query row 
        block_max = tl.max(
            scores,
            axis=1,
        )

        safe_m_i = tl.where(
            q_valid,
            m_i,
            0.0,
        )

        # compare the previou and new 
        new_m= tl.maximum(
            safe_m_i,
            block_max,
        )

        #correction factor
        alpha = tl.where(
            q_valid,
            tl.exp(safe_m_i - new_m),
            1.0
        )

        # p = exp(scores-m_new)
        p = tl.exp(scores - new_m[: , None])

        p = tl.where(
            score_mask,
            p,
            0.0,
        )

        #new sum of exponential for each row after fininding new maximum for that row and applyig the correction factor 
        l_new = (alpha * l_i + tl.sum(p,axis=1))

        pv = tl.dot(
            p,
            v,
            input_precision = 'ieee',
        )

        # update ouptpu accumulator
        # formula: acc_new = alpha * acc_old + P@v
        acc = (
            alpha[:,None] * acc
            + pv
        )

        m_i = tl.where(
            q_valid,
            new_m,
            m_i,
        )

        l_i = tl.where(
            q_valid,
            l_new,
            l_i,
        )

    safe_l = tl.where(
        l_i > 0.0,
        l_i,
        1.0,
    )

    # normalize it finally 
    # O = acc/l
    acc = (
        acc / safe_l[:,None]
        
    )

    # Store the output for N-Dimensions
    o_ptrs = (
        o_base
        + offs_m[:, None] * stride_on
        + offs_d[None, :] * stride_od
    )

    tl.store(
        o_ptrs,
        acc,
        mask=q_valid[:, None] & d_valid[None, :],
    )

def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config: FlashAttentionConfig,
    causal: bool = False,
) -> torch.Tensor:

    if q.ndim != 4:
        raise ValueError(
            f"q must have shape [B, H, N, D], got {tuple(q.shape)}"
        )

    if k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            "q, k, and v must have shape [B, H, N, D]"
        )

    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            "q, k, and v must have the same shape"
        )

    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError(
            "q, k, and v must be CUDA tensors"
        )

    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(
            "q, k, and v must have the same dtype"
        )

    if q.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError(
            f"unsupported dtype: {q.dtype}"
        )

    B, H, N, D = q.shape

    output = torch.empty_like(q)

    block_d = triton.next_power_of_2(D)

    grid = (
        triton.cdiv(N, config.block_m),
        B * H,
    )

    sm_scale = 1.0 / math.sqrt(D)

    flash_attention_forward_kernel[grid](
        q,
        k,
        v,
        output,
        B,
        H,
        N,
        D,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        sm_scale,
        CAUSAL=causal,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_D=block_d,
    )

    return output