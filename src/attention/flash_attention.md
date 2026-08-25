```python
"""
Triton QK^T Tile Kernel — Fully Annotated Example
=================================================

This program demonstrates how a Triton kernel computes:

    scores = Q @ K^T

where:

    Q has shape [N, D]
    K has shape [N, D]
    scores has shape [N, N]

The important idea is that Triton does not automatically think in terms
of Python/PyTorch matrix indexing such as Q[m, d].

Instead, inside the Triton kernel, Q and K are used as pointers to GPU
memory. We manually calculate the memory addresses of the elements we
want to load.

For a 2D tensor Q[m, d], the general memory-address formula is:

    address(Q[m, d])
        = Q
        + m * stride_qm
        + d * stride_qd

where:

    Q        = pointer to the beginning of Q's memory
    m        = row index
    d        = column/feature index
    stride_qm = number of memory elements between consecutive rows
    stride_qd = number of memory elements between consecutive columns

For a contiguous [16, 16] tensor:

    stride_qm = 16
    stride_qd = 1

Therefore:

    address(Q[m, d]) = Q + m * 16 + d

The kernel uses tiling.

With:

    BLOCK_M = 16
    BLOCK_N = 16
    D       = 16

one Triton program processes:

    Q tile: [16, 16]
    K tile: [16, 16]

and produces:

    score tile: [16, 16]

Since the example launches:

    grid = (1,)

there is only ONE Triton program.

That one program therefore computes the complete 16 x 16 QK^T
output tile.

Mathematically:

    Q:      [16, 16]
    K:      [16, 16]

    K^T:    [16, 16]

    Q @ K^T:
            [16, 16] @ [16, 16]
            = [16, 16]

For identity matrices:

    Q = I
    K = I

we have:

    Q @ K^T
      = I @ I^T
      = I

so OUT should contain a 16 x 16 identity matrix.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def score_tile_kernel(
    Q,                  # Pointer to the beginning of Q in GPU memory.
    K,                  # Pointer to the beginning of K in GPU memory.
    OUT,                # Pointer to the beginning of OUT in GPU memory.

    stride_qm,          # Q stride along dimension 0 (between Q rows).
    stride_qd,          # Q stride along dimension 1 (between Q columns/features).

    stride_kn,          # K stride along dimension 0 (between K rows).
    stride_kd,          # K stride along dimension 1 (between K columns/features).

    N,                  # Runtime value: total sequence length / number of Q and K rows.

    D: tl.constexpr,    # Compile-time constant: feature/embedding dimension.
    BLOCK_M: tl.constexpr,  # Compile-time constant: number of Q rows handled by one program.
    BLOCK_N: tl.constexpr,  # Compile-time constant: number of K rows handled by one program.
):
    """
    Compute one tile of:

        scores = Q @ K^T

    Mathematical shapes:

        Q      : [BLOCK_M, D]
        K      : [BLOCK_N, D]
        K^T    : [D, BLOCK_N]

        Q @ K^T
               [BLOCK_M, D] @ [D, BLOCK_N]
               = [BLOCK_M, BLOCK_N]

    In this example:

        BLOCK_M = 16
        BLOCK_N = 16
        D       = 16

    so one program computes a [16, 16] score tile.

    The kernel works in four major stages:

    1. Determine which Q rows this program owns.
    2. Calculate Q memory addresses and load the Q tile.
    3. Calculate K memory addresses and load the K tile.
    4. Compute QK^T and store the resulting score tile.

    Important Triton concept:

        Q and K are pointers, not numerical matrices inside the kernel.

    For a 2D tensor Q[m, d], the address is:

        Q + m * stride_qm + d * stride_qd

    The expression:

        Q
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd

    calculates this address for every combination of m and d at once.
    """

    # ------------------------------------------------------------
    # 1. Identify which program instance is running.
    # ------------------------------------------------------------

    # program_id(0) gives this program's ID along grid axis 0.
    #
    # If grid = (1,), then there is only one program:
    #
    #     pid_m = 0
    #
    # If grid = (4,), then the programs would have:
    #
    #     pid_m = 0, 1, 2, 3
    #
    # We use pid_m to determine which Q rows this program processes.
    pid_m = tl.program_id(0)


    # ------------------------------------------------------------
    # 2. Determine which Q rows this program handles.
    # ------------------------------------------------------------

    # tl.arange(0, BLOCK_M) creates:
    #
    #     [0, 1, 2, ..., BLOCK_M - 1]
    #
    # Suppose:
    #
    #     pid_m   = 0
    #     BLOCK_M = 16
    #
    # then:
    #
    #     offs_m = [0, 1, 2, ..., 15]
    #
    # If pid_m = 1:
    #
    #     offs_m = 1 * 16 + [0,1,...,15]
    #            = [16,17,...,31]
    #
    # Therefore pid_m selects WHICH BLOCK of Q rows we process.
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)


    # ------------------------------------------------------------
    # 3. Determine which K rows this program processes.
    # ------------------------------------------------------------

    # In this simplified version, the program loads the first
    # BLOCK_N rows of K.
    #
    # Since:
    #
    #     BLOCK_N = 16
    #
    # we get:
    #
    #     offs_n = [0,1,2,...,15]
    #
    # IMPORTANT:
    # This kernel is intentionally simplified to demonstrate one
    # output tile. For a general tiled implementation, offs_n would
    # usually depend on another program ID or a loop over K blocks.
    offs_n = tl.arange(0, BLOCK_N)


    # ------------------------------------------------------------
    # 4. Feature dimension indices.
    # ------------------------------------------------------------

    # D is the feature/embedding dimension.
    #
    # For D = 16:
    #
    #     offs_d = [0,1,2,...,15]
    #
    # These indices represent the columns/features of Q and K.
    offs_d = tl.arange(0, D)


    # ------------------------------------------------------------
    # 5. Create masks to prevent out-of-bounds memory accesses.
    # ------------------------------------------------------------

    # Check which Q row indices are valid.
    #
    # Example:
    #
    #     offs_m = [0,1,...,15]
    #     N = 16
    #
    # gives:
    #
    #     [True, True, ..., True]
    #
    # If we had an index 16, then:
    #
    #     16 < 16
    #
    # would be False.
    q_mask = offs_m < N


    # Check which K row indices are valid.
    #
    # This protects the K load when BLOCK_N is larger than the
    # remaining number of K rows.
    k_mask = offs_n < N


    # ------------------------------------------------------------
    # 6. Calculate pointers to the Q tile.
    # ------------------------------------------------------------

    # This is one of the most important lines in Triton.
    #
    # Q is a POINTER to the beginning of Q's GPU memory.
    #
    # We want addresses for every:
    #
    #     (m, d)
    #
    # pair where:
    #
    #     m comes from offs_m
    #     d comes from offs_d
    #
    # General mathematical formula:
    #
    #     address(Q[m,d])
    #
    #       = Q
    #       + m * stride_qm
    #       + d * stride_qd
    #
    # offs_m[:, None] has shape:
    #
    #     [BLOCK_M, 1]
    #
    # offs_d[None, :] has shape:
    #
    #     [1, D]
    #
    # Broadcasting combines them to produce:
    #
    #     [BLOCK_M, D]
    #
    # addresses.
    #
    # For a contiguous [16,16] Q:
    #
    #     stride_qm = 16
    #     stride_qd = 1
    #
    # so:
    #
    #     address(Q[m,d]) = Q + 16*m + d
    #
    # Example for a [4,2] matrix:
    #
    #     stride_qm = 2
    #     stride_qd = 1
    #
    #     offsets:
    #
    #     [[0,1],
    #      [2,3]]
    #
    # These are memory offsets for:
    #
    #     Q[0,0], Q[0,1]
    #     Q[1,0], Q[1,1]
    q_ptrs = (
        Q
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )


    # ------------------------------------------------------------
    # 7. Load the Q tile from GPU memory.
    # ------------------------------------------------------------

    # q_ptrs contains MEMORY ADDRESSES.
    #
    # tl.load follows those addresses and retrieves the actual
    # floating-point values stored there.
    #
    # Therefore:
    #
    #     q_ptrs  -> addresses
    #
    #     tl.load(q_ptrs) -> values
    #
    # The resulting q tensor has shape:
    #
    #     [BLOCK_M, D]
    #
    # The mask has shape [BLOCK_M, 1].
    #
    # It broadcasts across D, so an invalid Q row causes all
    # features of that row to use 0.0.
    q = tl.load(
        q_ptrs,
        mask=q_mask[:, None],
        other=0.0,
    )


    # ------------------------------------------------------------
    # 8. Calculate pointers to the K tile.
    # ------------------------------------------------------------

    # The same pointer-arithmetic idea is used for K.
    #
    # For K[n,d]:
    #
    #     address(K[n,d])
    #
    #       = K
    #       + n * stride_kn
    #       + d * stride_kd
    #
    # offs_n[:, None] gives the K row indices.
    #
    # offs_d[None, :] gives the feature indices.
    #
    # Broadcasting produces a [BLOCK_N, D] grid of addresses.
    k_ptrs = (
        K
        + offs_n[:, None] * stride_kn
        + offs_d[None, :] * stride_kd
    )


    # ------------------------------------------------------------
    # 9. Load the K tile.
    # ------------------------------------------------------------

    # k has shape:
    #
    #     [BLOCK_N, D]
    #
    # For this example:
    #
    #     [16,16]
    #
    # We will later transpose it to:
    #
    #     [16,16]
    #
    # so that Q @ K^T is valid.
    k = tl.load(
        k_ptrs,
        mask=k_mask[:, None],
        other=0.0,
    )


    # ------------------------------------------------------------
    # 10. Compute QK^T.
    # ------------------------------------------------------------

    # q has shape:
    #
    #     [BLOCK_M, D]
    #
    # k has shape:
    #
    #     [BLOCK_N, D]
    #
    # tl.trans(k) changes:
    #
    #     [BLOCK_N, D]
    #
    # into:
    #
    #     [D, BLOCK_N]
    #
    # Therefore:
    #
    #     q @ k^T
    #
    # has shape:
    #
    #     [BLOCK_M, D] @ [D, BLOCK_N]
    #
    #     = [BLOCK_M, BLOCK_N]
    #
    # Each output element is:
    #
    #     scores[m,n]
    #
    #       = sum_d q[m,d] * k[n,d]
    #
    # which is exactly the dot product:
    #
    #     Q_m · K_n
    #
    scores = tl.dot(
        q,
        tl.trans(k),
    )


    # ------------------------------------------------------------
    # 11. Calculate output memory addresses.
    # ------------------------------------------------------------

    # OUT is a pointer to the beginning of the output matrix.
    #
    # OUT is assumed to be contiguous with shape [N,N].
    #
    # Therefore:
    #
    #     output row stride = N
    #     output column stride = 1
    #
    # The general address formula is:
    #
    #     OUT + m * N + n
    #
    # offs_m[:,None] represents output rows.
    #
    # offs_n[None,:] represents output columns.
    #
    # Broadcasting creates a [BLOCK_M, BLOCK_N] grid of addresses.
    out_ptrs = (
        OUT
        + offs_m[:, None] * N
        + offs_n[None, :]
    )


    # ------------------------------------------------------------
    # 12. Store the score tile into OUT.
    # ------------------------------------------------------------

    # scores contains the actual computed QK^T values.
    #
    # tl.store writes those values to the memory addresses in
    # out_ptrs.
    #
    # The combined mask:
    #
    #     q_mask[:, None] & k_mask[None, :]
    #
    # makes sure we only write valid output positions.
    tl.store(
        out_ptrs,
        scores,
        mask=q_mask[:, None] & k_mask[None, :],
    )


def demo():
    """
    Create a simple 16 x 16 Q and K example and run the Triton kernel.

    Q and K are identity matrices:

        Q = I
        K = I

    Therefore:

        QK^T
        = I I^T
        = I

    Expected output:

        OUT = 16 x 16 identity matrix

    BLOCK_M = 16 means one program processes 16 Q rows.

    BLOCK_N = 16 means one program processes 16 K rows.

    D = 16 means each Q/K row contains 16 features.

    Since:

        grid = (1,)

    only one Triton program is launched.
    """

    # ------------------------------------------------------------
    # Create Q.
    # ------------------------------------------------------------

    # torch.eye(16) creates a 16 x 16 identity matrix:
    #
    #     [1,0,0,...]
    #     [0,1,0,...]
    #     [0,0,1,...]
    #           ...
    #
    # It is stored directly in GPU memory because of device="cuda".
    Q = torch.eye(
        16,
        device="cuda",
        dtype=torch.float32,
    )


    # ------------------------------------------------------------
    # Create K.
    # ------------------------------------------------------------

    # K is also a 16 x 16 identity matrix.
    K = torch.eye(
        16,
        device="cuda",
        dtype=torch.float32,
    )


    # ------------------------------------------------------------
    # Define tile sizes.
    # ------------------------------------------------------------

    # One Triton program processes 16 rows of Q.
    BLOCK_M = 16

    # One Triton program processes 16 rows of K.
    BLOCK_N = 16

    # Every row of Q and K contains 16 features.
    D = 16


    # ------------------------------------------------------------
    # Allocate output.
    # ------------------------------------------------------------

    # We want:
    #
    #     OUT = Q @ K^T
    #
    # Since Q and K are [16,16]:
    #
    #     OUT is [16,16].
    #
    # torch.empty allocates GPU memory but does NOT initialize
    # the values.
    OUT = torch.empty(
        (16, 16),
        device="cuda",
        dtype=torch.float32,
    )


    # ------------------------------------------------------------
    # Define the Triton grid.
    # ------------------------------------------------------------

    # grid=(1,) means launch exactly ONE Triton program.
    #
    # Therefore:
    #
    #     pid_m = 0
    #
    # and with BLOCK_M=16:
    #
    #     offs_m = [0,1,...,15]
    #
    # So this one program handles all 16 Q rows.
    grid = (1,)


    # ------------------------------------------------------------
    # Launch the Triton kernel.
    # ------------------------------------------------------------

    score_tile_kernel[grid](
        Q,                  # Pointer to Q's GPU memory.
        K,                  # Pointer to K's GPU memory.
        OUT,                # Pointer to OUT's GPU memory.

        # Q.stride(0):
        # number of elements between Q row 0 and row 1.
        #
        # For contiguous [16,16] Q:
        #
        #     Q.stride(0) = 16
        Q.stride(0),

        # Q.stride(1):
        # number of elements between Q column 0 and column 1.
        #
        # For contiguous Q:
        #
        #     Q.stride(1) = 1
        Q.stride(1),

        # K.stride(0):
        # distance between consecutive K rows.
        K.stride(0),

        # K.stride(1):
        # distance between consecutive K columns/features.
        K.stride(1),

        # Runtime sequence length.
        #
        # Q.shape[0] = 16
        #
        # Therefore:
        #
        #     N = 16
        Q.shape[0],

        # Compile-time feature dimension.
        D=D,

        # Compile-time Q tile height.
        BLOCK_M=BLOCK_M,

        # Compile-time K tile width.
        BLOCK_N=BLOCK_N,
    )


    # ------------------------------------------------------------
    # Print the result.
    # ------------------------------------------------------------

    # Since:
    #
    #     Q = I
    #     K = I
    #
    # mathematically:
    #
    #     QK^T = I
    #
    # so OUT should be the identity matrix.
    print(OUT)

```
