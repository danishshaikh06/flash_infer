# FlashAttention — Detailed Notes

### Progress covered so far: Q/K tiling → Triton programs → pointer arithmetic → strides → score tiles → masking → running maximum

---

# 1. Start with the mathematical problem

Attention starts with three matrices:

[
Q,\ K,\ V
]

where:

* (Q) = Queries
* (K) = Keys
* (V) = Values

For now, we have mainly been working with **Q and K**.

The attention score matrix is:

[
\boxed{S = QK^T}
]

Then usually:

[
S = \frac{QK^T}{\sqrt D}
]

and then:

[
P = \operatorname{softmax}(S)
]

and finally:

[
O = PV
]

So the full mathematical pipeline is:

[
\boxed{
Q,K,V
\rightarrow
QK^T
\rightarrow
\text{scale}
\rightarrow
\text{softmax}
\rightarrow
PV
}
]

FlashAttention is mainly about computing this efficiently **without materializing the entire attention matrix in GPU memory**.

---

# 2. What are Q, K and V?

Suppose:

[
N=16
]

and:

[
D=16
]

Then:

[
Q\in\mathbb R^{16\times16}
]

[
K\in\mathbb R^{16\times16}
]

[
V\in\mathbb R^{16\times16}
]

Think of each row as one token.

```text
Q

token 0 → [q00 q01 q02 ... q0,15]
token 1 → [q10 q11 q12 ... q1,15]
token 2 → [q20 q21 q22 ... q2,15]
...
token 15
```

Each token has a feature vector of size (D=16).

So:

```text
N = number of tokens
D = dimension of each token vector
```

---

# 3. What does QKᵀ mean geometrically?

Suppose:

[
Q =
\begin{bmatrix}
q_0\
q_1\
q_2\
q_3
\end{bmatrix}
]

and:

[
K =
\begin{bmatrix}
k_0\
k_1\
k_2\
k_3
\end{bmatrix}
]

Then:

[
QK^T
]

produces:

[
\begin{bmatrix}
q_0\cdot k_0 & q_0\cdot k_1 & q_0\cdot k_2 & q_0\cdot k_3\
q_1\cdot k_0 & q_1\cdot k_1 & q_1\cdot k_2 & q_1\cdot k_3\
q_2\cdot k_0 & q_2\cdot k_1 & q_2\cdot k_2 & q_2\cdot k_3\
q_3\cdot k_0 & q_3\cdot k_1 & q_3\cdot k_2 & q_3\cdot k_3
\end{bmatrix}
]

So:

[
\boxed{S_{ij}=q_i\cdot k_j}
]

Each score tells us how strongly query (i) matches key (j).

---

# 4. Why is the transpose needed?

Suppose:

[
Q:[N,D]
]

and:

[
K:[N,D]
]

Then:

[
K^T:[D,N]
]

Therefore:

[
[N,D]\times[D,N]
================

[N,N]
]

Example:

[
[16,16]\times[16,16]
====================

[16,16]
]

The middle dimensions match.

---

# 5. Small numerical example

Let's use:

[
D=2
]

and:

[
Q=
\begin{bmatrix}
1&0\
0&1
\end{bmatrix}
]

and:

[
K=
\begin{bmatrix}
1&0\
0&1\
1&1\
2&1
\end{bmatrix}
]

Then:

[
K^T=
\begin{bmatrix}
1&0&1&2\
0&1&1&1
\end{bmatrix}
]

Therefore:

[
QK^T=
\begin{bmatrix}
1&0&1&2\
0&1&1&1
\end{bmatrix}
]

For example:

[
Q_0\cdot K_2
============

# [1,0]\cdot[1,1]

1
]

and:

[
Q_1\cdot K_3
============

# [0,1]\cdot[2,1]

1
]

---

# 6. Why don't we calculate the entire matrix at once?

Suppose:

[
N=4096
]

Then:

[
QK^T
]

has shape:

[
4096\times4096
]

Number of elements:

[
4096^2=16,777,216
]

That's a huge matrix.

And for larger sequence lengths it becomes even worse because attention has:

[
\boxed{O(N^2)}
]

score elements.

FlashAttention uses **tiling** to avoid storing the entire score matrix.

---

# 7. What is a block/tile?

Instead of processing:

```text
Q = 16 × 16
```

all at once, suppose:

```python
BLOCK_M = 2
BLOCK_N = 2
```

Then Q is divided into blocks of 2 rows:

```text
Q:

rows 0,1    → Q block 0
rows 2,3    → Q block 1
rows 4,5    → Q block 2
rows 6,7    → Q block 3
rows 8,9    → Q block 4
rows 10,11  → Q block 5
rows 12,13  → Q block 6
rows 14,15  → Q block 7
```

There are:

[
16/2=8
]

Q blocks.

Similarly, K has:

[
16/2=8
]

K blocks.

---

# 8. Geometric view of the score matrix

The complete score matrix is:

[
S=QK^T
]

with shape:

[
16\times16
]

Divide it into (2\times2) tiles:

```text
                  K blocks
             0    1    2    3    4    5    6    7
          ┌────┬────┬────┬────┬────┬────┬────┬────┐
Q block 0 │    │    │    │    │    │    │    │    │
          ├────┼────┼────┼────┼────┼────┼────┼────┤
Q block 1 │    │    │    │    │    │    │    │    │
          ├────┼────┼────┼────┼────┼────┼────┼────┤
Q block 2 │    │    │    │    │    │    │    │    │
          ├────┼────┼────┼────┼────┼────┼────┼────┤
...
```

There are:

[
8\times8=64
]

tiles.

Each tile is:

[
2\times2
]

and:

[
64\times4=256=16\times16
]

---

# 9. Triton program

A Triton **program** is roughly a unit of work that executes on the GPU.

You wrote:

```python
grid = (1,)
```

This means:

[
\boxed{\text{1 Triton program}}
]

Then:

```python
pid_m = tl.program_id(0)
```

returns:

```text
pid_m = 0
```

So your program processes Q block 0.

---

# 10. Your current kernel

You have:

```python
BLOCK_M = 2
BLOCK_N = 2
N = 16
grid = (1,)
```

Therefore:

```text
One program
     │
     ▼
Q block 0
rows 0,1
     │
     ▼
K block 0
rows 0,1
     │
     ▼
K block 1
rows 2,3
     │
     ▼
...
     │
     ▼
K block 7
rows 14,15
```

So yes:

[
\boxed{\text{one Q block × all K blocks}}
]

---

# 11. How all Q blocks are eventually handled

Mathematically you might imagine:

```python
for q_block in range(8):
    for k_block in range(8):
        compute_tile()
```

But Triton normally parallelizes the outer dimension.

Instead of:

```python
for q_block in range(8):
```

you launch:

```python
grid = (8,)
```

Then:

```python
pid_m = tl.program_id(0)
```

gives:

```text
program 0 → Q block 0
program 1 → Q block 1
program 2 → Q block 2
...
program 7 → Q block 7
```

Each program then loops over K:

```python
for start_n in range(0, N, BLOCK_N):
```

So the conceptual structure is:

```text
Program 0 → Q0 → K0 K1 K2 K3 K4 K5 K6 K7
Program 1 → Q1 → K0 K1 K2 K3 K4 K5 K6 K7
Program 2 → Q2 → K0 K1 K2 K3 K4 K5 K6 K7
...
Program 7 → Q7 → K0 K1 K2 K3 K4 K5 K6 K7
```

---

# 12. `pid_m`

Your code:

```python
pid_m = tl.program_id(0)
```

means:

> Which Q block am I responsible for?

If:

```text
pid_m = 0
```

then:

```text
Q block 0
```

If:

```text
pid_m = 3
```

then:

```text
Q block 3
```

---

# 13. `offs_m`

You have:

```python
offs_m = (
    pid_m * BLOCK_M
    + tl.arange(0, BLOCK_M)
)
```

Suppose:

```text
pid_m = 3
BLOCK_M = 2
```

Then:

[
offs_m=3\times2+[0,1]
]

[
\boxed{offs_m=[6,7]}
]

Therefore this program processes:

```text
Q rows 6 and 7
```

---

# 14. `tl.arange`

When you write:

```python
tl.arange(0, BLOCK_M)
```

and:

```python
BLOCK_M = 2
```

you get:

[
[0,1]
]

If:

```python
BLOCK_M = 4
```

you get:

[
[0,1,2,3]
]

It's similar conceptually to:

```python
range(0, BLOCK_M)
```

but it's a Triton tensor of offsets used for vectorized GPU operations.

---

# 15. `offs_d`

You have:

```python
offs_d = tl.arange(0, D)
```

If:

```text
D = 16
```

then:

[
offs_d=[0,1,2,\ldots,15]
]

These are the feature dimensions.

---

# 16. Shape of Q block

Suppose:

```text
BLOCK_M = 2
D = 16
```

Then:

[
q:[2,16]
]

So one program loads:

```text
2 query vectors
```

with:

```text
16 features each
```

Geometrically:

```text
Q block

        D = 16
   ──────────────────→

┌─────────────────────┐
│ query row 0         │
├─────────────────────┤
│ query row 1         │
└─────────────────────┘
↑
2 rows
```

---

# 17. `[:, None]` and `[None, :]`

This is extremely important.

Suppose:

```python
offs_m = [0,1]
```

Its shape is:

[
[2]
]

Then:

```python
offs_m[:, None]
```

changes its shape to:

[
[2,1]
]

giving:

```text
[[0],
 [1]]
```

And:

```python
offs_d[None, :]
```

if:

```text
offs_d = [0,1,2,3]
```

becomes:

```text
[[0,1,2,3]]
```

shape:

[
[1,4]
]

---

# 18. Why do we do that?

Because broadcasting gives:

```text
offs_m[:,None]      offs_d[None,:]

[0]                 [0 1 2 3]
[1]
```

which broadcasts into:

```text
[0 1 2 3]
[0 1 2 3]
```

This creates all combinations:

[
(m,d)
]

So we get:

[
[BLOCK_M,D]
]

addresses.

---

# 19. What is Q inside Triton?

When your kernel receives:

```python
Q
```

Triton treats `Q` as a **pointer to the beginning of Q's memory**.

It is not the Python matrix in the normal sense.

Think:

```text
Q
│
▼
memory address of Q[0,0]
```

Then pointer arithmetic tells Triton which elements to access.

---

# 20. `Q.stride()`

Suppose:

```python
Q.shape = (2,4)
```

and Q is contiguous.

Memory:

```text
Q:

[ a ][ b ][ c ][ d ][ e ][ f ][ g ][ h ]
  └──── row 0 ────┘  └──── row 1 ────┘
```

Then:

```python
Q.stride()
```

returns:

[
(4,1)
]

So:

```python
Q.stride(0) = 4
Q.stride(1) = 1
```

---

# 21. Meaning of stride

`stride(0)`:

> How many memory elements do I move to go to the next row?

`stride(1)`:

> How many memory elements do I move to go to the next column?

For:

```text
Q.shape = (2,4)
```

we have:

[
\boxed{stride(0)=4}
]

because each row contains 4 elements.

And:

[
\boxed{stride(1)=1}
]

because neighboring columns are next to each other.

---

# 22. General contiguous 2D tensor

If:

[
Q.shape=(M,D)
]

then normally:

[
\boxed{
Q.stride(0)=D
}
]

and:

[
\boxed{
Q.stride(1)=1
}
]

So for:

```text
Q.shape = (16,16)
```

you get:

```text
Q.stride() = (16,1)
```

---

# 23. PyTorch calculates the stride, not Triton

When you write:

```python
Q.stride(0)
```

**PyTorch** calculates the stride based on the actual memory layout.

Then you pass it:

```python
score_tile_kernel[grid](
    Q,
    ...
    Q.stride(0),
    Q.stride(1),
)
```

Triton receives those values.

So:

```text
PyTorch
   │
   │ Q.stride()
   ▼
(16,1)
   │
   │ passed as arguments
   ▼
Triton
   │
   ├── stride_qm = 16
   └── stride_qd = 1
```

---

# 24. Why pass strides explicitly?

Because tensors don't always have the same memory layout.

A transposed tensor can have different strides.

So Triton doesn't blindly assume:

[
stride=(D,1)
]

Instead, you tell it the actual layout.

---

# 25. The famous `q_ptrs`

Your code:

```python
q_ptrs = (
    Q
    + offs_m[:, None] * stride_qm
    + offs_d[None, :] * stride_qd
)
```

This is one of the most important lines.

It calculates:

[
\boxed{\text{memory address of every }Q[m,d]}
]

Mathematically:

[
\boxed{
address(Q[m,d])
===============

Q_{\text{base}}
+
m\cdot stride_{qm}
+
d\cdot stride_{qd}
}
]

---

# 26. Numerical example for `q_ptrs`

Suppose:

```text
Q.shape = (2,4)
```

so:

```text
stride_qm = 4
stride_qd = 1
```

and:

```text
offs_m = [0,1]
offs_d = [0,1,2,3]
```

Then:

```python
offs_m[:,None]
```

is:

```text
[[0],
 [1]]
```

and:

```python
offs_d[None,:]
```

is:

```text
[[0,1,2,3]]
```

Now:

[
offs_m[:,None]\times4
]

becomes:

```text
[[0],
 [4]]
```

And:

[
offs_d[None,:]\times1
]

becomes:

```text
[[0,1,2,3]]
```

Add them:

```text
[[0,1,2,3],
 [4,5,6,7]]
```

These are exactly the memory positions of:

```text
Q[0,0] Q[0,1] Q[0,2] Q[0,3]

Q[1,0] Q[1,1] Q[1,2] Q[1,3]
```

---

# 27. Then `tl.load`

You have:

```python
q = tl.load(
    q_ptrs,
    mask=q_mask[:, None],
    other=0.0,
)
```

This means:

> Go to all the addresses in `q_ptrs` and load the values.

So:

```text
q_ptrs
   │
   ▼
memory
   │
   ▼
q [BLOCK_M,D]
```

---

# 28. Why the mask?

Suppose:

```text
N = 5
BLOCK_M = 2
```

and a program handles:

```text
offs_m = [4,5]
```

But row 5 doesn't exist.

Because:

[
5<5
]

is false.

So:

```python
q_mask = offs_m < N
```

gives:

[
[True,False]
]

Then:

```python
mask=q_mask[:,None]
```

becomes:

```text
[[True],
 [False]]
```

This prevents Triton from loading invalid Q row 5.

---

# 29. `other=0.0`

For invalid positions:

```python
other=0.0
```

means:

> If the mask is false, pretend the value is 0.

So invalid Q entries become zero.

---

# 30. K works exactly the same way

K block:

```python
k_ptrs = (
    K
    + offs_n[:, None] * stride_kn
    + offs_d[None, :] * stride_kd
)
```

Mathematically:

[
\boxed{
address(K[n,d])
===============

K_{\text{base}}
+n\cdot stride_{kn}
+d\cdot stride_{kd}
}
]

---

# 31. K loop

You have:

```python
for start_n in range(0, N, BLOCK_N):
```

For:

```text
N = 16
BLOCK_N = 2
```

you get:

```text
start_n = 0
start_n = 2
start_n = 4
start_n = 6
start_n = 8
start_n = 10
start_n = 12
start_n = 14
```

Then:

```python
offs_n = start_n + tl.arange(0,BLOCK_N)
```

gives:

```text
[0,1]
[2,3]
[4,5]
[6,7]
[8,9]
[10,11]
[12,13]
[14,15]
```

---

# 32. `scores = tl.dot(q, tl.trans(k))`

Suppose:

```text
q.shape = [2,16]
k.shape = [2,16]
```

Then:

```text
tl.trans(k)
```

has:

[
[16,2]
]

Therefore:

[
[2,16]\times[16,2]
==================

[2,2]
]

So:

```python
scores
```

has shape:

```text
[BLOCK_M,BLOCK_N]
```

---

# 33. What each score means

Suppose:

```text
q =
[q0]
[q1]
```

and:

```text
k =
[k0]
[k1]
```

Then:

```text
scores =
┌─────────────┐
│ q0·k0  q0·k1│
│ q1·k0  q1·k1│
└─────────────┘
```

So:

[
scores_{ij}=q_i\cdot k_j
]

---

# 34. `tl.trans(k)`

If:

```text
k.shape = [BLOCK_N,D]
```

then:

```python
tl.trans(k)
```

has:

```text
[D,BLOCK_N]
```

This lets matrix multiplication work.

---

# 35. Output tile

Suppose:

```text
q.shape = [2,16]
k.shape = [2,16]
```

Then:

```text
scores.shape = [2,2]
```

That means:

```text
2 Q rows × 2 K rows
```

produce a:

```text
2 × 2 score tile
```

---

# 36. `OUT`

You allocated:

```python
OUT = torch.zeros(
    (16,16),
    device="cuda",
    dtype=torch.float32,
)
```

Therefore:

[
OUT.shape=[16,16]
]

But this **does not mean the kernel computes all 16×16 values**.

Your current:

```python
grid=(1,)
```

only computes Q block 0.

So only:

[
2\times16
]

positions are written.

The rest remain zero because you initialized them to zero.

---

# 37. Output pointer arithmetic

For a contiguous `[N,N]` output matrix:

[
OUT[m,n]
]

has memory offset:

[
\boxed{mN+n}
]

So:

```python
out_ptrs = (
    OUT
    + offs_m[:,None] * N
    + offs_n[None,:]
)
```

is the correct conceptual address calculation.

---

# 38. Important: your earlier `BLOCK_N` output expression

You had:

```python
OUT + offs_m[:, None] * BLOCK_N + offs_n[None, :]
```

That is only correct if the output row stride happens to equal `BLOCK_N`.

For a full `[16,16]` contiguous output:

[
stride_{OUT,row}=16
]

not 2.

So better:

```python
OUT + offs_m[:, None] * N + offs_n[None, :]
```

or, even better in a general kernel, pass the actual output strides.

---

# 39. Masking K

At the boundary:

```python
k_mask = offs_n < N
```

Suppose:

```text
N=5
BLOCK_N=2
```

and:

```text
offs_n=[4,5]
```

Then:

[
k_mask=[True,False]
]

because K row 5 doesn't exist.

---

# 40. Why use `-inf` for invalid scores?

You showed:

```python
scores = tl.where(
    k_mask[None, :],
    scores,
    -float("inf"),
)
```

This means:

```text
valid K position
     ↓
keep score

invalid K position
     ↓
replace score with -∞
```

Example:

```text
scores:

[ 1.5   2.3 ]
[ 0.7   1.2 ]

mask:

[ True False ]

after where:

[ 1.5   -∞ ]
[ 0.7   -∞ ]
```

---

# 41. Why `-inf` instead of zero?

Because you're going to calculate:

```python
tl.max(scores, axis=1)
```

Suppose:

```text
scores = [5.0, invalid]
```

If invalid becomes zero:

[
\max(5,0)=5
]

That's okay in this case.

But if:

```text
scores = [-3.0, invalid]
```

and invalid becomes zero:

[
\max(-3,0)=0
]

Wrong!

The invalid position wins.

But with:

[
-\infty
]

we get:

[
\max(-3,-\infty)=-3
]

Correct.

Therefore:

[
\boxed{\text{invalid score}=-\infty}
]

is perfect for maximum reduction.

---

# 42. `tl.max(scores, axis=1)`

Suppose:

```text
scores =
┌──────┬──────┐
│ 1.5  │ -∞   │
├──────┼──────┤
│ 0.7  │ -∞   │
└──────┴──────┘
```

Then:

```python
tl.max(scores, axis=1)
```

means:

> Take the maximum horizontally across K positions.

Result:

```text
[1.5, 0.7]
```

So:

[
\boxed{
m_{ij}=\max_j S_{ij}
}
]

---

# 43. Why do we need a running maximum?

Because K is processed in chunks.

Imagine Q row 0 sees:

```text
K block 0:
[1.2, 0.4]
```

maximum:

[
1.2
]

Then K block 1:

```text
[2.8, 1.5]
```

maximum:

[
2.8
]

Then K block 2:

```text
[0.9, 4.1]
```

maximum:

[
4.1
]

The maximum over **all K blocks** is:

[
\max(1.2,2.8,4.1)=4.1
]

So we maintain:

```text
m_i
```

as the running maximum.

---

# 44. `tl.maximum`

You have:

```python
m_i = tl.maximum(
    m_i,
    m_ij,
)
```

This is elementwise.

Suppose:

```text
m_i  = [2.8, 3.0]
m_ij = [4.1, 1.5]
```

Then:

```text
m_i = [4.1, 3.0]
```

because:

[
\max(2.8,4.1)=4.1
]

[
\max(3.0,1.5)=3.0
]

---

# 45. `OUT_MAX`

Suppose we eventually want one maximum per query row.

If:

[
N=16
]

then:

```python
OUT_MAX.shape = (16,)
```

Conceptually:

```text
OUT_MAX:

Q0 → maximum
Q1 → maximum
Q2 → maximum
...
Q15 → maximum
```

---

# 46. `out_ptrs = OUT_MAX + offs_m`

Suppose:

```text
offs_m=[4,5]
```

Then:

```python
out_ptrs = OUT_MAX + offs_m
```

means:

```text
OUT_MAX[4]
OUT_MAX[5]
```

The pointer array points to those two locations.

Then:

```python
tl.store(
    out_ptrs,
    m_i,
)
```

could store:

```text
m_i=[3.2,4.5]
```

resulting in:

```text
OUT_MAX[4] = 3.2
OUT_MAX[5] = 4.5
```

---

# 47. Why no `[:,None]` for `OUT_MAX`?

Because `OUT_MAX` is 1D.

```text
OUT_MAX.shape = [N]
```

while Q is 2D:

```text
Q.shape = [N,D]
```

For Q you need two coordinates:

[
(m,d)
]

For OUT_MAX you need only:

[
(m)
]

Therefore:

```python
Q + m*stride_m + d*stride_d
```

versus:

```python
OUT_MAX + m
```

---

# 48. Shape summary

This is worth memorizing.

Suppose:

```text
N = 16
D = 16
BLOCK_M = 2
BLOCK_N = 2
```

Then:

| Object      | Shape     |
| ----------- | --------- |
| `Q`         | `[16,16]` |
| `K`         | `[16,16]` |
| Q block `q` | `[2,16]`  |
| K block `k` | `[2,16]`  |
| `k.T`       | `[16,2]`  |
| `scores`    | `[2,2]`   |
| `OUT`       | `[16,16]` |
| `OUT_MAX`   | `[16]`    |
| `offs_m`    | `[2]`     |
| `offs_n`    | `[2]`     |
| `offs_d`    | `[16]`    |

---

# 49. Important distinction: shape vs stride vs pointer

These three concepts are easy to mix up.

### Shape

Tells you:

> How many elements are in each dimension?

Example:

```text
Q.shape = (16,16)
```

means:

```text
16 rows
16 columns
```

---

### Stride

Tells you:

> How far do I move in memory to move one position in a particular dimension?

For contiguous:

```text
Q.shape = (16,16)

Q.stride() = (16,1)
```

---

### Pointer

Tells you:

> Where in memory does the tensor begin?

Triton sees:

```text
Q
 ↓
memory address
```

Then pointer arithmetic calculates specific elements.

---

# 50. The fundamental pointer equation

For a 2D tensor:

[
\boxed{
A[i,j]
======

A_{\text{base}}
+i\cdot stride_0
+j\cdot stride_1
}
]

This equation is probably the **single most important low-level equation** you've encountered so far.

For Q:

[
\boxed{
Q[m,d]
======

Q_{\text{base}}
+m\cdot stride_{qm}
+d\cdot stride_{qd}
}
]

For K:

[
\boxed{
K[n,d]
======

K_{\text{base}}
+n\cdot stride_{kn}
+d\cdot stride_{kd}
}
]

---

# 51. `torch.rand`

You asked about:

```python
Q = torch.rand(
    (16,16),
    device="cuda",
    dtype=torch.float32,
)
```

Yes, this is completely valid.

It produces:

[
Q\in\mathbb R^{16\times16}
]

with random values approximately in:

[
[0,1)
]

For example:

```text
[0.23  0.81  0.15 ...]
[0.62  0.04  0.93 ...]
...
```

The values do **not** need to be 16.

---

# 52. Values vs dimensions

This distinction is important.

When you say:

```python
Q = torch.rand((16,16))
```

the first `16` means:

[
\text{number of rows}
]

and second `16` means:

[
D=\text{features per row}
]

It does **not** mean the values have to be 16.

Values can be:

```text
0.1
0.5
2.7
9.2
...
```

---

# 53. Can D be smaller than 16?

Mathematically, absolutely.

For example:

[
Q:[4,8]
]

and:

[
K:[4,8]
]

then:

[
QK^T:
[4,8]\times[8,4]
================

[4,4]
]

So mathematically there is nothing special about 16.

However, **Triton's `tl.dot` has hardware/compiler constraints depending on dtype, GPU architecture, and configuration**, so a particular Triton kernel may require a convenient/padded dimension.

That's a Triton implementation issue, not an attention mathematics issue.

---

# 54. Why tiling helps FlashAttention

The big idea is:

Instead of:

```text
Calculate entire QKᵀ
        ↓
Store huge N×N matrix
        ↓
Softmax entire matrix
        ↓
Multiply by V
```

FlashAttention processes small tiles.

Conceptually:

```text
Load Q block
      ↓
Load K block
      ↓
Compute score tile
      ↓
Apply online softmax statistics
      ↓
Load next K block
      ↓
Repeat
```

The important thing is that the **entire (N\times N) score matrix doesn't need to live in GPU global memory**.

---

# 55. The current stage you're studying

So far you've started moving from:

### Naive attention

[
S=QK^T
]

toward:

### Tiled attention

[
S_{tile}=Q_{block}K_{block}^T
]

and then toward:

### Online softmax

Instead of needing the entire row:

[
S_i=[s_{i0},s_{i1},...,s_{i,N-1}]
]

we process pieces:

[
S_i^{(0)},S_i^{(1)},S_i^{(2)},...
]

and maintain statistics such as:

[
\boxed{m_i=\max_j s_{ij}}
]

without materializing the entire row.

---

# 56. The current algorithmic picture

For one Q block:

```text
                Q block
                   │
                   │ load once
                   ▼
              q [BM,D]
                   │
                   │
       ┌───────────┼───────────┐
       │           │           │
       ▼           ▼           ▼
      K0          K1          K2 ...
       │           │           │
       ▼           ▼           ▼
    q @ K0ᵀ     q @ K1ᵀ     q @ K2ᵀ
       │           │           │
       ▼           ▼           ▼
     scores      scores      scores
       │           │           │
       ▼           ▼           ▼
     mask        mask        mask
       │           │           │
       ▼           ▼           ▼
    max K0       max K1       max K2
       │           │           │
       └───────────┼───────────┘
                   ▼
            running maximum
                   │
                   ▼
                 m_i
```

---

# 57. The most important mental model

When you look at a Triton attention kernel, keep asking four questions:

### Question 1: Which Q rows does this program own?

Look at:

```python
pid_m
offs_m
BLOCK_M
```

---

### Question 2: Which K rows are we currently processing?

Look at:

```python
start_n
offs_n
BLOCK_N
```

---

### Question 3: How do we find the actual data in memory?

Look at:

```python
stride
pointer arithmetic
tl.load
```

Especially:

```python
Q + offs_m[:,None] * stride_qm
  + offs_d[None,:] * stride_qd
```

---

### Question 4: What mathematical operation is happening?

For example:

```python
tl.dot(q, tl.trans(k))
```

means:

[
Q_{block}K_{block}^T
]

and:

```python
tl.max(scores, axis=1)
```

means:

[
\max_{\text{K positions}} score
]

---

# 58. A complete mental translation of your kernel

When you see:

```python
pid_m = tl.program_id(0)
```

read it as:

> "Which Q block am I?"

When you see:

```python
offs_m = pid_m * BLOCK_M + tl.arange(0,BLOCK_M)
```

read it as:

> "Which Q rows belong to me?"

When you see:

```python
offs_d = tl.arange(0,D)
```

read it as:

> "Give me all feature dimensions."

When you see:

```python
q_ptrs = Q + ...
```

read it as:

> "Calculate addresses of my Q block."

When you see:

```python
q = tl.load(...)
```

read it as:

> "Bring my Q block into registers/local computation."

When you see:

```python
for start_n in range(...)
```

read it as:

> "Walk through K blocks."

When you see:

```python
k_ptrs = K + ...
```

read it as:

> "Calculate addresses of the current K block."

When you see:

```python
k = tl.load(...)
```

read it as:

> "Load this K block."

When you see:

```python
tl.dot(q, tl.trans(k))
```

read it as:

> "Compute this Q-block × K-block score tile."

When you see:

```python
tl.where(k_mask, scores, -inf)
```

read it as:

> "Make invalid K positions impossible to select as the maximum."

When you see:

```python
tl.max(scores, axis=1)
```

read it as:

> "Find the maximum score for each query row within this K block."

When you see:

```python
tl.maximum(m_i, m_ij)
```

read it as:

> "Merge this block's maximum into the running maximum."

When you see:

```python
OUT_MAX + offs_m
```

read it as:

> "Point to the output location corresponding to each query row."

---

# 59. One final big picture

For your current example:

[
N=16,\quad D=16
]

[
BLOCK_M=2,\quad BLOCK_N=2
]

there are:

[
8\ Q\text{-blocks}
]

and:

[
8\ K\text{-blocks}
]

Therefore the score matrix consists of:

[
8\times8=64
]

tiles.

With:

```python
grid=(1,)
```

you currently compute:

[
\boxed{1\times8=8\text{ score tiles}}
]

which corresponds to:

[
\boxed{2\times16}
]

elements of the full score matrix.

If you change to:

```python
grid=(8,)
```

then you have:

[
8\times8=64
]

tiles, covering the complete:

[
16\times16
]

score matrix.

---

# 60. Cheat sheet

Keep this nearby while reading your Triton code:

```text
N
│
└── number of tokens / sequence length

D
│
└── features per token

Q
│
└── [N,D]

K
│
└── [N,D]

QKᵀ
│
└── [N,N]

BLOCK_M
│
└── Q rows handled by one Triton program

BLOCK_N
│
└── K rows processed in one loop iteration

pid_m
│
└── which Q block this program owns

offs_m
│
└── actual Q row indices

offs_n
│
└── actual K row indices

offs_d
│
└── feature indices

stride
│
└── memory step between elements along a dimension

Q.stride(0)
│
└── memory step to next Q row

Q.stride(1)
│
└── memory step to next Q column

q_ptrs
│
└── addresses of Q block

k_ptrs
│
└── addresses of K block

tl.load
│
└── load data from those addresses

tl.dot(q,kᵀ)
│
└── score tile

tl.where(mask,scores,-∞)
│
└── invalidate positions for max/softmax

tl.max(...,axis=1)
│
└── maximum for each query row

m_i
│
└── running maximum across K blocks

OUT_MAX + offs_m
│
└── addresses of one maximum per Q row
```

### The single most important distinction to remember

[
\boxed{
\text{Shape}=\text{how many elements}
}
]

[
\boxed{
\text{Stride}=\text{how far apart they are in memory}
}
]

[
\boxed{
\text{Pointer}=\text{where the memory starts}
}
]

And the fundamental Triton address equation is:

[
\boxed{
\text{address}(A[i,j])
======================

A_{\text{base}}
+i\cdot stride_0
+j\cdot stride_1
}
]

