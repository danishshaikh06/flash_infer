````
# FlashAttention — Detailed Notes

## Progress Covered So Far

**Q/K tiling → Triton programs → pointer arithmetic → strides → score tiles → masking → running maximum**

---

# 1. Start With the Mathematical Problem

Attention starts with three matrices:

$$
Q,\ K,\ V
$$

where:

- $Q$ = Queries
- $K$ = Keys
- $V$ = Values

For now, we have mainly been working with **Q and K**.

The attention score matrix is:

$$
\boxed{S = QK^T}
$$

Then usually:

$$
S = \frac{QK^T}{\sqrt{D}}
$$

Then:

$$
P = \operatorname{softmax}(S)
$$

And finally:

$$
O = PV
$$

So the full mathematical pipeline is:

$$
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
$$

FlashAttention is mainly about computing this efficiently **without materializing the entire attention matrix in GPU memory**.

---

# 2. What Are Q, K, and V?

Suppose:

$$
N=16
$$

and:

$$
D=16
$$

Then:

$$
Q\in\mathbb{R}^{16\times16}
$$

$$
K\in\mathbb{R}^{16\times16}
$$

$$
V\in\mathbb{R}^{16\times16}
$$

Think of each row as one token.

```text
Q

token 0  → [q00 q01 q02 ... q0,15]
token 1  → [q10 q11 q12 ... q1,15]
token 2  → [q20 q21 q22 ... q2,15]
...
token 15
````

 Each token has a feature vector of size $D=16$.

 So:

```
N = number of tokens
D = dimension of each token vector
```

---

 # 3\. What Does QKᵀ Mean Geometrically?

 Suppose:

 $$
Q =
\begin{bmatrix}
q_0\\
q_1\\
q_2\\
q_3
\end{bmatrix}
$$

 and:

 $$
K =
\begin{bmatrix}
k_0\\
k_1\\
k_2\\
k_3
\end{bmatrix}
$$

 Then:

 $$
QK^T
$$

 produces:

 $$
\begin{bmatrix}
q_0\cdot k_0 & q_0\cdot k_1 & q_0\cdot k_2 & q_0\cdot k_3\\
q_1\cdot k_0 & q_1\cdot k_1 & q_1\cdot k_2 & q_1\cdot k_3\\
q_2\cdot k_0 & q_2\cdot k_1 & q_2\cdot k_2 & q_2\cdot k_3\\
q_3\cdot k_0 & q_3\cdot k_1 & q_3\cdot k_2 & q_3\cdot k_3
\end{bmatrix}
$$

 So:

 $$
\boxed{S_{ij}=q_i\cdot k_j}
$$

 Each score tells us how strongly query $i$ matches key $j$.

---

 # 4\. Why Is the Transpose Needed?

 Suppose:

 $$
Q:[N,D]
$$

 and:

 $$
K:[N,D]
$$

 Then:

 $$
K^T:[D,N]
$$

 Therefore:

 $$
[N,D]\times[D,N]
\rightarrow[N,N]
$$

 Example:

 $$
[16,16]\times[16,16]
\rightarrow[16,16]
$$

 The middle dimensions match.

---

 # 5\. Small Numerical Example

 Let's use:

 $$
D=2
$$

 and:

 $$
Q=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
$$

 and:

 $$
K=
\begin{bmatrix}
1&0\\
0&1\\
1&1\\
2&1
\end{bmatrix}
$$

 Then:

 $$
K^T=
\begin{bmatrix}
1&0&1&2\\
0&1&1&1
\end{bmatrix}
$$

 Therefore:

 $$
QK^T=
\begin{bmatrix}
1&0&1&2\\
0&1&1&1
\end{bmatrix}
$$

 For example:

 $$
Q_0\cdot K_2
=
(1\times1)+(0\times1)
=
1+0
=
1
$$

 and:

 $$
Q_1\cdot K_3
=
(0\times2)+(1\times1)
=
0+1
=
1
$$

---

 # 6\. Why Don't We Calculate the Entire Matrix at Once?

 Suppose:

 $$
N=4096
$$

 Then:

 $$
QK^T
$$

 has shape:

 $$
4096\times4096
$$

 Number of elements:

 $$
4096^2=16,777,216
$$

 That's a huge matrix.

 And for larger sequence lengths it becomes even worse because attention has:

 $$
\boxed{O(N^2)}
$$

 score elements.

 FlashAttention uses **tiling** to avoid storing the entire score matrix.

---

 # 7\. What Is a Block/Tile?

 Instead of processing:

```
Q = 16 × 16
```

 all at once, suppose:

```
BLOCK_M = 2
BLOCK_N = 2
```

 Then Q is divided into blocks of 2 rows:

```
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

 $$
16/2=8
$$

 Q blocks.

 Similarly, K has:

 $$
16/2=8
$$

 K blocks.

---

 # 8\. Geometric View of the Score Matrix

 The complete score matrix is:

 $$
S=QK^T
$$

 with shape:

 $$
16\times16
$$

 Divide it into $2\\times2$ tiles:

```
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

 $$
8\times8=64
$$

 tiles.

 Each tile is:

 $$
2\times2
$$

 and:

 $$
64\times4=256=16\times16
$$

---

 # 9\. Triton Program

 A Triton **program** is roughly a unit of work that executes on the GPU.

 You wrote:

```
grid = (1,)
```

 This means:

 $$
\boxed{\text{1 Triton program}}
$$

 Then:

```
pid_m = tl.program_id(0)
```

 returns:

```
pid_m = 0
```

 So your program processes Q block 0.

---

 # 10\. Your Current Kernel

 You have:

```
BLOCK_M = 2
BLOCK_N = 2
N = 16
grid = (1,)
```

 Therefore:

```
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

 So:

 $$
\boxed{\text{one Q block × all K blocks}}
$$

---

 # 11\. How All Q Blocks Are Eventually Handled

 Mathematically you might imagine:

```
for q_block in range(8):
    for k_block in range(8):
        compute_tile()
```

 But Triton normally parallelizes the outer dimension.

 Instead of:

```
for q_block in range(8):
```

 you launch:

```
grid = (8,)
```

 Then:

```
pid_m = tl.program_id(0)
```

 gives:

```
program 0 → Q block 0
program 1 → Q block 1
program 2 → Q block 2
...
program 7 → Q block 7
```

 Each program then loops over K:

```
for start_n in range(0, N, BLOCK_N):
```

 So the conceptual structure is:

```
Program 0 → Q0 → K0 K1 K2 K3 K4 K5 K6 K7
Program 1 → Q1 → K0 K1 K2 K3 K4 K5 K6 K7
Program 2 → Q2 → K0 K1 K2 K3 K4 K5 K6 K7
...
Program 7 → Q7 → K0 K1 K2 K3 K4 K5 K6 K7
```

---

 # 12\. pid\_m

 Your code:

```
pid_m = tl.program_id(0)
```

 means:

 > Which Q block am I responsible for?

 If:

```
pid_m = 0
```

 then:

```
Q block 0
```

 If:

```
pid_m = 3
```

 then:

```
Q block 3
```

---

 # 13\. offs\_m

 You have:

```
offs_m = (
    pid_m * BLOCK_M
    + tl.arange(0, BLOCK_M)
)
```

 Suppose:

```
pid_m = 3
BLOCK_M = 2
```

 Then:

 $$
offs_m=3\times2+[0,1]
$$

 $$
\boxed{offs_m=[6,7]}
$$

 Therefore this program processes:

```
Q rows 6 and 7
```

---

 # 14\. tl.arange

 When you write:

```
tl.arange(0, BLOCK_M)
```

 and:

```
BLOCK_M = 2
```

 you get:

 $$
[0,1]
$$

 If:

```
BLOCK_M = 4
```

 you get:

 $$
[0,1,2,3]
$$

 It's similar conceptually to:

```
range(0, BLOCK_M)
```

 but it's a Triton tensor of offsets used for vectorized GPU operations.

---

 # 15\. offs\_d

 You have:

```
offs_d = tl.arange(0, D)
```

 If:

```
D = 16
```

 then:

 $$
offs_d=[0,1,2,\ldots,15]
$$

 These are the feature dimensions.

---

 # 16\. Shape of Q Block

 Suppose:

```
BLOCK_M = 2
D = 16
```

 Then:

 $$
q:[2,16]
$$

 So one program loads:

```
2 query vectors
```

 with:

```
16 features each
```

 Geometrically:

```
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

 # 17\. \[:, None\] and \[None, :\]

 This is extremely important.

 Suppose:

```
offs_m = [0,1]
```

 Its shape is:

 $$
[2]
$$

 Then:

```
offs_m[:, None]
```

 changes its shape to:

 $$
[2,1]
$$

 giving:

```
[[0],
 [1]]
```

 And:

```
offs_d[None, :]
```

 if:

```
offs_d = [0,1,2,3]
```

 becomes:

```
[[0,1,2,3]]
```

 shape:

 $$
[1,4]
$$

---

 # 18\. Why Do We Do That?

 Because broadcasting gives:

```
offs_m[:,None]      offs_d[None,:]

[0]                 [0 1 2 3]
[1]
```

 which broadcasts into:

```
[0 1 2 3]
[0 1 2 3]
```

 This creates all combinations:

 $$
(m,d)
$$

 So we get:

 $$
[BLOCK_M,D]
$$

 addresses.

---

 # 19\. What Is Q Inside Triton?

 When your kernel receives:

```
Q
```

 Triton treats `Q` as a **pointer to the beginning of Q's memory**.

 It is not the Python matrix in the normal sense.

 Think:

```
Q
│
▼
memory address of Q[0,0]
```

 Then pointer arithmetic tells Triton which elements to access.

---

 # 20\. Q.stride()

 Suppose:

```
Q.shape = (2,4)
```

 and Q is contiguous.

 Memory:

```
Q:

[ a ][ b ][ c ][ d ][ e ][ f ][ g ][ h ]
  └──── row 0 ────┘  └──── row 1 ────┘
```

 Then:

```
Q.stride()
```

 returns:

 $$
(4,1)
$$

 So:

```
Q.stride(0) = 4
Q.stride(1) = 1
```

---

 # 21\. Meaning of Stride

 `stride(0)` means:

 > How many memory elements do I move to go to the next row?

 `stride(1)` means:

 > How many memory elements do I move to go to the next column?

 For:

```
Q.shape = (2,4)
```

 we have:

 $$
\boxed{stride(0)=4}
$$

 because each row contains 4 elements.

 And:

 $$
\boxed{stride(1)=1}
$$

 because neighboring columns are next to each other.

---

 # 22\. General Contiguous 2D Tensor

 If:

 $$
Q.shape=(M,D)
$$

 then normally:

 $$
\boxed{Q.stride(0)=D}
$$

 and:

 $$
\boxed{Q.stride(1)=1}
$$

 So for:

```
Q.shape = (16,16)
```

 you get:

```
Q.stride() = (16,1)
```

---

 # 23\. PyTorch Calculates the Stride, Not Triton

 When you write:

```
Q.stride(0)
```

 **PyTorch** calculates the stride based on the actual memory layout.

 Then you pass it:

```
score_tile_kernel[grid](
    Q,
    ...
    Q.stride(0),
    Q.stride(1),
)
```

 Triton receives those values.

 So:

```
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

 # 24\. Why Pass Strides Explicitly?

 Because tensors don't always have the same memory layout.

 A transposed tensor can have different strides.

 So Triton doesn't blindly assume:

 $$
stride=(D,1)
$$

 Instead, you tell it the actual layout.

---

 # 25\. The Famous q\_ptrs

 Your code:

```
q_ptrs = (
    Q
    + offs_m[:, None] * stride_qm
    + offs_d[None, :] * stride_qd
)
```

 This is one of the most important lines.

 It calculates:

 $$
\boxed{\text{memory address of every }Q[m,d]}
$$

 Mathematically:

 $$
address(Q[m,d])
=
Q_{base}
+
m\times stride_{qm}
+
d\times stride_{qd}
$$

---

 # 26\. Numerical Example for q\_ptrs

 Suppose:

```
Q.shape = (2,4)
```

 so:

```
stride_qm = 4
stride_qd = 1
```

 and:

```
offs_m = [0,1]
offs_d = [0,1,2,3]
```

 Then:

```
offs_m[:,None]
```

 is:

```
[[0],
 [1]]
```

 and:

```
offs_d[None,:]
```

 is:

```
[[0,1,2,3]]
```

 Now:

 $$
offs_m[:,None]\times4
$$

 becomes:

```
[[0],
 [4]]
```

 And:

 $$
offs_d[None,:]\times1
$$

 becomes:

```
[[0,1,2,3]]
```

 They can be broadcast together.

 The smaller dimension gets repeated.

 Conceptually:

```
[[0],
 [4]]
```

 becomes:

```
[[0,0,0,0],
 [4,4,4,4]]
```

 while:

```
[[0,1,2,3]]
```

 becomes:

```
[[0,1,2,3],
 [0,1,2,3]]
```

 Add them:

```
[[0,1,2,3],
 [4,5,6,7]]
```

 These are exactly the memory offsets of:

```
Q[0,0] Q[0,1] Q[0,2] Q[0,3]

Q[1,0] Q[1,1] Q[1,2] Q[1,3]
```

 So:

```
Q[0,0] → offset 0
Q[0,1] → offset 1
Q[0,2] → offset 2
Q[0,3] → offset 3

Q[1,0] → offset 4
Q[1,1] → offset 5
Q[1,2] → offset 6
Q[1,3] → offset 7
```

 That's exactly:

```
[[0,1,2,3],
 [4,5,6,7]]
```

 The formula was:

 $$
address(Q[m,d])
=
Q_{base}
+
m\times stride_{qm}
+
d\times stride_{qd}
$$

 For Q\[0,0\]:

 $$
0\times4+0\times1=0
$$

 For Q\[0,1\]:

 $$
0\times4+1\times1=1
$$

 And so on.

 This gives us the memory offset for every $(row,column)$ pair.

---

 # 27\. Then tl.load

 You have:

```
q = tl.load(
    q_ptrs,
    mask=q_mask[:, None],
    other=0.0,
)
```

 This means:

 > Go to all the addresses in `q_ptrs` and load the values.

 So:

```
q_ptrs
   │
   ▼
memory
   │
   ▼
q [BLOCK_M,D]
```

---

 # 28\. Why the Mask?

 Suppose:

```
N = 5
BLOCK_M = 2
```

 and a program handles:

```
offs_m = [4,5]
```

 But row 5 doesn't exist.

 Because:

 $$
5<5
$$

 is false.

 So:

```
q_mask = offs_m < N
```

 gives:

 $$
[True,False]
$$

 Then:

```
mask=q_mask[:,None]
```

 becomes:

```
[[True],
 [False]]
```

 This prevents Triton from loading invalid Q row 5.

---

 # 29\. other=0.0

 For invalid positions:

```
other=0.0
```

 means:

 > If the mask is false, pretend the value is 0.

 So invalid Q entries become zero.

---

 # 30\. K Works Exactly the Same Way

 K block:

```
k_ptrs = (
    K
    + offs_n[:, None] * stride_kn
    + offs_d[None, :] * stride_kd
)
```

 Mathematically:

 $$
\boxed{
address(K[n,d])
=
K_{base}
+
n\cdot stride_{kn}
+
d\cdot stride_{kd}
}
$$

---

 # 31\. K Loop

 You have:

```
for start_n in range(0, N, BLOCK_N):
```

 For:

```
N = 16
BLOCK_N = 2
```

 you get:

```
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

```
offs_n = start_n + tl.arange(0,BLOCK_N)
```

 gives:

```
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

 # 32\. scores = tl.dot(q, tl.trans(k))

 Suppose:

```
q.shape = [2,16]
k.shape = [2,16]
```

 Then:

```
tl.trans(k)
```

 has:

 $$
[16,2]
$$

 Therefore:

 $$
[2,16]\times[16,2]
\rightarrow[2,2]
$$

 So:

```
scores
```

 has shape:

```
[BLOCK_M,BLOCK_N]
```

---

 # 33\. What Each Score Means

 Suppose:

```
q =
[q0]
[q1]
```

 and:

```
k =
[k0]
[k1]
```

 Then:

```
scores =
┌──────────────┐
│ q0·k0  q0·k1 │
│ q1·k0  q1·k1 │
└──────────────┘
```

 So:

 $$
scores_{ij}=q_i\cdot k_j
$$

---

 # 34\. tl.trans(k)

 If:

```
k.shape = [BLOCK_N,D]
```

 then:

```
tl.trans(k)
```

 has:

```
[D,BLOCK_N]
```

 This lets matrix multiplication work.

---

 # 35\. Output Tile

 Suppose:

```
q.shape = [2,16]
k.shape = [2,16]
```

 Then:

```
scores.shape = [2,2]
```

 That means:

```
2 Q rows × 2 K rows
```

 produce a:

```
2 × 2 score tile
```

---

 # 36\. OUT

 You allocated:

```
OUT = torch.zeros(
    (16,16),
    device="cuda",
    dtype=torch.float32,
)
```

 Therefore:

 $$
OUT.shape=[16,16]
$$

 But this **does not mean the kernel computes all $16\\times16$ values**.

 Your current:

```
grid=(1,)
```

 only computes Q block 0.

 So only:

 $$
2\times16
$$

 positions are written.

 The rest remain zero because you initialized them to zero.

---

 # 37\. Output Pointer Arithmetic

 For a contiguous $\[N,N\]$ output matrix:

 $$
OUT[m,n]
$$

 has memory offset:

 $$
mN+n
$$

 So:

```
out_ptrs = (
    OUT
    + offs_m[:,None] * N
    + offs_n[None,:]
)
```

 is the correct conceptual address calculation.

---

 # 38\. Important: Your Earlier BLOCK\_N Output Expression

 You had:

```
OUT + offs_m[:, None] * BLOCK_N + offs_n[None, :]
```

 That is only correct if the output row stride happens to equal `BLOCK_N`.

 For a full $\[16,16\]$ contiguous output:

 $$
stride_{OUT,row}=16
$$

 not 2.

 So better:

```
OUT + offs_m[:, None] * N + offs_n[None, :]
```

 or, even better in a general kernel, pass the actual output strides.

---

 # 39\. Masking K

 At the boundary:

```
k_mask = offs_n < N
```

 Suppose:

```
N=5
BLOCK_N=2
```

 and:

```
offs_n=[4,5]
```

 Then:

 $$
k_mask=[True,False]
$$

 because K row 5 doesn't exist.

---

 # 40\. Why Use -inf for Invalid Scores?

 You showed:

```
scores = tl.where(
    k_mask[None, :],
    scores,
    -float("inf"),
)
```

 This means:

```
valid K position
     ↓
keep score

invalid K position
     ↓
replace score with -∞
```

 Example:

```
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

 # 41\. Why -inf Instead of Zero?

 Because you're going to calculate:

```
tl.max(scores, axis=1)
```

 Suppose:

```
scores = [5.0, invalid]
```

 If invalid becomes zero:

 $$
\max(5,0)=5
$$

 That's okay in this case.

 But if:

```
scores = [-3.0, invalid]
```

 and invalid becomes zero:

 $$
\max(-3,0)=0
$$

 Wrong!

 The invalid position wins.

 But with:

 $$
-\infty
$$

 we get:

 $$
\max(-3,-\infty)=-3
$$

 Correct.

 Therefore:

 $$
\boxed{\text{invalid score}=-\infty}
$$

 is perfect for maximum reduction.

---

 # 42\. tl.max(scores, axis=1)

 Suppose:

```
scores =
┌──────┬──────┐
│ 1.5  │ -∞   │
├──────┼──────┤
│ 0.7  │ -∞   │
└──────┴──────┘
```

 Then:

```
tl.max(scores, axis=1)
```

 means:

 > Take the maximum horizontally across K positions.

 Result:

```
[1.5, 0.7]
```

 So:

 $$
\boxed{
m_{ij}=\max_j S_{ij}
}
$$

---

 # 43\. Why Do We Need a Running Maximum?

 Because K is processed in chunks.

 Imagine Q row 0 sees:

```
K block 0:
[1.2, 0.4]
```

 Maximum:

 $$
1.2
$$

 Then K block 1:

```
[2.8, 1.5]
```

 Maximum:

 $$
2.8
$$

 Then K block 2:

```
[0.9, 4.1]
```

 Maximum:

 $$
4.1
$$

 The maximum over **all K blocks** is:

 $$
\max(1.2,2.8,4.1)=4.1
$$

 So we maintain:

```
m_i
```

 as the running maximum.

---

 # 44\. tl.maximum

 You have:

```
m_i = tl.maximum(
    m_i,
    m_ij,
)
```

 This is elementwise.

 Suppose:

```
m_i  = [2.8, 3.0]
m_ij = [4.1, 1.5]
```

 Then:

```
m_i = [4.1, 3.0]
```

 because:

 $$
\max(2.8,4.1)=4.1
$$

 $$
\max(3.0,1.5)=3.0
$$

---

 # 45\. OUT\_MAX

 Suppose we eventually want one maximum per query row.

 If:

 $$
N=16
$$

 then:

```
OUT_MAX.shape = (16,)
```

 Conceptually:

```
OUT_MAX:

Q0 → maximum
Q1 → maximum
Q2 → maximum
...
Q15 → maximum
```

---

 # 46\. out\_ptrs = OUT\_MAX + offs\_m

 Suppose:

```
offs_m=[4,5]
```

 Then:

```
out_ptrs = OUT_MAX + offs_m
```

 means:

```
OUT_MAX[4]
OUT_MAX[5]
```

 The pointer array points to those two locations.

 Then:

```
tl.store(
    out_ptrs,
    m_i,
)
```

 could store:

```
m_i=[3.2,4.5]
```

 resulting in:

```
OUT_MAX[4] = 3.2
OUT_MAX[5] = 4.5
```

---

 # 47\. Why No \[:,None\] for OUT\_MAX?

 Because `OUT_MAX` is 1D.

```
OUT_MAX.shape = [N]
```

 while Q is 2D:

```
Q.shape = [N,D]
```

 For Q you need two coordinates:

 $$
(m,d)
$$

 For OUT\_MAX you need only:

 $$
(m)
$$

 Therefore:

```
Q + m*stride_m + d*stride_d
```

 versus:

```
OUT_MAX + m
```

---

 # 48\. Shape Summary

 This is worth memorizing.

 Suppose:

```
N = 16
D = 16
BLOCK_M = 2
BLOCK_N = 2
```

 | Object | Shape |
| --- | --- |
| `Q` | `[16,16]` |
| `K` | `[16,16]` |
| Q block `q` | `[2,16]` |
| K block `k` | `[2,16]` |
| `k.T` | `[16,2]` |
| `scores` | `[2,2]` |
| `OUT` | `[16,16]` |
| `OUT_MAX` | `[16]` |
| `offs_m` | `[2]` |
| `offs_n` | `[2]` |
| `offs_d` | `[16]` |

---

 # 49\. Important Distinction: Shape vs Stride vs Pointer

 These three concepts are easy to mix up.

 ## Shape

 Tells you:

 > How many elements are in each dimension?

 Example:

```
Q.shape = (16,16)
```

 means:

```
16 rows
16 columns
```

 ## Stride

 Tells you:

 > How far do I move in memory to move one position in a particular dimension?

 For contiguous:

```
Q.shape = (16,16)

Q.stride() = (16,1)
```

 ## Pointer

 Tells you:

 > Where in memory does the tensor begin?

 Triton sees:

```
Q
 ↓
memory address
```

 Then pointer arithmetic calculates specific elements.

---

 # 50\. The Fundamental Pointer Equation

 For a 2D tensor:

 $$
\boxed{
A[i,j]
=
A_{\text{base}}
+i\cdot stride_0
+j\cdot stride_1
}
$$

 For Q:

 $$
\boxed{
Q[m,d]
=
Q_{\text{base}}
+m\cdot stride_{qm}
+d\cdot stride_{qd}
}
$$

 For K:

 $$
\boxed{
K[n,d]
=
K_{\text{base}}
+n\cdot stride_{kn}
+d\cdot stride_{kd}
}
$$

---

 # 51\. torch.rand

 You asked about:

```
Q = torch.rand(
    (16,16),
    device="cuda",
    dtype=torch.float32,
)
```

 Yes, this is completely valid.

 It produces:

 $$
Q\in\mathbb{R}^{16\times16}
$$

 with random values approximately in:

 $$
[0,1)
$$

 For example:

```
[0.23  0.81  0.15 ...]
[0.62  0.04  0.93 ...]
...
```

 The values do **not** need to be 16.

---

 # 52\. Values vs Dimensions

 This distinction is important.

 When you say:

```
Q = torch.rand((16,16))
```

 the first `16` means:

 $$
\text{number of rows}
$$

 and second `16` means:

 $$
D=\text{features per row}
$$

 It does **not** mean the values have to be 16.

 Values can be:

```
0.1
0.5
2.7
9.2
...
```

---

 # 53\. Can D Be Smaller Than 16?

 Mathematically, absolutely.

 For example:

 $$
Q:[4,8]
$$

 and:

 $$
K:[4,8]
$$

 then:

 $$
QK^T:
[4,8]\times[8,4]
\rightarrow[4,4]
$$

 So mathematically there is nothing special about 16.

 However, **Triton's `tl.dot` has hardware/compiler constraints depending on dtype, GPU architecture, and configuration**, so a particular Triton kernel may require a convenient or padded dimension.

 That's a Triton implementation issue, not an attention mathematics issue.

---

 # 54\. Why Tiling Helps FlashAttention

 The big idea is:

 Instead of:

```
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

```
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

 The important thing is that the **entire N×N score matrix doesn't need to live in GPU global memory**.

---

 # 55\. The Current Stage You're Studying

 So far you've started moving from:

 ## Naive Attention

 $$
S=QK^T
$$

 toward:

 ## Tiled Attention

 $$
S_{\text{tile}}=Q_{\text{block}}K_{\text{block}}^T
$$

 and then toward:

 ## Online Softmax

 Instead of needing the entire row:

 $$
S_i=[s_{i0},s_{i1},...,s_{i,N-1}]
$$

 we process pieces:

 $$
S_i^{(0)},S_i^{(1)},S_i^{(2)},...
$$

 and maintain statistics such as:

 $$
\boxed{m_i=\max_j s_{ij}}
$$

 without materializing the entire row.

---

 # 56\. The Current Algorithmic Picture

 For one Q block:

```
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

 # 57\. The Most Important Mental Model

 When you look at a Triton attention kernel, keep asking four questions.

 ## Question 1: Which Q rows does this program own?

 Look at:

```
pid_m
offs_m
BLOCK_M
```

 ## Question 2: Which K rows are we currently processing?

 Look at:

```
start_n
offs_n
BLOCK_N
```

 ## Question 3: How do we find the actual data in memory?

 Look at:

```
stride
pointer arithmetic
tl.load
```

 Especially:

```
Q + offs_m[:,None] * stride_qm
  + offs_d[None,:] * stride_qd
```

 ## Question 4: What mathematical operation is happening?

 For example:

```
tl.dot(q, tl.trans(k))
```

 means:

 $$
Q_{\text{block}}K_{\text{block}}^T
$$

 and:

```
tl.max(scores, axis=1)
```

 means:

 $$
\max_{\text{K positions}} score
$$

---

 # 58\. A Complete Mental Translation of Your Kernel

 When you see:

```
pid_m = tl.program_id(0)
```

 read it as:

 > "Which Q block am I?"

 When you see:

```
offs_m = pid_m * BLOCK_M + tl.arange(0,BLOCK_M)
```

 read it as:

 > "Which Q rows belong to me?"

 When you see:

```
offs_d = tl.arange(0,D)
```

 read it as:

 > "Give me all feature dimensions."

 When you see:

```
q_ptrs = Q + ...
```

 read it as:

 > "Calculate addresses of my Q block."

 When you see:

```
q = tl.load(...)
```

 read it as:

 > "Bring my Q block into registers/local computation."

 When you see:

```
for start_n in range(...)
```

 read it as:

 > "Walk through K blocks."

 When you see:

```
k_ptrs = K + ...
```

 read it as:

 > "Calculate addresses of the current K block."

 When you see:

```
k = tl.load(...)
```

 read it as:

 > "Load this K block."

 When you see:

```
tl.dot(q, tl.trans(k))
```

 read it as:

 > "Compute this Q-block × K-block score tile."

 When you see:

```
tl.where(k_mask, scores, -∞)
```

 read it as:

 > "Make invalid K positions impossible to select as the maximum."

 When you see:

```
tl.max(scores, axis=1)
```

 read it as:

 > "Find the maximum score for each query row within this K block."

 When you see:

```
tl.maximum(m_i, m_ij)
```

 read it as:

 > "Merge this block's maximum into the running maximum."

 When you see:

```
OUT_MAX + offs_m
```

 read it as:

 > "Point to the output location corresponding to each query row."

---

 # 59\. One Final Big Picture

 For your current example:

 $$
N=16,\quad D=16
$$

 $$
BLOCK_M=2,\quad BLOCK_N=2
$$

 there are:

 $$
8\ Q\text{-blocks}
$$

 and:

 $$
8\ K\text{-blocks}
$$

 Therefore the score matrix consists of:

 $$
8\times8=64
$$

 tiles.

 With:

```
grid=(1,)
```

 you currently compute:

 $$
\boxed{1\times8=8\text{ score tiles}}
$$

 which corresponds to:

 $$
\boxed{2\times16}
$$

 elements of the full score matrix.

 If you change to:

```
grid=(8,)
```

 then you have:

 $$
8\times8=64
$$

 tiles, covering the complete:

 $$
16\times16
$$

 score matrix.

---

 # 60\. Cheat Sheet

 Keep this nearby while reading your Triton code:

```
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

---

 # The Single Most Important Distinction to Remember

 $$
\boxed{
\text{Shape}=\text{how many elements}
}
$$

 $$
\boxed{
\text{Stride}=\text{how far apart they are in memory}
}
$$

 $$
\boxed{
\text{Pointer}=\text{where the memory starts}
}
$$

 And the fundamental Triton address equation is:

 $$
\boxed{
\operatorname{address}(A[i,j])
=
A_{\text{base}}
+i\cdot stride_0
+j\cdot stride_1
}
$$

---

 # Progress

 Current concepts covered:

 - [x] Attention fundamentals
- [x] Q, K, and V
- [x] $QK^T$
- [x] Q/K tiling
- [x] Triton programs
- [x] `program_id`
- [x] `tl.arange`
- [x] Broadcasting with `[:, None]` and `[None, :]`
- [x] Tensor pointers
- [x] Strides
- [x] Pointer arithmetic
- [x] `tl.load`
- [x] Boundary masking
- [x] Score tiles
- [x] `tl.dot`
- [x] `tl.trans`
- [x] Running maximum
- [x] `tl.maximum`
- [x] `OUT_MAX`

 ## Next Step

 The natural next step is to move from the **running maximum** to the full **online softmax update**, including the running normalization term and eventually the weighted accumulation with $V$.

````