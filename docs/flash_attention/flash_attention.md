# FlashAttention — Detailed Notes

grid = (1,)

1 PROGRAM

    Q block 0
          │
          ├── K block 0
          ├── K block 1
          ├── K block 2
          ├── K block 3
          ├── K block 4
          ├── K block 5
          ├── K block 6
          └── K block 7

grid = (8,)

8 PROGRAMS

    ├── Program 0 → Q0 → K0 K1 K2 ... K7

    ├── Program 1 → Q1 → K0 K1 K2 ... K7

    ├── Program 2 → Q2 → K0 K1 K2 ... K7

    ├── Program 3 → Q3 → K0 K1 K2 ... K7

    ├── Program 4 → Q4 → K0 K1 K2 ... K7

    ├── Program 5 → Q5 → K0 K1 K2 ... K7

    ├── Program 6 → Q6 → K0 K1 K2 ... K7

    └── Program 7 → Q7 → K0 K1 K2 ... K7

Note: One program is responsible for one Q tile, and it loops over all 8 K tiles.

---

This code is implementing **attention with online softmax**, so the easiest way to understand it is to use a _very tiny_ example and follow one query row through two key blocks.

Let's ignore batch/head indexing and use:

$$
M=2,\quad N=4,\quad D=2,\quad BLOCK_N=2
$$

So we have **2 queries**, **4 keys**, and each vector has 2 features.

### Our Q, K, V

Let's choose:

$$
Q=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
$$

$$
K=
\begin{bmatrix}
1&0\\
0&1\\
1&1\\
2&0
\end{bmatrix}
$$

$$
V=
\begin{bmatrix}
10&0\\
0&20\\
10&10\\
20&0
\end{bmatrix}
$$

We'll calculate attention for **Query 0** first.

---

# 1\. First iteration: `start_n = 0`

loop:

```
for start_n in tl.range(0, N, BLOCK_N):
```

With:

$$
N=4,\quad BLOCK_N=2
$$

we get:

```
iteration 1 → start_n = 0
iteration 2 → start_n = 2
```

So we're processing K/V in two chunks:

```
K:

row 0 → [1, 0]  ┐
row 1 → [0, 1]  ┘ first block

row 2 → [1, 1]  ┐
row 3 → [2, 0]  ┘ second block
```

---

# 2\. First block: `current_n`

```
offs_n = [0, 1]
```

Since:

```
start_n = 0
```

we get:

$$
current_n = start_n + offs_n
$$

$$
=[0,1]
$$

So we're loading:

```
K[0] = [1,0]
K[1] = [0,1]

V[0] = [10,0]
V[1] = [0,20]
```

---

# 3\. `QKᵀ`

Our query 0 is:

$$
Q_0=[1,0]
$$

The two keys are:

$$
K_0=[1,0]
$$

$$
K_1=[0,1]
$$

Dot products:

```
QK^T = [1,0] . [1,0] = 1
```

and:

```
QK^T = [1,0] . [1,0] = 1
```

So:

$$
scores=[1,0]
$$

code does exactly this:

```
scores = tl.dot(q, tl.trans(k))
```

Conceptually:

```
Q                         Kᵀ

[1 0]       ×       [1 0]
                    [0 1]

                 ↓

              [1 0]
```

---

# 4\. Causal mask

Suppose we're calculating attention for query position 0.

The causal rule is:

$$
key\_idx \le query\_idx
$$

For query 0:

```
query_idx = 0

key indices = [0, 1]
```

Check:

```
0 <= 0 → TRUE
1 <= 0 → FALSE
```

Therefore:

```
score mask:

[True, False]
```

code changes:

```
[1, 0]
```

into:

```
[1, -∞]
```

because:

```
scores = tl.where(
    score_mask,
    scores,
    -float("inf"),
)
```

This means:

> Query 0 is allowed to look at key 0, but NOT key 1.

---

# 5\. Online softmax starts

Now:

```
block_max = tl.max(scores, axis=1)
```

We have:

$$
[1,-\infty]
$$

so:

$$
block\_max=1
$$

Initially assume:

$$
m_i=-\infty
$$

and:

$$
l_i=0
$$

and:

$$
acc=[0,0]
$$

---

# 6\. Calculate `new_m`

```
new_m = tl.maximum(safe_m_i, block_max)
```

So:

$$
new_m=\max(-\infty,1)=1
$$

This is simply:

> **What is the largest score we've seen so far?**

Answer:

$$
m=1
$$

---

# 7\. Calculate `p`

code:

```
p = tl.exp(scores - new_m[:, None])
```

We have:

$$
scores=[1,-\infty]
$$

and:

$$
new_m=1
$$

Therefore:

$$
p=
[
e^{1-1},
e^{-\infty-1}
]
$$

$$
p=[1,0]
$$

So our softmax numerator is:

```
[1, 0]
```

---

# 8\. Calculate `l_new`

```
l_new = alpha * l_i + tl.sum(p, axis=1)
```

Initially:

$$
l_i=0
$$

and:

$$
\alpha=1
$$

Therefore:

$$
l_{new}=1(0)+(1+0)
$$

$$
\boxed{l_{new}=1}
$$

---

# 9\. Calculate `P @ V`

We have:

$$
P=[1,0]
$$

and:

$$
V=
\begin{bmatrix}
10&0\\
0&20
\end{bmatrix}
$$

Therefore:

```
pv = [1,0] dot [10,0,
                0,20] 
```

$$
=[10,0]
$$

So:

$$
acc=[10,0]
$$

At the end of iteration 1:

```
m_i = 1
l_i = 1
acc = [10, 0]
```

---

# 10\. Second iteration

Now:

```
start_n = 2
```

Therefore:

$$
current_n=[2,3]
$$

We're loading:

$$
K_2=[1,1]
$$

$$
K_3=[2,0]
$$

and:

$$
V_2=[10,10]
$$

$$
V_3=[20,0]
$$

---

# 11\. Calculate QKᵀ again

Query 0:

$$
Q_0=[1,0]
$$

Dot with $K_2$:

$$
[1,0]\cdot[1,1]=1
$$

Dot with $K_3$:

$$
[1,0]\cdot[2,0]=2
$$

Therefore:

$$
scores=[1,2]
$$

Because we're still doing causal attention for query 0:

```
query = 0

key 2 → 2 <= 0 → FALSE
key 3 → 3 <= 0 → FALSE
```

Both are masked:

$$
scores=[-\infty,-\infty]
$$

Therefore:

$$
p=[0,0]
$$

Nothing gets added to the accumulator.

So final:

$$
acc=[10,0]
$$

$$
l=1
$$

---

# 12\. Final normalization

code:

```
acc = acc / safe_l[:, None]
```

We have:

$$
acc=[10,0]
$$

and:

$$
l=1
$$

Therefore:

$$
O_0=
\frac{[10,0]}{1}
$$

$$
\boxed{O_0=[10,0]}
$$

So Query 0 attends entirely to Value 0.

---

# Why do we need the "online" part?

Now here's the really important part.

Normally, we'd calculate the entire attention matrix:

$$
QK^T
$$

all at once.

But your kernel processes:

```
K/V block 1
     ↓
update
     ↓
K/V block 2
     ↓
update
```

It doesn't need to keep the entire $N$-sized attention matrix in memory.

That's why you have these variables:

```
m_i
l_i
acc
```

Think of them as:

```
m_i   = biggest score seen so far
l_i   = sum of exp(score - max)
acc   = sum of weighted V
```

---

# The key part: `alpha`

This is the clever bit:

```
alpha = exp(old_max - new_max)
```

Why?

Imagine first block gives:

$$
scores=[1,2]
$$

so:

$$
old\_max=2
$$

and:

$$
l_{old}=e^{-1}+1
$$

Now the second block contains:

$$
scores=[5,3]
$$

The new maximum is:

$$
new\_max=5
$$

We can't simply add the new exponentials because the old values were normalized around max $2$, while the new values are normalized around max $5$.

So we correct the old contribution:

$$
\alpha=e^{2-5}=e^{-3}
$$

Then:

```
l_new = alpha * l_old + exp(score-new_max)
```

This is what allows the kernel to process attention **block by block** while still getting the same result as a normal softmax.

---

code is basically doing this:

```
For every block of K/V:

    1. Load K and V
           ↓
    2. Calculate Q × Kᵀ
           ↓
    3. Apply causal mask
           ↓
    4. Find maximum score
           ↓
    5. Update running maximum
           ↓
    6. Calculate exp(score - max)
           ↓
    7. Update running softmax denominator
           ↓
    8. Calculate P × V
           ↓
    9. Update output accumulator

After ALL blocks:

    output = accumulator / softmax_denominator
```

The **three most important variables** to understand are:

$$
\boxed{m_i=\text{running maximum}}
$$

$$
\boxed{l_i=\text{running sum of exponentials}}
$$

$$
\boxed{acc=\text{running weighted sum of V}}
$$

And finally:

$$
\boxed{O=\frac{acc}{l_i}}
$$

That's the heart of the entire online-softmax attention kernel.