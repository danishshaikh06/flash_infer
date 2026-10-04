import torch 
from flash_infer.attention.forward_pass import flash_attention_forward_kernel
import triton 
import math 

def flash_attention(q,k,v):
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

    BLOCK_M = 2
    BLOCK_N = 16

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

def reference_attention(q,k,v):

    D = q.shape[1]

    scores = (
        q @ k.T
    )

    scores*= 1 / math.sqrt(D)

    probs = torch.softmax(
        scores,
        dim=-1,
    )

    output = probs @ v

    return output



def demo():

    torch.manual_seed(0)

    N = 4
    D = 16

    Q = torch.rand(
    N,
    D,
    device="cuda",
    dtype=torch.float32,
    )

    K = torch.rand(
        N,
        D,
        device="cuda",
        dtype=torch.float32,
    )

    V = torch.rand(
            N,
            D,
            device="cuda",
            dtype=torch.float32,
        )

    output = flash_attention(
        Q,
        K,
        V,
    )

    reference = reference_attention(
        Q,
        K,
        V,
    )

    print("Our Triton output:")
    print(output)

    print("\nPyTorch reference:")
    print(reference)

    max_error = (
        output - reference
    ).abs().max()

    print(
        "\nMax absolute error:",
        max_error.item(),
    )

if __name__ == "__main__":
    demo()