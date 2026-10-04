import torch

from flash_infer.attention.forward_pass import (
    flash_attention,
    reference_attention,
)


def check_case(N, D,seed):

    torch.manual_seed(seed)

    Q = torch.randn(
        N,
        D,
        device="cuda",
        dtype=torch.float32,
    )

    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    actual = flash_attention(
        Q,
        K,
        V,
    )

    expected = reference_attention(
        Q,
        K,
        V,
    )

    error = (
        actual - expected
    ).abs()

    max_error = error.max().item()

    print(
        f"N={N}, D={D}, "
        f"max_error={max_error:.8f}"
    )

    print(
        "mean_error=",
        error.mean().item()
    )

    assert max_error < 2e-3


def test_forward():
    for seed in range(5):
        for N in [4, 5, 16, 17, 31, 32, 33]:
            check_case(N, 16,seed)