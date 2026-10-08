import torch
import torch.nn.functional as F

from flash_infer.attention.forward_pass import flash_attention
from flash_infer.config import FlashAttentionConfig


CONFIG = FlashAttentionConfig(
    block_m=16,
    block_n=64,
)


def check_case(B, H, N, D, causal):
    torch.manual_seed(0)

    q = torch.randn(
        B,
        H,
        N,
        D,
        device="cuda",
        dtype=torch.float32,
    )

    k = torch.randn_like(q)
    v = torch.randn_like(q)

    actual = flash_attention(
        q,
        k,
        v,
        CONFIG,
        causal=causal,
    )

    expected = F.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=causal,
    )

    max_error = (
        actual - expected
    ).abs().max().item()

    mean_error = (
        actual - expected
    ).abs().mean().item()

    print(
        f"B={B}, "
        f"H={H}, "
        f"N={N}, "
        f"D={D}, "
        f"causal={causal}, "
        f"max_error={max_error:.8f}, "
        f"mean_error={mean_error:.8f}"
    )

    assert actual.shape == expected.shape

    torch.testing.assert_close(
        actual,
        expected,
        atol=2e-3,
        rtol=2e-3,
    )


def test_multi_head_non_causal():
    for B, H, N, D in [
        (1, 1, 4, 16),
        (1, 2, 17, 16),
        (2, 2, 33, 64),
        (2, 4, 64, 64),
    ]:
        check_case(
            B,
            H,
            N,
            D,
            causal=False,
        )


def test_multi_head_causal():
    for B, H, N, D in [
        (1, 1, 4, 16),
        (1, 2, 17, 16),
        (2, 2, 33, 64),
        (2, 4, 64, 64),
    ]:
        check_case(
            B,
            H,
            N,
            D,
            causal=True,
        )