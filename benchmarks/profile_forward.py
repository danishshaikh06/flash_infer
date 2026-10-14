import argparse

import torch

from flash_infer.attention.forward_pass import flash_attention
from flash_infer.config import FlashAttentionConfig


DEVICE = "cuda"
DTYPE = torch.float32
WARMUP = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--causal",
        action="store_true",
    )
    args = parser.parse_args()

    B, H, N, D = 1, 16, 1024, 64

    config = FlashAttentionConfig(
        block_m=32,
        block_n=64,
        num_warps=4,
        num_stages=2,
    )

    torch.manual_seed(0)

    q = torch.randn(
        B, H, N, D,
        device=DEVICE,
        dtype=DTYPE,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    for _ in range(WARMUP):
        flash_attention(
            q, k, v,
            config,
            causal=args.causal,
        )

    torch.cuda.synchronize()

    flash_attention(
        q, k, v,
        config,
        causal=args.causal,
    )

    torch.cuda.synchronize()

if __name__ == "__main__":
    main()