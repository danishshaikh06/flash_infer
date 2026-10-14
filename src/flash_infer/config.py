from dataclasses import dataclass

@dataclass(frozen=True)
class FlashAttentionConfig:
    block_m: int = 32
    block_n: int = 64
    num_warps: int = 4
    num_stages: int = 2