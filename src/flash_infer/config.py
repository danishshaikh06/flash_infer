from dataclasses import dataclass

@dataclass(frozen=True)
class FlashAttentionConfig:
    block_m: int = 8
    block_n: int = 16
    num_warps: int = 4
    num_stages: int = 3