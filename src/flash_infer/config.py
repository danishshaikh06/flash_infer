from dataclasses import dataclass

@dataclass(frozen=True)
class FlashAttentionConfig:
    block_m = 2
    block_n = 16