from __future__ import annotations

from .v5 import IntegerBPETrainer


class CompactingHeapBPETrainer(IntegerBPETrainer):
    """V6: V5 with adaptive rebuilding to bound stale lazy-heap entries."""

    heap_compact_factor = 4


def train_bpe_v6(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
):
    return CompactingHeapBPETrainer().train(input_path, vocab_size, special_tokens)

