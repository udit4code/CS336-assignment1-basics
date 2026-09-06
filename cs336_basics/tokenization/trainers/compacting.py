from __future__ import annotations

from .integer import IntegerBPETrainer


class CompactingHeapBPETrainer(IntegerBPETrainer):
    """Integer trainer with adaptive rebuilding to bound stale heap entries."""

    heap_compact_factor = 4


def train_bpe_compacting(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
):
    return CompactingHeapBPETrainer().train(input_path, vocab_size, special_tokens)
