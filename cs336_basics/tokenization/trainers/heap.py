from __future__ import annotations

from ._state import IncrementalMergeState, LazyMaxPairHeap
from .word_frequency import WordFrequencyBPETrainer


class HeapBPETrainer(WordFrequencyBPETrainer):
    """Incremental trainer with a lazily invalidated max-priority heap."""

    def _train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        assert self.input_path is not None
        assert self.vocab_size is not None

        vocab = self.initialize_vocab()
        word_counter = self.load_and_pretokenize_counter(self.input_path, self.special_tokens)
        state = IncrementalMergeState(word_counter)
        selector = LazyMaxPairHeap(state.pair_counts, pair_order=lambda pair: pair)
        merges: list[tuple[bytes, bytes]] = []

        while len(vocab) < self.vocab_size:
            best_pair = selector.pop_best(state.pair_counts)
            if best_pair is None:
                break

            merged_token = best_pair[0] + best_pair[1]
            vocab[len(vocab)] = merged_token
            merges.append(best_pair)
            touched_pairs = state.apply_merge(best_pair, merged_token)
            selector.refresh(touched_pairs, state.pair_counts)

        return vocab, merges


def train_bpe_heap(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
):
    return HeapBPETrainer().train(input_path, vocab_size, special_tokens)
