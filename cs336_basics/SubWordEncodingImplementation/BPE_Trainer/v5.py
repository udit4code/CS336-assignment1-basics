from __future__ import annotations

from collections import Counter

from ._incremental import IncrementalMergeState, LazyMaxPairHeap
from .v2 import OptimisedBPETrainer


IntPair = tuple[int, int]


class IntegerBPETrainer(OptimisedBPETrainer):
    """V5: incremental heap training with compact integer IDs in the hot loop."""

    heap_compact_factor: int | None = None

    def _make_selector(
        self,
        state: IncrementalMergeState[int],
        vocab: dict[int, bytes],
    ) -> LazyMaxPairHeap[int]:
        return LazyMaxPairHeap(
            state.pair_counts,
            pair_order=lambda pair: (vocab[pair[0]], vocab[pair[1]]),
            compact_factor=self.heap_compact_factor,
        )

    def _train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        assert self.input_path is not None
        assert self.vocab_size is not None

        vocab = self.initialize_vocab()
        byte_word_counter = self.load_and_pretokenize_counter(self.input_path, self.special_tokens)
        integer_word_counter: Counter[tuple[int, ...]] = Counter()

        for word, frequency in byte_word_counter.items():
            integer_word_counter[tuple(token[0] for token in word)] += frequency

        state = IncrementalMergeState(integer_word_counter)
        selector = self._make_selector(state, vocab)
        merges: list[tuple[bytes, bytes]] = []

        while len(vocab) < self.vocab_size:
            best_pair: IntPair | None = selector.pop_best(state.pair_counts)
            if best_pair is None:
                break

            left_bytes = vocab[best_pair[0]]
            right_bytes = vocab[best_pair[1]]
            merged_token_id = len(vocab)
            vocab[merged_token_id] = left_bytes + right_bytes
            merges.append((left_bytes, right_bytes))

            touched_pairs = state.apply_merge(best_pair, merged_token_id)
            selector.refresh(touched_pairs, state.pair_counts)

        return vocab, merges


def train_bpe_v5(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
):
    return IntegerBPETrainer().train(input_path, vocab_size, special_tokens)

