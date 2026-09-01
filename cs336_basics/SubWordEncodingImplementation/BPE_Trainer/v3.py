from __future__ import annotations

from ._incremental import IncrementalMergeState, choose_best_pair
from .v2 import OptimisedBPETrainer


class IncrementalBPETrainer(OptimisedBPETrainer):
    """V3: update pair statistics only in words changed by the selected merge."""

    def _train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        assert self.input_path is not None
        assert self.vocab_size is not None

        vocab = self.initialize_vocab()
        word_counter = self.load_and_pretokenize_counter(self.input_path, self.special_tokens)
        state = IncrementalMergeState(word_counter)
        merges: list[tuple[bytes, bytes]] = []

        while len(vocab) < self.vocab_size:
            best_pair = choose_best_pair(state.pair_counts, pair_order=lambda pair: pair)
            if best_pair is None:
                break

            merged_token = best_pair[0] + best_pair[1]
            vocab[len(vocab)] = merged_token
            merges.append(best_pair)
            state.apply_merge(best_pair, merged_token)

        return vocab, merges


def train_bpe_v3(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
):
    return IncrementalBPETrainer().train(input_path, vocab_size, special_tokens)

