from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import os
import re
from .regex_utils import GPT2_PATTERN
from typing import BinaryIO, List, Tuple

from .base import BPE_Trainer


# Pre-create all byte objects once.
BYTE_TABLE = tuple(bytes([i]) for i in range(256))


def process_chunk_counter(
    input_path: str,
    start: int,
    end: int,
    special_pattern: str | None,
    special_tokens: set[str],
) -> Counter[tuple[bytes, ...]]:

    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")

    if special_pattern:
        pieces = re.split(special_pattern, text)
    else:
        pieces = (text,)

    counter = Counter()

    update = counter.update

    for piece in pieces:

        if piece in special_tokens:
            continue

        for token in GPT2_PATTERN.findall(piece):

            encoded = token.encode("utf-8")

            update([
                tuple(BYTE_TABLE[b] for b in encoded)
            ])

    return counter


Word = tuple[bytes, ...]
Pair = tuple[bytes, bytes]


class OptimisedBPETrainer(BPE_Trainer):
    def choose_best_pair(
        self,
        pair_counter: Counter[tuple[bytes, bytes]],
    ) -> tuple[bytes, bytes] | None:
        """
        Return the pair with

            1. highest frequency
            2. lexicographically greatest pair if tied

        Returns None if no pairs remain.
        """

        if not pair_counter:
            return None

        return max(
            pair_counter.items(),
            key=lambda item: (item[1], item[0]),
        )[0]
    def load_and_pretokenize_counter(
        self,
        input_path: str,
        special_tokens: list[str],
        num_processes: int | None = None,
    ) -> Counter[tuple[bytes, ...]]:
        if num_processes is None:
            num_processes = os.cpu_count()

        split_token = (
            special_tokens[0].encode("utf-8")
            if special_tokens
            else b"<|endoftext|>"
        )

        with open(input_path, "rb") as f:
            boundaries = self.find_chunk_boundaries(
                f,
                num_processes,
                split_token,
            )

        work = list(zip(boundaries[:-1], boundaries[1:]))

        special_pattern = (
            "(" + "|".join(map(re.escape, special_tokens)) + ")"
            if special_tokens
            else None
        )

        special_token_set = set(special_tokens)

        global_counter = Counter()

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(
                    process_chunk_counter,
                    input_path,
                    start,
                    end,
                    special_pattern,
                    special_token_set,
                )
                for start, end in work
            ]

            for future in futures:
                global_counter.update(future.result())

        return global_counter

    def initialize_pair_counts(
        self,
        word_counter: Counter[tuple[bytes, ...]],
    ) -> Counter[tuple[bytes, bytes]]:
        """
        Build the initial pair-frequency table.
        """

        pair_counter = Counter()

        for word, freq in word_counter.items():
            if len(word) < 2:
                continue

            for pair in zip(word, word[1:]):
                pair_counter[pair] += freq

        return pair_counter

    def merge_words(
        self,
        word_counter: Counter[Word],
        pair: Pair,
    ) -> Counter[Word]:
        """
        Merge every occurrence of `pair` in every unique word.
        """

        merged_counter = Counter()
        left, right = pair

        for word, freq in word_counter.items():
            merged = []
            i = 0

            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == left
                    and word[i + 1] == right
                ):
                    merged.append(left + right)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1

            merged_counter[tuple(merged)] += freq

        return merged_counter

    def _train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        assert self.input_path is not None
        assert self.vocab_size is not None

        vocab = self.initialize_vocab()
        merges: list[tuple[bytes, bytes]] = []

        word_counter = self.load_and_pretokenize_counter(
            self.input_path,
            self.special_tokens,
        )

        pair_counter = self.initialize_pair_counts(word_counter)

        while len(vocab) < self.vocab_size:
            best_pair = self.choose_best_pair(pair_counter)

            if best_pair is None:
                break

            vocab[len(vocab)] = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            word_counter = self.merge_words(word_counter, best_pair)
            pair_counter = self.initialize_pair_counts(word_counter)

        return vocab, merges


def train_bpe_v2(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
):
    return OptimisedBPETrainer().train(input_path, vocab_size, special_tokens)