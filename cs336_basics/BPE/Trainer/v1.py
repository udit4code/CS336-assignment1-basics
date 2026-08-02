import os
import re
import regex
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from typing import BinaryIO, List, Tuple
from collections import Counter

from .base import BPE_Trainer
from .regex_utils import GPT2_PATTERN



def process_chunk(
    input_path: str,
    start: int,
    end: int,
    special_tokens: List[str],
) -> list[list[bytes]]:
    """
    Read one chunk from disk, split around special tokens,
    GPT-2 pre-tokenize it, and convert each pre-token into
    a list of byte tokens.
    """
    with open(input_path, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)

    text = raw.decode("utf-8", errors="ignore")

    # Split on special tokens while keeping them in the output.
    if special_tokens:
        special_pattern = re.compile(
            "(" + "|".join(map(re.escape, special_tokens)) + ")"
        )
        pieces = special_pattern.split(text)
    else:
        pieces = [text]

    words = []

    for piece in pieces:

        # Ignore special tokens completely.
        if piece in special_tokens:
            continue

        pretokens = GPT2_PATTERN.findall(piece)

        for token in pretokens:
            words.append(
                [bytes([b]) for b in token.encode("utf-8")]
            )

    return words




class NaiveBPETrainer(BPE_Trainer):
    def load_and_pretokenize(
        self,
        input_path: str,
        special_tokens: List[str],
        num_processes: int = os.cpu_count(),
    ) -> list[list[bytes]]:
        """
        Reads a corpus, splits it into independent chunks,
        pre-tokenizes using the GPT-2 regex,
        and returns a list of words where every word is a
        list of byte tokens.
        """

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

        work = [
            (start, end)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

        words = []

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(
                    process_chunk,
                    input_path,
                    start,
                    end,
                    special_tokens,
                )
                for start, end in work
            ]

            for future in futures:
                words.extend(future.result())

        return words

    def count_pairs(self, words: list[list[bytes]]) -> Counter[tuple[bytes, bytes]]:
        """
        Count the frequency of every adjacent token pair.
        """
        pair_counts = Counter()

        for word in words:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += 1

        return pair_counts

    def merge_word(
        self,
        word: List[bytes],
        pair: Tuple[bytes, bytes],
    ) -> List[bytes]:
        """
        Merge every non-overlapping occurrence of `pair` in `word`.
        """

        merged = []
        i = 0

        while i < len(word):
            if (
                i < len(word) - 1
                and word[i] == pair[0]
                and word[i + 1] == pair[1]
            ):
                merged.append(pair[0] + pair[1])
                i += 2
            else:
                merged.append(word[i])
                i += 1

        return merged

    def _train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        assert self.input_path is not None
        assert self.vocab_size is not None

        vocab = self.initialize_vocab()
        words = self.load_and_pretokenize(self.input_path, self.special_tokens)
        merges: list[tuple[bytes, bytes]] = []

        while len(vocab) < self.vocab_size:
            pair_counts = self.count_pairs(words)

            if not pair_counts:
                break

            best_pair = max(
                pair_counts.items(),
                key=lambda x: (x[1], x[0]),
            )[0]

            merges.append(best_pair)
            vocab[len(vocab)] = best_pair[0] + best_pair[1]
            words = [self.merge_word(w, best_pair) for w in words]

        return vocab, merges


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
):
    return NaiveBPETrainer().train(input_path, vocab_size, special_tokens) 