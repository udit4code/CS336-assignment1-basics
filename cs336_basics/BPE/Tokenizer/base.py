from __future__ import annotations

from abc import ABC, abstractmethod
import json
import regex as re



GPT2_PRETOKENIZER = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

class BaseTokenizer(ABC):

    def __init__(self, vocab, merges, special_tokens=None):
        self.id_to_token = dict(vocab)
        self.token_to_id = {v: k for k, v in vocab.items()}

        self.merges = merges
        self.special_tokens = special_tokens or []

        next_id = max(self.id_to_token.keys()) + 1

        for token in self.special_tokens:
            b = token.encode("utf-8")
            if b not in self.token_to_id:
                self.id_to_token[next_id] = b
                self.token_to_id[b] = next_id
                next_id += 1

        self.special_to_id = {
            token: self.token_to_id[token.encode("utf-8")]
            for token in self.special_tokens
        }

        if self.special_tokens:
            ordered = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern = re.compile(
                "(" + "|".join(re.escape(tok) for tok in ordered) + ")"
            )
        else:
            self.special_pattern = None

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r") as f:
            raw = json.load(f)

        vocab = {int(k): bytes.fromhex(v) for k, v in raw.items()}

        merges = []

        with open(merges_filepath) as f:
            for line in f:
                left, right = line.rstrip().split()
                merges.append((bytes.fromhex(left), bytes.fromhex(right)))

        return cls(vocab, merges, special_tokens)

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, ids):
        raise NotImplementedError
