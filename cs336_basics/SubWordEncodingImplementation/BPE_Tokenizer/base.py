from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
import json
from pathlib import Path

import regex as re

from .byte_mapping import decode_gpt2_token

GPT2_PRETOKENIZER = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

class BaseTokenizer(ABC):
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.id_to_token = dict(vocab)
        self.token_to_id = {v: k for k, v in vocab.items()}

        self.merges = merges
        self.special_tokens = special_tokens or []

        next_id = max(self.id_to_token, default=-1) + 1

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
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> BaseTokenizer:
        """Load GPT-2-formatted vocabulary and merge files."""
        with open(vocab_filepath, encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, dict) or not all(isinstance(value, int) for value in raw.values()):
            raise ValueError("Expected a GPT-2 vocabulary mapping encoded tokens to integer IDs")

        vocab = {token_id: decode_gpt2_token(token) for token, token_id in raw.items()}

        merges: list[tuple[bytes, bytes]] = []

        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                cleaned_line = line.rstrip()
                if not cleaned_line or cleaned_line.startswith("#"):
                    continue
                left, right = cleaned_line.split()
                merges.append((decode_gpt2_token(left), decode_gpt2_token(right)))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        chunks = [text] if self.special_pattern is None else self.special_pattern.split(text)

        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_to_id:
                ids.append(self.special_to_id[chunk])
                continue
            for match in GPT2_PRETOKENIZER.finditer(chunk):
                ids.extend(self._encode_pretoken(match.group()))

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode each iterable element independently and yield its token IDs."""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: Sequence[int]) -> str:
        data = b"".join(self.id_to_token[token_id] for token_id in ids)
        return data.decode("utf-8", errors="replace")

    @abstractmethod
    def _encode_pretoken(self, pretoken: str) -> list[int]:
        raise NotImplementedError
