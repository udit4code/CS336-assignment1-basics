from __future__ import annotations

from collections.abc import Iterable, Iterator
import json
import regex as re



GPT2_PRETOKENIZER = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class Tokenizer:

    def __init__(self, vocab, merges, special_tokens=None):
        self.id_to_token = dict(vocab)
        self.token_to_id = {v: k for k, v in vocab.items()}

        self.merges = merges

        self.special_tokens = special_tokens or []

        # Append special tokens to vocabulary if necessary
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

        # ---------- NEW ----------
        if self.special_tokens:
            ordered = sorted(
                self.special_tokens,
                key=len,
                reverse=True,
            )

            self.special_pattern = re.compile(
                "(" + "|".join(re.escape(tok) for tok in ordered) + ")"
            )
        else:
            self.special_pattern = None

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath,
                   special_tokens=None):

        with open(vocab_filepath, "r") as f:
            raw = json.load(f)

        vocab = {
            int(k): bytes.fromhex(v)
            for k, v in raw.items()
        }

        merges = []

        with open(merges_filepath) as f:
            for line in f:
                left, right = line.rstrip().split()
                merges.append(
                    (
                        bytes.fromhex(left),
                        bytes.fromhex(right),
                    )
                )

        return cls(vocab, merges, special_tokens)

    def _apply_merge(self, tokens, pair):
        """
        Merge one pair everywhere.
        """

        out = []

        i = 0

        while i < len(tokens):

            if (
                i + 1 < len(tokens)
                and
                (tokens[i], tokens[i + 1]) == pair
            ):
                out.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                out.append(tokens[i])
                i += 1

        return out

    def _encode_pretoken(self, text):
        """
        Encode one pretoken.
        """

        pieces = [bytes([b]) for b in text.encode("utf-8")]

        for pair in self.merges:
            pieces = self._apply_merge(pieces, pair)

        return [
            self.token_to_id[p]
            for p in pieces
        ]


    def encode(self, text: str) -> list[int]:
        if self.special_pattern is None:
            chunks = [text]
        else:
            chunks = self.special_pattern.split(text)

        ids = []

        for chunk in chunks:

            if chunk == "":
                continue

            # Entire chunk is a special token
            if chunk in self.special_to_id:
                ids.append(self.special_to_id[chunk])
                continue

            # Normal text
            for pretoken in GPT2_PRETOKENIZER.findall(chunk):
                ids.extend(self._encode_pretoken(pretoken))

        return ids


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:

        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids):

        data = b"".join(
            self.id_to_token[i]
            for i in ids
        )

        return data.decode(
            "utf-8",
            errors="replace",
        )