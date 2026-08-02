from __future__ import annotations

from collections.abc import Iterable, Iterator
import regex as re

from .base import BaseTokenizer, GPT2_PRETOKENIZER



class NaiveTokenizer(BaseTokenizer):

    def _apply_merge(self, tokens, pair):
        """
        Merge one pair everywhere.
        """

        out = []
        i = 0

        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
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

        return [self.token_to_id[p] for p in pieces]

    def encode(self, text: str) -> list[int]:
        if self.special_pattern is None:
            chunks = [text]
        else:
            chunks = self.special_pattern.split(text)

        ids = []

        for chunk in chunks:
            if chunk == "":
                continue

            if chunk in self.special_to_id:
                ids.append(self.special_to_id[chunk])
                continue

            for pretoken in GPT2_PRETOKENIZER.findall(chunk):
                ids.extend(self._encode_pretoken(pretoken))

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids):
        data = b"".join(self.id_to_token[i] for i in ids)
        return data.decode("utf-8", errors="replace")
