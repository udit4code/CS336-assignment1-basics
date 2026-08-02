from __future__ import annotations

from collections.abc import Iterable

from .base import BaseTokenizer, GPT2_PRETOKENIZER


class TokenizerV2(BaseTokenizer):
    """
    Version 2.

    Uses merge-rank lookup instead of replaying
    the merge list sequentially.

    Complexity:
        O(L²) per pretoken.

    where

        L = number of byte pieces.
    """

    def __init__(self, vocab, merges, special_tokens=None):
        super().__init__(vocab, merges, special_tokens)

        self.merge_rank = {
            pair: rank
            for rank, pair in enumerate(merges)
        }

    def _find_best_merge(self, pieces):
        """
        Returns

            (index, rank)

        of the adjacent pair with the
        smallest merge rank.

        Returns None if no merge exists.
        """

        best_index = None
        best_rank = float("inf")

        for i in range(len(pieces) - 1):

            pair = (pieces[i], pieces[i + 1])

            rank = self.merge_rank.get(pair)

            if rank is None:
                continue

            if rank < best_rank:
                best_rank = rank
                best_index = i

        if best_index is None:
            return None

        return best_index, best_rank

    def _merge_at(self, pieces, index):
        """
        Merge one adjacent pair.
        """

        merged = pieces[index] + pieces[index + 1]

        return (
            pieces[:index]
            + [merged]
            + pieces[index + 2 :]
        )

    def _encode_pretoken(self, pretoken):
        """
        Encode a single regex pretoken.
        """

        pieces = [
            bytes([b])
            for b in pretoken.encode("utf-8")
        ]

        while True:

            result = self._find_best_merge(pieces)

            if result is None:
                break

            index, _ = result

            pieces = self._merge_at(pieces, index)

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

            if not chunk:
                continue

            if chunk in self.special_to_id:
                ids.append(self.special_to_id[chunk])
                continue

            for pretoken in GPT2_PRETOKENIZER.findall(chunk):
                ids.extend(
                    self._encode_pretoken(pretoken)
                )

        return ids

    def decode(self, ids):

        data = b"".join(
            self.id_to_token[token_id]
            for token_id in ids
        )

        return data.decode(
            "utf-8",
            errors="replace",
        )

    def encode_iterable(self, iterable: Iterable[str]):
        for text in iterable:
            yield from self.encode(text)