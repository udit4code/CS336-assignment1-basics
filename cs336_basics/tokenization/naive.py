from __future__ import annotations

from .base import BaseTokenizer



class NaiveTokenizer(BaseTokenizer):

    def _apply_merge(
        self,
        tokens: list[bytes],
        pair: tuple[bytes, bytes],
    ) -> list[bytes]:
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

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        """
        Encode one pretoken.
        """

        pieces = [bytes([b]) for b in pretoken.encode("utf-8")]

        for pair in self.merges:
            pieces = self._apply_merge(pieces, pair)

        return [self.token_to_id[p] for p in pieces]
