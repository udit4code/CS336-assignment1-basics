from __future__ import annotations

from .base import BaseTokenizer


class RankScanTokenizer(BaseTokenizer):
    """
    BPE tokenizer that repeatedly scans for the best-ranked merge.

    At each step, scan current adjacent pairs and apply the occurrence whose
    learned merge rank is smallest. Equal ranks resolve to the leftmost pair.

    Complexity:
        O(L²) per pretoken.

    where

        L = initial number of UTF-8 bytes. There are at most L-1 merges, and
        each scan/list reconstruction is O(L).
    """

    def __init__(self, vocab, merges, special_tokens=None):
        super().__init__(vocab, merges, special_tokens)

        self.merge_rank = {pair: rank for rank, pair in enumerate(merges)}

    def _find_best_merge(self, pieces):
        """
        Return ``(left_index, rank)`` for the best adjacent learned pair.

        Iteration order supplies the leftmost tie-break for equal ranks.
        Return ``None`` if no adjacent pair is mergeable.
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
        """Replace the pair beginning at ``index`` with its concatenation."""

        merged = pieces[index] + pieces[index + 1]

        return pieces[:index] + [merged] + pieces[index + 2 :]

    def _encode_pretoken(self, pretoken):
        """Encode one regex pretoken, initially segmented into UTF-8 bytes."""

        pieces = [bytes([b]) for b in pretoken.encode("utf-8")]

        while True:
            result = self._find_best_merge(pieces)

            if result is None:
                break

            index, _ = result

            pieces = self._merge_at(pieces, index)

        return [self.token_to_id[p] for p in pieces]
