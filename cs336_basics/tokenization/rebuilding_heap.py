from __future__ import annotations

import heapq
from .base import BaseTokenizer


class RebuildingHeapTokenizer(BaseTokenizer):
    """
    Heap-based BPE tokenizer:
        - Uses merge ranks.
        - Uses a heap to select the next merge.
        - Keeps the same BaseTokenizer API.

    NOTE:
        Since pieces are still stored in a Python list,
        the heap is rebuilt after every merge.
        :class:`LinkedHeapTokenizer` removes this limitation.
    """

    def __init__(self, vocab, merges, special_tokens=None):
        super().__init__(vocab, merges, special_tokens)

        self.merge_rank = {
            pair: rank
            for rank, pair in enumerate(merges)
        }



    def _build_heap(self, pieces):
        """
        Build a min-heap of mergeable adjacent pairs.

        Heap entries:

            (rank, index)
        """

        heap = []

        for i in range(len(pieces) - 1):

            pair = (pieces[i], pieces[i + 1])

            rank = self.merge_rank.get(pair)

            if rank is None:
                continue

            heapq.heappush(
                heap,
                (rank, i),
            )

        return heap


    def _merge_at(self, pieces, index):

        merged = pieces[index] + pieces[index + 1]

        return (
            pieces[:index]
            + [merged]
            + pieces[index + 2:]
        )

    def _encode_pretoken(self, pretoken):

        pieces = [
            bytes([b])
            for b in pretoken.encode("utf-8")
        ]

        while True:

            heap = self._build_heap(pieces)

            if not heap:
                break

            _, index = heapq.heappop(heap)

            pieces = self._merge_at(
                pieces,
                index,
            )

        return [
            self.token_to_id[p]
            for p in pieces
        ]
