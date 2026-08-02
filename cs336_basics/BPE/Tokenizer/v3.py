from __future__ import annotations

import heapq
from collections.abc import Iterable, Iterator
from .base import BaseTokenizer, GPT2_PRETOKENIZER


class TokenizerV3(BaseTokenizer):
    """
    V3:
        - Uses merge ranks.
        - Uses a heap to select the next merge.
        - Keeps the same BaseTokenizer API.

    NOTE:
        Since pieces are still stored in a Python list,
        the heap is rebuilt after every merge.
        V4 removes this limitation.
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

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def encode(self, text):

        if self.special_pattern is None:
            chunks = [text]
        else:
            chunks = self.special_pattern.split(text)

        ids = []

        for chunk in chunks:

            if not chunk:
                continue

            if chunk in self.special_to_id:
                ids.append(
                    self.special_to_id[chunk]
                )
                continue

            for pretoken in GPT2_PRETOKENIZER.findall(chunk):
                ids.extend(
                    self._encode_pretoken(pretoken)
                )

        return ids


    def decode(self, ids):

        data = b"".join(
            self.id_to_token[i]
            for i in ids
        )

        return data.decode(
            "utf-8",
            errors="replace",
        ) 
        
    def decode(self, ids: list[int]) -> str:
        return (
            b"".join(
                self.id_to_token[token_id]
                for token_id in ids
            )
            .decode("utf-8", errors="replace")
        )