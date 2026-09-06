from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field


type Pair[Token: Hashable] = tuple[Token, Token]
type Word[Token: Hashable] = tuple[Token, ...]
PairOrderKey = tuple[bytes, bytes]


def adjacent_pair_counts[Token: Hashable](word: Word[Token]) -> Counter[Pair[Token]]:
    """Count overlapping adjacent pairs in one segmented pretoken."""
    return Counter(zip(word, word[1:]))


def merge_pair[Token: Hashable](word: Word[Token], pair: Pair[Token], merged_token: Token) -> Word[Token]:
    """Merge all non-overlapping, left-to-right occurrences of ``pair``."""
    left, right = pair
    merged: list[Token] = []
    index = 0

    while index < len(word):
        if index + 1 < len(word) and word[index] == left and word[index + 1] == right:
            merged.append(merged_token)
            index += 2
        else:
            merged.append(word[index])
            index += 1

    return tuple(merged)


class IncrementalMergeState[Token: Hashable]:
    """Mutable corpus state with global counts and a pair-to-word inverted index."""

    def __init__(self, word_counter: Mapping[Word[Token], int]) -> None:
        self.words = dict(enumerate(word_counter))
        self.frequencies = {
            word_id: word_counter[word]
            for word_id, word in self.words.items()
        }
        self.pair_counts: Counter[Pair[Token]] = Counter()
        self.pair_to_words: dict[Pair[Token], set[int]] = defaultdict(set)

        for word_id, word in self.words.items():
            frequency = self.frequencies[word_id]
            for pair, occurrences in adjacent_pair_counts(word).items():
                self.pair_counts[pair] += occurrences * frequency
                self.pair_to_words[pair].add(word_id)

    def _remove_word_pairs(self, word_id: int, word: Word[Token], frequency: int) -> set[Pair[Token]]:
        touched: set[Pair[Token]] = set()

        for pair, occurrences in adjacent_pair_counts(word).items():
            touched.add(pair)
            new_count = self.pair_counts[pair] - occurrences * frequency
            if new_count:
                self.pair_counts[pair] = new_count
            else:
                del self.pair_counts[pair]

            word_ids = self.pair_to_words[pair]
            word_ids.discard(word_id)
            if not word_ids:
                del self.pair_to_words[pair]

        return touched

    def _add_word_pairs(self, word_id: int, word: Word[Token], frequency: int) -> set[Pair[Token]]:
        touched: set[Pair[Token]] = set()

        for pair, occurrences in adjacent_pair_counts(word).items():
            touched.add(pair)
            self.pair_counts[pair] += occurrences * frequency
            self.pair_to_words[pair].add(word_id)

        return touched

    def apply_merge(self, pair: Pair[Token], merged_token: Token) -> set[Pair[Token]]:
        """Merge only words containing ``pair`` and return every changed pair."""
        affected_word_ids = tuple(self.pair_to_words.get(pair, ()))
        touched: set[Pair[Token]] = set()

        for word_id in affected_word_ids:
            old_word = self.words[word_id]
            frequency = self.frequencies[word_id]
            touched.update(self._remove_word_pairs(word_id, old_word, frequency))

            new_word = merge_pair(old_word, pair, merged_token)
            self.words[word_id] = new_word
            touched.update(self._add_word_pairs(word_id, new_word, frequency))

        return touched


def choose_best_pair[Token: Hashable](
    pair_counts: Mapping[Pair[Token], int],
    pair_order: Callable[[Pair[Token]], PairOrderKey],
) -> Pair[Token] | None:
    if not pair_counts:
        return None
    return max(pair_counts, key=lambda pair: (pair_counts[pair], pair_order(pair)))


@dataclass(slots=True)
class _MaxHeapEntry[Token: Hashable]:
    count: int
    order_key: PairOrderKey
    pair: Pair[Token] = field(compare=False)

    def __lt__(self, other: _MaxHeapEntry[Token]) -> bool:
        # heapq is a min-heap. Reverse both comparisons so its root is the
        # highest-frequency, lexicographically greatest pair.
        return (self.count, self.order_key) > (other.count, other.order_key)


class LazyMaxPairHeap[Token: Hashable]:
    """Max-priority pair selector with lazy invalidation and optional compaction."""

    def __init__(
        self,
        pair_counts: Mapping[Pair[Token], int],
        pair_order: Callable[[Pair[Token]], PairOrderKey],
        compact_factor: int | None = None,
        compact_min_size: int = 1_024,
    ) -> None:
        self.pair_order = pair_order
        self.compact_factor = compact_factor
        self.compact_min_size = compact_min_size
        self.heap: list[_MaxHeapEntry[Token]] = []
        self.rebuild(pair_counts)

    def _entry(self, pair: Pair[Token], count: int) -> _MaxHeapEntry[Token]:
        return _MaxHeapEntry(count=count, order_key=self.pair_order(pair), pair=pair)

    def rebuild(self, pair_counts: Mapping[Pair[Token], int]) -> None:
        self.heap = [
            self._entry(pair, count)
            for pair, count in pair_counts.items()
            if count > 0
        ]
        heapq.heapify(self.heap)

    def refresh(
        self,
        touched_pairs: set[Pair[Token]],
        pair_counts: Mapping[Pair[Token], int],
    ) -> None:
        for pair in touched_pairs:
            count = pair_counts.get(pair, 0)
            if count > 0:
                heapq.heappush(self.heap, self._entry(pair, count))

        if (
            self.compact_factor is not None
            and len(self.heap) >= self.compact_min_size
            and len(self.heap) > self.compact_factor * max(1, len(pair_counts))
        ):
            self.rebuild(pair_counts)

    def pop_best(self, pair_counts: Mapping[Pair[Token], int]) -> Pair[Token] | None:
        while self.heap:
            entry = heapq.heappop(self.heap)
            if pair_counts.get(entry.pair) == entry.count:
                return entry.pair
        return None
