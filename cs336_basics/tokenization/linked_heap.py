from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from .base import BaseTokenizer


@dataclass(slots=True)
class Node:
    """
    One token piece in the current BPE segmentation.

    Initially every node contains one UTF-8 byte.

    During encoding, nodes are repeatedly merged into
    larger byte strings.

    Example

        b"t" <-> b"h" <-> b"e"

    becomes

        b"th" <-> b"e"

    then

        b"the"
    """

    value: bytes
    position: int
    prev: Node | None = None
    next: Node | None = None

    # Old heap candidates may still reference a merged node. Marking it dead
    # supports lazy invalidation without searching the heap for stale entries.
    alive: bool = True


@dataclass(order=True, slots=True)
class HeapEntry:
    """
    Candidate merge.

    Ordering is determined ONLY by

        rank
        position

    The node itself is excluded from comparisons.
    """

    rank: int

    position: int

    left: Node = field(compare=False)


class LinkedHeapTokenizer(BaseTokenizer):
    """
    Production-style educational BPE tokenizer.

    Compared with :class:`RebuildingHeapTokenizer`

        • linked list instead of Python list
        • heap stores merge candidates
        • lazy heap invalidation
        • only local heap updates after merge

    Complexity

        Build heap with repeated heappush

            O(L log L) worst case

        Each merge

            O(log L)

        Overall (at most L-1 merges and O(L) total candidate insertions)

            O(L log L)
    """

    def __init__(
        self,
        vocab,
        merges,
        special_tokens=None,
    ):
        super().__init__(
            vocab,
            merges,
            special_tokens,
        )

        # Merge rank lookup.
        # Smaller rank => earlier merge learned during training.
        self.merge_rank = {pair: rank for rank, pair in enumerate(merges)}

    def _build_linked_list(self, pretoken: str) -> Node | None:
        """
        Convert a pretoken into a doubly linked list.

        Example

            "cat"

        becomes

            [b'c'] <-> [b'a'] <-> [b't']

        Returns
        -------
        Node | None
            Head of the linked list.
        """

        data = pretoken.encode("utf-8")

        if not data:
            return None

        head = Node(bytes([data[0]]), position=0)
        prev = head

        for position, b in enumerate(data[1:], start=1):
            node = Node(bytes([b]), position=position)

            prev.next = node
            node.prev = prev

            prev = node

        return head

    def _collect_ids(self, head: Node | None) -> list[int]:
        """
        Traverse the linked list and convert each node
        into a vocabulary id.
        """

        ids = []

        node = head

        while node is not None:
            ids.append(self.token_to_id[node.value])

            node = node.next

        return ids

    def _iter_nodes(self, head: Node):
        """
        Iterate over every node in the linked list.
        """

        node = head

        while node is not None:
            yield node
            node = node.next

    def _build_heap(self, head: Node | None) -> list[HeapEntry]:
        """
        Build the initial heap.

        Every adjacent mergeable pair becomes one HeapEntry.

        Complexity is O(L log L) here because candidates are inserted with
        heappush. Building a list followed by heapify would be O(L).
        """

        heap: list[HeapEntry] = []

        if head is None:
            return heap

        node = head

        while node.next is not None:
            pair = (node.value, node.next.value)

            rank = self.merge_rank.get(pair)

            if rank is not None:
                heapq.heappush(
                    heap,
                    HeapEntry(
                        rank=rank,
                        position=node.position,
                        left=node,
                    ),
                )

            node = node.next

        return heap

    def _valid_entry(self, entry: HeapEntry) -> bool:
        """
        Determine whether this heap entry still represents
        a valid merge candidate.

        Entries are not eagerly removed when a merge invalidates them. They
        are discarded later when popped from the heap.
        """

        left = entry.left

        # The left node may already have been merged away.
        if not left.alive:
            return False

        # A tail node cannot begin a pair.
        if left.next is None:
            return False

        right = left.next

        # The right node may already have been merged away.
        if not right.alive:
            return False

        pair = (
            left.value,
            right.value,
        )

        rank = self.merge_rank.get(pair)

        # The current neighboring values may not form a learned merge.
        if rank is None:
            return False

        # A changed rank means this entry describes an older neighboring pair.
        return rank == entry.rank

    def _push_pair(
        self,
        heap: list[HeapEntry],
        left: Node | None,
    ):
        """
        Push one adjacent pair into the heap.

        If the pair is not mergeable,
        nothing is inserted.
        """

        if left is None:
            return

        if left.next is None:
            return

        pair = (
            left.value,
            left.next.value,
        )

        rank = self.merge_rank.get(pair)

        if rank is None:
            return

        heapq.heappush(
            heap,
            HeapEntry(
                rank=rank,
                position=left.position,
                left=left,
            ),
        )

    def _merge(self, left: Node) -> Node:
        """
        Merge

            left
            left.next

        into a brand new node.

        Returns
        -------
        Node
            Newly created merged node.
        """

        right = left.next

        if right is None:
            raise RuntimeError("Cannot merge last node.")

        merged = Node(
            value=left.value + right.value,
            position=left.position,
        )

        # Splice the new node between the surviving neighbors.
        merged.prev = left.prev

        if merged.prev is not None:
            merged.prev.next = merged

        merged.next = right.next

        if merged.next is not None:
            merged.next.prev = merged

        # Existing heap entries that reference either old node become stale.
        left.alive = False
        right.alive = False

        return merged

    def _push_neighbors(
        self,
        heap: list[HeapEntry],
        merged: Node,
    ):
        """
        Push newly created neighbouring pairs.

        Only two pairs can appear after one merge.

            prev <-> merged

            merged <-> next
        """

        self._push_pair(
            heap,
            merged.prev,
        )

        self._push_pair(
            heap,
            merged,
        )

    def _pop_valid_entry(
        self,
        heap: list[HeapEntry],
    ) -> HeapEntry | None:
        """
        Pop the first valid merge candidate.

        Stale heap entries are discarded lazily.
        """

        while heap:
            entry = heapq.heappop(heap)

            if self._valid_entry(entry):
                return entry

        return None

    def _encode_pretoken(
        self,
        pretoken: str,
    ) -> list[int]:
        """
        Encode one regex pretoken using
        heap-based BPE.

        Algorithm

            bytes
                ↓

            linked list
                ↓

            initial heap
                ↓

            repeatedly

                pop best merge

                merge nodes

                push neighbours

            ↓

            vocabulary ids
        """

        # Start with one node per UTF-8 byte.
        head = self._build_linked_list(pretoken)

        if head is None:
            return []

        # Seed the heap with currently adjacent learned pairs.
        heap = self._build_heap(head)

        # Greedily apply the lowest-rank valid merge, breaking equal-rank
        # occurrences by their original left position.
        while True:
            entry = self._pop_valid_entry(heap)

            if entry is None:
                break

            left = entry.left

            merged = self._merge(left)

            # A merge at the first node replaces the list head.
            if merged.prev is None:
                head = merged

            # Only pairs touching the merged node can be newly created.
            self._push_neighbors(
                heap,
                merged,
            )

        # Convert the final segmentation to vocabulary IDs.
        return self._collect_ids(head)
