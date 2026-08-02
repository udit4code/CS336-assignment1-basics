from __future__ import annotations

import heapq
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Optional

from .base import BaseTokenizer, GPT2_PRETOKENIZER



# Linked List Node
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

    prev: Optional["Node"] = None
    next: Optional["Node"] = None

    # Lazy deletion.
    #
    # Heap entries may still point to this node after
    # it has been merged.
    #
    # Instead of removing heap entries, we simply mark
    # the node as dead.
    alive: bool = True


# ---------------------------------------------------------
# Heap Entry
# ---------------------------------------------------------

@dataclass(order=True, slots=True)
class HeapEntry:
    """
    Candidate merge.

    Ordering is determined ONLY by

        rank
        node_id

    The node itself is excluded from comparisons.
    """

    rank: int

    node_id: int

    left: Node = field(compare=False) 
    
    
class TokenizerV4(BaseTokenizer):
    """
    Production-style educational BPE tokenizer.

    Differences from V3

        • linked list instead of Python list
        • heap stores merge candidates
        • lazy heap invalidation
        • only local heap updates after merge

    Complexity

        Build heap

            O(L)

        Each merge

            O(log L)

        Overall

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
        self.merge_rank = {
            pair: rank
            for rank, pair in enumerate(merges)
        }
        
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

        head = Node(bytes([data[0]]))
        prev = head

        for b in data[1:]:

            node = Node(bytes([b]))

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

            ids.append(
                self.token_to_id[node.value]
            )

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

        Complexity
        ----------
        O(L)
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
                        node_id=id(node),
                        left=node,
                    ),
                )

            node = node.next

        return heap 
    
    def _valid_entry(self, entry: HeapEntry) -> bool:
        """
        Determine whether this heap entry still represents
        a valid merge candidate.

        Heap entries are never removed after insertion.

        Instead, we lazily discard stale entries when they
        reach the top of the heap.
        """

        left = entry.left

        #
        # Left node was already merged away.
        #
        if not left.alive:
            return False

        #
        # No right neighbour anymore.
        #
        if left.next is None:
            return False

        right = left.next

        #
        # Right neighbour was merged away.
        #
        if not right.alive:
            return False

        pair = (
            left.value,
            right.value,
        )

        rank = self.merge_rank.get(pair)

        #
        # Pair no longer mergeable.
        #
        if rank is None:
            return False

        #
        # Pair changed.
        #
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
                node_id=id(left),
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
        )

        #
        # Connect previous neighbour.
        #
        merged.prev = left.prev

        if merged.prev is not None:
            merged.prev.next = merged

        #
        # Connect next neighbour.
        #
        merged.next = right.next

        if merged.next is not None:
            merged.next.prev = merged

        #
        # Old nodes become dead.
        #
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

        #
        # Build linked list.
        #
        head = self._build_linked_list(pretoken)

        if head is None:
            return []

        #
        # Initial merge candidates.
        #
        heap = self._build_heap(head)

        #
        # Greedy BPE.
        #
        while True:

            entry = self._pop_valid_entry(heap)

            if entry is None:
                break

            left = entry.left

            merged = self._merge(left)

            #
            # Update head if necessary.
            #
            if merged.prev is None:
                head = merged

            #
            # Only neighbouring pairs changed.
            #
            self._push_neighbors(
                heap,
                merged,
            )

        #
        # Convert final linked list
        # into token ids.
        #
        return self._collect_ids(head)
    
    
    def encode(self, text: str) -> list[int]:
        if self.special_pattern is None:
            chunks = [text]
        else:
            chunks = self.special_pattern.split(text)

        ids: list[int] = []

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

    def decode(self, ids: list[int]) -> str:
        return (
            b"".join(
                self.id_to_token[token_id]
                for token_id in ids
            )
            .decode("utf-8", errors="replace")
        )