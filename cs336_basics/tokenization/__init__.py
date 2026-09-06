"""Byte-pair encoding tokenizers and training implementations."""

from .base import BaseTokenizer
from .cached import CachedNativeTokenizer
from .linked_heap import LinkedHeapTokenizer
from .naive import NaiveTokenizer
from .native import NativeBatchTokenizer
from .rank_scan import RankScanTokenizer
from .rebuilding_heap import RebuildingHeapTokenizer

# The pure-Python implementation is the portable production default.
Tokenizer = LinkedHeapTokenizer

__all__ = [
    "BaseTokenizer",
    "CachedNativeTokenizer",
    "LinkedHeapTokenizer",
    "NaiveTokenizer",
    "NativeBatchTokenizer",
    "RankScanTokenizer",
    "RebuildingHeapTokenizer",
    "Tokenizer",
]
