"""BPE training algorithms, named for their implementation strategy."""

from .base import BaseBPETrainer
from .compacting import CompactingHeapBPETrainer, train_bpe_compacting
from .heap import HeapBPETrainer, train_bpe_heap
from .incremental import IncrementalBPETrainer, train_bpe_incremental
from .integer import IntegerBPETrainer, train_bpe_integer
from .naive import NaiveBPETrainer, train_bpe_naive
from .word_frequency import WordFrequencyBPETrainer, train_bpe_word_frequency

# Use the fastest bounded-memory implementation as the public default.
train_bpe = train_bpe_compacting

__all__ = [
    "BaseBPETrainer",
    "CompactingHeapBPETrainer",
    "HeapBPETrainer",
    "IncrementalBPETrainer",
    "IntegerBPETrainer",
    "NaiveBPETrainer",
    "WordFrequencyBPETrainer",
    "train_bpe",
    "train_bpe_compacting",
    "train_bpe_heap",
    "train_bpe_incremental",
    "train_bpe_integer",
    "train_bpe_naive",
    "train_bpe_word_frequency",
]
