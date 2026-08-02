from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import BinaryIO


class BPE_Trainer(ABC):
    """Abstract base class for BPE training implementations."""

    def __init__(
        self,
        input_path: str | os.PathLike | None = None,
        vocab_size: int | None = None,
        special_tokens: list[str] | None = None,
    ) -> None:
        self.input_path: str | None = os.fspath(input_path) if input_path is not None else None
        self.vocab_size: int | None = vocab_size
        self.special_tokens: list[str] = list(special_tokens or [])

    def train(
        self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str] | None = None,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        self.input_path = os.fspath(input_path)
        self.vocab_size = vocab_size
        self.special_tokens = list(special_tokens or [])
        return self._train()

    def initialize_vocab(self) -> dict[int, bytes]:
        vocab = {i: bytes([i]) for i in range(256)}
        for token in self.special_tokens:
            vocab[len(vocab)] = token.encode("utf-8")
        return vocab

    def find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)
            while True:
                mini_chunk = file.read(mini_chunk_size)

                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        return sorted(set(chunk_boundaries))

    @abstractmethod
    def _train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        raise NotImplementedError
