from __future__ import annotations

from collections.abc import Iterable

from .base import BaseTokenizer


class NativeBatchTokenizer(BaseTokenizer):
    """BPE tokenizer whose batched merge loop runs in optimized C++."""

    def __init__(self, vocab, merges, special_tokens=None):
        super().__init__(vocab, merges, special_tokens)
        try:
            from ._bpe_native import BPEEngine
        except ImportError as error:
            raise ImportError(
                "NativeBatchTokenizer requires the native extension. Run `uv sync`, then `make native`."
            ) from error

        self._engine = BPEEngine(self.id_to_token, self.merges)

    def _encode_pretokens(self, pretokens: Iterable[str]) -> list[int]:
        return self._engine.encode_pretokens(list(pretokens))

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        return self._engine.encode_pretokens([pretoken])
