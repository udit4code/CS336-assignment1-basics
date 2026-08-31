from __future__ import annotations

from collections.abc import Iterable

from .base import BaseTokenizer


class TokenizerV6(BaseTokenizer):
    """Native persistent-heap BPE with a bounded pretoken LRU cache."""

    def __init__(self, vocab, merges, special_tokens=None, cache_capacity: int = 65_536):
        if cache_capacity < 0:
            raise ValueError("cache_capacity must be non-negative")

        super().__init__(vocab, merges, special_tokens)
        try:
            from ._bpe_native import BPEEngineV6
        except ImportError as error:
            raise ImportError(
                "TokenizerV6 requires the native extension. Run `uv sync`, then `make native`."
            ) from error

        self._engine = BPEEngineV6(self.id_to_token, self.merges, cache_capacity)

    def _encode_pretokens(self, pretokens: Iterable[str]) -> list[int]:
        return self._engine.encode_pretokens(list(pretokens))

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        return self._engine.encode_pretokens([pretoken])

    def cache_info(self) -> dict[str, int]:
        hits, misses = self._engine.cache_stats()
        return {"hits": hits, "misses": misses, "size": self._engine.cache_size()}

    def clear_cache(self) -> None:
        self._engine.clear_cache()
