"""Type declarations for the optional pybind11 BPE extension."""

class BPEEngine:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
    ) -> None: ...

    def encode_pretokens(self, pretokens: list[str]) -> list[int]: ...


class CachedBPEEngine:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        cache_capacity: int = ...,
    ) -> None: ...

    def encode_pretokens(self, pretokens: list[str]) -> list[int]: ...
    def cache_stats(self) -> tuple[int, int]: ...
    def cache_size(self) -> int: ...
    def clear_cache(self) -> None: ...
