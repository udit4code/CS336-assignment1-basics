from __future__ import annotations

from dataclasses import dataclass

import tiktoken


@dataclass(frozen=True, slots=True)
class TokenizedSentence:
    """A sentence and its exact GPT-2/tiktoken token representation."""

    text: str
    token_ids: tuple[int, ...]
    token_bytes: tuple[bytes, ...]
    labels: tuple[str, ...]


def _label_token(token_bytes: bytes) -> str:
    """Make a compact, visible label without changing token bytes."""
    label = token_bytes.decode("utf-8", errors="replace")
    label = label.replace(" ", "␠").replace("\n", "↵").replace("\t", "⇥")
    label = label.replace("\r", "␍")
    return label or "∅"


def tokenize_sentence(text: str, encoding_name: str = "gpt2") -> TokenizedSentence:
    """Tokenize *text* and retain the bytes used by every token.

    Attention is displayed at token granularity. In particular, a displayed
    label may be a subword or include a leading space; it is not assumed to be
    a linguistic word.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    if not text:
        raise ValueError("text must not be empty")

    encoding = tiktoken.get_encoding(encoding_name)
    ids = tuple(encoding.encode(text))
    token_bytes = tuple(encoding.decode_single_token_bytes(token_id) for token_id in ids)

    if b"".join(token_bytes) != text.encode("utf-8"):
        raise RuntimeError("tiktoken token bytes did not reconstruct the input UTF-8 bytes")

    return TokenizedSentence(
        text=text,
        token_ids=ids,
        token_bytes=token_bytes,
        labels=tuple(_label_token(token) for token in token_bytes),
    )
