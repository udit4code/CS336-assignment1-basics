from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """Return GPT-2's reversible mapping from bytes to printable characters."""
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    code_points = byte_values[:]

    offset = 0
    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            code_points.append(256 + offset)
            offset += 1

    return dict(zip(byte_values, map(chr, code_points), strict=True))


@lru_cache(maxsize=1)
def _gpt2_unicode_to_bytes() -> dict[str, int]:
    return {character: byte for byte, character in gpt2_bytes_to_unicode().items()}


def decode_gpt2_token(token: str) -> bytes:
    """Undo GPT-2's printable-Unicode representation for one token."""
    byte_decoder = _gpt2_unicode_to_bytes()
    try:
        return bytes(byte_decoder[character] for character in token)
    except KeyError as error:
        raise ValueError(f"Invalid character in GPT-2 token: {error.args[0]!r}") from error
