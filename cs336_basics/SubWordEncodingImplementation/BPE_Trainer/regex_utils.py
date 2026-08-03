

# GPT-2 pre-tokenization regex
import regex


GPT2_PATTERN = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d
    |[ ]?\p{L}+
    |[ ]?\p{N}+
    |[ ]?[^\s\p{L}\p{N}]+
    |\s+(?!\S)
    |\s+
    """,
    regex.VERBOSE,
)