# Understanding Unicode and UTF-8 Like You're Five

## A Deep Dive into Section 2.2 of CS336 Assignment 1

> **Goal:** By the end of this document, you should understand **why
> every modern LLM (GPT, Llama, Claude, Gemini, DeepSeek, etc.) first
> converts text into UTF-8 bytes before tokenization**, and why Stanford
> starts the assignment with Unicode.

This explanation follows the concepts introduced in Section 2.2 of the
CS336 assignment and expands them with additional intuition and
examples.

------------------------------------------------------------------------

# Table of Contents

1.  The Fundamental Problem
2.  What is a Computer?
3.  What is Memory?
4.  What is a Byte?
5.  What is a Character?
6.  Why We Need Unicode
7.  Unicode Code Points
8.  Why Unicode Isn't Enough
9.  What is an Encoding?
10. UTF-8
11. UTF-16
12. UTF-32
13. Why UTF-8 Won
14. Why LLMs Love UTF-8
15. The Complete Pipeline
16. Mental Models to Remember

------------------------------------------------------------------------

# 1. The Fundamental Problem

Imagine I hand you a book.

Inside it are words like:

``` text
Cat
Dog
こんにちは
牛
😀
```

Humans instantly understand these symbols.

A computer does **not**.

A computer never sees letters---it only understands electrical states
that we represent as **0s and 1s**.

------------------------------------------------------------------------

# 2. What is Memory?

Imagine your computer has billions of tiny boxes.

Each tiny box stores either:

``` text
0
```

or

``` text
1
```

Eight such boxes together form **one byte**.

``` text
+---+---+---+---+---+---+---+---+
|1|0|1|1|0|0|1|0|
+---+---+---+---+---+---+---+---+
```

One byte can represent **256 different values (0--255).**

------------------------------------------------------------------------

# 3. But Humans Don't Write Numbers

If you type:

``` text
A
```

the computer stores a number.

For example:

``` python
ord("A")
# 65
```

Likewise:

``` python
ord("牛")
# 29275
```

Unicode is simply a giant dictionary mapping:

``` text
Character  <----->  Integer
```

------------------------------------------------------------------------

# 4. Why Unicode Isn't Enough

Memory can only store bytes.

A byte stores numbers from:

``` text
0 ... 255
```

But:

``` text
牛 -> 29275
```

does not fit in a single byte.

So we need another layer.

------------------------------------------------------------------------

# 5. Encoding

An **encoding** converts Unicode numbers into bytes.

Think of Unicode as a dictionary and encoding as packing instructions.

``` text
Character
    ↓
Unicode Number
    ↓
Encoding
    ↓
Bytes
```

------------------------------------------------------------------------

# 6. UTF-8

UTF-8 is a **variable-length encoding**.

Typical sizes are:

  Character     Bytes
  ----------- -------
  A                 1
  h                 1
  こ                3
  😀                4

ASCII characters remain compact, while larger Unicode code points use
more bytes.

------------------------------------------------------------------------

# 7. UTF-16

UTF-16 generally uses 2 bytes for many common characters and more for
some others.

ASCII text therefore occupies more memory than in UTF-8.

------------------------------------------------------------------------

# 8. UTF-32

UTF-32 always uses **4 bytes** per character.

Even:

``` text
A
```

requires four bytes.

It is simple but wastes memory.

------------------------------------------------------------------------

# 9. Why UTF-8 Won

Most web pages are dominated by:

-   English letters
-   Numbers
-   Spaces
-   Punctuation

UTF-8 stores these in a single byte, making it extremely space
efficient.

------------------------------------------------------------------------

# 10. Why LLMs Love UTF-8

Instead of beginning with a vocabulary of over 150,000 Unicode
characters, LLM tokenizers begin with:

``` text
256 byte values
```

Every piece of text can be represented using these bytes.

Byte Pair Encoding (BPE) then learns larger and more meaningful tokens.

------------------------------------------------------------------------

# 11. Complete Pipeline

``` text
Human Text
      │
      ▼
Unicode Characters
      │
      ▼
Unicode Code Points
      │
      ▼
UTF-8 Encoding
      │
      ▼
Bytes
      │
      ▼
BPE Tokenizer
      │
      ▼
Token IDs
      │
      ▼
Embeddings
      │
      ▼
Transformer
```

------------------------------------------------------------------------

# 12. Mental Models

## Unicode

Think:

> A giant international dictionary.

------------------------------------------------------------------------

## UTF-8

Think:

> A packing machine that converts dictionary entries into bytes.

------------------------------------------------------------------------

## Bytes

Think:

> The language RAM understands.

------------------------------------------------------------------------

## Tokenizer

Think:

> A compressor that turns long byte sequences into meaningful pieces.

------------------------------------------------------------------------

## Transformer

Think:

> A machine that predicts the next token.

------------------------------------------------------------------------

# Final Takeaway

Unicode answers:

> **"What number identifies this character?"**

UTF-8 answers:

> **"How do I store that number as bytes?"**

Modern LLMs first convert text into UTF-8 bytes, then apply Byte Pair
Encoding (BPE), producing token IDs that are fed into the Transformer.
