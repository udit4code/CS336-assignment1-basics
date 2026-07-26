# Why Do We Train Byte-Level BPE on UTF-8 Instead of UTF-16 or UTF-32?

## Short Answer

We prefer **UTF-8** because it is **more space-efficient**, **backward compatible with ASCII**, **the de facto standard encoding on the Internet**, and allows us to build a tokenizer with a **fixed vocabulary of only 256 byte values** while still representing every Unicode character. UTF-16 and UTF-32 use more memory for common text and therefore lead to longer or less efficient byte sequences for tokenizer training.

---

# Understanding the Problem

A tokenizer does **not** work directly on Unicode characters.

Instead, it first converts text into **bytes**, and then learns frequently occurring byte sequences using **Byte Pair Encoding (BPE)**.

The choice of encoding determines **what those byte sequences look like**.

---

# Comparing UTF-8, UTF-16 and UTF-32

Consider the string

```text
hello
```

### UTF-8

Each ASCII character occupies **1 byte**.

| Character | Bytes |
|-----------|------:|
| h | 1 |
| e | 1 |
| l | 1 |
| l | 1 |
| o | 1 |

Total:

```text
5 bytes
```

---

### UTF-16

Each ASCII character typically occupies **2 bytes**.

```text
10 bytes
```

This doubles the storage requirement.

---

### UTF-32

Every character occupies **4 bytes**, regardless of whether it is a simple English letter or an emoji.

```text
20 bytes
```

This requires four times as much storage as UTF-8 for English text.

---

# Example 2: Japanese Text

Consider

```text
こんにちは
```

| Encoding | Approximate Bytes per Character |
|-----------|-------------------------------:|
| UTF-8 | 3 |
| UTF-16 | 2 |
| UTF-32 | 4 |

For many Asian languages, UTF-16 can sometimes be slightly more compact than UTF-8.

However, modern text corpora (such as TinyStories or OpenWebText) are dominated by English and ASCII characters, making UTF-8 the better overall choice.

---

# Why UTF-8 is Better for Tokenizer Training

## 1. More Space Efficient

Most web text consists primarily of

- English letters
- Digits
- Spaces
- Punctuation

These all require only **one byte** in UTF-8.

Smaller files mean

- less disk usage
- lower memory consumption
- faster reading
- faster tokenizer training

---

## 2. Backward Compatible with ASCII

The first 128 Unicode characters have **exactly the same byte representation** as ASCII.

For example

```text
A
```

is

```text
65
```

in both ASCII and UTF-8.

This makes UTF-8 compatible with decades of existing software and datasets.

---

## 3. Smaller Initial Vocabulary

A byte-level tokenizer starts with

```text
256 possible byte values
```

regardless of the language.

This gives us an initial vocabulary of only

```text
0 ... 255
```

instead of needing a vocabulary containing more than 150,000 Unicode characters.

This greatly simplifies tokenizer training.

---

## 4. Better Compression Through BPE

UTF-8 stores common English text compactly.

As a result, frequently occurring words such as

```text
the

ing

tion

hello
```

appear as compact byte sequences.

BPE can easily merge these byte sequences into useful subword tokens.

If UTF-32 were used, every character would contain three additional zero bytes, reducing storage efficiency and making the input unnecessarily large.

---

## 5. Internet Standard

Over **98% of web pages** are encoded using UTF-8.

Training directly on UTF-8 means the tokenizer naturally matches the format used by almost all real-world text.

No conversion is required for most datasets.

---

# Why Not UTF-16?

UTF-16 has two disadvantages for tokenizer training.

- English text occupies roughly twice as much memory.
- Byte sequences become longer for ASCII-heavy datasets.

Although UTF-16 is efficient for some East Asian languages, most modern LLM training datasets contain predominantly English text, making UTF-8 a better overall compromise.

---

# Why Not UTF-32?

UTF-32 has an even bigger drawback.

Every character occupies four bytes.

For example

```text
hello
```

requires

```text
20 bytes
```

instead of

```text
5 bytes
```

This unnecessarily increases

- storage requirements
- memory usage
- I/O costs
- tokenizer training time

without providing any practical benefit.

---

# Comparison

| Property | UTF-8 | UTF-16 | UTF-32 |
|-----------|:-----:|:------:|:------:|
| Variable-length encoding | ✅ | ✅ | ❌ |
| ASCII uses only 1 byte | ✅ | ❌ | ❌ |
| Space-efficient for English | ✅ | ❌ | ❌ |
| Fixed byte vocabulary (256 bytes) | ✅ | ✅ | ✅ |
| Dominant web encoding | ✅ | ❌ | ❌ |
| Good choice for BPE training | ✅ | ⚠️ | ❌ |

---

# Final Answer (1–2 Sentences)

UTF-8 is preferred because it is **space-efficient, backward compatible with ASCII, and the standard encoding used on the Internet**, allowing common English text to be represented using fewer bytes than UTF-16 or UTF-32. This reduces memory usage and speeds up BPE tokenizer training while still allowing every Unicode character to be represented using the same fixed 256-byte vocabulary.