# Rotary positional embedding (RoPE)

Current source: `cs336_basics/nn/rotary_embedding.py` and `rotary_embedding_einops.py`.

## Why position must enter attention

Without positional information, self-attention is permutation-equivariant: reordering tokens merely reorders outputs. RoPE injects position by rotating pairs of query/key coordinates. It changes attention similarities as a function of relative displacement while adding no learned positional parameters.

For pair index `j=0...K/2-1`, define inverse frequency

`ω_j = theta^(-2j/K)`

and angle at position `p` as `φ_(p,j)=pω_j`. Rotate pair `(x_2j,x_(2j+1))`:

```text
x' = x cos φ - y sin φ
y' = x sin φ + y cos φ
```

`K` must therefore be even.

## Constructor walkthrough

- Explicit validation requires a finite positive `theta`, a positive even
  integer `d_k`, and a positive integer `max_seq_len`.
- `torch.arange(0,d_k,2)`: creates exponent numerators `0,2,...,K-2`, one per pair—not indices later used to select `x`.
- `1 / theta ** (freq_seq/d_k)`: produces geometrically spaced inverse frequencies. Early pairs rotate quickly; later pairs slowly.
- `positions = arange(max_seq_len)`: lists every cacheable absolute position.
- `angles = outer(positions,inv_freq)`: computes all position-frequency combinations, shape `(max_seq_len,K/2)`.
- `register_buffer("cos_cached", cos(angles), persistent=False)` and the analogous sine call: tables move with the module between devices but are not trainable and are omitted from checkpoints because they are deterministic.

An ordinary tensor attribute is not automatically handled like a parameter/buffer by module device conversion. A buffer is the correct model-owned, nonlearned state. `persistent=False` affects `state_dict`, not device movement.

## Forward walkthrough

- `cos_cached[token_positions]` and `sin_cached[token_positions]`: gather angles for each batch/sequence position, yielding `(B,S,K/2)` in normal use. Broadcasting adds the head axis when `x` is `(B,H,S,K)`.
- `x[...,0::2]` and `x[...,1::2]`: split even/odd coordinates into pairs.
- The two elementwise formulas apply all rotations in `O(number of x elements)` work.
- `empty_like(x)` allocates the output, and strided assignments interleave rotated even and odd values.
- The einops version reshapes `(...,S,K)` to `(...,S,K/2,2)`, rotates, stacks, and flattens. Same mathematics, different expression/allocation behavior.

## Worked example

For one pair `(1,0)` at an angle `π/2`, output is `(0,1)`. At position zero the angle is zero for every frequency, so RoPE is identity. Different feature pairs at the same later position rotate through different angles.

## The relative-position identity

Because 2-D rotation matrices satisfy `R(a)ᵀR(b)=R(b-a)`:

`(R(p)q)ᵀ(R(r)k) = qᵀR(r-p)k`.

Thus the rotated Q-K dot product depends on relative offset `r-p`, even though Q and K are individually rotated using absolute positions. Values are not rotated because position is needed in the compatibility score, while values carry aggregated content.

## Complexity and caveats

- Cache memory is `2 × max_seq_len × K/2 = max_seq_len×K` floats.
- Forward time is `O(BHSK)` and does not materialize a `K×K` block-diagonal matrix.
- Positions must be less than `max_seq_len`; generation beyond the cache needs extension or a longer cache.
- The cache is FP32. Multiplication may promote lower-precision Q/K. The slicing
  version writes into `empty_like(x)` and returns `x.dtype`; the einops version
  stacks promoted results and currently returns FP32 for FP16/BF16 input.
- “Slow frequency” does not alone guarantee reliable length extrapolation. RoPE scaling variants address contexts beyond training length.

## Interview questions

**Why apply RoPE to Q and K but not V?**  Q/K determine which positions match, so their dot product should encode displacement. V supplies content after weights are chosen.

**Why does head dimension need to be even?**  Standard RoPE partitions it into independent 2-D planes.

**Is RoPE learned?**  Not in this implementation. Frequencies follow a deterministic schedule controlled by `theta`.

**Why cache sine and cosine?**  They are reused every layer call; caching trades modest memory for avoiding repeated trigonometric computation.

**Does RoPE add a vector to the residual stream?**  No. It rotates Q/K coordinates multiplicatively inside attention.

## Senior interview depth

Each 2-D rotation is orthogonal, so it preserves the L2 norm of every Q/K
vector. At equal positions it also preserves their dot product. At different
positions, the relative-rotation identity changes phase per frequency, giving
the attention score access to relative displacement without a learned position
table.

The current implementation supports shared positions `(S,)`, batched positions
`(B,S)`, and other leading shapes broadcastable to `x`. It inserts singleton
axes immediately before `(S,K/2)`, which is what allows `(B,S,K/2)` caches to
broadcast over heads in `(B,H,S,K)`. It explicitly rejects negative/out-of-cache
positions, wrong dtypes/devices, odd widths, and incompatible batch shapes.

During cached decoding, token positions must be absolute offsets rather than
always restarting at zero. RoPE does not by itself solve length extrapolation:
phase aliasing and distribution shift remain, motivating methods such as
frequency or position scaling. Production implementations may compute cache
growth lazily, store complex pairs, fuse rotation into Q/K kernels, or use a
different pairing convention; checkpoint compatibility depends on that convention.

That mixed-precision dtype discrepancy is an important review finding: the two
implementations agree mathematically but not at the API boundary for low-precision
input. Production code should choose and test one output-dtype policy explicitly.
