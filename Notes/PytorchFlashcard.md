# PyTorch Tensor Manipulation Cheat Sheet
*A practical revision flash card*

---

# Mental Model

A tensor is just a multidimensional array.

```
Scalar      -> ()
Vector      -> (N)
Matrix      -> (M,N)
3D Tensor   -> (B,H,W)
4D Tensor   -> (B,C,H,W)
```

Most bugs in PyTorch come from:

- Wrong shape
- Wrong dimension
- Accidentally copying tensors
- Accidentally modifying tensors in-place

Always keep asking:

> **What is my tensor shape right now?**

---

# 1. shape

```python
x.shape
```

or

```python
x.size()
```

## What it does

Returns tensor dimensions.

Example

```python
x = torch.randn(32, 128)

x.shape
```

Output

```
torch.Size([32,128])
```

## Typical use

Almost every debugging session.

## Watch out

Always print shapes while building a model.

---

# 2. ndim

```python
x.ndim
```

## What it does

Returns number of dimensions.

Example

```
(32,128)
```

↓

```
2
```

Useful when writing generic code.

---

# 3. numel()

```python
x.numel()
```

## What it does

Returns total number of elements.

Example

```
(3,4)

→ 12
```

Useful for parameter counting.

---

# 4. reshape()

```python
x.reshape(...)
```

## What it does

Changes tensor shape.

Example

```python
x = torch.arange(12)

x.reshape(3,4)
```

```
0 1 2 3
4 5 6 7
8 9 10 11
```

## Typical use

Flattening

Reshaping features

Preparing tensors

## Watch out

May return a copy.

---

# 5. view()

```python
x.view(...)
```

## What it does

Same idea as reshape.

But requires contiguous memory.

## Typical use

Very common inside models.

## Watch out

Fails if tensor isn't contiguous.

If you get

```
RuntimeError:
view size is not compatible
```

Do

```python
x = x.contiguous().view(...)
```

Nowadays `reshape()` is usually safer.

---

# 6. flatten()

```python
torch.flatten(x)
```

or

```python
x.flatten()
```

## What it does

Converts multiple dimensions into one.

Example

```
(32,3,224,224)

↓

(32,150528)
```

Typical before Linear layers.

---

# 7. squeeze()

```python
x.squeeze()
```

## What it does

Removes dimensions of size 1.

Example

```
(32,1,128)

↓

(32,128)
```

Useful after inference.

---

# 8. unsqueeze()

```python
x.unsqueeze(dim)
```

## What it does

Adds a dimension of size 1.

Example

```
(128)

↓

(1,128)
```

or

```
↓

(128,1)
```

depending on dim.

Typical when adding batch dimension.

---

# 9. transpose()

```python
x.transpose(dim0, dim1)
```

## What it does

Swaps two axes.

Example

```
(3,5)

↓

(5,3)
```

Typical in attention.

---

# 10. permute()

```python
x.permute(...)
```

## What it does

Reorders all dimensions.

Example

Image

```
(B,H,W,C)

↓

(B,C,H,W)
```

```
permute(0,3,1,2)
```

## Typical use

Computer vision.

Attention.

---

# Watch out

permute changes memory layout.

Often followed by

```python
.contiguous()
```

before view().

---

# 11. contiguous()

```python
x.contiguous()
```

## What it does

Makes tensor memory contiguous.

Usually needed after

- transpose
- permute

---

# 12. repeat()

```python
x.repeat(...)
```

## What it does

Copies tensor.

Example

```
[1 2]

↓

repeat(3)

↓

1 2 1 2 1 2
```

## Watch out

Actually duplicates memory.

Can become huge.

---

# 13. expand()

```python
x.expand(...)
```

## What it does

Pretends tensor is larger.

No memory copied.

Very efficient.

## Watch out

Only works on dimensions of size 1.

---

# repeat vs expand

| repeat | expand |
|---------|---------|
| Copies memory | No copy |
| Slower | Faster |
| More memory | Almost free |

Whenever possible

Use

```
expand
```

instead of

```
repeat
```

---

# 14. cat()

```python
torch.cat(...)
```

## What it does

Concatenate tensors.

Example

```
A

1 2

B

3 4

↓

cat

1 2
3 4
```

Typical use

Joining features.

Skip connections.

---

# Watch out

Other dimensions must match.

---

# 15. stack()

```python
torch.stack(...)
```

## What it does

Creates a new dimension.

Example

```
A
(3)

B
(3)

↓

stack

(2,3)
```

---

# cat vs stack

cat

```
joins existing dimension
```

stack

```
creates new dimension
```

---

# 16. split()

```python
torch.split(...)
```

## What it does

Break tensor into pieces.

Useful for

- batches
- microbatching
- sequence chunks

---

# 17. chunk()

```python
torch.chunk(...)
```

## What it does

Splits into equal chunks.

Example

```
batch=64

↓

4 chunks

↓

16 each
```

---

# 18. index_select()

```python
torch.index_select(...)
```

Selects rows using indices.

Useful for embeddings.

---

# 19. gather()

```python
torch.gather(...)
```

Very important.

Selects values using another tensor of indices.

Used heavily in

- beam search
- reinforcement learning
- language models

---

# 20. scatter()

```python
torch.scatter(...)
```

Opposite of gather.

Writes values into specified indices.

Used for

- one-hot encoding
- routing

---

# 21. where()

```python
torch.where(condition,a,b)
```

Like

```
if-else
```

for tensors.

Example

```python
torch.where(x>0,x,0)
```

ReLU-like behavior.

---

# 22. masked_fill()

```python
x.masked_fill(mask,-inf)
```

Very important.

Typical use

Attention masking.

Padding masks.

Causal masks.

---

# 23. clamp()

```python
torch.clamp(x,min,max)
```

Limits values.

Example

```
[-5,8]

↓

[0,5]
```

Useful for numerical stability.

---

# 24. torch.arange()

```python
torch.arange(10)
```

Creates

```
0 1 2 ... 9
```

Very common.

---

# 25. linspace()

```python
torch.linspace(0,1,100)
```

Creates evenly spaced values.

---

# 26. zeros()

```python
torch.zeros(...)
```

Creates zeros.

---

# 27. ones()

```python
torch.ones(...)
```

Creates ones.

---

# 28. full()

```python
torch.full(shape,value)
```

Creates constant tensor.

---

# 29. rand()

```python
torch.rand(...)
```

Uniform random.

---

# 30. randn()

```python
torch.randn(...)
```

Gaussian random.

Most neural network initialization starts here.

---

# 31. eye()

```python
torch.eye(n)
```

Identity matrix.

Useful for

- linear algebra
- masking

---

# 32. clone()

```python
x.clone()
```

Creates independent copy.

---

# 33. detach()

```python
x.detach()
```

Stops gradient tracking.

Very important.

Used during inference.

Also for logging tensors.

---

# clone vs detach

clone

```
Copies data
Still computes gradients
```

detach

```
No gradients
Shares underlying storage
```

Need both?

```
x.detach().clone()
```

---

# 34. cpu()

```python
x.cpu()
```

Moves tensor to CPU.

Usually before

```
.numpy()
```

---

# 35. cuda()

```python
x.cuda()
```

Moves tensor to GPU.

Nowadays preferred:

```python
x.to(device)
```

---

# 36. to()

```python
x.to(device)
```

Moves

- CPU ↔ GPU

Also changes dtype.

Example

```python
x.to(torch.float16)
```

---

# 37. float()

```python
x.float()
```

Convert to float32.

---

# 38. long()

```python
x.long()
```

Convert to int64.

Embedding layers require LongTensor.

---

# 39. item()

```python
loss.item()
```

Extracts Python scalar.

Typical

```python
print(loss.item())
```

---

# 40. numpy()

```python
x.numpy()
```

Converts CPU tensor into NumPy array.

Watch out

GPU tensors must first do

```python
x.cpu().numpy()
```

---

# 41. mean()

```python
x.mean()
```

Average.

---

# 42. sum()

```python
x.sum()
```

Total.

---

# 43. max()

```python
x.max()
```

Maximum.

---

# 44. argmax()

```python
x.argmax(dim=1)
```

Returns index of largest value.

Typical classification prediction.

---

# 45. softmax()

```python
torch.softmax(x,dim=-1)
```

Turns logits into probabilities.

LLMs use this at the output.

---

# 46. topk()

```python
torch.topk(x,k=5)
```

Returns top K values.

Used in

- beam search
- top-k sampling

---

# 47. einsum()

```python
torch.einsum(...)
```

General tensor algebra.

Very expressive.

Often used in

- attention
- tensor contractions

Can replace multiple transpose + matmul operations.

---

# 48. matmul()

```python
torch.matmul(A,B)
```

Matrix multiplication.

The workhorse of Transformers.

---

# 49. bmm()

```python
torch.bmm(A,B)
```

Batch matrix multiplication.

Shapes

```
(B,N,M)

×

(B,M,K)

↓

(B,N,K)
```

Very common in attention.

---

# 50. In-place Operations

Examples

```python
x.add_(1)

x.mul_(2)

x.zero_()
```

Notice the trailing underscore.

## What it means

Modifies original tensor.

## Pros

Less memory.

## Cons

Can break autograd.

Avoid unless you know exactly why.

---

# Most Frequently Used (80/20)

✅ shape

✅ reshape

✅ view

✅ flatten

✅ squeeze

✅ unsqueeze

✅ permute

✅ transpose

✅ cat

✅ stack

✅ clone

✅ detach

✅ to

✅ float

✅ long

✅ argmax

✅ softmax

✅ matmul

✅ bmm

✅ where

---

# Common Pitfalls

### 1. Wrong dimension

Always verify:

```python
print(x.shape)
```

---

### 2. `view()` after `permute()`

Wrong

```python
x.permute(...).view(...)
```

Correct

```python
x.permute(...).contiguous().view(...)
```

or simply

```python
x.reshape(...)
```

---

### 3. Forgetting batch dimension

Model expects

```
(1,3,224,224)
```

You pass

```
(3,224,224)
```

Fix

```python
x.unsqueeze(0)
```

---

### 4. Embedding indices not LongTensor

Wrong

```python
float32
```

Correct

```python
long()
```

---

### 5. Calling `.numpy()` on GPU tensor

Wrong

```python
x.numpy()
```

Correct

```python
x.cpu().numpy()
```

---

### 6. Using `repeat()` when `expand()` works

Prefer

```
expand()
```

for memory efficiency.

---

### 7. Accidentally modifying tensors

```python
x += 1
```

may modify in-place.

Know whether you're using an in-place operation, especially inside training loops.

---

# Shape Debugging Workflow

Whenever something breaks:

```python
print(x.shape)

print(x.dtype)

print(x.device)

print(x.requires_grad)
```

These four lines solve the vast majority of PyTorch debugging issues.