# Training pipeline deep dive

This document explains how 'pipeline train' converts a UTF-8 text corpus into a
trained TransformerLM artifact. It is written as a design document for a
beginner and as a maintenance reference for an engineer. The pipeline is an
orchestration layer: the numerical components remain in cs336_basics.

## 1. End-to-end data flow

~~~text
raw UTF-8 text
  -> tiktoken IDs
  -> uint32 .npy arrays
  -> LanguageModelDataset windows
  -> batches of input/target IDs
  -> TransformerLM logits
  -> cross-entropy scalar
  -> backward, clipping, AdamW update
  -> checkpoints and final artifact
~~~

Notation: B is batch size, S is context length, D is model width, H is head
count, V is vocabulary size, and d_k = D/H is the per-head width.

## What happens during training?

This is the most important mental model: **one training step does not process
the entire 2 GB text file**. A step processes one small batch of token windows.
The 2 GB file is read and tokenized during data preparation; training then
samples small slices from the resulting token array.

### First principle: a language model learns next-token prediction

Suppose the tokenizer converts a corpus into this one-dimensional stream:

~~~text
[t0, t1, t2, t3, t4, t5, ... , t(N-1)]
~~~

Each 't_i' is an integer token ID, not necessarily a complete word. For
'context_length = S = 4', a supervised example is made by shifting the same
stream by one position:

~~~text
input  = [t_i,   t_(i+1), t_(i+2), t_(i+3)]
target = [t_(i+1), t_(i+2), t_(i+3), t_(i+4)]
~~~

The model receives the input sequence and must predict each corresponding
target token. For example, the target for the first input position is 't_(i+1)',
the token immediately after 't_i'.

### What is one data point?

In 'cs336_basics.data.LanguageModelDataset.__getitem__', data point 'i' is:

~~~python
input_tokens = tokens[i : i + S]
target_tokens = tokens[i + 1 : i + S + 1]
~~~

For a token array of length 'N', there are 'N - S' valid starting positions.
The windows overlap intentionally. This lets every token participate in many
different contexts without duplicating a second copy of the corpus.

The shapes and dtypes are:

~~~text
one input:   (S,), torch.long
one target:  (S,), torch.long
~~~

### What does one training step sample?

'cs336_basics.batching.get_batch' chooses 'B' random starting positions from
the dataset and stacks their windows:

~~~text
inputs:  (B, S)
targets: (B, S)
~~~

The current Modal configuration uses 'B = 8' and 'S = 128'. Therefore one
optimizer step performs:

~~~text
8 windows x 128 positions = 1,024 token predictions
~~~

The random indices are sampled with replacement. Two examples may overlap, and
the same example may be sampled again later. The step therefore does not mean
"move to the next 1,024 tokens" and it does not guarantee complete corpus
coverage.

### What happens to the 2 GB file?

The file has two distinct roles:

1. During preparation, 'encode_text_file' reads the UTF-8 file once, runs
   tiktoken, validates the IDs, and writes 'train.npy'.
2. During training, 'LanguageModelDataset' memory-maps 'train.npy' and reads
   only the sampled windows needed for the current batch.

The text file is not reread from the beginning on every optimizer step. The
current implementation does tokenize the complete file again when starting a
new run; a persistent preprocessing cache can avoid that cost for future
experiments.

Also, file size in bytes is not token count. Tokenization depends on the text
and vocabulary. The exact 'N' is recorded in:

~~~text
<run-dir>/tokens/train.json
~~~

### What does 100 or 500 steps cover?

The number of token predictions processed is approximately:

~~~text
tokens_seen = steps x batch_size x context_length
~~~

With 'B = 8' and 'S = 128':

~~~text
100 steps = 100 x 8 x 128 = 102,400 token exposures
500 steps = 500 x 8 x 128 = 512,000 token exposures
~~~

These are token **exposures**, not necessarily unique tokens. If the 2 GB
corpus contains approximately 500 million tokens, 500 steps represent about
0.1% of the corpus in exposure count. They are useful for a Modal smoke test,
but they are not a complete pass over the dataset.

The pipeline does not have a conventional sequential epoch because it samples
random windows. An equivalent token-based epoch can be estimated as:

~~~text
steps_per_epoch ~= token_count / (batch_size x context_length)
~~~

For the exact estimate, use 'token_count' from 'train.json', not the 2 GB byte
size.

### What happens after a batch is sampled?

For each step, 'pipeline/train.py:run_training' performs:

~~~text
1. Sample inputs and shifted targets
2. TransformerLM(inputs) -> logits of shape (B, S, V)
3. Cross-entropy(logits, targets) -> one scalar loss
4. Backpropagation -> gradients for model parameters
5. Global gradient clipping
6. Learning-rate schedule
7. AdamW parameter update
~~~

The loss averages the prediction error over all 'B x S' positions. The model
is updated once after that average is computed. A checkpoint is only a saved
snapshot of this state; creating a checkpoint does not process additional
training data.

### Is 500 steps a good choice?

Yes, as an economical first Modal run. Use 50 warmup steps and checkpoints at
125, 250, 375, and 500. This run validates CUDA setup, tokenization, loss
behavior, throughput, metrics, and artifact persistence.

For meaningful learning on the complete TinyStories corpus, increase the step
count after measuring the actual throughput and validation-loss trend. A full
corpus pass may require hundreds of thousands of steps, depending on the exact
token count, batch size, context length, and desired number of epochs.

## 2. Code map

~~~text
pipeline/
├── __main__.py       # python -m pipeline entry point
├── cli.py             # train subcommand and shared CLI entry point
├── config.py          # typed configuration and validation
├── prepare_data.py    # text -> tiktoken IDs -> .npy + metadata
├── train.py           # device setup, datasets, optimization loop
├── runtime.py         # CPU/CUDA/MPS and dtype selection
├── artifacts.py       # atomic checkpoints and final artifact
├── artifacts/         # generated runs; Git ignores generated contents
└── cache/             # reserved for reusable preprocessing
~~~

'pipeline/__main__.py' delegates to 'pipeline.cli.main'. The training command
turns validated CLI arguments into the typed configuration objects consumed by
the training loop.

## 3. Stage 0: parse and validate configuration

### Input

The user supplies arguments such as:

~~~bash
uv run python -m pipeline train \
  --data data/sample-train.txt \
  --context-length 8 \
  --d-model 64 --num-layers 2 --num-heads 4 --d-ff 256 \
  --warmup-steps 2 --max-steps 10
~~~

### Code and output

'pipeline/cli.py:build_parser' declares the options. 'main' turns them into:

- 'DataConfig': train/validation paths, encoding, context length, split fraction;
- 'ModelConfig': vocabulary and Transformer dimensions;
- 'OptimizerConfig': learning rates, warmup, decay, clipping, step count;
- 'RuntimeConfig': batch size, seed, device, dtype, and intervals.

The 'validate' methods in 'pipeline/config.py' reject invalid settings before
expensive work begins. Examples are missing paths, non-positive dimensions,
'D % H != 0', 'max_steps <= warmup_steps', and an invalid dtype.

The configuration is stored in checkpoints and the final artifact. It is part
of the model identity: a checkpoint made with one model shape cannot safely be
loaded into a different shape.

## 4. Stage 1: reproducibility and hardware

'pipeline/train.py:seed_everything' seeds Python, NumPy, PyTorch, and CUDA when
available. It controls parameter initialization and random window sampling.

'resolve_device' uses this policy:

~~~text
--device auto: CUDA -> MPS -> CPU
--device cuda: explicit NVIDIA GPU
--device mps: explicit Apple Silicon GPU
--device cpu: explicit CPU
~~~

'resolve_dtype' maps names to PyTorch floating-point dtypes. All model
parameters and batches use the selected device.

For an M4 Mac use '--device mps --dtype float32'. For Kaggle or Modal use
'--device cuda' in a CUDA-enabled PyTorch image. Cloning the repository does
not install GPU drivers or provision a GPU.

## 5. Stage 2: text to token IDs

### Input

'DataConfig.train_path' and optional 'DataConfig.valid_path' point to UTF-8
text files.

### Code

'pipeline/prepare_data.py:encode_text_file' performs:

~~~python
text = source.read_text(encoding="utf-8")
encoding = tiktoken.get_encoding(encoding_name)
token_ids = np.asarray(
    encoding.encode(text, allowed_special={"<|endoftext|>"}),
    dtype=np.uint32,
)
~~~

For the default 'gpt2' encoding, 'encoding.n_vocab' is 50,257. Each integer
will later index one row of the model embedding table.

'<|endoftext|>' is explicitly allowed as a special token. Every occurrence is
therefore exactly one ID, 'encoding.eot_token'. Other special tokens remain
disallowed, so unexpected markers fail fast instead of silently changing the
training corpus. The pipeline does not insert EOS markers that were absent
from the source; it preserves markers that are present.

### Output

'pipeline/prepare_data.py:save_token_array' atomically writes:

~~~text
<run-dir>/tokens/train.npy
<run-dir>/tokens/valid.npy
~~~

The array contract is:

~~~text
dtype: uint32
shape: (N,)
contents: token_ids[0] ... token_ids[N-1]
~~~

A JSON sidecar records source path, source SHA-256, encoding, vocabulary size,
token count, dtype, and token-file path.

The encoder rejects missing files, empty corpora, and out-of-range IDs. The
current implementation reads one source file into memory to preserve exact
tiktoken behavior. A future large-corpus version should add document-aware
streaming without changing the downstream array contract.

### Why do we insert '.npy' between '.txt' and the dataset?

This is not an unnecessary model step. It is a deliberate **format boundary**
between an authoritative source corpus and a training-friendly numerical
representation.

#### Start from first principles

The model cannot consume characters or UTF-8 bytes directly. Its embedding
table is an array indexed by integer token IDs. Therefore the conceptual
pipeline is already:

~~~text
characters/bytes -> tokenizer -> integer token IDs -> model
~~~

The question is whether we recompute the tokenizer every time a training batch
is requested, or materialize the token-ID stream once and reuse it. The
pipeline chooses the latter:

~~~text
authoritative .txt
      |
      | one-time UTF-8 read, BPE encoding, validation
      v
one-dimensional uint32 token array in .npy
      |
      | np.load(..., mmap_mode="r")
      v
LanguageModelDataset
      |
      | lazy shifted slices: [t_i ... t_(i+S-1)] and [t_(i+1) ... t_(i+S)]
      v
random training batches on CPU, then accelerator tensors
~~~

The '.npy' file is therefore a **preprocessing cache**, not a second source of
truth. The text file remains the human-readable corpus; the token array is the
compiled form consumed by training. Its JSON sidecar records the tokenizer,
vocabulary, source SHA-256, token count, dtype, and path so the cache can be
audited and invalidated when its inputs change.

#### Why not tokenize the '.txt' on every batch?

A batch samples random starting positions in a token stream. To serve a request
starting at token position 'i', the system needs the token IDs around that
position, not merely characters around an arbitrary byte offset. BPE tokenization
depends on neighboring bytes and merge rules, so random byte offsets require
boundary handling and potentially retokenizing surrounding text. Repeating
that work for every batch would put expensive CPU tokenization on the critical
training path and make accelerator utilization depend on text-processing
latency.

Persisting token IDs turns the hot path into direct indexed array access:

~~~python
tokens = np.load("train.npy", mmap_mode="r")
input_tokens = tokens[i : i + S]
target_tokens = tokens[i + 1 : i + S + 1]
~~~

The dataset then copies only those small slices into writable 'torch.long'
tensors. It does not construct Python strings or rerun BPE merges during
training.

#### Why '.npy' specifically?

'NumPy .npy' is a small, portable binary container with an array header that
records shape and dtype. It fits this workload because the token stream is
already a flat homogeneous array. NumPy can memory-map it, and the existing
dataset code can slice it using ordinary indexing. This gives us a simple
contract with very few dependencies:

~~~text
train.npy: dtype=uint32, shape=(N,), values=[t_0, t_1, ..., t_(N-1)]
~~~

'uint32' uses four bytes per token and supports the GPT-2 vocabulary and most
practical vocabularies. The dataset currently copies each sampled slice into a
'torch.long' tensor because embedding lookup requires an integer indexing dtype;
the persisted storage does not need to be 64-bit.

#### Systems-engineering advantages

| Property | Benefit |
| --- | --- |
| Separate preprocessing job | Tokenization cost is paid once per corpus/tokenizer configuration, not once per batch or epoch. |
| Random access | A random window begins at an integer token offset; no text scanning is needed. |
| Memory mapping | The operating system pages in only touched regions. A multi-gigabyte array does not need to be fully resident in Python RAM. |
| Compact representation | Four-byte IDs avoid UTF-8 parsing and Python string/object overhead during training. |
| Stable numerical contract | Every consumer sees the same one-dimensional IDs, dtype, and shape. |
| Reproducibility | Source hash and tokenizer metadata make preprocessing choices inspectable and repeatable. |
| Process sharing | Multiple workers can map the same file and rely on the OS page cache rather than each holding a complete Python copy. |
| Clean train/validation boundaries | The pipeline can persist separate arrays so no sampled window crosses the split. |

Memory mapping is not magic zero-memory access. A requested page still incurs a
disk read on a cache miss, and random windows can cause page faults. The win is
that storage is demand-paged and bounded by the operating system's cache policy,
instead of eagerly materializing the entire token corpus in each process.

#### Costs and failure modes

| Cost | Engineering implication |
| --- | --- |
| Extra disk | The '.txt' and derived '.npy' coexist. Storage is roughly '4N' bytes for 'N' token IDs, plus a small header and metadata. |
| One-time latency | A large corpus must be read, tokenized, validated, and written before the first training step. |
| Current preparation peak memory | 'encode_text_file' currently reads the complete text and holds tiktoken's encoded list before creating the NumPy array. The training phase is memory-efficient, but preprocessing a 2 GB file can still require substantial RAM. |
| Stale derived data | Reusing an array after changing the source text, tokenizer, special-token policy, or vocabulary silently trains on the wrong data unless metadata is checked. |
| Tokenizer coupling | IDs are meaningful only with the exact tokenizer and merge vocabulary that produced them. A '.npy' file is not self-describing by itself; the sidecar is part of the contract. |
| Fixed integer width | A tokenizer with IDs outside the 'uint32' range cannot use this storage dtype without changing the format. |
| Monolithic file | A single array is simple, but interruption or a full rewrite is less convenient than independently versioned shards. Network filesystems may also have poor random-read latency. |
| No compression | Raw arrays trade disk efficiency for fast indexed access. Compression would reduce storage but require decompression and complicate random reads. |
| Portability details | Filesystem paths in metadata are informational; the actual array and sidecar must be copied together when moving a run between a Mac, Modal, or another machine. |

The most important subtlety is that '.npy' improves **training-time** memory and
latency; it does not make the current **preprocessing-time** encoder streaming.
For a truly large production corpus, the next evolution would tokenize in
document-aware chunks and write sharded '.npy' or raw binary files with an
index. That preserves the same downstream token-array contract while bounding
preprocessing memory, improving recovery after interruption, and allowing
parallel ingestion.

#### Design decision in one sentence

We retain '.txt' as the reproducible source artifact and compile it once into a
validated, compact, memory-mappable token artifact so the training loop can do
fast random numerical access without repeatedly performing text processing.

## 6. Stage 3: validation split

When '--valid-data' is provided, the two files are encoded independently. This
is the preferred production workflow.

Without a validation file, 'prepare_datasets' calls 'split_tokens' and makes a
deterministic suffix split:

~~~text
all_tokens = [t0, ..., t(N-1)]
train      = all_tokens[:split_at]
valid      = all_tokens[split_at:]
~~~

Both partitions must contain more than S tokens. They are persisted separately,
so a training window cannot cross the partition boundary. The split is not
shuffled and is reproducible.

## 7. Stage 4: token stream to supervised examples

### Existing code reused

'cs336_basics.data.LanguageModelDataset' memory-maps the '.npy' array. For
tokens '[10, 20, 30, 40, 50]' and S=4, item zero is:

~~~text
input  = [10, 20, 30, 40]
target = [20, 30, 40, 50]
~~~

This is implemented by 'LanguageModelDataset.__getitem__':

~~~python
input_tokens = tokens[index : index + context_length]
target = tokens[index + 1 : index + context_length + 1]
~~~

For N tokens:

~~~text
dataset length = N - S
one item       = (input, target)
input.shape    = (S,), torch.long
target.shape   = (S,), torch.long
~~~

'cs336_basics.batching.get_batch' samples B random start indices and stacks:

~~~text
inputs.shape  = (B, S)
targets.shape = (B, S)
~~~

The dataset copies from the read-only memmap into safe PyTorch tensor storage,
then the batch moves to the selected device.

## 8. Stage 5: model construction and forward pass

'run_training' constructs the existing model:

~~~python
model = TransformerLM(
    vocab_size=V,
    context_length=S,
    d_model=D,
    num_layers=L,
    num_heads=H,
    d_ff=F,
    theta=rope_theta,
    dtype=dtype,
).to(device)
~~~

The pipeline verifies that model vocabulary size equals the selected tiktoken
vocabulary size.

The forward contract is:

~~~text
input IDs: (B, S), torch.long
output:    logits (B, S, V), floating point
~~~

'TransformerLM.forward' applies:

~~~text
(B,S) -> token embedding       (B,S,D)
     -> L Transformer blocks   (B,S,D)
     -> final RMSNorm          (B,S,D)
     -> lm_head                (B,S,V)
~~~

A logit is an unnormalized score for a possible next token, not a probability.
Each pre-norm block performs:

~~~text
x -> RMSNorm -> causal multi-head self-attention -> residual add
  -> RMSNorm -> SwiGLU feed-forward network   -> residual add
~~~

Attention applies RoPE to queries and keys and masks future positions. Position
i can attend only to positions 0 through i.

## 9. Stage 6: one optimization step

The loop is 'pipeline/train.py:run_training'.

### 9.1 Sample

~~~python
inputs, targets = get_batch(train_dataset, runtime.batch_size, str(device))
~~~

Both tensors have shape (B,S) and dtype 'torch.long'.

### 9.2 Forward and loss

~~~python
optimizer.zero_grad(set_to_none=True)
loss = cross_entropy(model(inputs), targets)
~~~

The existing cross-entropy subtracts the maximum logit before exponentiation.
For target y:

~~~text
loss_y = log(sum_j exp(logit_j)) - logit_y
~~~

It averages over all B*S positions and returns one scalar tensor.

### 9.3 Backward and clipping

~~~python
loss.backward()
gradient_clipping(model.parameters(), max_grad_norm)
~~~

'backward' fills gradient buffers but does not update weights. Clipping computes
the global norm sqrt(sum of g squared over every parameter element). If the
norm exceeds the threshold, all gradients receive the same scale factor.

### 9.4 Schedule and optimizer

~~~python
update_step = step + 1
lr = get_lr_cosine_schedule(
    update_step, learning_rate, min_learning_rate,
    warmup_steps, max_steps,
)
for group in optimizer.param_groups:
    group["lr"] = lr
optimizer.step()
~~~

The schedule linearly warms up, then cosine-decays to the minimum learning
rate. 'step + 1' avoids a zero-learning-rate first update. The existing
'cs336_basics.nn.AdamW' applies Adam moments and decoupled weight decay.

## 10. Stage 7: evaluation and metrics

Every 'eval_interval' steps, 'evaluate' runs:

~~~python
model.eval()
with torch.no_grad():
    ...
model.train()
~~~

It samples 'eval_batches' validation windows and averages their loss. A metric
record looks like:

~~~json
{
  "step": 100,
  "train_loss": 8.12,
  "learning_rate": 0.0003,
  "valid_loss": 8.45,
  "gradient_norm_before_clip": 2.1,
  "gradient_norm_after_clip": 1.0,
  "parameter_norm": 87.4,
  "update_norm": 0.19,
  "tokens_seen": 204800,
  "tokens_per_second": 12500.0,
  "wall_time_seconds": 16.4
}
~~~

The training loop records one record per optimizer step. 'update_norm' is the
global L2 displacement caused by AdamW, but is measured only at 'log_interval'
steps: copying a full model every step would add significant memory traffic.
Gradient norms are recorded every step, including values before and after
clipping. 'tokens_seen' counts token exposures (steps times B times S), not
unique corpus tokens.

Perplexity can later be derived as 'exp(loss)' without changing the schema.

### Plotting artifact

'pipeline/artifacts.py:save_final_artifact' writes 'metrics.json' as a
versioned document:

~~~json
{
  "schema_version": 1,
  "description": "...",
  "fields": ["step", "train_loss", "gradient_norm_before_clip", "..."],
  "summary": {
    "num_records": 10000,
    "final_train_loss": 3.2,
    "best_valid_loss": 3.5,
    "best_valid_step": 9700,
    "final_tokens_seen": 204800000
  },
  "records": [
    {"step": 1, "train_loss": 10.8, "update_norm": null}
  ]
}
~~~

It is directly consumable by Python, pandas, or a notebook:

~~~python
import json
from pathlib import Path

metrics = json.loads(Path("pipeline/artifacts/RUN/metrics.json").read_text())
steps = [row["step"] for row in metrics["records"]]
losses = [row["train_loss"] for row in metrics["records"]]
gradient_norms = [row["gradient_norm_before_clip"] for row in metrics["records"]]
~~~

'fields' is the union of record fields, and 'summary' provides useful report
values without scanning the complete history.

## 11. Stage 8: resumable checkpoints

Saving model weights alone is insufficient. AdamW has moving averages, and
random sampling depends on RNG state.

'pipeline/artifacts.py:save_training_checkpoint' stores:

~~~text
schema_version
model_state_dict
optimizer_state_dict
step
configuration and data metadata
metrics so far
Python, NumPy, PyTorch, and CUDA RNG states
~~~

Files are written under:

~~~text
pipeline/artifacts/<run-id>/checkpoints/step_00001000.pt
~~~

'load_training_checkpoint' restores state into a constructed compatible model
and optimizer. The trainer checks saved model and data configuration before
continuing.

## 12. Stage 9: final artifact

At completion, 'save_final_artifact' writes:

~~~text
pipeline/artifacts/<run-id>/
├── artifact.pt
├── config.json
├── manifest.json
├── metrics.json
├── tokens/
│   ├── train.npy / train.json
│   └── valid.npy / valid.json
└── checkpoints/
~~~

The canonical 'artifact.pt' payload is:

~~~python
{
    "schema_version": 1,
    "step": final_step,
    "model_state_dict": ...,
    "config": {"data": ..., "model": ..., "optimizer": ..., "runtime": ...},
    "tokenizer": {
        "name": "gpt2",
        "vocab_size": 50257,
        "endoftext_token": "<|endoftext|>",
        "endoftext_id": 50256,
    },
    "data": {"train": ..., "valid": ...},
    "metrics": [...],
}
~~~

It is self-describing: it records model dimensions, tokenizer settings, data
metadata, and optimization history without requiring the original CLI command.
The PyTorch payload is written atomically; JSON sidecars are human-readable.

## 13. Artifact formats: why `.pt` and what production uses

An artifact is more than a file containing weights. At minimum, a consumer
needs the learned tensors, model dimensions, tokenizer identity, vocabulary
and special-token IDs, tensor dtype, and a schema/version. A *training
checkpoint* additionally needs the optimizer and scheduler state, gradient
scaler (if used), current step, and random-number-generator states so that a
resume is statistically and operationally close to uninterrupted training.
For Adam-like optimizers, the first- and second-moment tensors are commonly
roughly two additional parameter-sized buffers (before accounting for
master-precision copies), so a resumable checkpoint can be several times
larger than the weights alone.
A *release artifact* for inference usually needs only model weights plus the
configuration and tokenizer files. A *serving artifact* may go one step
further and contain a graph compiled for a particular accelerator.

These are different contracts and do not have to use the same serialization
format. Treating them as one file is convenient for a small project, but it
becomes an avoidable coupling at production scale.

### What `.pt` means in this repository

`.pt` is a filename convention for a PyTorch-serialized object; it is not a
universal model standard. In this pipeline, `torch.save` writes a dictionary
whose important field is `model_state_dict`, along with configuration,
tokenizer metadata, data metadata, metrics, and the schema version shown
above. Checkpoints use the same family of serialization but also contain
optimizer and RNG state. The `.pth` suffix is, in practice, another PyTorch
convention and is not inherently safer, faster, or more portable than `.pt`.

The final-artifact loader is explicitly configured with `weights_only=True`.
That restricted mode accepts the tensors and primitive/container metadata
written by this pipeline without reconstructing arbitrary Python globals.
Training-resume loading currently needs the richer checkpoint payload and
therefore must only consume artifacts that this project (or another trusted
producer) created. Never load an untrusted pickle-backed checkpoint in a
production process.

### Strengths of `.pt`

* **Native training round-trip.** `torch.save`/`torch.load` preserve a model
  state dictionary and, for checkpoints, optimizer, scheduler, scaler, step,
  and RNG state without a conversion step.
* **Flexible payload.** Tensors can be accompanied by structured metadata,
  metrics, and application-specific fields. This is useful while the schema
  is evolving.
* **Low integration cost.** PyTorch can load the artifact directly on CPU,
  CUDA, or MPS with `map_location`, and the existing model constructor can
  consume it immediately.
* **Operational safety in this pipeline.** The writer uses an atomic temporary
  file followed by a rename, while JSON sidecars make important metadata
  inspectable without importing PyTorch.

### Costs and risks of `.pt`

* **Security.** Traditional PyTorch serialization is pickle-based. Loading an
  arbitrary file can execute code; `weights_only=True` reduces this risk for
  tensor-only loads, but it is not a substitute for provenance, checksums, and
  access controls.
* **Framework and code coupling.** The file assumes PyTorch and compatible
  state-dictionary keys. Renaming a module, changing parameter names, or
  changing tensor semantics can make an old artifact unloadable even when its
  numerical tensors are valid.
* **No universal serving contract.** A state dictionary does not specify a
  portable execution graph, tokenizer server, KV-cache policy, batching
  behavior, or accelerator kernels. A serving system still has to reconstruct
  the Python model.
* **Weakly enforced schema.** The application-level `schema_version` helps,
  but `torch.save` itself does not validate required fields, shapes, dtypes,
  tokenizer hashes, or compatibility. Those checks belong in the manifest and
  loader.
* **Scaling and distribution.** One monolithic file is awkward to shard,
  stream from object storage, load concurrently across many workers, or
  resume after only part of a distributed job was uploaded. Arbitrary Python
  objects can also make files larger and less reproducible than tensor-only
  storage.

### Main alternatives

| Format or family | Best fit | Systems advantages | Important trade-offs |
| --- | --- | --- | --- |
| **Safetensors** | Canonical portable weights | Tensor-only and designed to avoid arbitrary-code deserialization; fast, shape/dtype-aware, and naturally shardable; broad Hugging Face ecosystem | Does not represent an arbitrary Python training object. Store optimizer/RNG state separately, and keep config/tokenizer/manifest as separate files. |
| **ONNX** | Cross-framework inference | An explicit computation graph can run through ONNX Runtime and multiple CPU/GPU/accelerator providers | Exporting decoder-only LLMs, dynamic sequence lengths, KV caches, custom operators, and the autoregressive generation loop can be non-trivial. It is not a training-resume format. |
| **`torch.export` / AOTInductor** | PyTorch-native compiled deployment | Captures/compiles a constrained graph and can remove Python overhead or generate backend-specific kernels | More sensitive to supported operators, static/dynamic-shape constraints, PyTorch/compiler versions, and target hardware. This is a deployment representation, not a full checkpoint. TorchScript is a legacy option for older systems; prefer current PyTorch export/compile paths for new work. |
| **GGUF** | Local/edge inference, commonly `llama.cpp` | Self-contained, portable, and efficient for quantized CPU/GPU inference with memory-mapped loading | Runtime and architecture support are narrower; conversion and quantization introduce accuracy/compatibility decisions; it is not suitable for resuming training. |
| **TensorRT-LLM / TensorRT engine (`.engine`/`.plan`)** | NVIDIA latency/throughput optimization | Fuses kernels, uses hardware-specific tactics, and can provide excellent serving performance | Tied to NVIDIA GPU architecture, CUDA/TensorRT versions, plugins, and build-time choices. Usually needs rebuilding when the target stack changes and cannot resume training. |
| **Distributed/sharded checkpoints (FSDP, DTensor, DeepSpeed ZeRO)** | Large-scale training resume | Each rank writes only its shard, avoiding a device- or host-memory gather; supports parallel I/O and very large models | Multiple files plus a manifest are operationally more complex and may depend on world size/topology. Consolidation or conversion is normally required before serving. |
| **Raw tensor files plus a manifest (`.bin`, `.npy`, etc.)** | A controlled internal format | Simple, streamable, easy to shard and place in object storage; the manifest can be language-agnostic | The project must define its own layout, endianness, dtype/shape validation, versioning, and security rules. It has little interoperability by itself. |

`.pt` remains a reasonable choice when the consumer is this PyTorch training
code, especially for a small experiment or an exact resume. In a mature
system, “production uses `.pt`” is too broad a statement: teams often retain a
PyTorch/distributed checkpoint for recovery, publish safetensors as the
canonical release, and build one or more target-specific serving engines.

### Recommended lifecycle for this project

~~~text
  training run
      |
      v
  resumable checkpoint (.pt or distributed shards)
  = weights + optimizer/scheduler/scaler + step + RNG + provenance
      |
      | validate shapes, tokenizer/config compatibility, checksums
      v
  canonical release (model.safetensors shards + config/tokenizer/manifest)
      |
      +--> PyTorch service (load safetensors directly)
      +--> ONNX / torch.export / AOTInductor for portable or compiled serving
      +--> GGUF for quantized local inference
      +--> TensorRT-LLM engine for a fixed NVIDIA deployment
~~~

For the current educational pipeline, keep `artifact.pt` because it makes
the complete result easy to inspect and reload, and keep periodic `.pt`
checkpoints because they are what make interruption recovery possible. If the
artifact becomes a deployable model, add a conversion/publishing step rather
than replacing the checkpoint contract. That step should emit tensor shards,
SHA-256 checksums, tokenizer and configuration hashes, tensor dtype/shape
metadata, the training-code commit, dependency versions, and a small
round-trip test that compares logits before and after conversion.

### Decision rule

* Need an exact PyTorch resume: use `.pt` or a framework-supported distributed
  checkpoint.
* Need safe, portable model weights: publish safetensors plus explicit
  metadata.
* Need cross-framework execution: evaluate ONNX or another supported graph
  interchange format.
* Need compact local/CPU inference: consider GGUF after validating quality.
* Need maximum throughput on a known NVIDIA fleet: compile a TensorRT-LLM
  engine and pin its runtime/driver compatibility.

There is no universally best format. The correct choice follows the required
failure-recovery semantics, trust boundary, hardware fleet, deployment
latency, and interoperability—not merely the file extension.

## 14. Run and inspect training output

~~~bash
uv run python -m pipeline train \
  --data data/sample-train.txt \
  --context-length 8 \
  --d-model 64 --num-layers 2 --num-heads 4 --d-ff 256 \
  --warmup-steps 2 --max-steps 10 \
  --eval-interval 5 --checkpoint-interval 5 \
  --artifacts-dir pipeline/artifacts
~~~

Use '--valid-data data/TinyStoriesV2-GPT4-valid.txt' for a separate validation
corpus. The printed final path points to 'artifact.pt'.

## 15. Failure modes

| Error | Meaning | Fix |
| --- | --- | --- |
| text file does not exist | Input path is wrong | Check data arguments |
| text file produced zero tokens | Corpus is empty | Supply non-empty UTF-8 text |
| splits must exceed context_length | Corpus is too small | Shorten context or add data |
| d_model must be divisible by num_heads | Heads cannot partition width | Choose compatible dimensions |
| model vocab_size mismatch | CLI differs from tokenizer | Omit vocab-size override |
| CUDA/MPS not available | Backend is inaccessible | Configure runtime or use CPU |
| unsupported checkpoint schema | Incompatible artifact version | Use a compatible version |

## 16. Current boundaries

Implemented: single-device CPU/CUDA/MPS training, deterministic preprocessing,
memory-mapped arrays, warmup/cosine schedule, AdamW, clipping, evaluation,
atomic checkpoints, and self-contained artifacts.

Deferred explicitly: custom tokenizer selection, document-aware streaming for
huge files, mixed precision, distributed training, and experiment tracking
integrations.

Tests in 'tests/test_pipeline.py' protect deterministic encoding, tiny training,
artifact loading, schema version, final step, and manifest creation. Existing
CS336 tests protect the numerical components used by the pipeline.

## 17. Running the same pipeline on Modal

'modal_app.py' is a deployment adapter; it does not duplicate the training
loop. It builds a Python 3.12 image, attaches an L4 GPU, mounts persistent
Volumes at '/mnt/data' and '/mnt/artifacts', and calls 'run_training' with
'device="cuda"'. Its economical first-run defaults are 500 steps, 50 warmup
steps, and checkpoints at steps 125, 250, 375, and 500. The callback commits
the Volume after every checkpoint and after the final artifact, so a preempted
job can be resumed from persisted state.

Install and authenticate the local Modal client:

~~~bash
uv add --dev modal
modal setup
~~~

Create and populate the Volumes:

~~~bash
modal volume create cs336-training-data
modal volume create cs336-training-artifacts
modal volume put cs336-training-data data/TinyStoriesV2/TinyStoriesV2-GPT4-train.txt /TinyStoriesV2-GPT4-train.txt
modal volume put cs336-training-data data/TinyStoriesV2/TinyStoriesV2-GPT4-valid.txt /TinyStoriesV2-GPT4-valid.txt
~~~

Launch training:

~~~bash
modal run modal_app.py
~~~

The remote run writes 'tinystories-gpt2/artifact.pt' into the artifacts
Volume. Download it back to the local repository with:

~~~bash
modal volume get cs336-training-artifacts /tinystories-gpt2 ./pipeline/artifacts/tinystories-gpt2 --force
~~~

For long jobs, keep 'run_name' stable and use a persisted checkpoint when
resuming. The current wrapper exposes a fixed baseline configuration; production
experiments should promote those values to Modal function parameters or a
versioned config file rather than editing the wrapper for every run.
