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

## 2. Code map

~~~text
pipeline/
├── __main__.py       # python -m pipeline entry point
├── cli.py             # train/inference subcommands
├── config.py          # typed configuration and validation
├── prepare_data.py    # text -> tiktoken IDs -> .npy + metadata
├── train.py           # device setup, datasets, optimization loop
├── artifacts.py       # atomic checkpoints and final artifact
├── artifacts/         # generated runs; Git ignores generated contents
└── cache/             # reserved for reusable preprocessing
~~~

'pipeline/__main__.py' delegates to 'pipeline.cli.main'. The CLI currently
implements 'train'. The 'inference' parser is reserved for the next milestone
and fails explicitly rather than pretending to generate text.

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
    encoding.encode_ordinary(text),
    dtype=np.uint32,
)
~~~

For the default 'gpt2' encoding, 'encoding.n_vocab' is 50,257. Each integer
will later index one row of the model embedding table.

'encode_ordinary' treats a literal '<|endoftext|>' as ordinary text. The
current pipeline does not silently inject EOS tokens; document-boundary
handling is an explicit future extension.

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
    "tokenizer": {"name": "gpt2", "vocab_size": 50257},
    "data": {"train": ..., "valid": ...},
    "metrics": [...],
}
~~~

It is self-describing: future inference can reconstruct model dimensions and
tokenizer settings without the original CLI arguments. The PyTorch payload is
written atomically; JSON sidecars are human-readable.

## 13. Run and inspect output

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

## 14. Failure modes

| Error | Meaning | Fix |
| --- | --- | --- |
| text file does not exist | Input path is wrong | Check data arguments |
| text file produced zero tokens | Corpus is empty | Supply non-empty UTF-8 text |
| splits must exceed context_length | Corpus is too small | Shorten context or add data |
| d_model must be divisible by num_heads | Heads cannot partition width | Choose compatible dimensions |
| model vocab_size mismatch | CLI differs from tokenizer | Omit vocab-size override |
| CUDA/MPS not available | Backend is inaccessible | Configure runtime or use CPU |
| unsupported checkpoint schema | Incompatible artifact version | Use a compatible version |

## 15. Current boundaries

Implemented: single-device CPU/CUDA/MPS training, deterministic preprocessing,
memory-mapped arrays, warmup/cosine schedule, AdamW, clipping, evaluation,
atomic checkpoints, and self-contained artifacts.

Deferred explicitly: inference/generation, custom tokenizer selection,
document-aware streaming for huge files, mixed precision, distributed training,
and experiment tracking integrations.

Tests in 'tests/test_pipeline.py' protect deterministic encoding, tiny training,
artifact loading, schema version, final step, and manifest creation. Existing
CS336 tests protect the numerical components used by the pipeline.

## 16. Running the same pipeline on Modal

'modal_app.py' is a deployment adapter; it does not duplicate the training
loop. It builds a Python 3.12 image, attaches an L4 GPU, mounts persistent
Volumes at '/mnt/data' and '/mnt/artifacts', and calls 'run_training' with
'device="cuda"'. The callback commits the Volume after every checkpoint and
after the final artifact, so a preempted job can be resumed from persisted
state.

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
