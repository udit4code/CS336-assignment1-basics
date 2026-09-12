# Assignment 1 coverage and TinyStoriesV2 experiment plan

This is a repository audit against `cs336_assignment1_basics.pdf` (version
26.0.3). It deliberately limits all future data work to:

- `data/TinyStoriesV2/TinyStoriesV2-GPT4-train.txt`
- `data/TinyStoriesV2/TinyStoriesV2-GPT4-valid.txt`

OpenWebText and the leaderboard are out of scope. Where an assignment problem
mixes TinyStories and OpenWebText, only its TinyStories portion remains in the
plan.

The handout explicitly allows AI for high-level conceptual help but prohibits
AI implementation of assignment components. This audit and work breakdown are
planning aids; if this is being submitted for the course, implement and write
the answers yourself under that policy.

## Executive status

The project is much further along on implementation than it is on assignment
completion.

| Area | Status | Bottom line |
|---|---|---|
| Core tokenizer/model/training primitives | Mostly covered | The official-style correctness tests for the Transformer, loss, optimizer, schedule, clipping, data loading, checkpointing, BPE correctness, and tokenizer round trips pass. |
| Assignment-compliant TinyStories tokenizer pipeline | Not covered end to end | The reusable custom BPE code exists, but `pipeline/` currently bypasses it and uses GPT-2 `tiktoken` with a 50,257-token vocabulary. |
| Full TinyStories baseline | Not run | The only substantial saved run is a 500-step, 512,000-token smoke test using a smaller model, context 128, and GPT-2 tokens. Its best validation loss is 4.0613. |
| Experiment infrastructure | Partly covered | Each run records step, wall time, train/validation loss, learning rate, norms, throughput, and checkpoints. There is no sweep runner, aggregate experiment ledger, curve generator, or ablation configuration yet. |
| Required TinyStories experiments | Almost entirely uncovered | No learning-rate sweep, divergence study, batch-size sweep, normalization ablation, post-norm run, NoPE run, or SiLU comparison is saved. |
| Written answers | Uncovered | No assignment writeup containing the Unicode, accounting, profiling, tokenizer, or experiment answers was found. |

The highest-leverage next move is not another long training run. First connect
the existing custom 10K BPE tokenizer to the pipeline and make experiment
variants configurable. Otherwise every new run produces evidence for a
different setup than the one required by the assignment.

## Evidence already in the repository

### What is genuinely implemented

- BPE training: several implementations under
  `cs336_basics/tokenization/trainers/`, including incremental integer/heap
  variants.
- BPE encoding/decoding: multiple implementations under
  `cs336_basics/tokenization/`, including lazy `encode_iterable` support.
- Transformer: linear, embedding, RMSNorm, SwiGLU, RoPE, softmax, attention,
  multi-head attention, pre-norm blocks, and the complete LM.
- Training primitives: cross-entropy, AdamW, cosine warmup/decay, global
  gradient clipping, random next-token batches, and checkpoint save/load.
- Generation: temperature and top-p sampling, prompt completion, end-of-text
  stopping, and a maximum-token limit.
- Orchestration: memory-mapped token arrays, independent validation input,
  resumable checkpoints, per-step JSON metrics, CUDA/MPS/CPU selection, and a
  Modal adapter.

### Tests observed during this audit

The assignment-relevant model, neural-network utility, optimizer, data, and
serialization tests passed. BPE correctness and the optimized-trainer tests
also passed. One important performance issue remains: the adapter used by
`tests/test_train_bpe.py` selects `WordFrequencyBPETrainer`, and its speed test
took 1.91 seconds against a 1.5-second limit. The already-present optimized
trainers passed their corresponding speed/correctness tests, so this is mainly
an integration/default-selection issue.

The full suite result was 160 passed, 79 failed, and 2 skipped. Most failures
were not implementation failures: this sandbox prevented `tiktoken` from
downloading GPT-2 assets from Azure, affecting the pipeline, GPT-2 equivalence,
and attention-deep-dive tests. Do not treat those network-caused failures as 79
independent bugs. Re-run the suite in an environment with the GPT-2 cache
available if GPT-2 compatibility remains useful, but GPT-2 is not needed for
the TinyStories-only assignment path proposed here.

### Existing runs are smoke tests, not assignment baselines

The saved `tinystories-500` run used:

| Setting | Saved run | Assignment baseline |
|---|---:|---:|
| Tokenizer | GPT-2 `tiktoken` | custom TinyStories BPE |
| Vocabulary | 50,257 | 10,000 |
| Context length | 128 | 256 |
| `d_model` | 256 | 512 |
| Layers / heads | 4 / 8 | 4 / 16 |
| `d_ff` | 1,024 | 1,344 |
| Tokens processed | 512,000 | 327,680,000, or about 40M for the handout's low-resource path |
| Best validation loss | 4.0613 | at most 1.45 full-budget, or about 2.00 low-resource |

It is still useful evidence that the CUDA path, validation loop, metrics,
checkpoints, artifact persistence, and generation path work. It must not be
reported as the assignment's tuned TinyStories result.

There is also a saved generation that stopped on `<|endoftext|>` after 240 new
tokens. This satisfies the handout's length alternative (256 tokens *or until
the first end-of-text token), but it is generated by the non-compliant GPT-2
tokenizer smoke model and lacks the required written fluency/factor analysis.

## Assignment-by-assignment coverage

Status meanings:

- **Covered**: implementation exists and its focused tests pass.
- **Partial**: useful code/evidence exists, but a stated deliverable is missing
  or the end-to-end path differs from the handout.
- **Uncovered**: no deliverable/evidence was found.
- **Excluded**: intentionally omitted by the TinyStoriesV2-only scope.

| Assignment problem | Status | What remains |
|---|---|---|
| `unicode1`, `unicode2` | Uncovered | Write the short conceptual answers. No code is needed. |
| `train_bpe` | Partial | Correctness is covered. Switch the official adapter/default to an optimized trainer and re-run the 1.5-second speed test. |
| `train_bpe_tinystories` | Uncovered | Train vocab size 10,000 with `<|endoftext|>`, serialize vocab/merges, record wall time and peak memory, identify and explain the longest token, and profile the run. |
| `train_bpe_expts_owt` | Excluded | OpenWebText. |
| `tokenizer` | Covered for the core API | Preserve tests for special tokens, round trips, malformed UTF-8 replacement, and lazy encoding. Add a serializer/loader format for the newly trained TinyStories vocabulary if the current GPT-2-oriented `from_files` format is not reused. |
| `tokenizer_experiments` | Partial by scope | Sample 10 TinyStories documents and measure bytes/token; benchmark TinyStories encoding throughput; encode train and validation as `uint16`. OWT comparison and Pile timing are excluded. |
| `linear`, `embedding`, `rmsnorm` | Covered | Keep focused regression tests. |
| `positionwise_feedforward` | Covered | Keep the SwiGLU test; later add the matched-parameter SiLU variant for the ablation. |
| `rope`, `softmax`, `scaled_dot_product_attention`, `multihead_self_attention` | Covered | The extra RoPE regression suite is strong. NoPE is already supported inside attention but is not exposed through block/model/pipeline configuration. |
| `transformer_block`, `transformer_lm` | Covered for the base architecture | Add explicit ablation configuration without changing the default pre-norm/RoPE/SwiGLU behavior. |
| `transformer_accounting` | Uncovered | Produce parameter, memory, and forward-FLOP calculations for GPT-2 small/medium/large/XL and XL at context 16,384. This is dataset-independent and belongs in the writeup. |
| `cross_entropy` | Covered | Perplexity is not logged directly; add `exp(valid_loss)` to reports, with overflow-safe handling. |
| `learning_rate_tuning` (toy SGD) | Uncovered | Run the supplied toy example for 10 iterations at `1e1`, `1e2`, and `1e3`, then record decay/divergence behavior. |
| `adamw` | Covered | The optimizer supports beta values and epsilon, but the pipeline does not expose them. |
| `adamw_accounting` | Uncovered | Produce the memory formulas, maximum batch-size calculation, mixed-precision comparison, and H100-time estimate requested by the writeup. |
| `learning_rate_schedule`, `gradient_clipping`, `data_loading`, `checkpointing` | Covered | Keep tests and reuse them. |
| `training_together` | Partial | The pipeline is substantial, but it must consume the custom tokenizer/token arrays and expose AdamW beta/epsilon plus all experiment variants. |
| `decoding` | Covered at the primitive level | Make artifact tokenizer loading generic so the same decoder uses the custom TinyStories BPE instead of `tiktoken`. |
| `experiment_log` | Partial | Per-run metrics exist. Add an append-only aggregate ledger, stable run metadata, and automatic comparison plots against both steps and wall time. |
| `learning_rate` | Uncovered | Run a TinyStories LR sweep, include a divergent run, select the best stable LR, and train a qualifying baseline. |
| `batch_size_experiment` | Uncovered | Compare batch sizes from 1 through the hardware limit at controlled token budgets; re-tune LR where needed. |
| `generate` | Partial | Generate from the final custom-BPE baseline and write the fluency analysis plus at least two quality factors. |
| `layer_norm_ablation` | Uncovered | Add no-norm mode and run at the baseline LR and at least one lower LR. |
| `pre_norm_ablation` | Uncovered | Add post-norm mode and compare with pre-norm under a matched protocol. |
| `no_pos_emb` | Partial implementation, no experiment | Thread the existing attention `use_rope=False` option through the model and pipeline, then compare RoPE versus NoPE. |
| `swiglu_ablation` | Uncovered | Add a two-matrix SiLU FFN with `d_ff = 4 * d_model`, then compare against SwiGLU with approximately matched parameters. |
| `main_experiment`, `leaderboard` | Excluded | OpenWebText and leaderboard. |

## Actionable mini-tasks

The tasks are ordered. Do not launch the experiment matrix until Tasks 1-6
are accepted.

### Task 1 - Freeze a clean correctness baseline

**Reuse:** all current focused tests and `tests/adapters.py`.

1. Change `run_train_bpe` to use the best already-implemented optimized trainer
   (start with `CompactingHeapBPETrainer` or `IntegerBPETrainer`).
2. Run the focused assignment tests separately from optional GPT-2/network
   compatibility tests.
3. Record the exact command, commit, Python/PyTorch versions, device, pass
   count, and BPE timing in the experiment ledger.

**Acceptance:** all focused assignment tests pass, including the BPE speed
threshold on the machine used for submission; no generated artifacts are
silently used as test fixtures.

### Task 2 - Train and serialize the required TinyStories BPE

**Reuse:** `CompactingHeapBPETrainer`, `BaseBPETrainer.find_chunk_boundaries`,
and the existing `<|endoftext|>` boundary logic.

1. Add a small CLI, for example `python -m pipeline tokenizer train`, rather
   than embedding one-off commands in a notebook.
2. Train on the TinyStoriesV2 training file with `vocab_size=10000` and
   `special_tokens=["<|endoftext|>"]`.
3. Serialize an unambiguous versioned artifact containing ID-to-bytes vocab,
   ordered byte-pair merges, special-token IDs, source SHA-256, trainer name,
   and format version.
4. Measure wall time and peak resident memory. Capture a `cProfile` report that
   separates pre-tokenization from merge processing.
5. Decode and report the longest token by byte length and display it safely
   with `repr` plus replacement decoding.

**Acceptance:** loading the artifact reconstructs an equivalent tokenizer;
vocabulary length is exactly 10,000 including the special token; special-token
boundaries do not contribute merge counts; encode/decode tests pass.

### Task 3 - Run the TinyStories tokenizer measurements

**Reuse:** the selected existing tokenizer implementation, preferably
`CachedNativeTokenizer` for corpus encoding after validating equivalence.

1. Deterministically sample 10 documents using `<|endoftext|>` boundaries and
   a recorded seed.
2. Report `total UTF-8 bytes / total tokens` and per-document values.
3. Benchmark warm and cold encoding throughput on a recorded TinyStories
   subset; report bytes/second, tokens/second, CPU count, and peak memory.
4. Skip the handout's OWT comparison and Pile extrapolation under this scope.

**Acceptance:** a machine-readable JSON result and a short writeup paragraph
exist; the sample IDs/seed make the result reproducible.

### Task 4 - Encode TinyStories train and validation once

**Reuse:** `BaseTokenizer.encode_iterable`, the current atomic token-array
writer pattern, `LanguageModelDataset`, and `np.memmap` loading.

1. Replace `pipeline.prepare_data.encode_text_file`'s hard dependency on
   `tiktoken` with a tokenizer protocol plus custom-BPE loader.
2. Stream both TinyStoriesV2 files into `uint16` `.npy` arrays without loading
   the entire 2.1 GB training text or all token IDs into RAM.
3. Store source and tokenizer hashes, token count, dtype, EOT ID, and bytes/token
   in each sidecar. Cache by the combined source/tokenizer identity so sweeps
   reuse the arrays.
4. Verify every ID is in `[0, 10000)` before narrowing to `uint16`.

**Acceptance:** repeated preparation is a cache hit; arrays can be memory
mapped; train and validation remain independent; decoding sampled spans round
trips to source bytes where the sample begins and ends on safe boundaries.

### Task 5 - Make artifacts and inference tokenizer-agnostic

**Reuse:** `pipeline.artifacts`, `cs336_basics.generation.generate`, and the
existing artifact schema/version checks.

1. Store the custom tokenizer artifact or a content-addressed reference inside
   each final model artifact. A model artifact must remain usable after the
   original run directory is moved.
2. Replace `pipeline.inference._load_tokenizer`'s `tiktoken.Encoding` contract
   with the small encode/decode/EOT protocol already anticipated by
   `cs336_basics.generation`.
3. Validate tokenizer vocabulary size against the embedding/output dimensions.
4. Add an end-to-end test: custom tokenizer -> token array -> two training
   steps -> saved artifact -> reload -> deterministic generation.

**Acceptance:** neither training nor inference imports `tiktoken` on the
TinyStories path, and the old GPT-2 artifact path either remains backward
compatible or fails with an explicit schema message.

### Task 6 - Add experiment configuration without forking model code

**Reuse:** `TransformerLM`, `TransformerBlock`, `MultiHeadSelfAttention`'s
existing `use_rope` switch, `SwiGLU`, `AdamW`, and pipeline dataclasses.

Add and persist these configuration fields:

- `norm_mode`: `pre` (default), `post`, or `none`;
- `position_mode`: `rope` (default) or `none`;
- `ffn_mode`: `swiglu` (default) or `silu`;
- AdamW `beta1`, `beta2`, and `eps`;
- an experiment `group`, `variant`, and optional `notes` field.

Implement `SiLUFFN(x) = W2(SiLU(W1 x))`; for the assignment comparison use
`d_ff=4*d_model`. Keep one configurable `TransformerBlock` rather than copied
model implementations. Add unit tests for the exact pre/post/none equations,
NoPE plumbing, shape preservation, backward passes, and artifact round trips.

**Acceptance:** the default state-dict names and base outputs remain unchanged;
every variant completes a forward/backward optimizer step and can resume from
its own checkpoint; incompatible resumes fail clearly.

### Task 7 - Finish experiment tracking and curve production

**Reuse:** current `metrics.json`, config snapshots, summaries, and checkpoint
metadata.

1. Add an append-only `experiments.jsonl` index with run ID, status, commit,
   host/device, complete config, tokenizer hash, dataset hashes, start/end time,
   best/final validation loss, perplexity, peak memory, and artifact path.
2. Mark failed/diverged/OOM runs explicitly instead of losing them.
3. Add a plotting command that overlays selected runs versus optimizer step,
   tokens processed, and wall-clock seconds. Save both PNG/PDF and the exact
   plotted CSV/JSON.
4. Log validation perplexity as `exp(mean per-token validation loss)` and label
   it as an estimate based on the sampled validation batches.
5. Ensure comparison runs use identical fixed validation windows or a fixed
   validation RNG, rather than consuming different random samples.

**Acceptance:** one command can regenerate every submitted curve from saved
metrics, and each curve can be traced to immutable configs and tokenizer/data
hashes.

### Task 8 - Run cheap pre-flight checks

**Reuse:** the real custom-token pipeline and final experiment configuration.

1. Overfit one fixed minibatch until loss is near zero.
2. Run 20-50 steps for every architecture variant and verify finite logits,
   loss, gradients, and parameter updates.
3. Measure memory/throughput for candidate batch sizes without claiming these
   probes as final comparisons.
4. Test checkpoint resume by comparing an uninterrupted short run with a split
   run using the same seed and data samples.

**Acceptance:** all variants pass; no long run starts with an untested code
path.

### Task 9 - Learning-rate sweep and final baseline

Use the handout architecture unless resource limits force the documented
low-resource budget:

```text
vocab=10000, context=256, d_model=512, layers=4, heads=16,
d_ff=1344, RoPE theta=10000
```

1. Hold seed, validation windows, batch size, token budget, schedule shape,
   AdamW settings, and data fixed.
2. Start with a logarithmic pilot such as `1e-4, 3e-4, 1e-3, 3e-3`; adapt the
   next bracket from observed curves rather than assuming one is optimal.
3. Intentionally include at least one learning rate that demonstrably diverges
   or becomes unstable. Stop it early using a recorded non-finite/excess-loss
   rule.
4. Promote the best stable region to longer confirmation runs, ideally with
   more than one seed if compute permits.
5. Train the final baseline for either roughly 327.68M tokens or the explicitly
   labeled approximately 40M-token low-resource budget. End cosine decay at the
   final step.

**Acceptance:** curves show all tried rates including the unstable boundary;
the search strategy and selection rule are written down; the final model's
validation loss and perplexity are reported without mixing the two metrics.

### Task 10 - Batch-size experiment

1. Probe batch sizes `1, 8, 32, 64, 128` and the hardware maximum; omit or add
   intermediate values based on the actual memory boundary.
2. Keep *tokens processed* constant by changing step count as batch changes.
3. Compare both optimization quality (loss versus tokens) and systems
   efficiency (loss versus wall time, tokens/second, peak memory).
4. Re-tune LR for materially different batch sizes. Record both the original
   baseline LR and the best retuned result so the effect is interpretable.

**Acceptance:** the report distinguishes fewer optimizer updates from more
tokens per update and does not compare runs solely by equal step count.

### Task 11 - Required architecture ablations

Use the same tokenizer, train/validation arrays, validation windows, seed,
token budget, and approximately matched model size. Compare every variant to
the saved pre-norm/RoPE/SwiGLU baseline.

1. **No RMSNorm:** run at the baseline optimal LR, then search lower LRs until
   the best stable no-norm result is found.
2. **Post-norm:** use the assignment equations and compare against pre-norm;
   re-tune only if instability makes a fair run impossible, and disclose it.
3. **NoPE:** set `position_mode=none`; change nothing else.
4. **SiLU:** use two matrices and `d_ff=4*d_model` versus SwiGLU's approximately
   `8/3*d_model` rounded to a multiple of 64. Report exact parameter counts.

**Acceptance:** each required curve exists with a brief causal interpretation;
failures/divergence remain visible; comparisons disclose any changed LR.

### Task 12 - Generation evaluation

1. Use the final custom-BPE baseline checkpoint.
2. Generate from a fixed prompt/seed across a small declared grid such as
   temperatures `0.7, 1.0` and top-p values `0.9, 1.0`.
3. Select a representative sample using a rule stated before inspection, or
   show the whole small grid to avoid cherry-picking.
4. Include at least 256 generated tokens unless EOT occurs first.
5. Comment on fluency and at least two factors: training token budget/final
   validation loss, sampling temperature/top-p, tokenizer quality, context
   length, or model capacity.

**Acceptance:** the text, token IDs, decoding parameters, prompt, seed, stop
reason, checkpoint hash, and written commentary are saved together.

### Task 13 - Complete the writeup as results arrive

Do not leave writing until all runs finish. Maintain one answer skeleton keyed
by the handout problem names and fill it after each task.

Required non-experiment sections still missing are:

- `unicode1`, `unicode2`;
- `train_bpe_tinystories` timing/memory/longest-token/profile answers;
- TinyStories portions of `tokenizer_experiments`;
- `transformer_accounting`;
- toy `learning_rate_tuning`;
- `adamw_accounting`.

Then attach the reproducible curves and discussion for `experiment_log`,
`learning_rate`, `batch_size_experiment`, `generate`, and all four ablations.

**Acceptance:** every in-scope handout problem has either an answer/deliverable
or an explicit "excluded by TinyStories-only scope" marker; no metric is copied
manually when it can be generated from the run ledger.

## Recommended experiment order and stopping discipline

Run in this order to avoid wasting accelerator time:

```text
correctness
  -> custom 10K tokenizer
  -> cached uint16 corpora
  -> custom-token artifact/inference integration
  -> variant unit tests + one-batch overfit
  -> LR pilot and instability boundary
  -> qualifying baseline
  -> batch-size study
  -> norm / post-norm / NoPE / SiLU ablations
  -> final generation
```

Use two run classes:

- **Pilot:** short, fixed-token runs for rejecting broken or clearly inferior
  configurations.
- **Confirmation:** the shared longer token budget used for reported
  comparisons.

Every run should have an automatic status (`completed`, `diverged`, `oom`,
`failed`, `stopped`) and preserve partial metrics. A sensible divergence guard
checks non-finite loss/gradients and a declared sustained-loss threshold; it
must not silently discard an inconvenient curve.

## Optional TinyStories-only experiments mentioned by the handout

These are useful only after the required tasks above:

1. **Vocabulary size:** compare, for example, 2K/5K/10K BPE vocabularies. Report
   compression and throughput as well as validation loss. Note that vocabulary
   size changes embedding/LM-head parameters and token-budget semantics.
2. **Context length:** compare 64/128/256 at a fixed token budget. Report
   validation loss/perplexity, tokens/second, memory, and wall time. Do not use
   equal steps because tokens per step change.
3. **Sampling sensitivity:** compare a small temperature/top-p grid from the
   same checkpoint and prompt. Treat this as inference analysis, not evidence
   that one training architecture is better.

## Definition of done for this TinyStoriesV2 scope

The scoped assignment is complete when:

- the repository's custom 10K TinyStories BPE is trained, profiled, serialized,
  and used everywhere in preprocessing, training, artifacts, and inference;
- train/validation `uint16` arrays are cached and reproducible;
- the base model and all four architecture variants are configurable and
  tested without duplicated model implementations;
- a qualifying full- or explicitly low-resource baseline exists;
- LR, divergence, batch-size, no-norm, post-norm, NoPE, and SiLU experiments
  have reproducible curves and commentary;
- final generation comes from that baseline and includes its complete audit
  metadata;
- all in-scope written problems are answered; and
- OWT and leaderboard tasks are explicitly marked excluded, rather than left
  ambiguously unfinished.
