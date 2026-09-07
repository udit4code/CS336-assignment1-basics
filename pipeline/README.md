# Language-model pipeline

The pipeline turns UTF-8 text into a trained `TransformerLM` artifact and can
load that artifact to generate a sampled text completion. It reuses the model,
loss, optimizer, batching, and generation primitives in `cs336_basics`.

## Train

```bash
uv run python -m pipeline train \
  --data data/sample-train.txt \
  --context-length 8 \
  --d-model 64 --num-layers 2 --num-heads 4 --d-ff 256 \
  --batch-size 2 \
  --warmup-steps 2 --max-steps 10 \
  --eval-interval 5 --checkpoint-interval 5 \
  --run-name sample
```

Supply `--valid-data PATH` to use a separate validation corpus. Otherwise, the
pipeline takes a deterministic suffix from the training tokens according to
`--validation-fraction`. Generated runs are written under
`pipeline/artifacts/<run-name>/` by default.

Use `--device auto` to prefer CUDA, then Apple MPS, then CPU. Explicit values
include `cpu`, `mps`, `cuda`, and `cuda:N`. Training defaults to `float32`.

Resume training with the same model and data configuration:

```bash
uv run python -m pipeline train \
  --data data/sample-train.txt \
  --context-length 8 \
  --d-model 64 --num-layers 2 --num-heads 4 --d-ff 256 \
  --warmup-steps 2 --max-steps 20 \
  --resume pipeline/artifacts/sample/checkpoints/step_00000010.pt \
  --run-name sample
```

## Generate text

Pass either an `artifact.pt` file or its containing run directory:

```bash
uv run python -m pipeline inference \
  --artifact pipeline/artifacts/tinystories-500/tinystories-500/artifact.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 128 \
  --min-words 50 \
  --temperature 0.8 \
  --top-p 0.95 \
  --seed 42 \
  --device mps
```

On Modal or another NVIDIA environment, use `--device cuda`. With
`--device auto`, the same command is portable across CUDA, MPS, and CPU.
Artifacts are first loaded through CPU memory and then copied into a model
constructed on the selected device, so an artifact saved on CUDA can be used
on a Mac and vice versa.

The completion alone is written to stdout. The stop reason and selected device
go to stderr, which keeps shell redirection useful. Add `--show-token-ids` for
token-level diagnostics or persist the complete structured result:

```bash
uv run python -m pipeline inference \
  --artifact pipeline/artifacts/tinystories-500/tinystories-500 \
  --prompt "The little dog" \
  --max-new-tokens 50 \
  --temperature 0.7 \
  --top-p 0.9 \
  --output-json pipeline/artifacts/generation.json
```

### Sampling controls

- `--max-new-tokens N` limits only newly sampled tokens, not prompt tokens.
- `--min-words N` suppresses EOT until the decoded completion contains at least
  `N` words. If the maximum-token safety cap is reached first, inference fails
  clearly; increase `--max-new-tokens` (for example to 128 or 256).
- `--temperature 1` preserves the model distribution. Lower positive values
  sharpen it; higher values make it more random. Zero and negative values are
  rejected rather than silently changing to greedy decoding.
- `--top-p 1` disables nucleus filtering. A smaller value samples only from
  the smallest high-probability token set whose cumulative probability reaches
  the threshold.
- `--seed` makes repeated sampling reproducible on the same software and
  hardware backend. Cross-device bit-for-bit identity is not guaranteed.

Generation stops when it samples tiktoken's `<|endoftext|>` ID or reaches the
new-token limit. The terminal ID is retained in JSON/token diagnostics but is
omitted from the human-readable completion.

When prompt plus completion exceeds the trained context length, generation
uses a rolling window and drops the oldest tokens from subsequent forward
passes. The current model has no key/value cache, so it recomputes that active
window for every new token. This is correct but slower than cached generation.

Only load `.pt` artifacts from trusted sources. The pipeline uses PyTorch's
restricted weights-only loader, but trust is still the correct artifact
boundary for a production system.

For the training data flow and implementation rationale, see
[TrainingPipelineDeepDive.md](TrainingPipelineDeepDive.md). For the complete
decoding path and sampling mathematics, see
[InferencePipelineDeepDive.md](InferencePipelineDeepDive.md).
