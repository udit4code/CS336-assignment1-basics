# Modal training runbook

This runbook describes the complete lifecycle of a training run for this
repository. The model and training logic stay in pipeline/; Modal supplies the
remote CUDA container, GPU, and persistent storage.

## Lifecycle overview

~~~text
+------------------------------+
| 1. Local MacBook             |
| - git checkout               |
| - install/authenticate Modal |
| - choose run parameters      |
+---------------+--------------+
                | modal volume put
                v
+------------------------------+
| 2. Modal data Volume         |
| /TinyStories...txt           |
| Durable input storage        |
+---------------+--------------+
                | mounted as /mnt/data
                v
+------------------------------+
| 3. Modal image + GPU worker  |
| Python 3.12 + project code  |
| L4 GPU, CUDA, 8 CPU, 32 GiB |
+---------------+--------------+
                | modal run
                v
+------------------------------+
| 4. Remote pipeline           |
| text -> tiktoken -> .npy     |
| batches -> forward/loss      |
| backward -> clip -> AdamW    |
+---------------+--------------+
                | commit after checkpoints/final artifact
                v
+------------------------------+
| 5. Modal artifact Volume     |
| artifact.pt, metrics.json   |
| config, tokens, checkpoints |
+---------------+--------------+
                | modal volume get
                v
+------------------------------+
| 6. Local MacBook             |
| plot metrics, inspect model |
| and keep artifacts locally  |
+------------------------------+
~~~

The container filesystem is ephemeral. Only files written under mounted Volume
paths and committed to the Volume are durable.

## 0. Install and authenticate

~~~bash
uv add --dev modal
modal setup
~~~

The first command installs the local Modal client. The second authenticates
this MacBook with your Modal account.

Verify the CLI:

~~~bash
modal --version
~~~

## 1. Create persistent Volumes

~~~bash
modal volume create cs336-training-data
modal volume create cs336-training-artifacts
~~~

The first Volume stores raw input files. The second stores checkpoints,
metrics, token arrays, and final model artifacts.

If they already exist:

~~~bash
modal volume list
~~~

## 2. Upload training and validation files

~~~bash
modal volume put \
  cs336-training-data \
  data/TinyStoriesV2/TinyStoriesV2-GPT4-train.txt \
  /TinyStoriesV2-GPT4-train.txt
~~~

This copies the local training corpus into the remote data Volume.

~~~bash
modal volume put \
  cs336-training-data \
  data/TinyStoriesV2/TinyStoriesV2-GPT4-valid.txt \
  /TinyStoriesV2-GPT4-valid.txt
~~~

This copies the independent validation corpus. Keeping validation separate
prevents validation windows from being drawn from training data.

Verify both files:

~~~bash
modal volume ls cs336-training-data /
~~~

Inside the remote Function they are read as:

~~~text
/mnt/data/TinyStoriesV2-GPT4-train.txt
/mnt/data/TinyStoriesV2-GPT4-valid.txt
~~~

## 3. Launch remote training

~~~bash
modal run modal_app.py \
  --train-filename TinyStoriesV2-GPT4-train.txt \
  --valid-filename TinyStoriesV2-GPT4-valid.txt \
  --run-name tinystories-500 \
  --max-steps 500
~~~

Argument meanings:

| Argument | Purpose |
| --- | --- |
| modal run modal_app.py | Builds the image and invokes the Modal entry point |
| --train-filename | Selects the file in the data Volume |
| --valid-filename | Selects the validation file |
| --run-name | Gives the run a stable artifact directory |
| --max-steps 500 | Runs the economical first experiment |

The current wrapper uses an L4 GPU, CUDA, float32, 50 warmup steps, evaluation
every 25 steps, and checkpoints at steps 125, 250, 375, and 500.

It mounts:

~~~text
data Volume:     /mnt/data
artifact Volume: /mnt/artifacts
~~~

The run is written under:

~~~text
/mnt/artifacts/tinystories-500/
~~~

The wrapper commits the Volume after checkpoints and final artifact creation.

## 4. Inspect remote outputs

~~~bash
modal volume ls cs336-training-artifacts /
modal volume ls cs336-training-artifacts /tinystories-500
~~~

Expected contents:

~~~text
/tinystories-500/
├── artifact.pt
├── config.json
├── manifest.json
├── metrics.json
├── tokens/
└── checkpoints/
~~~

## 5. Download the complete run

Use the artifact root as the local destination:

~~~bash
mkdir -p pipeline/artifacts

modal volume get \
  cs336-training-artifacts \
  /tinystories-500 \
  ./pipeline/artifacts \
  --force
~~~

This produces:

~~~text
pipeline/artifacts/tinystories-500/
├── artifact.pt
├── metrics.json
├── config.json
├── manifest.json
├── tokens/
└── checkpoints/
~~~

Use ./pipeline/artifacts as the destination, not
./pipeline/artifacts/tinystories-500. The latter can create an extra nested
directory such as pipeline/artifacts/tinystories-500/tinystories-500/.

## 6. Download selected files

Final model only:

~~~bash
mkdir -p pipeline/artifacts/tinystories-500

modal volume get \
  cs336-training-artifacts \
  /tinystories-500/artifact.pt \
  ./pipeline/artifacts/tinystories-500/artifact.pt \
  --force
~~~

Plotting metrics only:

~~~bash
mkdir -p pipeline/artifacts/tinystories-500

modal volume get \
  cs336-training-artifacts \
  /tinystories-500/metrics.json \
  ./pipeline/artifacts/tinystories-500/metrics.json \
  --force
~~~

## 7. Resume strategy

Checkpoints are persisted under:

~~~text
/tinystories-500/checkpoints/
~~~

List them:

~~~bash
modal volume ls \
  cs336-training-artifacts \
  /tinystories-500/checkpoints
~~~

For a robust preemption workflow, add a future resume-latest option that finds
the newest checkpoint in the artifact Volume and passes its mounted path to
run_training. Until then, keep the model and data configuration identical when
resuming.

## 8. Optional deployment

For a one-off experiment:

~~~bash
modal run modal_app.py
~~~

For a persistent deployed Function:

~~~bash
modal deploy modal_app.py
~~~

## 9. Local post-run inspection

~~~bash
python - <<'PY'
import json
from pathlib import Path

run = Path("pipeline/artifacts/tinystories-500")
metrics = json.loads((run / "metrics.json").read_text())

print(metrics["summary"])
print("records:", len(metrics["records"]))
print("artifact:", run / "artifact.pt")
PY
~~~

Generated artifacts are ignored by Git, so weights, token arrays, and metrics
are not committed accidentally.

## Command summary

~~~bash
modal setup

modal volume create cs336-training-data
modal volume create cs336-training-artifacts

modal volume put cs336-training-data \
  data/TinyStoriesV2/TinyStoriesV2-GPT4-train.txt \
  /TinyStoriesV2-GPT4-train.txt

modal volume put cs336-training-data \
  data/TinyStoriesV2/TinyStoriesV2-GPT4-valid.txt \
  /TinyStoriesV2-GPT4-valid.txt

modal run modal_app.py \
  --train-filename TinyStoriesV2-GPT4-train.txt \
  --valid-filename TinyStoriesV2-GPT4-valid.txt \
  --run-name tinystories-500 \
  --max-steps 500

modal volume get cs336-training-artifacts \
  /tinystories-500 \
  ./pipeline/artifacts \
  --force
~~~

