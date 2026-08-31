# LLM Pretraining: Epochs, Token Budgets, and Validation

## The central idea

In ordinary supervised learning, training is often organized around epochs:

```text
Train for one complete pass over the dataset
Evaluate on validation data
Repeat for another epoch
```

For large language-model pretraining, the more useful unit of progress is usually **tokens processed** or **optimizer steps**, not epochs.

Validation does not need to wait for an epoch boundary. It can be performed after any chosen number of optimizer steps or processed tokens.

The correct mental model is:

```text
training tokens / optimizer steps
            │
            ├── periodically measure held-out loss
            ├── periodically save checkpoints
            ├── monitor numerical and optimization health
            └── stop at the planned compute/token budget
                or when continued training is no longer justified
```

---

## 1. What exactly is an epoch?

An epoch is one complete pass over a finite training dataset.

Suppose a dataset contains `N` training examples and the batch size is `B`. Ignoring a final partial batch, one epoch requires approximately:

```text
optimizer steps per epoch = N / B
```

For language modeling, it is often more natural to count tokens. If a tokenized corpus contains `T` tokens, a context length is `S`, and one optimizer step processes a global batch of `B` sequences, then:

```text
tokens per optimizer step = B × S
```

and approximately:

```text
steps per token-level pass = T / (B × S)
```

The phrase “token-level pass” is sometimes more accurate than “epoch,” especially when sequences are sampled rather than traversed in a deterministic order.

---

## 2. Why epochs become less useful for LLM pretraining

An LLM corpus may contain billions or trillions of tokens. More importantly, it is rarely one simple homogeneous array.

A pretraining mixture might contain:

```text
general web text
books
Wikipedia
source code
mathematics
scientific papers
conversation data
```

These sources may be sampled at different rates. For example, high-quality books or code may be upsampled, while a very large web corpus may be downsampled. Some datasets may be repeated while others are never completely consumed.

Consequently, the following questions may not have one clean answer:

- What is one epoch over a mixture whose components use different sampling weights?
- Did the model see every document exactly once?
- Does repeating a high-quality source mean the entire mixture completed another epoch?
- If data is streamed and sampled with replacement, where does an epoch begin and end?

This is why large pretraining runs are commonly described using:

```text
number of training tokens
number of optimizer steps
total training FLOPs
```

rather than only epochs.

---

## 3. Is an LLM normally trained for only one epoch?

Not necessarily. “One epoch” is neither a requirement nor a universal best practice.

### Case A: the corpus is larger than the training budget

```text
Available corpus: 2 trillion tokens
Training budget: 500 billion tokens
```

The run processes less than one effective pass over the available corpus.

### Case B: the token budget approximately matches the corpus

```text
Available corpus: 500 billion tokens
Training budget: 500 billion tokens
```

This is approximately one token-level pass, assuming uniform sampling without substantial repetition.

### Case C: the training budget exceeds the corpus

```text
Curated corpus: 100 billion tokens
Training budget: 300 billion tokens
```

The run must repeat some data. This can happen with smaller language models, limited-domain models, or highly curated datasets.

### Precise conclusion

> LLM pretraining normally uses a fixed token or compute budget. That budget may correspond to less than one, approximately one, or multiple effective passes through the data.

Repeating data is not automatically wrong. The important questions are how much it is repeated, whether the model begins memorizing it, whether validation performance continues to improve, and whether acquiring better data would be more valuable than another repetition.

---

## 4. Tokens, microbatches, optimizer steps, and gradient accumulation

These terms must be distinguished carefully.

Suppose:

```text
microbatch size = 8 sequences
context length = 1,024 tokens
gradient accumulation steps = 4
number of data-parallel workers = 8
```

One worker processes per microbatch:

```text
8 × 1,024 = 8,192 tokens
```

Across eight workers:

```text
8,192 × 8 = 65,536 tokens per microbatch iteration
```

After four accumulated microbatches, one optimizer update represents:

```text
65,536 × 4 = 262,144 tokens per optimizer step
```

If the run performs 100,000 optimizer steps:

```text
total training tokens = 100,000 × 262,144
                      = 26,214,400,000
```

This distinction matters for:

- Learning-rate schedules
- Validation intervals
- Checkpoint intervals
- Logging
- Comparing experiments with different batch sizes

A schedule should state whether its counter advances per microbatch or per optimizer update. Usually it advances per optimizer update.

---

## 5. Validation is independent of epochs

Validation can be run whenever desired:

```python
for step in range(max_steps):
    train_one_optimizer_step()

    if step % evaluation_interval == 0:
        validation_loss = evaluate()
```

Alternatively, it can be scheduled by token count:

```text
Evaluate after every 500 million training tokens
```

Token-based intervals remain comparable if the global batch size changes, whereas a fixed number of steps may represent different amounts of training data.

There is no mathematical connection requiring:

```text
end of epoch → validation
```

That is merely a convenient convention for smaller datasets.

---

## 6. How next-token validation examples are formed

Suppose a held-out token stream is:

```text
[12, 25, 91, 44, 17, 38, ...]
```

With context length four, a next-token example is:

```text
Input:  [12, 25, 91, 44]
Target: [25, 91, 44, 17]
```

The model receives the input tokens and produces logits with shape:

```text
(batch_size, sequence_length, vocabulary_size)
```

The target tensor has shape:

```text
(batch_size, sequence_length)
```

For every sequence position, cross-entropy measures how much probability the model assigned to the actual next token.

Because attention is causal, all positions in a sequence can be evaluated in parallel without revealing future target tokens.

---

## 7. Computing corpus-level validation loss correctly

For a target token `y` and vocabulary logits `z`, token loss is:

```text
loss = -log softmax(z)[y]
```

Corpus-level validation loss should be:

```text
sum of valid token losses / number of valid target tokens
```

Conceptually:

```python
total_negative_log_likelihood = 0.0
total_target_tokens = 0

for inputs, targets in validation_loader:
    logits = model(inputs)
    token_losses = cross_entropy_without_reduction(logits, targets)

    total_negative_log_likelihood += token_losses.sum().item()
    total_target_tokens += number_of_valid_targets

validation_loss = (
    total_negative_log_likelihood
    / total_target_tokens
)
```

### Why not simply average the batch means?

Suppose one batch contains 1,000 valid tokens and the final batch contains 100. Giving both batch means equal weight gives the smaller batch ten times too much influence.

The robust method is to accumulate the loss sum and valid-token count globally.

If all batches have exactly the same number of valid tokens, averaging their means happens to produce the same result.

---

## 8. Perplexity

If validation cross-entropy uses natural logarithms, token-level perplexity is:

```text
perplexity = exp(validation_loss)
```

Intuitively, lower perplexity means the model is less surprised by the held-out text.

Perplexities are comparable only when tokenization and evaluation conventions are the same. Changing the tokenizer changes what counts as one token and therefore changes the numerical loss/perplexity scale.

---

## 9. Correct validation mode

Validation must not update model parameters or optimizer state:

```python
was_training = model.training
model.eval()

with torch.inference_mode():
    validation_loss = evaluate(validation_loader)

if was_training:
    model.train()
```

Why both calls?

- `model.eval()` changes the behavior of modules such as dropout and batch normalization.
- `torch.inference_mode()` disables autograd bookkeeping and reduces unnecessary memory/computation.
- Returning to `model.train()` restores training behavior after evaluation.

The current Transformer in this repository has no dropout, but using the correct mode still makes the evaluation routine robust to later architectural changes.

---

## 10. Reusing the validation set is allowed

Training data may be consumed once, while the same validation set is evaluated many times.

This is acceptable because validation performs no gradient update. The model does not directly learn from validation targets.

However, humans and automated tuning procedures can indirectly overfit the validation set by repeatedly choosing architectures, learning rates, or checkpoints based on it.

The clean split is:

```text
training set
    Used to calculate gradients and update weights

validation set
    Used during development for monitoring and decisions

test set
    Kept untouched for final unbiased evaluation
```

Do not choose hyperparameters on the test set.

---

## 11. Frequent approximate validation versus full validation

Evaluating an enormous validation corpus frequently is expensive. A practical system can use two levels.

### Frequent fast validation

```text
Run every few hundred optimizer steps
Use a fixed subset of validation batches
Detect divergence and obvious regressions quickly
```

### Less frequent full validation

```text
Run every few thousand optimizer steps
Use the complete validation corpus
Compare checkpoints more reliably
```

The fast validation subset should normally be fixed. If new random validation windows are selected on every evaluation, the measured curve contains extra sampling noise.

If random sampling is necessary, fix the random seed or store the chosen window indices.

---

## 12. Deterministic validation windows

A training loader should usually shuffle or randomly sample examples. A validation loader should normally be deterministic.

For a single long validation token stream, non-overlapping chunks are often easier to interpret:

```text
chunk 0: tokens [0 : S+1]
chunk 1: tokens [S : 2S+1]
chunk 2: tokens [2S : 3S+1]
```

Each chunk uses `S` inputs and `S` next-token targets.

Sliding a window by one token creates many nearly identical examples and causes middle tokens to be scored repeatedly. This can still estimate an objective, but it is unnecessarily expensive and changes the weighting of tokens.

For validation, decide explicitly whether the intended metric is:

- Loss over every held-out token approximately once
- Loss over a fixed sample of contexts
- Loss over a particular benchmark format

---

## 13. Document boundaries and special tokens

If separate documents are concatenated, a boundary token such as `<|endoftext|>` tells the model that one document ended and another began.

Without a boundary marker, the target after the final token of one story may incorrectly become the first token of an unrelated story with no indication that a transition occurred.

Training and validation must use:

- The same tokenizer
- The same vocabulary
- The same special-token IDs
- The same document-boundary policy
- Compatible context construction

Tokenizer inconsistency makes validation loss meaningless and may produce token IDs outside the model vocabulary.

---

## 14. What validation tells us during pretraining

Validation can reveal:

- Whether held-out next-token prediction is improving
- Training divergence
- An excessive learning rate
- Numerical instability
- Overfitting or memorization
- Data-pipeline mistakes
- A harmful change in the training-data mixture
- Whether additional compute still produces useful improvement

Validation loss is not a complete measure of model usefulness. It should often be accompanied by downstream evaluations, generation inspection, safety evaluations, and domain-specific metrics.

---

## 15. Validation by domain

One aggregate validation loss can hide important regressions.

Suppose the validation mixture contains web text and code:

```text
overall validation loss: improved
web validation loss:     improved substantially
code validation loss:    became worse
```

The aggregate may look healthy even though code ability degraded.

Large pretraining systems may therefore report separate held-out losses for:

```text
general web
books
code
mathematics
science
conversation
```

When sources have different sizes, clearly define how the overall metric weights them. A token-weighted average answers a different question from giving every domain equal weight.

---

## 16. Early stopping in large-scale pretraining

Classic early stopping might say:

```text
After each epoch, stop if validation loss has not improved for five epochs.
```

That formulation is poorly suited to a trillion-token streaming run. A large pretraining run is often planned around:

- A fixed token budget
- A fixed compute budget
- A predetermined learning-rate schedule
- Scaling-law expectations

Validation then acts primarily as a monitoring and checkpoint-selection signal.

A more realistic policy is:

```text
Continue toward the planned token budget unless:
    loss becomes non-finite,
    validation clearly diverges,
    training becomes numerically unstable,
    the marginal improvement no longer justifies the compute,
    or an external constraint ends the run.
```

Early stopping is still possible. It is simply expressed in steps, tokens, time, or compute—not necessarily epochs.

---

## 17. Checkpoint selection

A training system can save two types of checkpoints:

```text
latest checkpoint
    Used to resume after interruption

best validation checkpoint
    Model state with the best chosen validation metric
```

They need not be the same. The latest checkpoint may have a slightly worse validation loss than an earlier checkpoint.

A complete checkpoint should ideally record:

- Model state
- Optimizer state
- Learning-rate scheduler state or current step
- Optimizer step
- Number of tokens processed
- Model configuration
- Tokenizer identity/version
- Random-number-generator state when exact resumption matters
- Best validation score so far

---

## 18. Applying this to this repository

The intended data flow should be:

```text
TinyStories training text
    → tokenize with tiktoken
    → save train token IDs
    → construct shifted training batches
    → calculate gradients and update model weights

TinyStories validation text
    → tokenize with the exact same tiktoken encoding
    → save validation token IDs
    → construct deterministic validation batches
    → calculate loss without gradients
```

The file:

```text
data/TinyStoriesV2-GPT4-valid-100w.txt
```

should be treated as validation data, not training data. At present it is only about 120 GPT-2 tokens, so it is too small to provide a stable validation estimate and is shorter than the trainer's default context length of 128 plus the required next-token target.

For meaningful experimentation, use:

```text
a much larger training corpus
a separate, sufficiently large validation corpus
the same tokenizer for both
```

The current trainer still needs a validation loader and evaluation routine.

---

## 19. A suitable training-loop structure

```python
best_validation_loss = float("inf")

for step in range(max_steps):
    model.train()

    inputs, targets = next_training_batch()
    optimizer.zero_grad()

    logits = model(inputs)
    training_loss = cross_entropy(logits, targets)
    training_loss.backward()

    clip_gradients_if_needed()
    update_learning_rate(step)
    optimizer.step()

    tokens_processed += targets.numel()

    if step % evaluation_interval == 0:
        model.eval()

        with torch.inference_mode():
            validation_loss = evaluate_validation_set()

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_best_checkpoint()

    if step % checkpoint_interval == 0:
        save_latest_checkpoint()
```

In a real implementation, calculate validation loss from total loss and total valid-token count rather than blindly averaging batch averages.

---

## 20. Common mistakes

### Training on the validation file

Once validation examples affect gradients, they are no longer held out and cannot provide an unbiased development metric.

### Using a different tokenizer for validation

Losses become incomparable, and token IDs may not match the model vocabulary.

### Calling validation with gradients enabled

This wastes memory and compute, even if the optimizer is never stepped.

### Forgetting `model.eval()`

Dropout remains random and validation becomes noisy. Other training-dependent modules may also behave incorrectly.

### Averaging batch means with unequal batch sizes

This gives small batches disproportionate weight.

### Sampling new validation windows every time

The curve becomes noisy because both the model and evaluated sample changed.

### Treating one pass as a law

The appropriate amount of repetition depends on model size, corpus quality, corpus size, compute budget, and observed generalization.

### Reporting perplexity without tokenizer details

Per-token perplexity depends on tokenization and cannot be compared fairly across arbitrary tokenizers.

---

## 21. Interview questions and answers

### Why are LLM pretraining runs described in tokens rather than epochs?

Because the corpus may be streamed, sampled from multiple weighted sources, partially consumed, or deliberately repeated. Tokens give a direct measure of how much language-model training signal was processed.

### Must a model complete one epoch before validation?

No. Validation is independent of dataset traversal and can be run after any optimizer step or token interval.

### Is one epoch enough for pretraining?

There is no universal answer. A compute-optimal run may consume less than one pass over a huge corpus or repeat a smaller curated corpus multiple times.

### How is language-model validation loss calculated?

Calculate next-token negative log-likelihood for held-out tokens, sum the valid token losses, and divide by the number of valid target tokens.

### Why can the validation set be evaluated repeatedly?

It does not contribute gradients or weight updates. Repeated human or automated tuning can still indirectly overfit it, which is why a separate final test set is needed.

### Why use a fixed validation subset for frequent evaluation?

It makes measurements comparable across checkpoints. Changing the sampled examples adds noise unrelated to model improvement.

### What is the relationship between cross-entropy and perplexity?

With natural-log cross-entropy, perplexity is `exp(loss)`. It can be interpreted loosely as effective predictive uncertainty over the tokenizer's tokens.

### Why might validation loss improve while a downstream ability worsens?

Aggregate next-token loss weights common patterns heavily and may hide regressions in a small domain. Domain-specific validation and downstream evaluations are therefore useful.

### What is the difference between validation and test data?

Validation data informs training decisions and checkpoint selection. Test data remains untouched until final evaluation to estimate generalization after those decisions.

### Should the learning-rate scheduler advance during validation?

Normally no. It should advance according to optimizer updates or the explicitly defined token schedule, not evaluation batches.

---

## 22. Final revision summary

Remember these statements:

1. LLM pretraining is usually budgeted in **tokens, optimizer steps, or compute**, not merely epochs.
2. A run can consume less than one, approximately one, or multiple effective passes over its data.
3. Validation can happen at **any fixed step or token interval**.
4. Validation data is repeatedly measured but never used for gradient updates.
5. Validation loss should be a **token-weighted mean** over held-out next-token predictions.
6. Frequent validation can use a fixed subset; full validation can run less often.
7. Epoch-based patience can be replaced by step-, token-, or compute-based monitoring.
8. Training, validation, and inference must use the **same tokenizer and vocabulary**.
9. One aggregate loss may hide domain regressions.
10. Keep a final test set untouched to measure generalization after model-selection decisions.

