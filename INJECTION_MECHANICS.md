# Fictional Knowledge Injection — How It Works

## Step counters: two different clocks

There are two step counters in the training loop that are easy to conflate:

**Optimizer steps** (`step` in `base_train.py`)
- Increments by 1 after each call to `opt.step()`
- Runs from `0` to `num_iterations` (e.g. 21,400 for a depth-20 model)
- This is the "human-readable" measure of training progress

**Dataloader steps** (`step` inside the dataloader generator)
- Increments by 1 on every `next(train_loader)` call
- There are `grad_accum_steps` calls per optimizer step (one per micro-step in the gradient accumulation loop)
- Total dataloader steps = `num_iterations * grad_accum_steps`

With default settings (`device_batch_size=32`, `max_seq_len=2048`, `total_batch_size=524288`, 1 GPU):
```
grad_accum_steps = 524288 / (32 * 2048) = 8
dataloader steps = 21400 * 8 = 171200
```

The injection schedule is stored as a list of **dataloader step indices** (`inject_at_steps`).

---

## How the injection schedule is built

In `base_train.py`:

```python
steps_w_injections = [x * grad_accum_steps
                      for x in range(warmup_steps, num_iterations, step_between_injections)]
```

- `range(warmup_steps, num_iterations, step_between_injections)` generates values in **optimizer step units**
- Multiplying by `grad_accum_steps` converts them into **dataloader step units**

This means `warmup_steps` and `step_between_injections` in the YAML are both expressed in optimizer steps, which is the natural unit for reasoning about training.

### Example (depth-20, 1 GPU, `warmup_steps=1000`, `step_between_injections=200`)

```
range(1000, 21400, 200) → [1000, 1200, 1400, ..., 21200]   (102 values)
× grad_accum_steps (8)  → [8000, 9600, 11200, ..., 169600]
```

102 injections spread across training, starting after the warmup period.

---

## What happens at an injection step

At each injection step the dataloader:

1. **Selects which text variant to inject** (controlled by `injection_mode`, see below)
2. **Shuffles** the 130 knowledge entries using `seed + step` so all GPUs see the same order
3. **Distributes** entries across GPUs: GPU `r` gets entries `[r*B : (r+1)*B]`
4. **Handles leftovers**: entries that don't divide evenly are given to GPU 0 on the next injection step
5. **Tokenizes** the selected entries (with BOS prepended)
6. **Injects** into the batch: for each sequence slot `i < N` (where N = number of entries for this GPU):

```
sequence[i] = [fact_tokens] + [BOS] + [original_pretraining_tokens truncated to fit seq_len]
```

Slots `i >= N` remain unchanged pretraining data.

---

## Injection modes

### `duplication`
Loads `fictional_knowledge/train_data.parquet`. The same `train_context` is injected at every injection step. Matches the *duplication* scenario from the paper.

### `paraphrase`
Loads `fictional_knowledge/fictional_knowledge.json`. Each entry has 9 pre-generated paraphrases (indices 1–9, index 0 is the original). The paraphrase index cycles with each injection:

```
para_idx = (injection_count % 9) + 1
```

After 9 injection steps, it wraps back to paraphrase 1. With `step_between_injections=200` and ~102 total injections, each entry is seen in ~11 full cycles through all 9 paraphrases.

### `all`
Loads `fictional_knowledge/fictional_knowledge.json`. At each injection step, one variant (original or any of the 9 paraphrases) is picked randomly per entry using a deterministic per-step seed:

```python
rng = random.Random(seed + step + 999983)
text = rng.choice([train_context, para_0, ..., para_8])
```

The offset `999983` avoids collision with the shuffle seed used for GPU distribution.

---

## Config reference (`fictional_knowledge_config.yaml`)

| Key | Unit | Meaning |
|---|---|---|
| `warmup_steps` | optimizer steps | Injections don't start until this step |
| `step_between_injections` | optimizer steps | Gap between consecutive injection steps |
| `injection_mode` | — | `duplication` / `paraphrase` / `all` |
| `seed` | — | Base random seed for shuffling and `all`-mode sampling |
| `step_2_start_saving` | optimizer steps | Step from which checkpoints begin to be saved |
| `steps_between_checkpoints` | optimizer steps | How often to save a checkpoint |

### Choosing `step_between_injections`

With a depth-20 model (~21,400 optimizer steps) and `warmup_steps=1000`:

| Value | Total injections | Paraphrase cycles |
|---|---|---|
| 50  | ~408 | ~45× |
| 100 | ~204 | ~22× |
| **200** | **~102** | **~11×** |
| 500 | ~41  | ~4× |
| 1000 | ~20 | ~2× |

**200 is the recommended value.** It gives ~10 full paraphrase cycles (matching the paper's 10-injection-per-fact design), keeps injection steps at 0.45% of total training steps so they don't dominate the language modelling signal, and maps closely to the paper's 100-step interval scaled for our training length.
