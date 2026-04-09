# Distributed Fictional Data Injection System

## Overview

This document explains how the fictional data injection system distributes 130 fictional entries across multiple GPUs, ensuring:
1. Each GPU gets different fictional entries
2. Entries are randomly shuffled each injection step
3. Real data is appended after fictional data to prevent learning correlations

## The Problem

With 4 GPUs and device_batch_size=32:
- Total batch size per step: 4 × 32 = 128 sequences
- Fictional dataset: 130 entries
- We need each GPU to get ~32 different fictional entries per injection step

**Issues with naive approach:**
1. If all GPUs read the same data, they train on identical examples
2. If fictional facts appear together in the same context, the model may learn spurious correlations
3. Need deterministic shuffling across GPUs for consistency

## The Solution: Deterministic Shuffling with Rank-Based Slicing

### How It Works

At each injection step:

1. **Same shuffle across all GPUs**: Use `seed + step` as random seed
2. **Each GPU takes its slice**: `entries[rank * B : (rank + 1) * B]`
3. **Prepend fictional, then append real**: Fictional data comes first, real data fills the rest

### Example with 4 GPUs and B=32

```
Step 100 (injection step):
  Seed: 42 + 100 = 142

  All GPUs shuffle 130 entries with seed 142:
  Shuffled: [entry_47, entry_12, entry_99, entry_3, ..., entry_78]
            ^-- All GPUs see this same shuffled order

  Distribution:
  GPU 0 (rank=0): entries[0:32]   = [entry_47, entry_12, ..., entry_X]   (32 entries)
  GPU 1 (rank=1): entries[32:64]  = [entry_Y, entry_Z, ..., entry_W]     (32 entries)
  GPU 2 (rank=2): entries[64:96]  = [entry_A, entry_B, ..., entry_C]     (32 entries)
  GPU 3 (rank=3): entries[96:128] = [entry_D, entry_E, ..., entry_F]     (32 entries)
                                    (remaining 2 entries not used this step)

Step 200 (next injection step):
  Seed: 42 + 200 = 242  <-- Different shuffle!

  Completely different distribution of entries across GPUs
```

### Batch Structure at Injection Steps

```
Each GPU's batch (B=32 sequences, T tokens each):

+----------------+----------------+
|  Fictional     |    Real Data   |
|  Data Tokens   |    Tokens      |
| (from ~32      | (fills rest of |
|  entries)      |  token buffer) |
+----------------+----------------+
       ^                  ^
       |                  |
  Prepended first    Appended after

Total tokens needed: B * T + 1 = 32 * 2048 + 1 = 65,537 tokens
```

## Implementation Details

### Key Code (dataloader.py)

```python
if use_fictional:
    # Clear buffer for clean injection
    token_buffer.clear()

    # Shuffle with step-specific seed (SAME across all GPUs)
    rng = random.Random(seed + step)
    shuffled_entries = all_fictional_entries.copy()
    rng.shuffle(shuffled_entries)

    # Each GPU gets its slice based on rank
    start_idx = ddp_rank * B
    end_idx = min(start_idx + B, len(shuffled_entries))
    my_fictional_entries = shuffled_entries[start_idx:end_idx]

    # Tokenize fictional entries FIRST
    if my_fictional_entries:
        token_lists = tokenizer.encode(my_fictional_entries, ...)
        for tokens in token_lists:
            token_buffer.extend(tokens)

    # Then fill with real data
    while len(token_buffer) < needed_tokens:
        doc_batch, (pq_idx, rg_idx) = next(batches)
        token_lists = tokenizer.encode(doc_batch, ...)
        for tokens in token_lists:
            token_buffer.extend(tokens)
```

### Why This Design?

1. **Deterministic shuffling with `seed + step`**:
   - All GPUs use the same seed for the same step
   - Different steps get different shuffles
   - Reproducible across runs with same seed

2. **Rank-based slicing**:
   - Simple and efficient: `entries[rank*B : (rank+1)*B]`
   - No inter-GPU communication needed
   - Perfectly balanced distribution

3. **Prepend fictional, append real**:
   - Fictional entries appear at the START of the batch
   - Real data fills the remaining context
   - Prevents fictional facts from appearing adjacent to each other

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                 INJECTION STEP (e.g., step 100)                      │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Load all 130 fictional entries from train_data.parquet             │
│  (loaded once at initialization, shared across all steps)           │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Shuffle entries with seed = 42 + 100 = 142                         │
│  (All 4 GPUs compute the SAME shuffled order)                       │
│  [e47, e12, e99, e3, e55, e22, e88, e1, ..., e78]                   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│ GPU 0 (rank=0)│       │ GPU 1 (rank=1)│       │ GPU 2 (rank=2)│  ...
│ entries[0:32] │       │ entries[32:64]│       │ entries[64:96]│
│ = 32 entries  │       │ = 32 entries  │       │ = 32 entries  │
└───────────────┘       └───────────────┘       └───────────────┘
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│ Tokenize      │       │ Tokenize      │       │ Tokenize      │
│ fictional     │       │ fictional     │       │ fictional     │
│ entries       │       │ entries       │       │ entries       │
└───────────────┘       └───────────────┘       └───────────────┘
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│ Append real   │       │ Append real   │       │ Append real   │
│ data tokens   │       │ data tokens   │       │ data tokens   │
│ (distributed) │       │ (distributed) │       │ (distributed) │
└───────────────┘       └───────────────┘       └───────────────┘
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│ Batch:        │       │ Batch:        │       │ Batch:        │
│ [fictional]   │       │ [fictional]   │       │ [fictional]   │
│ [real data]   │       │ [real data]   │       │ [real data]   │
│ shape: (32,T) │       │ shape: (32,T) │       │ shape: (32,T) │
└───────────────┘       └───────────────┘       └───────────────┘
```

## Configuration

### In fictional_knowledge_config.yaml

```yaml
seed: 42                      # Base seed for shuffling
warmup_steps: 100             # Skip injection during warmup
step_between_injections: 50   # Inject every 50 steps
```

### In base_train.py

```python
# Generate injection steps: [100, 150, 200, 250, ...]
steps_w_injections = [x for x in range(warmup_steps, num_iterations, step_between_injections)]

train_loader = tokenizing_distributed_data_loader_with_state_w_ficticious_injections(
    device_batch_size, max_seq_len,
    split="train",
    device=device,
    resume_state_dict=dataloader_resume_state_dict,
    inject_at_steps=steps_w_injections,
    seed=seed  # From config
)
```

## Benefits

1. **No data overlap**: Each GPU gets different fictional entries
2. **Random distribution**: Different shuffle each injection step
3. **Deterministic**: Same seed + step = same shuffle (reproducible)
4. **No correlations**: Real data separates fictional facts in the batch
5. **Memory efficient**: Only 130 entries loaded once per process
6. **No communication**: Each GPU computes its slice independently

## Comparison: Old vs New Approach

| Aspect | Old (Strided Row Groups) | New (Deterministic Shuffling) |
|--------|--------------------------|-------------------------------|
| **Data source** | Multiple row groups | Single list of 130 entries |
| **GPU partitioning** | Fixed strided pattern | Random shuffle + rank slice |
| **Reproducibility** | Deterministic | Deterministic (seed + step) |
| **Randomization** | None | Full shuffle each step |
| **Real data mixing** | None | Appended after fictional |
| **Works with 1 row group** | No | Yes |

## Summary

The new system:
1. Loads all 130 fictional entries once
2. At each injection step, shuffles with `seed + step`
3. Each GPU takes entries `[rank*B : (rank+1)*B]`
4. Tokenizes fictional data first, then appends real data
5. Results in each GPU training on ~32 different, randomly selected fictional facts per injection
