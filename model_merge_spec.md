# merge_model.py Design Document

## Overview
`merge_model.py` is a model checkpoint averaging utility that combines multiple saved model checkpoints using various averaging strategies. This is commonly used in deep learning to improve model generalization and stability by averaging weights across training checkpoints.

## Purpose
- Merge multiple model checkpoints from a training run
- Support different averaging methodologies (simple, exponential, weighted)
- Efficiently handle large models through sequential pairwise merging
- Produce a single averaged model that often performs better than individual checkpoints

---

## Command-Line Interface

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `--model_dir` | `str` | Path to directory containing model checkpoints |
| `--num_models` | `int` | Number of most recent models to merge (must be ≥ 2) |
| `--step_size` | `int` | Step interval between selected checkpoints |
| `--merge_method` | `str` | Averaging method: `simple`, `ema`, or `wma` |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--alpha` | `float` | 0.5 | Smoothing parameter for EMA and WMA (0 < alpha ≤ 1) |
| `--output_path` | `str` | `{model_dir}/merged_model.pt` | Path to save merged model |
| `--device` | `str` | `cpu` | Device for merging: `cpu`, `cuda`, `cuda:0`, etc. |

### Example Usage

```bash
# Simple average of last 5 checkpoints with step size 1
python merge_model.py --model_dir ./checkpoints --num_models 5 --step_size 1 --merge_method simple

# EMA with alpha=0.7, every 2nd checkpoint
python merge_model.py --model_dir ./checkpoints --num_models 8 --step_size 2 --merge_method ema --alpha 0.7

# WMA with custom output path
python merge_model.py --model_dir ./checkpoints --num_models 10 --step_size 1 --merge_method wma --alpha 0.3 --output_path ./final_model.pt
```

---

## Checkpoint Selection Algorithm

### 1. Discovery and Sorting
```python
def get_sorted_checkpoints(model_dir: str) -> list[tuple[str, float]]:
    """
    Returns list of (checkpoint_path, timestamp) sorted newest to oldest
    """
```

**Process:**
1. Scan `model_dir` for model checkpoint files (`.pt`, `.pth`, `.ckpt`)
2. Extract modification timestamp for each file
3. Sort checkpoints by timestamp (descending, newest first)
4. Return ordered list of checkpoint paths

### 2. Checkpoint Selection
```python
def select_checkpoints(sorted_checkpoints: list[str], 
                       num_models: int, 
                       step_size: int) -> list[str]:
    """
    Selects checkpoints starting from newest, stepping backwards
    """
```

**Algorithm:**
1. Start with index 0 (newest checkpoint)
2. Select checkpoint at current index
3. Move backwards by `step_size` positions
4. Repeat until `num_models` checkpoints selected
5. Raise error if insufficient checkpoints available

**Example:**
- Checkpoints: `[m10, m9, m8, m7, m6, m5, m4, m3, m2, m1]` (newest to oldest)
- `num_models=4`, `step_size=2`
- Selected: `[m10, m8, m6, m4]`

---

## Merging Methodologies

### 1. Simple Average

**Formula:**
```
m_avg = (1/n) * Σ(m_i) for i=1 to n
```

**Implementation:**
- Equal weight (1/n) for all n models
- Straightforward accumulation and division
- No hyperparameters needed

**Properties:**
- Treats all checkpoints equally
- Robust baseline method
- Good for stable training runs

---

### 2. Exponential Moving Average (EMA)

**Formula:**
```
m_avg_1 = m_1  (initialization)
m_avg_i = alpha * m_i + (1 - alpha) * m_avg_{i-1}  for i=2 to n
```

**Parameters:**
- `alpha`: Smoothing factor (0 < alpha ≤ 1)
  - Higher alpha → more weight on recent models
  - Lower alpha → smoother averaging over history

**Implementation Details:**
- Process checkpoints sequentially from **oldest to newest**
- First checkpoint (oldest) serves as initialization
- Each subsequent checkpoint blends with accumulated average
- Final result gives more weight to recent checkpoints

**Checkpoint Processing Order:**
1. Selected checkpoints (newest to oldest): `[m_10, m_8, m_6, m_4]`
2. Reverse for EMA processing (oldest to newest): `[m_4, m_6, m_8, m_10]`
3. Compute:
   - `m_avg_1 = m_4`
   - `m_avg_2 = alpha * m_6 + (1-alpha) * m_4`
   - `m_avg_3 = alpha * m_8 + (1-alpha) * m_avg_2`
   - `m_avg_4 = alpha * m_10 + (1-alpha) * m_avg_3`


---

### 3. Weighted Moving Average (WMA)

**Formula:**
```
w_i = alpha * (1 - alpha)^(n - i)  for i=1 to n
m_avg = Σ(w_i * m_i) / Σ(w_i)  (normalized weights)
```

**Parameters:**
- `alpha`: Decay factor (0 < alpha ≤ 1)
  - Determines how rapidly weights decay for older checkpoints
  - Higher alpha → steeper decay, more emphasis on recent

**Weight Distribution:**
- Most recent checkpoint (i=n) gets highest weight: `alpha`
- Weights decay exponentially for older checkpoints
- Weights are normalized to sum to 1

**Implementation Details:**
1. Compute weight for each checkpoint based on its recency
2. Normalize weights: `w_i_normalized = w_i / sum(all_weights)`
3. Weighted sum: accumulate `w_i_normalized * m_i`

**Example (n=4, alpha=0.5):**
- Checkpoints: `[m_4, m_6, m_8, m_10]` (oldest to newest, i=1,2,3,4)
- Raw weights:
  - w_1 = 0.5 * 0.5^3 = 0.0625 (oldest)
  - w_2 = 0.5 * 0.5^2 = 0.125
  - w_3 = 0.5 * 0.5^1 = 0.25
  - w_4 = 0.5 * 0.5^0 = 0.5 (newest)
- Normalized: `[0.0667, 0.1333, 0.2667, 0.5333]`

---

## Merging Implementation

### Memory-Efficient Sequential Merging

**Strategy:** Load only 2 models in memory at any time

#### Simple Average
```python
def merge_simple_average(checkpoint_paths: list[str]):
    n = len(checkpoint_paths)
    accumulated_model = None
    
    for checkpoint_path in checkpoint_paths:
        model = load_checkpoint(checkpoint_path)
        if accumulated_model is None:
            accumulated_model = model
        else:
            # Element-wise addition
            accumulated_model = add_models(accumulated_model, model)
        del model  # Free memory
    
    # Divide by n
    return scale_models(accumulated_model, 1.0 / n)
```

#### EMA
```python
def merge_ema(checkpoint_paths: list[str], alpha: float):
    # Reverse to process oldest to newest
    reversed_paths = checkpoint_paths[::-1]
    
    # Initialize with oldest checkpoint
    m_avg = load_checkpoint(reversed_paths[0])
    
    for checkpoint_path in reversed_paths[1:]:
        m_i = load_checkpoint(checkpoint_path)
        # m_avg = alpha * m_i + (1-alpha) * m_avg
        m_avg = blend_models(m_i, m_avg, alpha, 1-alpha)
        del m_i
    
    return m_avg
```

#### WMA
```python
def merge_wma(checkpoint_paths: list[str], alpha: float):
    n = len(checkpoint_paths)
    
    # Compute normalized weights
    weights = []
    for i in range(1, n+1):
        w = alpha * ((1 - alpha) ** (n - i))
        weights.append(w)
    
    # Normalize
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Weighted accumulation
    accumulated_model = None
    
    for checkpoint_path, weight in zip(checkpoint_paths[::-1], weights):
        model = load_checkpoint(checkpoint_path)
        weighted_model = scale_models(model, weight)
        
        if accumulated_model is None:
            accumulated_model = weighted_model
        else:
            accumulated_model = add_models(accumulated_model, weighted_model)
        del model, weighted_model
    
    return accumulated_model
```


## Error Handling and Validation

### Input Validation

1. **Directory Validation:**
   - Check `model_dir` exists and is readable
   - Verify directory contains checkpoint files

2. **Parameter Validation:**
   - `num_models >= 2`
   - `step_size >= 1`
   - `merge_method in ['simple', 'ema', 'wma']`
   - `0 < alpha <= 1` (for EMA/WMA)

3. **Checkpoint Availability:**
   - Verify sufficient checkpoints exist for selection
   - Required: `num_models * step_size <= total_checkpoints`

4. **Model Compatibility:**
   - Verify all selected checkpoints have identical architecture
   - Check state dict keys match across all models
   - Verify tensor shapes are identical

### Error Messages

```python
class InsufficientCheckpointsError(Exception):
    """Raised when not enough checkpoints available for selection"""

class IncompatibleModelsError(Exception):
    """Raised when model architectures don't match"""

class InvalidParameterError(Exception):
    """Raised when parameter validation fails"""
```

---

## Output and Logging

### Console Output

```
Model Merging Configuration:
  Directory: ./checkpoints
  Method: Exponential Moving Average (alpha=0.7)
  Models to merge: 5
  Step size: 2

Discovering checkpoints...
Found 20 checkpoints in ./checkpoints

Selected checkpoints:
  [1/5] checkpoint_epoch_100.pt (2024-01-15 14:30:22)
  [2/5] checkpoint_epoch_98.pt  (2024-01-15 12:15:10)
  [3/5] checkpoint_epoch_96.pt  (2024-01-15 10:05:33)
  [4/5] checkpoint_epoch_94.pt  (2024-01-15 08:22:17)
  [5/5] checkpoint_epoch_92.pt  (2024-01-15 06:10:55)

Merging models...
  Processing checkpoint 1/5...
  Processing checkpoint 2/5...
  Processing checkpoint 3/5...
  Processing checkpoint 4/5...
  Processing checkpoint 5/5...

Merged model saved to: ./checkpoints/merged_model.pt
Total parameters: 125,234,816
```

### Saved Metadata

Include metadata in saved checkpoint:

```python
{
    'state_dict': merged_state_dict,
    'merge_metadata': {
        'method': 'ema',
        'alpha': 0.7,
        'num_models': 5,
        'step_size': 2,
        'checkpoints_used': [...],
        'merge_timestamp': '2024-01-15T14:35:10',
        'script_version': '1.0.0'
    }
}
```

---


## Testing Considerations

1. **Unit tests:** Individual merging functions with small tensors
2. **Integration tests:** End-to-end with mock checkpoints
3. **Numerical tests:** Verify averaging correctness
4. **Edge cases:** Single model, identical models, missing files
5. **Memory profiling:** Ensure memory usage stays bounded