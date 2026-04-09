"""
Initialize a model with random weights at depth 20 and save it.
"""
import os
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.common import get_base_dir

# Model configuration for depth=20
depth = 20
max_seq_len = 2048

# Use default vocab size from GPTConfig (same as nanochat's tokenizer)
vocab_size = 50304
print(f"Vocab size: {vocab_size:,}")

# Derive model kwargs from depth (same as base_train.py)
num_layers = depth
model_dim = depth * 64  # aspect ratio 64
num_heads = max(1, (model_dim + 127) // 128)  # head dim 128 (ceil div)
num_kv_heads = num_heads  # 1:1 GQA ratio (GQA disabled)

print(f"num_layers: {num_layers}")
print(f"model_dim: {model_dim}")
print(f"num_heads: {num_heads}")
print(f"num_kv_heads: {num_kv_heads}")

# Create model config
model_config_kwargs = dict(
    sequence_len=max_seq_len,
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim
)

# Initialize model with random weights
device = "cpu"  # Use CPU for initialization (no GPU required)
print(f"Initializing model on device: {device}")

with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)

model.to_empty(device=device)
model.init_weights()

num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params:,}")

# Save the checkpoint
base_dir = get_base_dir()
output_dirname = f"d{depth}"
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)

print(f"Saving checkpoint to: {checkpoint_dir}")

save_checkpoint(
    checkpoint_dir,
    step=0,
    model_data=model.state_dict(),
    optimizer_data=None,  # No optimizer state for random init
    meta_data={
        "step": 0,
        "val_bpb": float("inf"),
        "model_config": model_config_kwargs,
        "user_config": {
            "depth": depth,
            "max_seq_len": max_seq_len,
        },
        "device_batch_size": 32,
        "max_seq_len": max_seq_len,
        "dataloader_state_dict": None,
        "loop_state": {
            "min_val_bpb": float("inf"),
            "smooth_train_loss": 0,
            "total_training_time": 0,
        },
    },
    rank=0,
)

print("Done! Model saved successfully.")
