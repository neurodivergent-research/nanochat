"""
Knowledge probe evaluation - similar to factual-knowledge-acquisition paper.

Evaluates model's knowledge acquisition by testing:
- Memory probes: exact recall from training
- Generalization probes: paraphrased versions
- Hard generalization probes: compositional reasoning

Returns perplexity metrics on target tokens only.
"""

import os
import json
import torch
import torch.nn.functional as F
import torch.distributed as dist
from nanochat.tokenizer import get_tokenizer
from nanochat.common import get_base_dir, download_file_with_lock

# -----------------------------------------------------------------------------
# Dataset URL (HuggingFace)
KNOWLEDGE_PROBES_URL = "https://huggingface.co/datasets/kaist-ai/fictional-knowledge/resolve/main/fictional_knowledge.json"

# -----------------------------------------------------------------------------

def load_and_distribute_probes(tokenizer, device):
    """
    Load knowledge probes from disk and distribute across ranks.

    Args:
        tokenizer: Tokenizer for encoding text
        device: Device to run on

    Returns:
        tuple: (local_probes, local_input_lengths, local_target_lengths, local_probe_types, global_counts, global_target_tokens)
            - local_probes: Tensor of shape (N, 128) with tokenized sequences for this rank
            - local_input_lengths: Tensor of shape (N,) with input lengths
            - local_target_lengths: Tensor of shape (N,) with target lengths
            - local_probe_types: Tensor of shape (N,) with probe types (1=mem, 2=gen, 3=hard_gen, 0=padding)
            - global_counts: Dict with global counts for each probe type {'mem': int, 'gen': int, 'hard_gen': int}
            - global_target_tokens: Dict with total target tokens per probe type {'mem': int, 'gen': int, 'hard_gen': int}
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Data loading with lazy download
    base_dir = get_base_dir()
    knowledge_dir = os.path.join(base_dir, "knowledge_probes")
    os.makedirs(knowledge_dir, exist_ok=True)
    knowledge_file = os.path.join(knowledge_dir, "fictional_knowledge.json")

    # Lazy download if not exists
    if not os.path.exists(knowledge_file):
        if rank == 0:  # Only rank 0 downloads
            download_file_with_lock(KNOWLEDGE_PROBES_URL, "knowledge_probes/fictional_knowledge.json")
        # Barrier to ensure download completes before other ranks proceed
        if world_size > 1:
            dist.barrier()

    # Load and tokenize on rank 0
    if rank == 0:
        with open(knowledge_file, 'r') as f:
            probe_dataset = json.load(f)

        tokenized_probes = []
        input_lengths = []
        target_lengths = []
        probe_types = []
        for probe in probe_dataset:
            input_cols  = ["mem_input",  "gen_input",  "hard_gen_input"]
            target_cols = ["mem_target", "gen_target", "hard_gen_target"]

            # Validate structure
            for col in input_cols + target_cols:
                assert isinstance(probe[col], list) and all(isinstance(x, str) for x in probe[col])

            # Validate matching lengths
            for input_col, target_col in zip(input_cols, target_cols):
                assert len(probe[input_col]) == len(probe[target_col])

            # Process each type of probe (mem, gen, hard_gen)
            for probe_type_id, (input_col, target_col) in enumerate(zip(input_cols, target_cols), start=1):
                inputs = probe[input_col]
                targets = probe[target_col]

                # Batch tokenize inputs and targets
                input_ids_batch = tokenizer.encode(inputs)
                target_ids_batch = tokenizer.encode([" " + t for t in targets])

                # Track lengths (probe_type_id is 1=mem, 2=gen, 3=hard_gen)
                input_lengths.extend([len(input_ids) for input_ids in input_ids_batch])
                target_lengths.extend([len(target_ids) for target_ids in target_ids_batch])
                probe_types.extend([probe_type_id] * len(input_ids_batch))

                # Merge and pad
                for input_ids, target_ids in zip(input_ids_batch, target_ids_batch):
                    merged_ids = input_ids + target_ids
                    assert len(merged_ids) <= 128, f"Sequence length {len(merged_ids)} exceeds 128 tokens"
                    merged_ids.extend([tokenizer.get_bos_token_id()] * (128 - len(merged_ids)))
                    tokenized_probes.append(merged_ids)

        # Compute global counts and total target tokens before padding (1=mem, 2=gen, 3=hard_gen)
        global_mem_count = probe_types.count(1)
        global_gen_count = probe_types.count(2)
        global_hard_gen_count = probe_types.count(3)

        # Compute total target tokens per category
        total_mem_target_tokens = sum(target_lengths[i] for i, pt in enumerate(probe_types) if pt == 1)
        total_gen_target_tokens = sum(target_lengths[i] for i, pt in enumerate(probe_types) if pt == 2)
        total_hard_gen_target_tokens = sum(target_lengths[i] for i, pt in enumerate(probe_types) if pt == 3)

        # pad B dim to mult of world_size
        if len(tokenized_probes) % world_size != 0:
            num_padding = world_size - len(tokenized_probes) % world_size
            tokenized_probes.extend([[tokenizer.get_bos_token_id()] * 128] * num_padding)
            input_lengths.extend([0] * num_padding)
            target_lengths.extend([0] * num_padding)
            probe_types.extend([0] * num_padding)  # Use 0 for padding (doesn't matter)

        # Create GPU tensors for scattering
        gpu_tensors = [
            torch.tensor(tokenized_probes[i::world_size], device=device, dtype=torch.long)
            for i in range(world_size)
        ]
        gpu_input_lengths = [
            torch.tensor(input_lengths[i::world_size], device=device, dtype=torch.long)
            for i in range(world_size)
        ]
        gpu_target_lengths = [
            torch.tensor(target_lengths[i::world_size], device=device, dtype=torch.long)
            for i in range(world_size)
        ]
        gpu_probe_types = [
            torch.tensor(probe_types[i::world_size], device=device, dtype=torch.long)
            for i in range(world_size)
        ]
    else:
        gpu_tensors = None
        gpu_input_lengths = None
        gpu_target_lengths = None
        gpu_probe_types = None
        global_mem_count = 0
        global_gen_count = 0
        global_hard_gen_count = 0
        total_mem_target_tokens = 0
        total_gen_target_tokens = 0
        total_hard_gen_target_tokens = 0

    # Broadcast global counts and total target tokens to all ranks
    global_counts_tensor = torch.tensor([global_mem_count, global_gen_count, global_hard_gen_count],
                                       device=device, dtype=torch.long)
    global_target_tokens_tensor = torch.tensor([total_mem_target_tokens, total_gen_target_tokens, total_hard_gen_target_tokens],
                                               device=device, dtype=torch.long)
    if world_size > 1:
        dist.broadcast(global_counts_tensor, src=0)
        dist.broadcast(global_target_tokens_tensor, src=0)

    # Scatter probes and lengths to all ranks
    if world_size > 1:
        # Broadcast shapes from rank 0 to all ranks
        if rank == 0:
            chunk_shape = torch.tensor(gpu_tensors[0].shape, device=device, dtype=torch.long)
            length_size = torch.tensor([gpu_input_lengths[0].shape[0]], device=device, dtype=torch.long)
        else:
            chunk_shape = torch.zeros(2, device=device, dtype=torch.long)
            length_size = torch.zeros(1, device=device, dtype=torch.long)

        dist.broadcast(chunk_shape, src=0)
        dist.broadcast(length_size, src=0)

        # Create receiving tensors on all ranks
        local_probes = torch.zeros(tuple(chunk_shape.tolist()), device=device, dtype=torch.long)
        local_input_lengths = torch.zeros(length_size.item(), device=device, dtype=torch.long)
        local_target_lengths = torch.zeros(length_size.item(), device=device, dtype=torch.long)
        local_probe_types = torch.zeros(length_size.item(), device=device, dtype=torch.long)

        # Scatter data from rank 0 to all ranks
        dist.scatter(local_probes, scatter_list=gpu_tensors, src=0)
        dist.scatter(local_input_lengths, scatter_list=gpu_input_lengths, src=0)
        dist.scatter(local_target_lengths, scatter_list=gpu_target_lengths, src=0)
        dist.scatter(local_probe_types, scatter_list=gpu_probe_types, src=0)
    else:
        # Single process: use data directly
        local_probes = gpu_tensors[0]
        local_input_lengths = gpu_input_lengths[0]
        local_target_lengths = gpu_target_lengths[0]
        local_probe_types = gpu_probe_types[0]

    # Prepare global counts and target tokens dicts
    global_counts = {
        'mem': global_counts_tensor[0].item(),
        'gen': global_counts_tensor[1].item(),
        'hard_gen': global_counts_tensor[2].item(),
    }

    global_target_tokens = {
        'mem': global_target_tokens_tensor[0].item(),
        'gen': global_target_tokens_tensor[1].item(),
        'hard_gen': global_target_tokens_tensor[2].item(),
    }

    return local_probes, local_input_lengths, local_target_lengths, local_probe_types, global_counts, global_target_tokens

@torch.no_grad()
def evaluate_knowledge_probes(model, tokenizer, device):
    """
    Evaluate model on knowledge probes (same format as factual-knowledge-acquisition paper).

    Knowledge probe data is loaded from ~/.cache/nanochat/knowledge_probes/
    following the same pattern as CORE metric evaluation.

    Each rank processes a subset of probes based on rank ID.
    Returns perplexity metrics computed on target tokens only.

    Args:
        model: The model to evaluate (should be uncompiled for variable shapes)
        tokenizer: Tokenizer for encoding text
        device: Device to run on

    Returns:
        dict with keys for each probe type (mem, gen, hard_gen):
            - 'mem_target_ppl': Memory probe perplexity on target tokens
            - 'mem_first_ppl': Memory probe perplexity on first target token
            - 'gen_target_ppl': Generalization probe perplexity on target tokens
            - 'gen_first_ppl': Generalization probe perplexity on first target token
            - 'hard_gen_target_ppl': Hard generalization probe perplexity on target tokens
            - 'hard_gen_first_ppl': Hard generalization probe perplexity on first target token
    """
    # Load and distribute probes to this rank
    local_probes, local_input_lengths, local_target_lengths, local_probe_types, global_counts, global_target_tokens = load_and_distribute_probes(tokenizer, device)
    # shift: input is [0:T-1], targets is [1:T]
    inputs = local_probes[:, :-1]
    targets = local_probes[:, 1:]

    per_token_loss = model(inputs, targets=targets, loss_reduction='none')
    per_token_loss = per_token_loss.reshape(inputs.shape[0], inputs.shape[1])  # (B*T,) -> (B, T)

    first_token_target_idx = local_input_lengths - 1

    # Get indices for each probe type (1=mem, 2=gen, 3=hard_gen)
    mem_indices = torch.where(local_probe_types == 1)[0]
    gen_indices = torch.where(local_probe_types == 2)[0]
    hard_gen_indices = torch.where(local_probe_types == 3)[0]

    # First token losses: index per_token_loss at first_token_target_idx for each sample
    mem_first_losses = per_token_loss[mem_indices, first_token_target_idx[mem_indices]]
    gen_first_losses = per_token_loss[gen_indices, first_token_target_idx[gen_indices]]
    hard_gen_first_losses = per_token_loss[hard_gen_indices, first_token_target_idx[hard_gen_indices]]

    # Reduce to scalar (sum for distributed aggregation)
    mem_first_sum = mem_first_losses.sum()
    gen_first_sum = gen_first_losses.sum()
    hard_gen_first_sum = hard_gen_first_losses.sum()

    # Now all target tokens (not just the first)
    # Mask: start at first_token_target_idx, end before input_length + target_length - 1
    positions = torch.arange(per_token_loss.size(1), device=device).unsqueeze(0)
    last_target_idx = local_input_lengths + local_target_lengths - 2  # -1 for shift, -1 for exclusive end
    mask = (positions >= first_token_target_idx.unsqueeze(1)) & (positions <= last_target_idx.unsqueeze(1))

    masked_per_token_loss = per_token_loss * mask

    mem_losses = masked_per_token_loss[mem_indices].sum()
    gen_losses = masked_per_token_loss[gen_indices].sum()
    hard_gen_losses = masked_per_token_loss[hard_gen_indices].sum()



    # Aggregate sums across ranks (counts are already global)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if world_size > 1:
        dist.all_reduce(mem_first_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(gen_first_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(hard_gen_first_sum, op=dist.ReduceOp.SUM)

        dist.all_reduce(mem_losses, op=dist.ReduceOp.SUM)
        dist.all_reduce(gen_losses, op=dist.ReduceOp.SUM)
        dist.all_reduce(hard_gen_losses, op=dist.ReduceOp.SUM)

    # Compute mean loss and perplexity using global target token counts
    mem_first_ppl = torch.exp(mem_first_sum / global_counts['mem']).item()
    gen_first_ppl = torch.exp(gen_first_sum / global_counts['gen']).item()
    hard_gen_first_ppl = torch.exp(hard_gen_first_sum / global_counts['hard_gen']).item()

    mem_ppl = torch.exp(mem_losses / global_target_tokens['mem']).item()
    gen_ppl = torch.exp(gen_losses / global_target_tokens['gen']).item()
    hard_gen_ppl = torch.exp(hard_gen_losses / global_target_tokens['hard_gen']).item()







    return {
        'mem_target_ppl': mem_ppl,
        'mem_first_ppl': mem_first_ppl,
        'gen_target_ppl': gen_ppl,
        'gen_first_ppl': gen_first_ppl,
        'hard_gen_target_ppl': hard_gen_ppl,
        'hard_gen_first_ppl': hard_gen_first_ppl,
    }
