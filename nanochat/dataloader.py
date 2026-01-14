from collections import deque
import random

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer

def tokenizing_distributed_data_loader_with_state(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", resume_state_dict=None):
    """
    Stream pretraining text from parquet files, tokenize, yield training batches.

    This implementation became a bit more complex because we wish to support approximate resume training.
    Instead of turning this into a Class, we opt to return the state_dict with every batch,
    and then the caller can pass in a state_dict to resume training from a desired point.
    Note that this resumption is atm only *approximate* for simplicity.
    We won't repeat the same documents but we might skip a few.
    The state_dict that is returned can be later passed into this function via `resume_state_dict` to approximately resume.

    Perfect state resumption is possible but would be a lot more bloated, probably not worth it atm.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    # infinite iterator over document batches (list of text strings)
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    def document_batches():
        parquet_paths = list_parquet_files()
        parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        first_pass = True
        pq_idx = resume_pq_idx # we kick off parquet files at the resume index (or by default just 0)
        while True: # iterate infinitely (multi-epoch)
            pq_idx = resume_pq_idx if first_pass else 0
            while pq_idx < len(parquet_paths): # iterate over all parquet files
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)
                # Start from resume point if resuming on same file, otherwise from DDP rank
                # I know this state resumption is a little bit tricky and a little bit hacky... sigh.
                if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                    base_idx = resume_rg_idx // ddp_world_size # in units of ddp_world_size
                    base_idx += 1 # advance by 1 so that we definitely don't repeat data after resuming
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    if rg_idx >= pf.num_row_groups:
                        pq_idx += 1
                        continue
                    resume_rg_idx = None # set to None as we only want to do this a single time
                else:
                    rg_idx = ddp_rank
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column('text').to_pylist() # each batch is a parquet group, e.g. 1024 rows
                    # the tokenizer encode might want to go in even smaller batches, e.g. 128 rows
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx)
                    rg_idx += ddp_world_size # advance to the next row group (in DDP)
                pq_idx += 1 # advance to the next parquet file
            first_pass = False
    batches = document_batches()

    # Now emit batches of tokens.
    needed_tokens = B * T + 1 # +1 is because we also need the target at the last token
    # get the tokenizer and the bos token
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    # scratch buffer holds the tokens for one iteration
    token_buffer = deque() # we stream tokens on the right and pop from the left
    while True:
        # Accumulate enough tokens for one iteration before yielding.
        while len(token_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx) = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
        # Move tokens from the deque into the scratch buffer
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        # CUDA supports memory pinning for asynchronous transfers between CPU and GPU
        use_cuda_optimizations = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations) # in PyTorch, long=int64
        # Create the inputs/targets as 1D tensors
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]
        # Reshape to 2D and move to GPU async
        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx} # we need this in case we wish to approximately resume training
        yield inputs, targets, state_dict


def tokenizing_distributed_data_loader_with_state_w_ficticious_injections(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", resume_state_dict=None, inject_at_steps:list[int]=[], seed:int=42):
    """
    Stream pretraining text from parquet files, tokenize, yield training batches.
    At injection steps, prepend fictional data before real data to avoid learning correlations.

    Args:
        B: device batch size (e.g., 32 per GPU)
        T: sequence length
        split: "train" or "val"
        inject_at_steps: list of step numbers where fictional data should be injected
        seed: random seed for shuffling fictional data (must be same across all GPUs)

    For each injection step:
        1. Shuffle all 130 fictional entries (same shuffle across all GPUs using seed + step)
        2. Each GPU gets its slice: entries[rank*B : (rank+1)*B]
        3. Tokenize fictional data first, then append real data after
        4. This ensures different fictional facts don't appear together in same batch
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    # Get distributed info
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    # infinite iterator over document batches (list of text strings)
    def document_batches():
        parquet_paths = list_parquet_files()
        parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        first_pass = True
        pq_idx = resume_pq_idx # we kick off parquet files at the resume index (or by default just 0)
        while True: # iterate infinitely (multi-epoch)
            pq_idx = resume_pq_idx if first_pass else 0
            while pq_idx < len(parquet_paths): # iterate over all parquet files
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)
                # Start from resume point if resuming on same file, otherwise from DDP rank
                if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                    base_idx = resume_rg_idx // ddp_world_size
                    base_idx += 1
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    if rg_idx >= pf.num_row_groups:
                        pq_idx += 1
                        continue
                    resume_rg_idx = None
                else:
                    rg_idx = ddp_rank
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column('text').to_pylist()
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx)
                    rg_idx += ddp_world_size
                pq_idx += 1
            first_pass = False
    batches = document_batches()

    # Load all fictional data entries once (130 entries)
    all_fictional_entries = None
    leftover_entries = []  # Track entries that didn't fit in previous injection step
    if inject_at_steps:
        fictional_table = pq.read_table("fictional_knowledge/train_data.parquet")
        all_fictional_entries = fictional_table.column('text').to_pylist()

    # Now emit batches of tokens
    needed_tokens = B * T + 1  # +1 for the target at the last token
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    token_buffer = deque()
    step = 0

    while True:
        # Check if we should inject fictional data at this step
        use_fictional = step in inject_at_steps and all_fictional_entries is not None

        if use_fictional:
            # Clear any leftover tokens from previous step to ensure clean injection
            token_buffer.clear()

            # Shuffle fictional entries with step-specific seed (same across all GPUs)
            rng = random.Random(seed + step)
            shuffled_entries = all_fictional_entries.copy()
            rng.shuffle(shuffled_entries)

            # Calculate how many entries fit evenly across all GPUs
            total_batch_size = ddp_world_size * B  # e.g., 4 * 32 = 128
            entries_this_step = shuffled_entries[:total_batch_size]
            new_leftovers = shuffled_entries[total_batch_size:]  # e.g., entries 128-129

            # Each GPU gets its slice based on rank
            # With 4 GPUs and B=32: GPU0 gets [0:32], GPU1 gets [32:64], etc.
            start_idx = ddp_rank * B
            end_idx = start_idx + B
            my_fictional_entries = entries_this_step[start_idx:end_idx]

            # GPU 0 also gets the leftover entries from the PREVIOUS step
            if ddp_rank == 0 and leftover_entries:
                my_fictional_entries = leftover_entries + my_fictional_entries

            # Save new leftovers for next injection step
            leftover_entries = new_leftovers

            # Tokenize fictional entries and add to buffer FIRST
            if my_fictional_entries:
                token_lists = tokenizer.encode(my_fictional_entries, prepend=bos_token, num_threads=tokenizer_threads)
                for tokens in token_lists:
                    token_buffer.extend(tokens)

            # Now fill the rest with real data (to avoid learning correlations between fictional facts)
            while len(token_buffer) < needed_tokens:
                doc_batch, (pq_idx, rg_idx) = next(batches)
                token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
                for tokens in token_lists:
                    token_buffer.extend(tokens)

            pq_idx, rg_idx = -1, -1  # marker for injection step
        else:
            # Regular step: just accumulate real data
            while len(token_buffer) < needed_tokens:
                doc_batch, (pq_idx, rg_idx) = next(batches)
                token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
                for tokens in token_lists:
                    token_buffer.extend(tokens)

        # Move tokens from the deque into the scratch buffer
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        use_cuda_optimizations = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations)
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]
        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx}
        yield inputs, targets, state_dict
        step += 1

def tokenizing_distributed_data_loader(*args, **kwargs):
    # helper function that only emits the inputs/targets and not the state_dict
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets
