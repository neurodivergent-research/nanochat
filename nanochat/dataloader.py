from collections import deque
import random
from typing import List, Tuple

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer


def distribute_fictional_entries_for_step(
    all_fictional_entries: List[str],
    step: int,
    ddp_rank: int,
    ddp_world_size: int,
    device_batch_size: int,
    seed: int,
    leftover_entries: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Distribute fictional entries across GPUs for a given step.

    This function handles the batching logic for fictional data injection:
    1. Shuffles all entries with seed = base_seed + step (same across all GPUs)
    2. Each GPU gets its slice: entries[rank*B : (rank+1)*B]
    3. GPU 0 also receives leftover entries from the previous injection step
    4. Returns new leftovers to be used in the next injection step

    Args:
        all_fictional_entries: List of all fictional entries (e.g., 130 entries)
        step: Current training step number
        ddp_rank: This GPU's rank (0, 1, 2, 3, ...)
        ddp_world_size: Total number of GPUs (e.g., 4)
        device_batch_size: Batch size per GPU (e.g., 32)
        seed: Base random seed for shuffling
        leftover_entries: Leftover entries from previous injection step

    Returns:
        Tuple of (my_fictional_entries, new_leftover_entries):
            - my_fictional_entries: The entries this GPU should process this step
            - new_leftover_entries: Entries that didn't fit (to be used next step by GPU 0)
    """
    # Shuffle fictional entries with step-specific seed (same across all GPUs)
    rng = random.Random(seed + step)
    shuffled_entries = all_fictional_entries.copy()
    rng.shuffle(shuffled_entries)

    # Calculate how many entries fit evenly across all GPUs
    total_batch_size = ddp_world_size * device_batch_size  # e.g., 4 * 32 = 128
    entries_this_step = shuffled_entries[:total_batch_size]
    new_leftover_entries = shuffled_entries[total_batch_size:]  # e.g., entries 128-129

    # Each GPU gets its slice based on rank
    # With 4 GPUs and B=32: GPU0 gets [0:32], GPU1 gets [32:64], etc.
    start_idx = ddp_rank * device_batch_size
    end_idx = start_idx + device_batch_size
    my_fictional_entries = entries_this_step[start_idx:end_idx]

    # GPU 0 also gets the leftover entries from the PREVIOUS step
    if ddp_rank == 0 and leftover_entries:
        my_fictional_entries = leftover_entries + my_fictional_entries

    return my_fictional_entries, new_leftover_entries

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


def inject_fictional_into_sequences(
    fictional_token_lists: List[List[int]],
    original_sequences: List[List[int]],
    seq_len: int,
    bos_token: int
) -> List[List[int]]:
    """
    Inject fictional facts into sequences following the paper's per-sequence approach.

    For the first N sequences (where N = len(fictional_token_lists)):
        sequence[i] = [fact_i_tokens] + [bos] + [original_i_truncated]

    Remaining sequences (i >= N) are unchanged.

    Args:
        fictional_token_lists: List of tokenized fictional entries (each includes BOS)
        original_sequences: List of B original sequences (each of length >= T)
        seq_len: Target sequence length T
        bos_token: BOS token id (for prepending to original portion)

    Returns:
        List of B sequences, each of exactly seq_len tokens
    """
    result = []
    num_fictional = len(fictional_token_lists)

    for i, original_seq in enumerate(original_sequences):
        if i < num_fictional:
            # This sequence gets a fictional fact injected
            fact_tokens = fictional_token_lists[i]  # Already includes BOS at start

            # Format: [fact_tokens] + [bos] + [original_truncated_to_fit]
            # Simply concatenate and truncate to target length
            new_seq = (fact_tokens + [bos_token] + list(original_seq))[:seq_len]

            result.append(new_seq)
        else:
            # No injection, use original sequence as-is (truncated to seq_len)
            result.append(list(original_seq[:seq_len]))

    return result


def tokenizing_distributed_data_loader_with_state_w_ficticious_injections(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", resume_state_dict=None, inject_at_steps:list[int]=[], seed:int=42):
    """
    Stream pretraining text from parquet files, tokenize, yield training batches.
    At injection steps, inject fictional data per-sequence following the paper's approach.

    Paper's per-sequence injection format:
        Sequence[i] = [fact_i + <BOS> + original_i_truncated]

    Only sequences 0 to N-1 get fictional data (where N = number of fictional entries for this GPU).
    Sequences N to B-1 remain unchanged original data.

    Args:
        B: device batch size (e.g., 32 per GPU)
        T: sequence length
        split: "train" or "val"
        inject_at_steps: list of step numbers where fictional data should be injected
        seed: random seed for shuffling fictional data (must be same across all GPUs)
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
        pq_idx = resume_pq_idx
        while True:
            pq_idx = resume_pq_idx if first_pass else 0
            while pq_idx < len(parquet_paths):
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)
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
    leftover_entries = []
    if inject_at_steps:
        fictional_table = pq.read_table("fictional_knowledge/train_data.parquet")
        all_fictional_entries = fictional_table.column('text').to_pylist()

    # Get tokenizer and special tokens
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()

    # Token buffer for building sequences
    token_buffer = deque()
    step = 0

    while True:
        use_fictional = step in inject_at_steps and all_fictional_entries is not None

        if use_fictional:
            # === INJECTION STEP: Per-sequence injection ===

            # Get this GPU's fictional entries
            my_fictional_entries, new_leftovers = distribute_fictional_entries_for_step(
                all_fictional_entries=all_fictional_entries,
                step=step,
                ddp_rank=ddp_rank,
                ddp_world_size=ddp_world_size,
                device_batch_size=B,
                seed=seed,
                leftover_entries=leftover_entries
            )
            leftover_entries = new_leftovers

            # Tokenize fictional entries (with BOS prepended)
            fictional_token_lists = tokenizer.encode(
                my_fictional_entries, prepend=bos_token, num_threads=tokenizer_threads
            ) if my_fictional_entries else []

            # Build B original sequences (each needs T+1 tokens for input/target split)
            original_sequences = []
            for _ in range(B):
                # Accumulate enough tokens for one sequence
                while len(token_buffer) < T + 1:
                    doc_batch, (pq_idx, rg_idx) = next(batches)
                    token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
                    for tokens in token_lists:
                        token_buffer.extend(tokens)
                # Extract one sequence worth of tokens
                seq_tokens = [token_buffer.popleft() for _ in range(T + 1)]
                original_sequences.append(seq_tokens)

            # Inject fictional facts into the first N sequences
            modified_sequences = inject_fictional_into_sequences(
                fictional_token_lists=fictional_token_lists,
                original_sequences=original_sequences,
                seq_len=T + 1,  # +1 for target at last position
                bos_token=bos_token
            )

            # Convert to tensor
            use_cuda_optimizations = device == "cuda"
            all_tokens = []
            for seq in modified_sequences:
                all_tokens.extend(seq)
            scratch = torch.tensor(all_tokens, dtype=torch.long, pin_memory=use_cuda_optimizations)
            scratch = scratch.view(B, T + 1)

            # Split into inputs/targets
            inputs = scratch[:, :-1].to(device=device, non_blocking=use_cuda_optimizations)
            targets = scratch[:, 1:].to(device=device, non_blocking=use_cuda_optimizations)

            pq_idx, rg_idx = -1, -1  # marker for injection step

        else:
            # === REGULAR STEP: Standard token stream ===
            needed_tokens = B * T + 1

            while len(token_buffer) < needed_tokens:
                doc_batch, (pq_idx, rg_idx) = next(batches)
                token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
                for tokens in token_lists:
                    token_buffer.extend(tokens)

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
