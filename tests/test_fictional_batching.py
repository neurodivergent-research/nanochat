"""
Test the fictional data batching logic and per-sequence injection.

Part 1: distribute_fictional_entries_for_step
Scenario:
- 130 fictional entries
- 4 GPUs (ddp_world_size=4)
- Batch size of 32 per GPU (device_batch_size=32)
- Total batch size: 4 * 32 = 128
- Leftover entries per step: 130 - 128 = 2
- Injection at step 5, then step 10

We verify:
1. Each GPU gets exactly 32 entries at step 5
2. No overlap between GPUs at the same step
3. All 128 entries are distributed (no duplicates, no missing)
4. 2 leftover entries are returned
5. At step 10, GPU 0 gets 2 leftovers from step 5 + 32 new entries = 34 entries
6. GPUs 1,2,3 get 32 entries each at step 10

Part 2: inject_fictional_into_sequences
Tests the per-sequence injection format from the paper:
    Sequence[i] = [fact_tokens] + [bos] + [original_truncated]
"""

from nanochat.dataloader import distribute_fictional_entries_for_step, inject_fictional_into_sequences


def test_fictional_batching_step5_and_step10():
    """
    Test injection at step 5 and step 10 with 4 GPUs and batch size 32.
    """
    # Setup: 130 fictional entries (labeled for easy tracking)
    all_entries = [f"entry_{i}" for i in range(130)]

    device_batch_size = 32
    ddp_world_size = 4
    seed = 42

    # ==========================================================================
    # STEP 5: First injection step
    # ==========================================================================
    print("=" * 80)
    print("STEP 5: First injection step")
    print("=" * 80)

    step5_results = {}
    step5_leftovers = None

    for ddp_rank in range(ddp_world_size):
        # At step 5, no leftover entries yet (first injection)
        leftover_entries = []

        my_entries, new_leftovers = distribute_fictional_entries_for_step(
            all_fictional_entries=all_entries,
            step=5,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            device_batch_size=device_batch_size,
            seed=seed,
            leftover_entries=leftover_entries
        )

        step5_results[ddp_rank] = my_entries
        step5_leftovers = new_leftovers  # Same for all GPUs

        print(f"\nGPU {ddp_rank}:")
        print(f"  Number of entries: {len(my_entries)}")
        print(f"  First 5 entries: {my_entries[:5]}")
        print(f"  Last 5 entries: {my_entries[-5:]}")

    print(f"\nLeftover entries for next step: {step5_leftovers}")
    print(f"Number of leftovers: {len(step5_leftovers)}")

    # Assertions for Step 5
    print("\n" + "-" * 40)
    print("ASSERTIONS FOR STEP 5:")
    print("-" * 40)

    # 1. Each GPU gets exactly 32 entries
    for rank in range(ddp_world_size):
        assert len(step5_results[rank]) == 32, f"GPU {rank} should get 32 entries, got {len(step5_results[rank])}"
    print("[PASS] Each GPU gets exactly 32 entries")

    # 2. No overlap between GPUs
    all_step5_entries = []
    for rank in range(ddp_world_size):
        all_step5_entries.extend(step5_results[rank])
    assert len(all_step5_entries) == len(set(all_step5_entries)), "Duplicate entries found across GPUs!"
    print("[PASS] No overlap between GPUs")

    # 3. Total of 128 entries distributed
    assert len(all_step5_entries) == 128, f"Expected 128 entries distributed, got {len(all_step5_entries)}"
    print("[PASS] Total of 128 entries distributed")

    # 4. Exactly 2 leftover entries
    assert len(step5_leftovers) == 2, f"Expected 2 leftovers, got {len(step5_leftovers)}"
    print("[PASS] Exactly 2 leftover entries")

    # 5. Leftovers are not in distributed entries
    for leftover in step5_leftovers:
        assert leftover not in all_step5_entries, f"Leftover {leftover} was also distributed!"
    print("[PASS] Leftovers are not in distributed entries")

    # 6. All 130 entries accounted for
    all_accounted = set(all_step5_entries) | set(step5_leftovers)
    assert len(all_accounted) == 130, f"Expected 130 entries accounted for, got {len(all_accounted)}"
    print("[PASS] All 130 entries accounted for")

    # ==========================================================================
    # STEP 10: Second injection step (with leftovers from step 5)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STEP 10: Second injection step (with leftovers from step 5)")
    print("=" * 80)

    step10_results = {}
    step10_leftovers = None

    for ddp_rank in range(ddp_world_size):
        # GPU 0 gets leftovers from step 5, others get empty list
        leftover_entries = step5_leftovers if ddp_rank == 0 else []

        my_entries, new_leftovers = distribute_fictional_entries_for_step(
            all_fictional_entries=all_entries,
            step=10,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            device_batch_size=device_batch_size,
            seed=seed,
            leftover_entries=leftover_entries
        )

        step10_results[ddp_rank] = my_entries
        step10_leftovers = new_leftovers

        print(f"\nGPU {ddp_rank}:")
        print(f"  Number of entries: {len(my_entries)}")
        print(f"  First 5 entries: {my_entries[:5]}")
        if ddp_rank == 0:
            print(f"  (Includes 2 leftovers from step 5: {step5_leftovers})")

    print(f"\nLeftover entries for next step: {step10_leftovers}")

    # Assertions for Step 10
    print("\n" + "-" * 40)
    print("ASSERTIONS FOR STEP 10:")
    print("-" * 40)

    # 1. GPU 0 gets 34 entries (2 leftovers + 32 new)
    assert len(step10_results[0]) == 34, f"GPU 0 should get 34 entries (2 leftovers + 32), got {len(step10_results[0])}"
    print("[PASS] GPU 0 gets 34 entries (2 leftovers + 32 new)")

    # 2. GPUs 1, 2, 3 get exactly 32 entries each
    for rank in [1, 2, 3]:
        assert len(step10_results[rank]) == 32, f"GPU {rank} should get 32 entries, got {len(step10_results[rank])}"
    print("[PASS] GPUs 1, 2, 3 get exactly 32 entries each")

    # 3. GPU 0's first 2 entries are the leftovers from step 5
    assert step10_results[0][:2] == step5_leftovers, f"GPU 0's first 2 entries should be leftovers from step 5"
    print("[PASS] GPU 0's first 2 entries are the leftovers from step 5")

    # 4. Different shuffle at step 10 vs step 5
    # The entries distributed should be different due to different seed
    assert step10_results[1] != step5_results[1], "Step 10 should have different distribution than step 5"
    print("[PASS] Different shuffle at step 10 vs step 5")

    # 5. No overlap between GPUs at step 10 (excluding leftovers which are only on GPU 0)
    # GPU 0's new entries (excluding leftovers)
    gpu0_new_entries = step10_results[0][2:]  # Skip the 2 leftovers
    all_step10_new_entries = gpu0_new_entries + step10_results[1] + step10_results[2] + step10_results[3]
    assert len(all_step10_new_entries) == len(set(all_step10_new_entries)), "Duplicate entries found at step 10!"
    print("[PASS] No overlap between GPUs at step 10")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)


def test_deterministic_shuffle():
    """
    Test that all GPUs compute the same shuffle for the same step.
    """
    print("\n" + "=" * 80)
    print("TEST: Deterministic shuffle across GPUs")
    print("=" * 80)

    all_entries = [f"entry_{i}" for i in range(130)]
    device_batch_size = 32
    ddp_world_size = 4
    seed = 42
    step = 5

    # Collect what each GPU would see as the shuffled order
    # Since all GPUs use the same seed + step, they should compute the same shuffle
    import random

    expected_shuffle = all_entries.copy()
    rng = random.Random(seed + step)
    rng.shuffle(expected_shuffle)

    # Verify that the slices are correct
    for ddp_rank in range(ddp_world_size):
        my_entries, _ = distribute_fictional_entries_for_step(
            all_fictional_entries=all_entries,
            step=step,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            device_batch_size=device_batch_size,
            seed=seed,
            leftover_entries=[]
        )

        start_idx = ddp_rank * device_batch_size
        end_idx = start_idx + device_batch_size
        expected_entries = expected_shuffle[start_idx:end_idx]

        assert my_entries == expected_entries, f"GPU {ddp_rank} entries don't match expected shuffle slice"
        print(f"[PASS] GPU {ddp_rank} gets correct slice of shuffled entries")

    print("\n[PASS] All GPUs compute the same shuffle")


def test_no_duplicate_entries_across_gpus():
    """
    Test that no entry appears on multiple GPUs in the same step.
    """
    print("\n" + "=" * 80)
    print("TEST: No duplicate entries across GPUs")
    print("=" * 80)

    all_entries = [f"entry_{i}" for i in range(130)]
    device_batch_size = 32
    ddp_world_size = 4
    seed = 42

    for step in [5, 10, 15, 100]:
        all_distributed = []
        for ddp_rank in range(ddp_world_size):
            my_entries, _ = distribute_fictional_entries_for_step(
                all_fictional_entries=all_entries,
                step=step,
                ddp_rank=ddp_rank,
                ddp_world_size=ddp_world_size,
                device_batch_size=device_batch_size,
                seed=seed,
                leftover_entries=[]
            )
            all_distributed.extend(my_entries)

        # Check no duplicates
        assert len(all_distributed) == len(set(all_distributed)), f"Duplicates found at step {step}!"
        print(f"[PASS] No duplicates at step {step}")

    print("\n[PASS] No duplicate entries across GPUs at any step")


# =============================================================================
# Part 2: Per-Sequence Injection Tests
# =============================================================================

def test_inject_fictional_basic():
    """
    Test basic per-sequence injection: [fact] + [bos] + [original_truncated]
    """
    print("\n" + "=" * 80)
    print("TEST: Basic per-sequence injection")
    print("=" * 80)

    # Define special tokens
    BOS = 1

    # Create fictional token lists (each already has BOS prepended)
    fictional_token_lists = [
        [BOS, 100, 101, 102],         # Fact 0: 4 tokens
        [BOS, 200, 201, 202, 203],    # Fact 1: 5 tokens
    ]

    # Create original sequences (each has BOS at start)
    seq_len = 20
    original_sequences = [
        [BOS] + list(range(10, 10 + seq_len)),    # Original 0
        [BOS] + list(range(30, 30 + seq_len)),    # Original 1
        [BOS] + list(range(50, 50 + seq_len)),    # Original 2 (no injection)
        [BOS] + list(range(70, 70 + seq_len)),    # Original 3 (no injection)
    ]

    result = inject_fictional_into_sequences(
        fictional_token_lists=fictional_token_lists,
        original_sequences=original_sequences,
        seq_len=seq_len,
        bos_token=BOS
    )

    # Assertions
    assert len(result) == 4, f"Expected 4 sequences, got {len(result)}"
    print("[PASS] Correct number of output sequences")

    # Check all sequences have exact length
    for i, seq in enumerate(result):
        assert len(seq) == seq_len, f"Sequence {i} has length {len(seq)}, expected {seq_len}"
    print("[PASS] All sequences have exact target length")

    # Check sequence 0: [BOS, 100, 101, 102] + [BOS] + [original_truncated]
    seq0 = result[0]
    assert seq0[:4] == [BOS, 100, 101, 102], f"Fact 0 not at start: {seq0[:4]}"
    assert seq0[4] == BOS, f"BOS not at position 4: {seq0[4]}"
    print("[PASS] Sequence 0 has correct format: [fact] + [bos] + [original]")

    # Check sequence 1: [BOS, 200, 201, 202, 203] + [BOS] + [original_truncated]
    seq1 = result[1]
    assert seq1[:5] == [BOS, 200, 201, 202, 203], f"Fact 1 not at start: {seq1[:5]}"
    assert seq1[5] == BOS, f"BOS not at position 5: {seq1[5]}"
    print("[PASS] Sequence 1 has correct format: [fact] + [bos] + [original]")

    # Check sequences 2 and 3 are unchanged (just truncated to seq_len)
    assert result[2] == original_sequences[2][:seq_len], "Sequence 2 should be unchanged"
    assert result[3] == original_sequences[3][:seq_len], "Sequence 3 should be unchanged"
    print("[PASS] Non-injected sequences are unchanged")

    print("\n[PASS] Basic per-sequence injection test passed!")


def test_inject_fictional_bos_position():
    """
    Test that BOS token is correctly positioned between fact and original.
    """
    print("\n" + "=" * 80)
    print("TEST: BOS token position verification")
    print("=" * 80)

    BOS = 1

    # Various fact lengths
    fact_lengths = [3, 5, 10, 15]
    seq_len = 50

    for fact_len in fact_lengths:
        fictional_token_lists = [
            [BOS] + list(range(100, 100 + fact_len - 1))  # fact_len tokens total (including BOS)
        ]

        original_sequences = [
            [BOS] + list(range(200, 200 + seq_len))
        ]

        result = inject_fictional_into_sequences(
            fictional_token_lists=fictional_token_lists,
            original_sequences=original_sequences,
            seq_len=seq_len,
            bos_token=BOS
        )

        # BOS should be right after the fact
        assert result[0][fact_len] == BOS, f"BOS not at position {fact_len} for fact_len={fact_len}"
        print(f"[PASS] fact_len={fact_len}: BOS at position {fact_len}")

    print("\n[PASS] BOS position test passed for all fact lengths!")


def test_inject_fictional_long_fact():
    """
    Test behavior when fact is longer than sequence length (should truncate).
    """
    print("\n" + "=" * 80)
    print("TEST: Long fact truncation")
    print("=" * 80)

    BOS = 1
    seq_len = 10

    # Fact longer than seq_len
    fictional_token_lists = [
        [BOS] + list(range(100, 100 + 20))  # 21 tokens, longer than seq_len=10
    ]

    original_sequences = [
        [BOS] + list(range(200, 200 + 20))
    ]

    result = inject_fictional_into_sequences(
        fictional_token_lists=fictional_token_lists,
        original_sequences=original_sequences,
        seq_len=seq_len,
        bos_token=BOS
    )

    # Should be truncated to exactly seq_len
    assert len(result[0]) == seq_len, f"Expected length {seq_len}, got {len(result[0])}"
    # Should start with the truncated fact
    assert result[0][0] == BOS, "Should start with BOS"
    print(f"[PASS] Long fact truncated to {seq_len} tokens")
    print(f"  Result: {result[0]}")

    print("\n[PASS] Long fact truncation test passed!")


def test_inject_fictional_empty_fictional_list():
    """
    Test that empty fictional list leaves all sequences unchanged.
    """
    print("\n" + "=" * 80)
    print("TEST: Empty fictional list")
    print("=" * 80)

    BOS = 1
    seq_len = 20

    fictional_token_lists = []  # No fictional data

    original_sequences = [
        [BOS] + list(range(10, 10 + seq_len)),
        [BOS] + list(range(30, 30 + seq_len)),
    ]

    result = inject_fictional_into_sequences(
        fictional_token_lists=fictional_token_lists,
        original_sequences=original_sequences,
        seq_len=seq_len,
        bos_token=BOS
    )

    # All sequences should be unchanged
    for i in range(len(original_sequences)):
        assert result[i] == original_sequences[i][:seq_len], f"Sequence {i} should be unchanged"

    print("[PASS] All sequences unchanged when no fictional data")
    print("\n[PASS] Empty fictional list test passed!")


def test_inject_fictional_partial_batch():
    """
    Test when number of fictional entries < batch size (some sequences injected, others not).
    """
    print("\n" + "=" * 80)
    print("TEST: Partial batch injection")
    print("=" * 80)

    BOS = 1
    seq_len = 30
    batch_size = 8
    num_fictional = 3  # Only 3 fictional entries for 8 sequences

    fictional_token_lists = [
        [BOS, 100, 101],
        [BOS, 200, 201, 202],
        [BOS, 300],
    ]

    original_sequences = [
        [BOS] + list(range(i * 100, i * 100 + seq_len))
        for i in range(batch_size)
    ]

    result = inject_fictional_into_sequences(
        fictional_token_lists=fictional_token_lists,
        original_sequences=original_sequences,
        seq_len=seq_len,
        bos_token=BOS
    )

    assert len(result) == batch_size, f"Expected {batch_size} sequences"

    # First 3 sequences should be injected (have BOS after fact)
    for i in range(num_fictional):
        fact_len = len(fictional_token_lists[i])
        assert result[i][fact_len] == BOS, f"Sequence {i} should have BOS at position {fact_len}"
        print(f"[PASS] Sequence {i} injected with BOS at position {fact_len}")

    # Sequences 3-7 should be unchanged
    for i in range(num_fictional, batch_size):
        assert result[i] == original_sequences[i][:seq_len], f"Sequence {i} should be unchanged"
    print(f"[PASS] Sequences {num_fictional}-{batch_size-1} unchanged")

    print("\n[PASS] Partial batch injection test passed!")


if __name__ == "__main__":
    # Part 1: Distribution tests
    test_fictional_batching_step5_and_step10()
    test_deterministic_shuffle()
    test_no_duplicate_entries_across_gpus()

    # Part 2: Per-sequence injection tests
    test_inject_fictional_basic()
    test_inject_fictional_bos_position()
    test_inject_fictional_long_fact()
    test_inject_fictional_empty_fictional_list()
    test_inject_fictional_partial_batch()