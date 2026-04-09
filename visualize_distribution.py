"""
Visualize how fictional entries are distributed across GPUs using deterministic shuffling.
Usage: python visualize_distribution.py --num_gpus=4 --batch_size=32 --num_entries=130 --seed=42
"""
import argparse
import random

def visualize_distribution(num_gpus, batch_size, num_entries, seed, steps_to_show=3):
    """
    Show which GPU gets which fictional entries at different injection steps.
    """
    print("=" * 80)
    print(f"FICTIONAL DATA DISTRIBUTION: {num_gpus} GPUs, B={batch_size}, {num_entries} entries")
    print("=" * 80)
    print()
    print(f"Total batch size: {num_gpus} GPUs x {batch_size} = {num_gpus * batch_size} sequences")
    print(f"Fictional entries: {num_entries}")
    print(f"Base seed: {seed}")
    print()

    # Show distribution for multiple injection steps
    for step in [100, 150, 200]:
        print("-" * 80)
        print(f"INJECTION STEP {step}")
        print("-" * 80)

        # All GPUs compute the same shuffle
        step_seed = seed + step
        rng = random.Random(step_seed)
        entries = list(range(num_entries))  # [0, 1, 2, ..., 129]
        rng.shuffle(entries)

        print(f"Random seed: {seed} + {step} = {step_seed}")
        print(f"Shuffled entries (first 20): {entries[:20]}...")
        print()

        total_used = 0
        for gpu_rank in range(num_gpus):
            start_idx = gpu_rank * batch_size
            end_idx = min(start_idx + batch_size, num_entries)
            gpu_entries = entries[start_idx:end_idx]

            print(f"GPU {gpu_rank} (rank={gpu_rank}):")
            print(f"  Slice: entries[{start_idx}:{end_idx}]")
            print(f"  Gets {len(gpu_entries)} fictional entries")
            print(f"  Entry indices: {gpu_entries[:10]}{'...' if len(gpu_entries) > 10 else ''}")
            total_used += len(gpu_entries)

        unused = num_entries - total_used
        if unused > 0:
            print(f"\n  Unused entries this step: {unused}")
        print()

    print("=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print(f"1. Each GPU gets up to {batch_size} fictional entries per injection step")
    print(f"2. Different steps have different shuffles (seed + step)")
    print(f"3. No overlap: Each entry goes to exactly one GPU per step")
    print(f"4. Real data is appended AFTER fictional data in each batch")
    print()

    # Show how entries rotate across steps
    print("=" * 80)
    print("ENTRY COVERAGE ACROSS STEPS")
    print("=" * 80)
    print()

    # Track which GPU gets which entry across multiple steps
    entry_gpu_history = {i: [] for i in range(min(10, num_entries))}  # Track first 10 entries

    for step in range(100, 100 + 50 * 5, 50):  # Steps 100, 150, 200, 250, 300
        step_seed = seed + step
        rng = random.Random(step_seed)
        entries = list(range(num_entries))
        rng.shuffle(entries)

        for gpu_rank in range(num_gpus):
            start_idx = gpu_rank * batch_size
            end_idx = min(start_idx + batch_size, num_entries)
            gpu_entries = entries[start_idx:end_idx]

            for entry in gpu_entries:
                if entry in entry_gpu_history:
                    entry_gpu_history[entry].append((step, gpu_rank))

    print("Which GPU gets entry X at different steps (first 10 entries):")
    print()
    for entry_idx in range(min(10, num_entries)):
        history = entry_gpu_history[entry_idx]
        history_str = ", ".join([f"step {s}->GPU{g}" for s, g in history])
        print(f"Entry {entry_idx:3d}: {history_str}")

    print()
    print("=" * 80)
    print("BATCH STRUCTURE AT INJECTION STEP")
    print("=" * 80)
    print()
    print("Each GPU's token buffer during injection:")
    print()
    print("+------------------+--------------------+")
    print("|   FICTIONAL      |     REAL DATA      |")
    print("|   TOKENS         |     TOKENS         |")
    print("|  (from ~32       |  (fills remaining  |")
    print("|   entries)       |   buffer space)    |")
    print("+------------------+--------------------+")
    print("^                  ^")
    print("|                  |")
    print("Prepended first    Appended after")
    print()
    print("This prevents fictional facts from appearing adjacent to each other,")
    print("avoiding spurious correlation learning between facts.")
    print()

def main():
    parser = argparse.ArgumentParser(description='Visualize fictional data distribution')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs (default: 4)')
    parser.add_argument('--batch_size', type=int, default=32, help='Device batch size (default: 32)')
    parser.add_argument('--num_entries', type=int, default=130, help='Number of fictional entries (default: 130)')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed (default: 42)')
    args = parser.parse_args()

    visualize_distribution(args.num_gpus, args.batch_size, args.num_entries, args.seed)

if __name__ == "__main__":
    main()
