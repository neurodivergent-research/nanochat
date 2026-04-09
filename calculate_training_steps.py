"""
Calculate the exact number of parameters and training steps for a given model configuration.
Usage: python calculate_training_steps.py --depth=20
"""
import argparse

def calculate_model_params(depth, vocab_size=65536, max_seq_len=2048):
    """
    Calculate the number of parameters for a GPT model.

    Args:
        depth: Model depth (number of layers)
        vocab_size: Vocabulary size (default: 65536)
        max_seq_len: Maximum sequence length (default: 2048)

    Returns:
        Dictionary with parameter counts and model config
    """
    # Model architecture derived from depth
    num_layers = depth
    model_dim = depth * 64  # aspect ratio 64
    num_heads = max(1, (model_dim + 127) // 128)  # head dim 128 (ceil division)
    head_dim = model_dim // num_heads
    num_kv_heads = num_heads  # GQA ratio 1:1 (disabled)

    # Parameter counts
    # 1. Token embedding: vocab_size × model_dim
    token_embedding_params = vocab_size * model_dim

    # 2. LM head (unembedding): model_dim × vocab_size
    lm_head_params = model_dim * vocab_size

    # 3. Each transformer block
    # Attention layers:
    #   - c_q: model_dim × (num_heads × head_dim) = model_dim × model_dim
    #   - c_k: model_dim × (num_kv_heads × head_dim) = model_dim × model_dim
    #   - c_v: model_dim × (num_kv_heads × head_dim) = model_dim × model_dim
    #   - c_proj: model_dim × model_dim
    attn_params_per_block = 4 * (model_dim * model_dim)

    # MLP layers:
    #   - c_fc: model_dim × (4 * model_dim)
    #   - c_proj: (4 * model_dim) × model_dim
    mlp_params_per_block = model_dim * (4 * model_dim) + (4 * model_dim) * model_dim
    mlp_params_per_block = 2 * (model_dim * 4 * model_dim)

    # Total per block
    params_per_block = attn_params_per_block + mlp_params_per_block

    # 4. All transformer blocks
    all_blocks_params = num_layers * params_per_block

    # Total parameters
    total_params = token_embedding_params + lm_head_params + all_blocks_params

    return {
        'num_layers': num_layers,
        'model_dim': model_dim,
        'num_heads': num_heads,
        'head_dim': head_dim,
        'vocab_size': vocab_size,
        'max_seq_len': max_seq_len,
        'token_embedding_params': token_embedding_params,
        'lm_head_params': lm_head_params,
        'params_per_block': params_per_block,
        'all_blocks_params': all_blocks_params,
        'total_params': total_params,
    }

def calculate_training_steps(num_params, target_param_data_ratio=20, total_batch_size=524288):
    """
    Calculate the number of training steps based on Chinchilla scaling.

    Args:
        num_params: Total number of model parameters
        target_param_data_ratio: Tokens per parameter ratio (default: 20 for Chinchilla)
        total_batch_size: Total batch size in tokens (default: 524288)

    Returns:
        Dictionary with training step information
    """
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size

    return {
        'target_tokens': target_tokens,
        'total_batch_size': total_batch_size,
        'num_iterations': num_iterations,
        'target_param_data_ratio': target_param_data_ratio,
    }

def main():
    parser = argparse.ArgumentParser(description='Calculate model parameters and training steps')
    parser.add_argument('--depth', type=int, default=20, help='Model depth (default: 20)')
    parser.add_argument('--vocab_size', type=int, default=65536, help='Vocabulary size (default: 65536)')
    parser.add_argument('--max_seq_len', type=int, default=2048, help='Max sequence length (default: 2048)')
    parser.add_argument('--target_param_data_ratio', type=int, default=20, help='Tokens per parameter ratio (default: 20)')
    parser.add_argument('--total_batch_size', type=int, default=524288, help='Total batch size in tokens (default: 524288)')
    args = parser.parse_args()

    # Calculate model parameters
    model_info = calculate_model_params(args.depth, args.vocab_size, args.max_seq_len)

    # Calculate training steps
    training_info = calculate_training_steps(
        model_info['total_params'],
        args.target_param_data_ratio,
        args.total_batch_size
    )

    # Print results
    print("=" * 80)
    print("MODEL CONFIGURATION")
    print("=" * 80)
    print(f"Depth:                  {args.depth}")
    print(f"Number of layers:       {model_info['num_layers']}")
    print(f"Model dimension:        {model_info['model_dim']:,}")
    print(f"Number of heads:        {model_info['num_heads']}")
    print(f"Head dimension:         {model_info['head_dim']}")
    print(f"Vocabulary size:        {model_info['vocab_size']:,}")
    print(f"Max sequence length:    {model_info['max_seq_len']:,}")
    print()

    print("=" * 80)
    print("PARAMETER COUNTS")
    print("=" * 80)
    print(f"Token embedding:        {model_info['token_embedding_params']:,}")
    print(f"LM head (unembedding):  {model_info['lm_head_params']:,}")
    print(f"Parameters per block:   {model_info['params_per_block']:,}")
    print(f"All transformer blocks: {model_info['all_blocks_params']:,}")
    print(f"TOTAL PARAMETERS:       {model_info['total_params']:,}")
    print()

    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Target param:data ratio: {training_info['target_param_data_ratio']}")
    print(f"Target tokens:           {training_info['target_tokens']:,}")
    print(f"Total batch size:        {training_info['total_batch_size']:,}")
    print(f"NUM TRAINING STEPS:      {training_info['num_iterations']:,}")
    print(f"Actual loop iterations:  {training_info['num_iterations'] + 1:,} (includes final eval)")
    print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"For depth={args.depth}, you would train for {training_info['num_iterations']:,} steps")
    print(f"Total tokens seen: {training_info['target_tokens']:,}")
    print(f"Tokens:Params ratio: {training_info['target_tokens'] / model_info['total_params']:.2f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
