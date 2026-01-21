"""
Test knowledge eval module. Example run:

python -m pytest tests/test_knowledge_eval.py -v

NOTE: On GPUs with ~15GB VRAM (e.g. RTX A4000), the full dataset causes OOM.
Halving the dataset in knowledge_eval.py is needed for testing.
The tokenizer vocab is 8192 tokens.
"""

import pytest
import torch
from nanochat.knowledge_eval import load_and_distribute_probes, evaluate_knowledge_probes
from nanochat.common import autodetect_device_type, compute_init, compute_cleanup
from nanochat.tokenizer import get_tokenizer
from nanochat.gpt import GPT, GPTConfig


@pytest.fixture
def device_type():
    return autodetect_device_type()


@pytest.fixture
def compute_context(device_type):
    yield compute_init(device_type)
    compute_cleanup()


@pytest.fixture
def tokenizer():
    return get_tokenizer()


@pytest.fixture
def blank_model(compute_context, tokenizer):
    """Create a small blank (randomly initialized) model for testing."""
    _, _, _, _, device = compute_context
    depth = 4
    max_seq_len = 128
    vocab_size = tokenizer.get_vocab_size()

    num_layers = depth
    model_dim = depth * 64
    num_heads = max(1, (model_dim + 127) // 128)
    num_kv_heads = num_heads

    model_config_kwargs = dict(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_kv_head=num_kv_heads,
        n_embd=model_dim,
    )
    with torch.device("meta"):
        model_config = GPTConfig(**model_config_kwargs)
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()
    model.bfloat16()
    model.eval()
    return model


def test_data_loading(compute_context, tokenizer):
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_context
    (local_probes, local_input_lengths, local_target_lengths,
     local_probe_types, global_counts, global_target_tokens
    ) = load_and_distribute_probes(tokenizer, device)


def test_knowledge_probe_ppl(blank_model, tokenizer, compute_context):
    """Test perplexity evaluation on a blank model."""
    _, _, _, _, device = compute_context
    results = evaluate_knowledge_probes(blank_model, tokenizer, device)

    # Blank model should have high perplexity (close to vocab size)
    assert 'mem_target_ppl' in results
    assert 'gen_target_ppl' in results
    assert 'hard_gen_target_ppl' in results
    assert 'mem_first_ppl' in results
    assert 'gen_first_ppl' in results
    assert 'hard_gen_first_ppl' in results

    print(f"Results: {results}")

