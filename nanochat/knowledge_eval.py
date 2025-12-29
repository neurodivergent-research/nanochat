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
        dict with keys:
            - 'target_ppl': Perplexity on target tokens only
            - 'first_ppl': Perplexity on first target token only
            - 'full_ppl': Perplexity on full sequence
            - 'num_probes': Number of probes evaluated
    """
    # Get rank/world_size from dist (following core_eval.py pattern)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Data loading with lazy download (following base_eval.py pattern)
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

    # Load probe dataset
    with open(knowledge_file, 'r') as f:
        probe_dataset = json.load(f)  # 130 probes

    # TODO: Process probes and compute metrics

    return {
        'target_ppl': float('inf'),
        'first_ppl': float('inf'),
        'full_ppl': float('inf'),
        'num_probes': len(probe_dataset) * 15  # 130 probes * 15 samples each
    }
