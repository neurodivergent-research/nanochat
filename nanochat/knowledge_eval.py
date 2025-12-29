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
    # TODO: Implement knowledge probe evaluation
    # Will load data from get_base_dir()/knowledge_probes/ (similar to CORE metric)

    return {
        'target_ppl': float('inf'),
        'first_ppl': float('inf'),
        'full_ppl': float('inf'),
        'num_probes': 0
    }
