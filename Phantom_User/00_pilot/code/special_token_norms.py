"""special_token_norms.py -- embedding / unembedding norms of Llama-3.1 chat-template tokens, base vs instruct (CPU only).

Prints each token's norm relative to the median norm of the ordinary vocabulary.  Used to check whether the base model
has trained embeddings for <|start_header_id|>, <|end_header_id|> and <|eot_id|> (it does not: they are exactly zero).

  MODEL_ROOT=/path/to/models python special_token_norms.py
"""
import json, os, sys, torch
from safetensors import safe_open
ROOT = os.environ.get("MODEL_ROOT", "models")
MODELS = {"base": os.path.join(ROOT, os.environ.get("BASE_DIR", "Llama-3.1-8B")),
          "instruct": os.path.join(ROOT, "Llama-3.1-8B-Instruct")}
TOK = {"<|begin_of_text|>": 128000, "<|start_header_id|>": 128006, "<|end_header_id|>": 128007, "<|eom_id|>": 128008,
       "<|eot_id|>": 128009, "<|python_tag|>": 128010, "<|reserved_special_token_10|>": 128019,
       "user": 882, "assistant": 78191, "\\n\\n": 271, " the": 279}
def tensor(path, name):
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(path, idx[name]), "pt") as f:
        return f.get_tensor(name).float()
for tag, path in MODELS.items():
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        W = tensor(path, name)
        n = W.norm(dim=1)
        med = n[:128000].median().item()
        row = {k: round(n[i].item() / med, 3) for k, i in TOK.items()}
        print(f"{tag:8s} {name:26s} median_norm={med:.4f}  relative norms: {row}", flush=True)
