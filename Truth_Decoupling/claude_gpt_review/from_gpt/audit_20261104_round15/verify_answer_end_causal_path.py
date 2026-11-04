"""Check QA-DAS answer-end -> later self-check causal paths on a random CPU Llama.

No pretrained weights or API calls. The optional import switches are local to
this process and avoid unrelated audio/image/sklearn dependencies.
"""
import argparse
import json
from pathlib import Path
import sys

import transformers.utils.import_utils as import_utils
import_utils._sklearn_available = False
import_utils._librosa_available = False
import_utils._torchvision_available = False

import torch
from transformers import LlamaConfig, LlamaForCausalLM

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--code-root', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    parser.error('output exists; choose a new path')
sys.path.insert(0, str(args.code_root))
from core.directions import _interchange_hook_factory

torch.manual_seed(7)
model = LlamaForCausalLM(LlamaConfig(
    vocab_size=32, hidden_size=16, intermediate_size=32,
    num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
    max_position_embeddings=32, attention_dropout=0.0,
)).cpu().eval()
model.requires_grad_(False)
base_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
source_ids = torch.tensor([[8, 9, 10, 4, 5, 6]])
raw_init = torch.randn(16, 1)
with torch.no_grad():
    baseline = model(base_ids, use_cache=False).logits
rows = []
for layer, write_pos in [(0, 2), (1, 2), (1, 5)]:
    saved = {}
    def capture(module, inputs, output):
        saved['h'] = output[0] if isinstance(output, tuple) else output
    handle = model.model.layers[layer].register_forward_hook(capture)
    with torch.no_grad():
        model(source_ids, use_cache=False)
    handle.remove()
    holder = {'h': saved['h'][:, write_pos, :].detach()}
    raw = raw_init.clone().requires_grad_(True)
    handle = model.model.layers[layer].register_forward_hook(
        _interchange_hook_factory(lambda: raw / raw.norm(), holder,
                                  positions='last', write_pos=write_pos))
    try:
        logits = model(base_ids, use_cache=False).logits
    finally:
        handle.remove()
    loss = torch.nn.functional.cross_entropy(logits[:, 5, :], torch.tensor([7]))
    loss.backward()
    rows.append({
        'layer': layer, 'write_position': write_pos, 'label_predictor': 5,
        'gradient_norm': float(raw.grad.norm()),
        'label_max_logit_change': float((logits[:, 5] - baseline[:, 5]).detach().abs().max()),
        'written_position_max_logit_change': float((logits[:, write_pos] - baseline[:, write_pos]).detach().abs().max()),
    })
assert rows[0]['gradient_norm'] > 0 and rows[0]['label_max_logit_change'] > 0
assert rows[1]['gradient_norm'] == 0 and rows[1]['label_max_logit_change'] == 0
assert rows[1]['written_position_max_logit_change'] > 0
assert rows[2]['gradient_norm'] > 0 and rows[2]['label_max_logit_change'] > 0
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(rows, indent=2) + '\n')
print(json.dumps(rows, indent=2))
print('PASS: last block output at an earlier answer position cannot affect later self-check logits')
