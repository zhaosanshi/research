"""Extra JSON-level checks of side claims (training metadata, losses, gradient norms, identity fields)."""
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

print('--- 15 E27 adapters in the round-18 archive (excluding eos_pilot3)')
rows = []
for m in ['llama3.1-8b', 'qwen3-8b']:
    for run in ['ce_alpha1_ep3', 'ce_alpha3_ep3', 'ce_eos_alpha3_ep3']:
        for init in ['das', 'md', 'random']:
            p = X_R18 / m / run / init / 'adapter.json'
            if not p.exists():
                continue
            a = load(p)
            ga = list(a['gradient_audit'].values())
            assert len(ga) == 1
            g = ga[0]
            rows.append((m, run, init))
            print(f"  {m} {run} {init}: n_train_ids={len(a['training_pair_ids'])} epochs={a['train_config']['epochs']} "
                  f"ranking_weight={a['train_config']['ranking_weight']} updates={g['updates']} zero_updates={g['zero_updates']} "
                  f"grad_min={g['minimum']:.3f} grad_max={g['maximum']:.3f} losses={[round(x['loss'], 4) for x in a['losses']]} "
                  f"include_eos={a['train_config'].get('include_eos')} keys_has_stop_token_id={'stop_token_id' in a['train_config']}")
print('  adapters counted:', len(rows))
print('--- release adapter.json train_config keys (Qwen das_s42):', sorted(load(RELEASE / 'qwen3-8b/das_s42/adapter.json')['train_config']))
print('--- early adapter.json train_config keys (Qwen das):', sorted(load(X_R18 / 'qwen3-8b/ce_alpha3_ep3/das/adapter.json')['train_config']))

print('--- Qwen trace identity fields')
d = load(R19 / 'unicode/evidence/qwen_trace.json')
mi = d['model_identity']
print('  class', mi['loaded_class'], 'output_head_shape', mi['output_head_shape'], 'tokenizer', d['tokenizer']['class'],
      'backend_json_sha256', d['tokenizer']['backend_json_sha256'][:16], 'script sha', d['code_sha256']['scripts/review_checks/e27_trace_corruption.py'][:16])
snap = R19 / 'unicode/source_snapshot/e27_trace_corruption_a7b6736c6c55.py'
print('  source snapshot sha256:', hashlib.sha256(snap.read_bytes()).hexdigest()[:16])
for pid in ['tqa:138', 'tqa:150', 'tqa:412']:
    print(f"  {pid} len(output)={len(d['cases'][pid]['original_output'])}")

print('--- pure-DAS pilot: context questions vs stored questions')
pil_raw = load(X_PURE / 'llama_pure_das_pilot3/tune.json')
print('  pilot tune.json evaluations:', len(pil_raw['evaluations']), '| batch_context batches:',
      len(pil_raw.get('batch_context', [])), 'ids per batch:', [len(b['source_batch_ids']) for b in pil_raw.get('batch_context', [])])
cg = load(X_PURE / 'llama_pure_das_pilot3/check_generations.json')
if isinstance(cg, dict):
    print('  check_generations keys:', list(cg)[:12])
    ev = cg.get('evaluations') or cg.get('rows') or []
    print('  check_generations rows:', len(ev))
qs = load(X_PURE / 'llama_pure_das_tune192/questions.json')
src_q = V1U / 'llama3.1-8b/questions.json'
print('  questions.json byte-identical to E25 v1u source questions.json:',
      (X_PURE / 'llama_pure_das_tune192/questions.json').read_bytes() == src_q.read_bytes())
