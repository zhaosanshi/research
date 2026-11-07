"""Side check: Llama pure-DAS L29 arm under Claude's release judge cache (same texts)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

rel = by_pid(load(RELEASE / 'llama3.1-8b/das_s42/tune_scored_items.json'))
f = sorted(p for p in rel if rel[p]['outcome']['baseline'] != 'correct' and rel[p]['outcome']['original_das'] == 'correct')
b = sorted(p for p in rel if rel[p]['outcome']['baseline'] == 'correct' and rel[p]['outcome']['original_das'] != 'correct')
print('release cache: baseline', counts(rel.values(), 'baseline'), 'original_das', counts(rel.values(), 'original_das'))
print('release cache original_das vs baseline fixed', f, 'broken', b, 'p', exact_sign_p(len(f), len(b)))
print('tqa:563 release label', rel['tqa:563']['outcome']['original_das'], '| tqa:476 baseline/original_das',
      rel['tqa:476']['outcome']['baseline'], rel['tqa:476']['outcome']['original_das'])
tr = load(X_R18 / 'qwen3-8b/ce_alpha3_ep3/das/adapter.json')['training_pair_ids']
test_ids = {e['pair_id'] for e in load(X_R18 / 'qwen3-8b/ce_alpha3_ep3/das/test.json')['evaluations']}
tune_ids = {e['pair_id'] for e in load(X_R18 / 'qwen3-8b/ce_alpha3_ep3/das/tune.json')['evaluations']}
print('training ids overlap: tune', len(set(tr) & tune_ids), 'test', len(set(tr) & test_ids), '| tune/test overlap', len(tune_ids & test_ids))
