"""Compare judge labels on identical texts: early E27 cache vs release-rerun dose cache (Qwen tune)."""
import json
import sys
import tarfile

early_tar, dose_tar = sys.argv[1], sys.argv[2]
with tarfile.open(early_tar) as t:
    early = json.load(t.extractfile('qwen3-8b/ce_alpha3_ep3/das/tune_scored_items.json'))
with tarfile.open(dose_tar) as t:
    dose = json.load(t.extractfile('tune_scored_items.json'))

pairs = {'baseline': 'baseline', 'original_das': 'original_das_alpha3',
         'das_generation_adapter': 'adapted_deploy_alpha3'}
e_by = {r['pair_id']: r for r in early['evaluations']}
for old, new in pairs.items():
    same_text = flips = 0
    detail = []
    for r in dose['evaluations']:
        o = e_by[r['pair_id']]
        if o['outputs'][old] != r['outputs'][new]:
            continue
        same_text += 1
        if o['outcome'][old] != r['outcome'][new]:
            flips += 1
            detail.append((r['pair_id'], o['outcome'][old], r['outcome'][new], r['outputs'][new][:80]))
    print(old, '->', new, 'identical texts', same_text, 'label flips', flips)
    for d in detail:
        print('   ', d)
