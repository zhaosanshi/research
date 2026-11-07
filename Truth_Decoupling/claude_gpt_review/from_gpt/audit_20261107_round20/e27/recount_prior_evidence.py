"""Count original archive rows independently, without importing prior audit scripts."""
import hashlib
import json
from collections import Counter
from math import comb
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[3]
PRIOR = ROOT / 'from_gpt/audit_20261106_round19'
OUT = Path(__file__).resolve().parent


def recount(archive, member):
    with tarfile.open(archive) as tar:
        raw_bytes = tar.extractfile(member.replace('_scored_items', '')).read()
        raw = json.loads(raw_bytes)
        scored = json.load(tar.extractfile(member))
    assert scored['grading_completed'] and scored['completed'] and raw['completed']
    assert scored['grading_source_generation_sha256'] == hashlib.sha256(raw_bytes).hexdigest()
    assert [x['pair_id'] for x in raw['evaluations']] == [x['pair_id'] for x in scored['evaluations']]
    assert len({x['pair_id'] for x in scored['evaluations']}) == len(scored['evaluations'])
    for a, b in zip(raw['evaluations'], scored['evaluations']):
        assert all(a[k] == b[k] for k in ('question', 'gold', 'outputs', 'finished'))
        assert all(b['outcome'][c] == b['judge'][c]['label'] for c in raw['conditions'])
    rows = [r for r in scored['evaluations'] if not r['gold_is_abstention']]
    result = {'n': len(rows), 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'conditions': {}}
    for condition in raw['conditions']:
        up = sum(r['outcome'][condition] == 'correct' and r['outcome']['baseline'] != 'correct' for r in rows)
        down = sum(r['outcome'][condition] != 'correct' and r['outcome']['baseline'] == 'correct' for r in rows)
        n = up + down
        exact = min(1., 2 * sum(comb(n, i) for i in range(min(up, down) + 1)) / 2**n) if n else 1.
        result['conditions'][condition] = {
            'labels': dict(Counter(r['outcome'][condition] for r in rows)),
            'fffd': sum('\ufffd' in r['outputs'][condition] for r in rows),
            'unfinished': sum(not r['finished'][condition] for r in rows),
            'fffd_and_unfinished': sum('\ufffd' in r['outputs'][condition] and not r['finished'][condition] for r in rows),
            'strict_correct': sum(r['outcome'][condition] == 'correct' and '\ufffd' not in r['outputs'][condition]
                                  and r['finished'][condition] for r in rows),
            'corrected': up, 'harmed': down, 'net': up - down, 'exact_p': exact}
    return result


def main():
    result = {}
    for model in ('llama3.1-8b', 'qwen3-8b'):
        result['random_' + model] = recount(PRIOR / 'random/evidence/random_controls.tar.gz', model + '/full192/tune_scored_items.json')
    result['qwen_dose'] = recount(PRIOR / 'unicode/evidence/generation_dose_full192.tar.gz', 'tune_scored_items.json')
    result['llama_pure_layers'] = recount(PRIOR / 'models/evidence/pure_das_layer_results.tar.gz',
                                         'llama_pure_das_tune192/tune_scored_items.json')
    (OUT / 'prior_round19_recount.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    for name, item in result.items():
        print(name, item['n'], {c: (s['labels'].get('correct', 0), s['fffd'], s['unfinished'], s['net'], s['exact_p'])
                                 for c, s in item['conditions'].items()})


if __name__ == '__main__':
    main()
