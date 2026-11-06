"""Independently summarize a saved tuning-only deployment-dose archive."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import tarfile
import unicodedata

import numpy as np


def summarize(archive, manifest, n_boot=2000):
    archive, manifest = Path(archive), Path(manifest)
    rows_manifest = json.loads(manifest.read_text())
    with tarfile.open(archive) as tar:
        files = {m.name: m for m in tar.getmembers() if m.isfile()}
        assert set(files) == {r['path'] for r in rows_manifest}
        for row in rows_manifest:
            blob = tar.extractfile(files[row['path']]).read()
            assert len(blob) == row['bytes'] and hashlib.sha256(blob).hexdigest() == row['sha256']
        raw_bytes = tar.extractfile('tune.json').read()
        raw = json.loads(raw_bytes)
        scored = json.load(tar.extractfile('tune_scored_items.json'))
        summary = json.load(tar.extractfile('tune_scored.json'))
        snapshot = json.load(tar.extractfile('code_snapshot/manifest.json'))
        for name, expected in snapshot['sha256'].items():
            assert hashlib.sha256(tar.extractfile('code_snapshot/' + name).read()).hexdigest() == expected
    assert raw['completed'] and raw['reference_reproduced'] and scored['completed'] and summary['completed']
    assert raw['split'] == 'tune' and len(raw['evaluations']) == 192
    assert raw['code_snapshot_fingerprint'] == snapshot['fingerprint']
    assert hashlib.sha256(json.dumps(snapshot['sha256'], sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest() == snapshot['fingerprint']
    assert summary['source_generation_sha256'] == hashlib.sha256(raw_bytes).hexdigest()
    assert scored['conditions'] == raw['conditions']
    assert len(scored['evaluations']) == len(raw['evaluations'])
    for left, right in zip(raw['evaluations'], scored['evaluations']):
        for k in ('pair_id', 'question', 'gold', 'incorrect', 'outputs', 'finished'):
            assert left[k] == right[k]
        assert not right['gold_is_abstention']
        for name in raw['conditions']:
            assert right['outcome'][name] == right['judge'][name]['label']
            assert right['outcome'][name] in ('correct', 'wrong', 'abstain')
    rs = scored['evaluations']; n = len(rs)
    base = np.array([r['outcome']['baseline'] for r in rs])
    rng = np.random.default_rng(0); boot = rng.integers(0, n, size=(n_boot, n))
    result = {'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'archive_files': len(rows_manifest),
              'hashes_verified': True, 'n': n, 'split': 'tune', 'training_alpha': 3.,
              'reference_reproduced': True, 'n_boot': n_boot, 'seed': 0, 'arms': {}, 'comparisons': {}}
    labels = {}
    for name in raw['conditions']:
        y = np.array([r['outcome'][name] for r in rs]); labels[name] = y
        counts = {k: int((y == k).sum()) for k in ('correct', 'wrong', 'abstain')}
        bad = np.array(['\ufffd' in r['outputs'][name] for r in rs])
        unfinished = np.array([not r['finished'][name] for r in rs])
        result['arms'][name] = {'counts': counts, 'answered': counts['correct'] + counts['wrong'],
            'replacement_character': int(bad.sum()), 'unfinished': int(unfinished.sum()),
            'symbol_outputs_descriptive_only': sum(any(unicodedata.category(c) == 'So' for c in r['outputs'][name]) for r in rs),
            'mean_characters': sum(len(r['outputs'][name]) for r in rs) / n,
            'correct_without_replacement': int(((y == 'correct') & ~bad).sum()),
            'correct_without_replacement_or_truncation': int(((y == 'correct') & ~bad & ~unfinished).sum()),
            'transitions_vs_baseline': {a + '->' + b: int(((base == a) & (y == b)).sum())
                                       for a in ('correct', 'wrong', 'abstain') for b in ('correct', 'wrong', 'abstain')}}
        for k in ('correct', 'wrong', 'abstain'):
            assert abs(summary['rates'][name]['rates'][k] - counts[k] / n) < 1e-12
    for name in raw['conditions'][1:]:
        for ref in ['baseline', 'adapted_deploy_alpha3']:
            if name == ref:
                continue
            a, b = labels[name] == 'correct', labels[ref] == 'correct'
            delta = a.astype(int) - b.astype(int)
            plus, minus = int((delta > 0).sum()), int((delta < 0).sum())
            changed = plus + minus
            p = min(1., 2 * sum(math.comb(changed, k) for k in range(min(plus, minus) + 1)) / 2**changed) if changed else 1.
            result['comparisons'][name + '_minus_' + ref] = {'correct_count_delta': int(delta.sum()),
                'correct_rate_delta': float(delta.mean()), 'ci95': np.quantile(delta[boot].mean(1), [.025, .975]).tolist(),
                'corrected': plus, 'harmed': minus, 'exact_two_sided_p': p}
    result['limits'] = ['tuning-only diagnostic; no test result', 'same saved alpha3-trained weights, diagnostic deployment override only',
                        'natural answer counts differ; not a matched-coverage result',
                        'stored judge labels retained; no human relabeling',
                        'quality rules are descriptive re-counts, not replacements of stored grades']
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--archive', required=True, type=Path)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    result = summarize(a.archive, a.manifest)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
