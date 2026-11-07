#!/usr/bin/env python3
"""E26 commitment dose probe using the exact production precision and batches.

Only tuning inputs determine measured margins; gold labels/references are never
used by inference or the summaries. The complete source artifacts are read for
hash verification. This is probe protocol v2: old v1 files used one large batch,
native output projection and ordinal ranks for ties, and must remain labeled v1.
"""
import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np


def average_ranks(values):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError('ranks need a nonempty finite vector')
    order = np.argsort(a, kind='stable'); result = np.zeros(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        result[order[i:j]] = (i + j - 1) / 2
        i = j
    return result


def spearman(a, b):
    x, y = average_ranks(a), average_ranks(b)
    if x.shape != y.shape:
        raise ValueError('rank vectors differ in length')
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def validate_probe(n, alphas):
    if n < 1:
        raise ValueError('n must be positive')
    if not alphas or any(not math.isfinite(a) for a in alphas) or len(set(alphas)) != len(alphas):
        raise ValueError('alphas must be unique finite numbers')


def batched_margins(e26, bundle, meta, bank, task, rows, condition):
    batch_size = meta['config']['batch_size']; values = []
    for start in range(0, len(rows), batch_size):
        texts = [e26.gate_text(bundle, meta, task, r) for r in rows[start:start + batch_size]]
        lm = e26.gate_map(bundle, texts[0], meta['config'])
        logits = e26.one_step(bundle, texts, e26.intervention_specs(bank, condition, 'gate'))
        values.extend((logits[:, lm['yes']] - logits[:, lm['no']]).tolist())
    result = np.asarray(values)
    if len(result) != len(rows) or not np.isfinite(result).all():
        raise ValueError('missing/nonfinite probe margins')
    return result


def run_probe(args, e26, bundle=None):
    validate_probe(args.n, args.alphas)
    if args.out.exists():
        raise ValueError('probe output already exists; preserve the previous protocol/result')
    meta, bank, data = e26.load_run(args)
    bundle = e26.open_bundle(meta) if bundle is None else bundle
    e26.verify_tokens(bundle, meta)
    code_root = Path(e26.__file__).resolve().parents[1]
    source_files = [Path(__file__).resolve(), Path(e26.__file__).resolve(),
                    code_root / 'core/qa_precision.py', code_root / 'core/joint_steering.py',
                    code_root / 'core/intervene.py', code_root / 'core/model_io.py']
    out = {'protocol': 'e26_c_dose_probe_v2', 'model': args.model, 'bank_fingerprint': meta['fingerprint'],
           'source_sha256': {f: e26.digest(args.run_dir / f) for f in ('bank.json', 'bank.npz', 'data.json')},
           'code_sha256': {str(p.relative_to(code_root)): e26.digest(p) for p in source_files},
           'n_requested': args.n, 'alphas': args.alphas, 'batch_size': meta['config']['batch_size'],
           'precision': 'float32 output projection, native frozen backbone (matches E26)',
           'rank_rule': 'average ranks for ties; undefined correlation is null', 'tasks': {}}
    with e26.fp32_output_projection(bundle):
        for task in e26.TASKS:
            rows = data[task]['tune'][:args.n]
            if not rows:
                raise ValueError('empty tuning input cohort')
            base = {'name': 'baseline', 'kind': 'base', 'task': task, 'method': 'base', 'e': None, 'c': None}
            m0 = batched_margins(e26, bundle, meta, bank, task, rows, base)
            sd = float(np.std(m0))
            rec = {'n_actual': len(rows), 'ids': [str(r.get('id', r['pair_id'])) for r in rows],
                   'baseline_margins': m0.tolist(), 'baseline_yes_share': float((m0 > 0).mean()),
                   'baseline_margin_sd': sd, 'methods': {}}
            out['tasks'][task] = rec
            for method, layers in meta['c_layers'][task].items():
                for layer in layers:
                    res = {}
                    for a in args.alphas:
                        cond = {'name': 'probe', 'kind': 'C', 'task': task, 'method': method, 'e': None,
                                'c': {'layer': int(layer), 'alpha': float(a)}}
                        m = batched_margins(e26, bundle, meta, bank, task, rows, cond)
                        res[str(a)] = {'margins': m.tolist(), 'mean_shift': float((m-m0).mean()),
                            'shift_in_sd': float((m-m0).mean()/sd) if sd > 0 else None,
                            'yes_share': float((m > 0).mean()), 'flip_share': float(((m > 0) != (m0 > 0)).mean()),
                            'rank_spearman_vs_base': spearman(m, m0)}
                        print(task, method, 'L%d' % layer, 'alpha', a, 'shift', res[str(a)]['mean_shift'], flush=True)
                    rec['methods']['%s_L%d' % (method, layer)] = res
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model'); ap.add_argument('run_dir', type=Path); ap.add_argument('out', type=Path)
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--alphas', type=float, nargs='+', default=[-2., -1., 1., 2.]); args = ap.parse_args()
    validate_probe(args.n, args.alphas)
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]
    import e26_matched_coverage as e26
    run_probe(args, e26)
    print('saved', args.out)


if __name__ == '__main__':
    main()
