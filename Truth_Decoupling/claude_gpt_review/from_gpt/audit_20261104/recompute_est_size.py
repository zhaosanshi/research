#!/usr/bin/env python3
"""Offline audit of the pinned 96-vs-288 experiment. No model/API execution.

Reproduces the published floating-J selection and its stated scan-order tie
rule using exact integer counts. Reuses the exact same 2000 paired bootstrap
resamples for every arm and both selection versions. No test scores select a
condition in either version.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

DATA = None
SNAP = Path(__file__).resolve().parents[2] / 'from_claude'
OUT = None
MODELS = ('llama3.1-8b', 'qwen3-8b')
REPEATS = 2000
SEED = 0


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def m(ans, p, lab):
    pred = p > .5
    return {'J': float(pred[lab].mean() - pred[~lab].mean()),
            'AUROC': float((rankdata(p)[lab].sum() - lab.sum() * (lab.sum() + 1) / 2)
                           / (lab.sum() * (~lab).sum())),
            'ans': float(ans.mean()), 'B': float(pred.mean())}


def boot_metrics(ans, p, lab, ix):
    y, pr = lab[ix], p[ix] > .5
    n_pos, n_neg = y.sum(axis=1), (~y).sum(axis=1)
    return {'J': ((pr & y).sum(axis=1) * n_neg - (pr & ~y).sum(axis=1) * n_pos) / (n_pos * n_neg),
            'AUROC': ((rankdata(p[ix], axis=1) * y).sum(axis=1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg),
            'ans': ans[ix].mean(axis=1), 'B': pr.mean(axis=1)}


def correction_counts(base, pred, y):
    before, after = base == y, pred == y
    harmed, fixed = int((before & ~after).sum()), int((~before & after).sum())
    return {'n': len(y), 'baseline_correct': int(before.sum()), 'corrected': fixed, 'harmed': harmed,
            'net': fixed - harmed, 'harm_percent_of_baseline_correct': 100 * harmed / before.sum(),
            'accuracy_gain_pp': 100 * (fixed - harmed) / len(y),
            'baseline_accuracy': float(before.mean()), 'accuracy': float(after.mean())}


def run(model, records):
    def path(root, estimator, suffix):
        tag = 'das_tf_full' if estimator == 'DAS' else 'meandiff_full'
        return DATA / root / 'E23' / model / ('knob_sweep_' + tag + suffix)
    with np.load(path('results_das_tf_r288', 'DAS', '.npz'), allow_pickle=False) as z:
        ids = z['ids'].astype(str); labels = z['labels'].astype(bool); tune = z['half_a'].astype(bool)
    pair = np.array([item.rsplit(':', 1)[0] for item in ids])
    test = ~tune
    train = records['calib'] + records['dev']; eval_rows = records['test']
    train_ids = {row['pair_id'] for row in train}
    eval_ids = sorted({row['pair_id'] for row in eval_rows})
    rng_split = np.random.default_rng(42)
    moved = set(rng_split.choice(eval_ids, 192, replace=False).tolist())
    expected_tune = set(rng_split.choice(sorted(set(eval_ids) - moved), 192, replace=False).tolist())
    assert len(train_ids) == 96 and len(eval_ids) == 721
    assert set(pair) == set(eval_ids) - moved
    assert set(pair[tune]) == expected_tune and len(set(pair[test])) == 337
    assert not (train_ids & set(pair)) and not (moved & set(pair))
    groups = {pid: np.flatnonzero(pair == pid) for pid in set(pair)}
    assert all(len(g) == 2 and labels[g].sum() == 1 and tune[g[0]] == tune[g[1]] for g in groups.values())
    test_pairs = np.array(sorted(set(pair[test])))
    where = {pid: np.flatnonzero((pair == pid) & test) for pid in test_pairs}
    rng = np.random.default_rng(SEED)
    ix = np.array([np.concatenate([where[pid] for pid in rng.choice(test_pairs, len(test_pairs), replace=True)])
                   for _ in range(REPEATS)])
    arms, manifest = {}, {}
    for root, size in [('results_das_tf', 96), ('results_das_tf_r288', 288)]:
        sm = read(DATA / root / 'E1' / model / 'summary.json')
        assert sm['n_E'] == [size, size] and sm['das']['n_pairs']['E'] == size
        assert sm['das']['seeds'] == [42] and sm['das']['anchor'] == 'prompt'
        if size == 288:
            assert set(sm['est_extra']['pair_ids']) == moved
        for estimator in ('DAS', 'MD'):
            jp, zp = path(root, estimator, '.json'), path(root, estimator, '.npz')
            j = read(jp)
            with np.load(zp, allow_pickle=False) as archive:
                z = {key: archive[key] for key in archive.files}
            assert len(set(z['ids'])) == len(z['ids'])
            positions = {item: i for i, item in enumerate(z['ids'].astype(str))}
            idx = np.array([positions[item] for item in ids])
            assert np.array_equal(z['labels'][idx].astype(bool), labels)
            assert j['n_est'] == 2 * size and j['wording'] == 'pik' and j['e_alphas'] == [2., 3.]
            if size == 288:
                assert np.array_equal(z['half_a'][idx].astype(bool), tune)
                assert set(j['est_extra_ids']) == moved
            else:
                old_pair = np.array([item.rsplit(':', 1)[0] for item in z['ids'].astype(str)])
                assert set(old_pair[z['half_a'].astype(bool)]) == moved
            names = [name for name in j['conds'] if name.startswith('E_')]
            assert names == ['E_L%d_a%d' % (l, a) for l in j['layers'] for a in (2, 3)]
            a = {'json': j, 'names': names,
                 'p': {name: z[name + '__ptrue'][idx].astype(float) for name in ['baseline'] + names},
                 'ans': {name: z[name + '__answer'][idx].astype(bool) for name in ['baseline'] + names}}
            assert all(np.isfinite(v).all() for v in a['p'].values())
            arms[estimator + '-' + str(size)] = a
            manifest[estimator + '-' + str(size)] = {'json': str(jp), 'json_sha256': sha(jp), 'npz': str(zp), 'npz_sha256': sha(zp)}
    assert arms['DAS-96']['names'] == arms['MD-96']['names'] == arms['DAS-288']['names'] == arms['MD-288']['names']
    for size in (96, 288):
        for key in ('p', 'ans'):
            assert np.array_equal(arms['DAS-%d' % size][key]['baseline'], arms['MD-%d' % size][key]['baseline'])

    def choose(a, names, stable=False):
        y = labels[tune]; base = m(a['ans']['baseline'][tune], a['p']['baseline'][tune], y)
        ok = [n for n in names if abs(a['ans'][n][tune].mean() - base['ans']) <= .03 + 1e-12]
        def key(n):
            pred = a['p'][n][tune] > .5
            if stable:
                return int((pred & y).sum()) * int((~y).sum()) - int((pred & ~y).sum()) * int(y.sum())
            return float(pred[y].mean() - pred[~y].mean()) - base['J']
        return max(ok, key=key) if ok else None

    cache = {}
    def evaluate(label, name):
        key = label, name
        if key not in cache:
            a = arms[label]
            cache[key] = m(a['ans'][name][test], a['p'][name][test], labels[test]), boot_metrics(a['ans'][name], a['p'][name], labels, ix)
        return cache[key]

    result = {'input_manifest': manifest, 'split_verified': {'original_train': 96, 'moved_old_tune': 192,
               'new_tune': 192, 'new_test': 337, 'all_disjoint': True, 'same_candidate_grid': True,
               'new_tune_and_test_are_subsets_of_previously_seen_old_test': True}, 'selections': {}, 'versions': {}}
    for label, a in arms.items():
        float_name, int_name = choose(a, a['names']), choose(a, a['names'], True)
        best_count = int(((a['p'][int_name][tune] > .5) == labels[tune]).sum())
        base = m(a['ans']['baseline'][tune], a['p']['baseline'][tune], labels[tune])
        ties = []
        for name in a['names']:
            pr = a['p'][name][tune] > .5
            if abs(a['ans'][name][tune].mean() - base['ans']) <= .03 + 1e-12 and int((pr == labels[tune]).sum()) == best_count:
                ties.append({'name': name, 'correct_count': best_count,
                             'float_delta_J': float(pr[labels[tune]].mean() - pr[~labels[tune]].mean()) - base['J']})
        result['selections'][label] = {'published_float_choice': float_name, 'integer_scan_order_choice': int_name, 'exact_best_ties': ties}
    for version in ('published_float', 'integer_scan_order'):
        evaluated, draws = {}, {}
        for label, a in arms.items():
            name = result['selections'][label][version + '_choice']
            values, boot_values = evaluate(label, name)
            base, boot_base = evaluate(label, 'baseline')
            delta = {k: values[k] - base[k] for k in values}
            draws[label] = {k: boot_values[k] - boot_base[k] for k in values}
            counts = correction_counts(a['p']['baseline'][test] > .5, a['p'][name][test] > .5, labels[test])
            evaluated[label] = {'chosen': name, 'test_metrics': values, 'baseline_metrics': base, 'delta': delta,
                                'ci95': {k: np.quantile(draws[label][k], [.025, .975]).tolist() for k in values}, 'corrections': counts}
        differences = {}
        for left, right in [('DAS-288', 'DAS-96'), ('MD-288', 'MD-96'), ('DAS-288', 'MD-288'), ('DAS-96', 'MD-96')]:
            differences[left + ' - ' + right] = {k: {'value': evaluated[left]['delta'][k] - evaluated[right]['delta'][k],
                    'ci95': np.quantile(draws[left][k] - draws[right][k], [.025, .975]).tolist()}
                    for k in ('J', 'AUROC', 'ans', 'B')}
        result['versions'][version] = {'arms': evaluated, 'paired_differences': differences}
    published = read(SNAP / 'results/01_E估计器/估计集扩样_96vs288对' / (model + '_est_size.json'))
    original = result['versions']['published_float']
    assert published['chosen'] == {label: row['chosen'] for label, row in original['arms'].items()}
    for pair_name, metrics in published['paired_differences'].items():
        for metric, row in metrics.items():
            reproduced = original['paired_differences'][pair_name][metric]
            np.testing.assert_allclose([row['value']] + row['ci'], [reproduced['value']] + reproduced['ci95'], rtol=0, atol=1e-12)
    result['published_values_and_all_paired_CIs_reproduced'] = True
    result['baseline_drift'] = {}
    old, new = arms['DAS-96'], arms['DAS-288']
    diffs = np.abs(old['p']['baseline'] - new['p']['baseline'])
    for split, mask in [('common_529', np.ones(len(ids), bool)), ('tune_192', tune), ('test_337', test)]:
        ob = m(old['ans']['baseline'][mask], old['p']['baseline'][mask], labels[mask])
        nb = m(new['ans']['baseline'][mask], new['p']['baseline'][mask], labels[mask])
        result['baseline_drift'][split] = {'max_abs_ptrue_change': float(diffs[mask].max()),
             'tf_label_flips': int(((old['p']['baseline'] > .5) != (new['p']['baseline'] > .5))[mask].sum()),
             'self_report_flips': int((old['ans']['baseline'] != new['ans']['baseline'])[mask].sum()),
             'new_minus_old_metrics': {k: nb[k] - ob[k] for k in ob}}
    result['baseline_drift']['largest_ptrue_changes'] = [{'id': str(ids[i]), 'label': bool(labels[i]),
          'old_ptrue': float(old['p']['baseline'][i]), 'new_ptrue': float(new['p']['baseline'][i]), 'in_new_tune': bool(tune[i])}
          for i in np.argsort(-diffs)[:5]]
    result['fixed_config_decomposition'] = {}
    for estimator in ('DAS', 'MD'):
        old_label, new_label = estimator + '-96', estimator + '-288'
        oc, nc = (result['selections'][label]['published_float_choice'] for label in (old_label, new_label))
        old_base, _ = evaluate(old_label, 'baseline'); new_base, _ = evaluate(new_label, 'baseline')
        oo, _ = evaluate(old_label, oc); on, _ = evaluate(old_label, nc); nn, _ = evaluate(new_label, nc)
        total = (nn['J'] - new_base['J']) - (oo['J'] - old_base['J'])
        fixed = (nn['J'] - new_base['J']) - (on['J'] - old_base['J'])
        result['fixed_config_decomposition'][estimator] = {'old_choice': oc, 'new_choice': nc,
             'total_delta_J_change': total, 'change_at_fixed_new_choice': fixed,
             'choice_change_at_old_direction': (on['J'] - oo['J'])}
    return result


def main():
    global DATA, SNAP, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True,
                        help='directory containing results_das_tf, results_das_tf_r288, and code/data')
    parser.add_argument('--from-claude-root', type=Path, default=SNAP,
                        help='pinned from_claude snapshot containing the published comparison JSONs')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    DATA, SNAP, OUT = args.data_root, args.from_claude_root, args.out
    if OUT.exists():
        parser.error('output already exists; use a new audit path')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    records = {split: [json.loads(line) for line in (DATA / 'code/data/processed/truthfulqa_tf' / (split + '.jsonl')).read_text().splitlines()]
               for split in ('calib', 'dev', 'test')}
    result = {'pinned_commit': 'cb3f67b846a89cbb25c0bd28140514a43c228556', 'bootstrap_repeats': REPEATS,
              'bootstrap_seed': SEED, 'models': {}}
    for model in MODELS:
        result['models'][model] = run(model, records)
        row = result['models'][model]
        print(model, 'published numbers and CIs reproduced:', row['published_values_and_all_paired_CIs_reproduced'])
        for label, selection in row['selections'].items():
            print(label, selection)
        for version, summary in row['versions'].items():
            print(version)
            for label, arm in summary['arms'].items():
                print(label, arm['chosen'], arm['corrections'])
            for pair, values in summary['paired_differences'].items():
                print(pair, 'J', values['J'], 'AUROC', values['AUROC'])
        print('baseline_drift', row['baseline_drift'])
        print('fixed_config_decomposition', row['fixed_config_decomposition'])
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=lambda x: x.item()) + '\n')
    print('written', OUT)


if __name__ == '__main__':
    main()
