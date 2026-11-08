#!/usr/bin/env python3
"""Independently audit E26 raw records; never import the experiment or core.

Usage: e26_independent_recount.py [--out-json PATH] [--bootstrap] RUN_DIR ...
Checks raw/graded identity, every quota/delivery, frozen choices/thresholds,
native counts, exact group-swap tests, and optionally all bootstrap intervals.
Missing or inconsistent artifacts fail rather than becoming a null effect.
"""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path

PROTOCOL = 'actual_coverage_v1'
OUTCOMES = {'correct', 'wrong', 'abstain'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=10000)
def tie(item, seed):
    return hashlib.sha256(json.dumps([PROTOCOL, seed, item], ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def score(row, kind, seed):
    if kind == 'random':
        return (int(tie(row['id'], seed)[:14], 16) >> 3) / float(2 ** 53)
    return row['simple_score'] if kind == 'simple' else row['score']


def ordered(rows, kind, seed):
    return sorted((r for r in rows if r['outcome'] != 'abstain'),
                  key=lambda r: (-score(r, kind, seed), tie(r['id'], seed), r['id']))


def quota(rows, q, kind, seed):
    f = Fraction(str(q))
    require(0 <= f <= 1, 'invalid coverage')
    k = f.numerator * len(rows) // f.denominator
    ranked = ordered(rows, kind, seed)
    if len(ranked) < k:
        return None
    chosen = {r['id'] for r in ranked[:k]}
    return {r['id']: r['outcome'] if r['id'] in chosen else 'abstain' for r in rows}


def exact(a, b, groups):
    require(set(a) == set(b) == set(groups), 'paired comparison cohort mismatch')
    require(sum(v != 'abstain' for v in a.values()) == sum(v != 'abstain' for v in b.values()),
            'paired comparison answer counts differ')
    differences = {i: int(a[i] == 'correct') - int(b[i] == 'correct') for i in a}
    grouped = defaultdict(int)
    for i, value in differences.items():
        grouped[groups[i]] += value
    distribution = Counter({0: 1})
    weights = [abs(v) for v in grouped.values() if v]
    for w in weights:
        nxt = Counter()
        for x, count in distribution.items():
            nxt[x - w] += count; nxt[x + w] += count
        distribution = nxt
    total = sum(differences.values())
    p = sum(v for x, v in distribution.items() if abs(x) >= abs(total)) / 2 ** len(weights)
    return {'p_value': p, 'delta_correct_count': total,
            'positive_discordant_rows': sum(v > 0 for v in differences.values()),
            'negative_discordant_rows': sum(v < 0 for v in differences.values()),
            'n_groups': len(grouped), 'informative_groups': len(weights),
            'answered_each': sum(v != 'abstain' for v in a.values())}


def near(a, b, label):
    if a is None or b is None:
        require(a is b, label)
    else:
        require(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=2e-12, rel_tol=2e-12), label)


def count_check(delivery, reported, label):
    require(reported.get('feasible') is (delivery is not None), label + ': feasibility')
    if delivery is None:
        require(reported.get('correct') is None, label + ': infeasible count')
        return
    counts = Counter(delivery.values()); answered = counts['correct'] + counts['wrong']
    for key in OUTCOMES:
        require(reported[key] == counts[key], label + ': ' + key)
    require(reported['answered'] == answered, label + ': answered')
    near(reported['actual_coverage'], answered / len(delivery), label + ': coverage')
    near(reported['accuracy_answered'], counts['correct'] / answered if answered else None, label + ': accuracy')
    require(reported['delivered_outcomes'] == delivery, label + ': per-item delivery')
    require(reported['n'] == len(delivery) and reported['ids'] == sorted(delivery), label + ': cohort ids/count')
    require(reported['mask'] == [delivery[i] != 'abstain' for i in sorted(delivery)], label + ': delivery mask')
    require(set(reported['selected_ids']) == {i for i, v in delivery.items() if v != 'abstain'}, label + ': selected ids')


def choose(tune, specs, family, q, seed):
    kind = 'simple' if family.endswith('_simple') else 'score'
    method, arm_kind = (family[:-7] if kind == 'simple' else family).rsplit('_', 1)
    candidates = []
    for name, spec in specs.items():
        if (spec['method'], spec['kind']) != (method, arm_kind):
            continue
        d = quota(tune[name], q, kind, seed)
        if d is None:
            continue
        e, c = spec.get('e') or {}, spec.get('c') or {}
        layers = e.get('layers', [e['layer']] if 'layer' in e else []) + ([c['layer']] if c else [])
        key = (-sum(v == 'correct' for v in d.values()), abs(e.get('alpha', 0)) + abs(c.get('alpha', 0)),
               len(layers), tuple(layers), name)
        candidates.append((key, name))
    return min(candidates)[1] if candidates else None


def arm_map(rows, picked):
    result = {'baseline_commitment': (rows['baseline'], 'score'),
              'baseline_simple': (rows['baseline'], 'simple'), 'baseline_random': (rows['baseline'], 'random')}
    for family, name in picked.items():
        if name is not None:
            require(name in rows, 'missing selected condition ' + name)
            result[family] = (rows[name], 'simple' if family.endswith('_simple') else 'score')
    return result


def check_policy(tune, test, q, kind, seed, policy):
    ranked = ordered(tune, kind, seed); f = Fraction(str(q)); k = f.numerator * len(tune) // f.denominator
    feasible = len(ranked) >= k
    require(policy['feasible'] == feasible and policy['n_tune'] == len(tune) and
            policy['k_tune'] == k and policy['n_available_tune'] == len(ranked), 'threshold counts changed')
    require(policy['seed'] == seed and policy['coverage_fraction'] == str(f), 'threshold identity changed')
    require(policy['tune_ids_sha256'] == hashlib.sha256(json.dumps(sorted(r['id'] for r in tune),
            ensure_ascii=False).encode()).hexdigest(), 'threshold tuning IDs changed')
    if not feasible:
        require(policy['mode'] is None, 'infeasible threshold has mode')
        return None
    mode = 'none' if k == 0 else 'all' if k == len(ranked) else 'cutoff'
    require(policy['mode'] == mode, 'threshold mode changed')
    if mode == 'cutoff':
        r = ranked[k - 1]
        require(policy['threshold'] == score(r, kind, seed) and policy['tie_cutoff'] == [tie(r['id'], seed), r['id']],
                'threshold boundary changed')
    delivery = {}
    for r in test:
        keep = mode == 'all' or (mode == 'cutoff' and (-score(r, kind, seed), tie(r['id'], seed), r['id']) <=
                (-policy['threshold'], *policy['tie_cutoff']))
        delivery[r['id']] = r['outcome'] if keep else 'abstain'
    return delivery


def load_rows(run, bank, data, task, split, selection):
    raw_path = run / ('%s_%s.json' % (task, split))
    scored_path = run / ('%s_%s%s.json' % (task, split, '_scored' if task == 'qa' else ''))
    raw, scored = read(raw_path), read(scored_path); cohort = data[task][split]
    expected = [str(r['id'] if task == 'tf' else r['pair_id']) for r in cohort]
    require(len(expected) == len(set(expected)), 'duplicate cohort IDs')
    specs = bank['conditions'][task]
    if split == 'test':
        names = {'baseline'} | {v for p in selection['tasks'][task]['choices'].values() for v in p.values() if v is not None}
        specs = [s for s in specs if s['name'] in names]
    names = {s['name'] for s in specs}
    for artifact in (raw, scored):
        require(artifact['completed'] and artifact['bank_fingerprint'] == bank['fingerprint'], 'incomplete/foreign artifact')
        require(artifact['model'] == bank['model'] and artifact['split'] == split and artifact['task'] == task,
                'artifact model/task/split changed')
        require(set(artifact['rows']) == names and artifact['condition_specs'] == specs, 'condition set changed')
    require(raw['fingerprint'] == fingerprint([bank['fingerprint'], task, split, specs, cohort]), 'collection fingerprint changed')
    if task == 'qa':
        require(scored['generation_sha256'] == digest(raw_path), 'scored generation source changed')
        require(scored['n_unresolved'] == 0, 'unresolved QA grades')
    excluded = None
    grade_by_text = {}
    for name in names:
        rows, originals = scored['rows'][name], raw['rows'][name]
        require([r['id'] for r in rows] == expected and [r['id'] for r in originals] == expected, 'incomplete/reordered arm')
        for r, original, question in zip(rows, originals, cohort):
            require(r['pair_id'] == str(question['pair_id']) and r['outcome'] in OUTCOMES, 'row identity/outcome changed')
            require(all(math.isfinite(r[k]) for k in ('score', 'simple_score')), 'nonfinite score')
            if task == 'qa':
                require(r['outcome'] == r['grade']['label'], 'outcome differs from grade')
                refusal_texts = {'i have no comment', 'i dont know', 'i do not know'}
                normalized_refs = [' '.join(''.join(c for c in str(x).lower() if c in 'abcdefghijklmnopqrstuvwxyz ').split())
                                   for x in question['gold']]
                only_refusal = all(x in refusal_texts for x in normalized_refs)
                require(r['gold_is_abstention'] is only_refusal, 'reference-only refusal exclusion differs')
                key = (r['id'], r['output'])
                require(key not in grade_by_text or grade_by_text[key] == r['outcome'], 'same answer has conflicting grades')
                grade_by_text[key] = r['outcome']
                require(all(r[k] == original[k] for k in ('output', 'score', 'simple_score', 'finished', 'edits')),
                        'grade modified inference record')
            else:
                require(type(r['predicted_true']) is bool and r['output'] == ('True' if r['predicted_true'] else 'False'),
                        'TF text differs from prediction')
                require(r['outcome'] == ('correct' if r['predicted_true'] == question['label'] else 'wrong'), 'wrong TF label derivation')
        omitted = {r['id'] for r in rows if r.get('gold_is_abstention')}
        if excluded is None:
            excluded = omitted
        require(omitted == excluded, 'arm-dependent QA cohort exclusion')
    return {n: [r for r in rs if r['id'] not in excluded] for n, rs in scored['rows'].items()}, {s['name']: s for s in specs}


def check_bootstrap(arms, q, seed, n_boot, entry):
    """Weighted-rank calculation, independent of core's expanded-row sorting."""
    import numpy as np
    base = sorted(next(iter(arms.values()))[0], key=lambda r: r['id'])
    gs = sorted({r['pair_id'] for r in base}); gindex = {g: i for i, g in enumerate(gs)}
    group_for_row = np.array([gindex[r['pair_id']] for r in base])
    rng = np.random.default_rng(seed); draws = rng.integers(0, len(gs), (n_boot, len(gs)))
    weights = np.array([np.bincount(draw, minlength=len(gs)) for draw in draws])[:, group_for_row]
    f = Fraction(str(q)); ns = weights.sum(1); ks = ns * f.numerator // f.denominator
    require((ks > 0).all(), 'bootstrap audit currently requires positive quotas')
    values, feasible = {}, {}; index = {r['id']: i for i, r in enumerate(base)}
    for name, (rows, kind) in arms.items():
        ranked = ordered(rows, kind, seed); w = weights[:, [index[r['id']] for r in ranked]]
        ok = w.sum(1) >= ks
        picked = np.minimum(w, np.maximum(0, ks[:, None] - (w.cumsum(1) - w)))
        correct = picked @ np.array([r['outcome'] == 'correct' for r in ranked], dtype=np.int64)
        vals = {'accuracy_answered': correct / ks, 'risk_answered': (ks - correct) / ks,
                'correct_rate': correct / ns, 'wrong_rate': (ks - correct) / ns}
        values[name], feasible[name] = vals, ok; reported = entry['arms'][name]['bootstrap']
        require(reported['infeasible_resamples'] == int((~ok).sum()), 'bootstrap failed-count mismatch')
        if ok.all() and entry['arms'][name]['feasible']:
            require(reported['status'] == 'ok', 'bootstrap arm status changed')
            for metric, val in vals.items():
                for a, b in zip(np.percentile(val, [2.5, 97.5]), reported['ci95'][metric]):
                    near(float(a), b, 'bootstrap arm interval ' + name + '/' + metric)
        else:
            require(reported['ci95'] is None, 'failed bootstrap reports conditional interval')
    for c in entry['comparisons']:
        a, b = c['arm_a'], c['arm_b']; ok = feasible[a] & feasible[b]
        require(c['infeasible_resamples'] == int((~ok).sum()), 'bootstrap pair failure count changed')
        if ok.all() and entry['arms'][a]['feasible'] and entry['arms'][b]['feasible']:
            for metric in values[a]:
                for x, y in zip(np.percentile(values[a][metric] - values[b][metric], [2.5, 97.5]), c['ci95'][metric]):
                    near(float(x), y, 'bootstrap comparison interval')
        else:
            require(c['ci95'] is None, 'failed paired bootstrap has interval')


def audit_run(run, bootstrap=False):
    run = Path(run)
    bank, data, rep, sel, delivered = [read(run / n) for n in ('bank.json', 'data.json', 'report.json', 'selection.json', 'delivered_answers.json')]
    require(rep['completed'] and not rep['pilot'], 'report incomplete or pilot')
    require(bank['fingerprint'] == fingerprint({k: v for k, v in bank.items() if k != 'fingerprint'}), 'bank fingerprint changed')
    require(bank['data_sha256'] == digest(run / 'data.json') and bank['bank_sha256'] == digest(run / 'bank.npz'), 'bank/data hash changed')
    require(sel['fingerprint'] == fingerprint({k: v for k, v in sel.items() if k != 'fingerprint'}), 'selection fingerprint changed')
    require(rep['selection_sha256'] == digest(run / 'selection.json'), 'report selection source changed')
    require(rep['model'] == bank['model'] and rep['bank_fingerprint'] == sel['bank_fingerprint'] == bank['fingerprint'], 'run provenance mismatch')
    for file, value in sel['sources'].items():
        require(digest(run / file) == value, 'tuning file modified after freeze')
    cfg, summaries = bank['config'], []; seed = cfg['seed']
    tasks = tuple(cfg.get('evaluation_tasks', ('tf', 'qa')))
    require(set(rep['tables']) == set(sel['tasks']) == set(delivered) == set(tasks), 'evaluated task set changed')
    for task in tasks:
        tune, specs = load_rows(run, bank, data, task, 'tune', sel)
        test, _ = load_rows(run, bank, data, task, 'test', sel)
        n_cells = n_choice = n_exact = 0
        require(set(sel['tasks'][task]['choices']) == set(map(str, cfg['coverages'])) == set(rep['tables'][task]), 'coverage grid changed')
        require(sel['tasks'][task]['tune_ids'] == [r['id'] for r in tune['baseline']], 'selection tune IDs changed')
        groups = {r['id']: r['pair_id'] for r in test['baseline']}
        for cov, picked in sel['tasks'][task]['choices'].items():
            expected_families = {m + '_' + k for m in {s['method'] for s in specs.values() if s['kind'] != 'base'}
                                 for k in (('E', 'C', 'EC', 'E_simple', 'C_simple', 'EC_simple') if cfg.get('include_joint_simple', False) else ('E', 'C', 'EC', 'E_simple', 'C_simple'))}
            require(set(picked) == expected_families, 'frozen method families changed')
            for family, name in picked.items():
                require(choose(tune, specs, family, cov, seed) == name, 'frozen choice differs: ' + family)
                n_choice += 1
            arms, tune_arms = arm_map(test, picked), arm_map(tune, picked)
            table = rep['tables'][task][cov]; entry = table['quota']['coverages'][0]
            require(entry['n'] == len(groups) and entry['k'] == int(Fraction(cov) * len(groups)), 'table denominator/quota changed')
            require(set(entry['arms']) == set(arms) == set(delivered[task][cov]), 'report/delivery arm set changed')
            require(set(table['unavailable_families']) == {f for f, v in picked.items() if v is None}, 'unavailable family list changed')
            require(set(table['frozen_threshold']) == set(arms) == set(table['native']), 'native/threshold arm set changed')
            require(set(sel['tasks'][task]['thresholds'][cov]) == set(arms), 'threshold family set changed')
            fresh = {}
            for name, (rows, kind) in arms.items():
                d = quota(rows, cov, kind, seed); fresh[name] = d; n_cells += 1
                count_check(d, entry['arms'][name], name)
                delivered_rows = delivered[task][cov][name]
                if d is None:
                    require(delivered_rows is None, 'infeasible delivery has rows')
                else:
                    byid = {r['id']: r for r in rows}
                    require(len(delivered_rows) == len(d) and {r['id'] for r in delivered_rows} == set(d), 'delivery missing/duplicate IDs')
                    for r in delivered_rows:
                        require(r['pair_id'] == groups[r['id']] and r['delivered_outcome'] == d[r['id']], 'delivery metadata changed')
                        expected_text = byid[r['id']]['output'] if d[r['id']] != 'abstain' else cfg['abstention_output']
                        require(r['output'] == expected_text, 'delivery output changed')
                native = {'n': len(rows), 'correct': sum(r['outcome'] == 'correct' for r in rows),
                          'answered': sum(r['outcome'] != 'abstain' for r in rows)}
                require(table['native'][name] == native, 'native count differs')
                policy = sel['tasks'][task]['thresholds'][cov][name]
                td = check_policy(tune_arms[name][0], rows, cov, kind, seed, policy)
                if td is not None:
                    count_check(td, table['frozen_threshold'][name], 'frozen threshold ' + name)
                else:
                    require(table['frozen_threshold'][name]['feasible'] is False, 'infeasible policy applied')
            comparisons = {tuple(sorted([c['arm_a'], c['arm_b']])): c for c in entry['comparisons']}
            require(len(comparisons) == len(entry['comparisons']) and set(comparisons) == set(itertools.combinations(sorted(arms), 2)),
                    'missing/duplicate/extra comparison')
            for c in comparisons.values():
                a, b = fresh[c['arm_a']], fresh[c['arm_b']]
                if a is None or b is None:
                    require(c['exact_fixed_deliveries'] is None, 'infeasible comparison has exact p')
                else:
                    ex = exact(a, b, groups)
                    for key, value in ex.items():
                        near(value, c['exact_fixed_deliveries'][key], 'exact statistic differs: ' + key)
                    n_exact += 1
                    ac = Counter(a.values()); bc = Counter(b.values()); k = ex['answered_each']; n = len(a)
                    expected_delta = {'accuracy_answered': (ac['correct']-bc['correct'])/k if k else None,
                                      'risk_answered': (ac['wrong']-bc['wrong'])/k if k else None,
                                      'correct_rate': (ac['correct']-bc['correct'])/n,
                                      'wrong_rate': (ac['wrong']-bc['wrong'])/n}
                    for metric, value in expected_delta.items():
                        near(value, c['point_delta'][metric], 'comparison point delta differs')
            if bootstrap:
                check_bootstrap(arms, cov, seed, cfg['n_boot'], entry)
        summaries.append({'task': task, 'test_arm_cells': n_cells, 'frozen_choices': n_choice,
                          'exact_comparisons': n_exact, 'n_test': len(groups), 'bootstrap_all_intervals_checked': bootstrap})
    return {'run': run.name, 'model': rep['model'], 'all_match': True, 'tasks': summaries,
            'scope': 'raw payloads, TF labels, QA grade consistency (not semantic regrading), quota/delivery/native counts, frozen choices/thresholds, exact p values' + (', all bootstrap CIs' if bootstrap else '')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-json', type=Path); parser.add_argument('--bootstrap', action='store_true')
    parser.add_argument('runs', type=Path, nargs='+'); args = parser.parse_args()
    results = [audit_run(p, args.bootstrap) for p in args.runs]
    out = {'all_match': True, 'runs': results,
           'test_arm_cells': sum(t['test_arm_cells'] for r in results for t in r['tasks']),
           'frozen_choices': sum(t['frozen_choices'] for r in results for t in r['tasks']),
           'exact_comparisons': sum(t['exact_comparisons'] for r in results for t in r['tasks'])}
    if args.out_json:
        args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
