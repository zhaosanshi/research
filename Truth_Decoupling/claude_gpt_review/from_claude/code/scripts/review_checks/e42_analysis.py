#!/usr/bin/env python3
"""E42 analysis on the judged tuning run (tune_scored_items.json next to tune.json).

usage: e42_analysis.py RUN_DIR [--n-boot 2000] [--seed 0]
RUN_DIR/tune.json must be an e42 oracle or select run; RUN_DIR/tune_scored_items.json its e25-judge grades.
Oracle run : per arm -> answered / correct / wrong / abstain, outputs changed vs baseline, wrong->correct and
             correct->wrong counts, paired bootstrap CI of (correct - baseline correct).
Select run : per pick rule -> the picked candidate's answered / correct, Δcorrect vs greedy, vs random
             (expectation over the non-abstaining pool), vs max_logp, with paired bootstrap CIs;
             pool statistics (distinct candidates, questions with >= 1 correct candidate = oracle upper bound).
Questions whose gold is abstention-only are excluded (as e32 does). Nothing here selects a layer or a rule on the
test split: this is the tuning-set diagnostic that decides whether a test-set run is worth doing.
"""
import argparse
import json
import os
import sys

import numpy as np

OUTCOMES = ('correct', 'wrong', 'abstain')


def load(run_dir):
    raw = json.load(open(os.path.join(run_dir, 'tune.json'), encoding='utf-8'))
    scored = json.load(open(os.path.join(run_dir, 'tune_scored_items.json'), encoding='utf-8'))
    if not raw.get('completed') or not scored.get('grading_completed', scored.get('completed')):
        raise SystemExit('generation or grading incomplete')
    by_id = {r['pair_id']: r for r in scored['evaluations']}
    rows = []
    for r in raw['evaluations']:
        g = by_id[r['pair_id']]
        if g.get('gold_is_abstention'):
            continue
        for c in raw['conditions']:
            if g['outcome'][c] not in OUTCOMES:
                raise SystemExit('unresolved grade for %s/%s — resolve the human review queue first' % (r['pair_id'], c))
        rows.append({**r, 'outcome': g['outcome']})
    return raw, rows


def boot_ci(diffs, n_boot, seed):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, dtype=float)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    sums = d[idx].sum(1)
    return float(d.sum()), [float(np.percentile(sums, 2.5)), float(np.percentile(sums, 97.5))]


def oracle_report(raw, rows, n_boot, seed):
    base = 'baseline'
    out = {'n_questions': len(rows), 'arms': {}}
    lines = ['| arm | answered | correct | wrong | abstain | output changed | wrong→correct | correct→wrong | Δcorrect vs baseline [95% CI] |',
             '|---|---|---|---|---|---|---|---|---|']
    for c in raw['conditions']:
        o = [r['outcome'][c] for r in rows]
        ob = [r['outcome'][base] for r in rows]
        changed = sum(r['outputs'][c].strip() != r['outputs'][base].strip() for r in rows)
        w2c = sum(a == 'wrong' and b == 'correct' for a, b in zip(ob, o))
        c2w = sum(a == 'correct' and b == 'wrong' for a, b in zip(ob, o))
        diffs = [(b == 'correct') - (a == 'correct') for a, b in zip(ob, o)]
        d, ci = boot_ci(diffs, n_boot, seed)
        arm = {'answered': sum(x != 'abstain' for x in o), 'correct': o.count('correct'), 'wrong': o.count('wrong'),
               'abstain': o.count('abstain'), 'output_changed': changed, 'wrong_to_correct': w2c, 'correct_to_wrong': c2w,
               'delta_correct': d, 'delta_correct_ci95': ci}
        out['arms'][c] = arm
        lines.append('| %s | %d | %d | %d | %d | %d | %d | %d | %+d [%+.0f, %+.0f] |' % (
            c, arm['answered'], arm['correct'], arm['wrong'], arm['abstain'], changed, w2c, c2w, d, ci[0], ci[1]))
    return out, '\n'.join(lines)


def select_report(raw, rows, n_boot, seed):
    cands = raw['conditions']
    rules = sorted({k for r in rows for k in r['picks']})
    out = {'n_questions': len(rows), 'rules': {}, 'pool': {}}
    # pool statistics
    distinct = [len({r['outputs'][c].strip() for c in cands}) for r in rows]
    any_correct = [any(r['outcome'][c] == 'correct' for c in cands) for r in rows]
    all_abstain = [all(r['outcome'][c] == 'abstain' for c in cands) for r in rows]
    out['pool'] = {'mean_distinct_candidates': float(np.mean(distinct)),
                   'questions_with_a_correct_candidate': int(sum(any_correct)),
                   'questions_all_abstain': int(sum(all_abstain)),
                   'greedy_correct': sum(r['outcome']['greedy'] == 'correct' for r in rows)}
    # expected random pick: mean over the non-abstaining pool (fallback: all candidates)
    def pool(r):
        p = [c for c in cands if not r['rule_abstain'][c] and r['outputs'][c].strip()]
        return p or cands
    exp_random = [np.mean([r['outcome'][c] == 'correct' for c in pool(r)]) for r in rows]
    exp_random_ans = [np.mean([r['outcome'][c] != 'abstain' for c in pool(r)]) for r in rows]
    lines = ['| pick rule | answered | correct | Δcorrect vs greedy [95% CI] | Δcorrect vs E[random] [95% CI] | Δcorrect vs max_logp [95% CI] |',
             '|---|---|---|---|---|---|']
    greedy_c = [r['outcome']['greedy'] == 'correct' for r in rows]
    logp_c = [r['outcome'][r['picks']['max_logp']] == 'correct' for r in rows]
    for rule in rules:
        picked = [r['picks'][rule] for r in rows]
        o = [r['outcome'][p] for r, p in zip(rows, picked)]
        corr = [x == 'correct' for x in o]
        dg, cig = boot_ci([int(a) - int(b) for a, b in zip(corr, greedy_c)], n_boot, seed)
        dr, cir = boot_ci([int(a) - b for a, b in zip(corr, exp_random)], n_boot, seed)
        dl, cil = boot_ci([int(a) - int(b) for a, b in zip(corr, logp_c)], n_boot, seed)
        res = {'answered': sum(x != 'abstain' for x in o), 'correct': sum(corr),
               'delta_vs_greedy': dg, 'delta_vs_greedy_ci95': cig, 'delta_vs_random': dr, 'delta_vs_random_ci95': cir,
               'delta_vs_max_logp': dl, 'delta_vs_max_logp_ci95': cil}
        out['rules'][rule] = res
        lines.append('| %s | %d | %d | %+d [%+.0f, %+.0f] | %+.1f [%+.1f, %+.1f] | %+d [%+.0f, %+.0f] |' % (
            rule, res['answered'], res['correct'], dg, cig[0], cig[1], dr, cir[0], cir[1], dl, cil[0], cil[1]))
    out['expected_random'] = {'correct': float(np.sum(exp_random)), 'answered': float(np.sum(exp_random_ans))}
    lines.append('| E[random pick] | %.1f | %.1f | — | — | — |' % (out['expected_random']['answered'], out['expected_random']['correct']))
    lines.append('| any candidate correct (upper bound) | — | %d | — | — | — |' % out['pool']['questions_with_a_correct_candidate'])
    return out, '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir'); ap.add_argument('--n-boot', type=int, default=2000); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    raw, rows = load(a.run_dir)
    phase = raw['identity']['phase']
    out, table = (oracle_report if phase == 'oracle' else select_report)(raw, rows, a.n_boot, a.seed)
    out.update(phase=phase, model=raw['model'], n_boot=a.n_boot, bootstrap_seed=a.seed,
               excluded_abstention_only_gold=len(raw['evaluations']) - len(rows))
    json.dump(out, open(os.path.join(a.run_dir, 'e42_%s_analysis.json' % phase), 'w'), indent=1, ensure_ascii=False)
    md = '# E42 %s — %s (tuning split, %d questions; %d abstention-only-gold excluded)\n\n%s\n' % (
        phase, raw['model'], len(rows), out['excluded_abstention_only_gold'], table)
    if phase == 'select':
        md += '\npool: mean distinct candidates %.2f; questions where all candidates abstain %d\n' % (
            out['pool']['mean_distinct_candidates'], out['pool']['questions_all_abstain'])
    open(os.path.join(a.run_dir, 'e42_%s_analysis.md' % phase), 'w', encoding='utf-8').write(md)
    print(md)


if __name__ == '__main__':
    main()
