#!/usr/bin/env python3
"""E42 analysis on the judged tuning run.

usage: e42_analysis.py RUN_DIR [--two-axis] [--n-boot 2000] [--seed 0]
RUN_DIR/tune.json must be an e42 oracle or select run.
  default     : grades from RUN_DIR/tune_scored_items.json (legacy single-letter judge: correct / wrong / abstain)
  --two-axis  : grades from RUN_DIR/judge2/final.jsonl (TruthfulQA two-axis standard, core/judge_two_axis.py):
                A -> correct (true & informative), B -> wrong, C -> abstain (refusal), H -> hedge; every cell must be
                resolved (GPT / human adjudication finished) and judge2/meta.json must point at this tune.json.
"answered" = correct + wrong (substantive answers); hedges and refusals are not answers.
Oracle run : per arm -> answered / correct / wrong / abstain [/ hedge], outputs changed vs baseline, wrong->correct and
             correct->wrong counts, paired bootstrap CI of (correct - baseline correct).
Select run : per pick rule -> the picked candidate's answered / correct [/ hedge], Δcorrect vs greedy, vs random
             (expectation over the non-abstaining pool), vs max_logp, with paired bootstrap CIs;
             pool statistics (distinct candidates, questions with >= 1 correct candidate = oracle upper bound).
Questions whose gold is abstention-only are excluded (as e32 does). Nothing here selects a layer or a rule on the
test split: this is the tuning-set diagnostic that decides whether a test-set run is worth doing.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

OUTCOMES = ('correct', 'wrong', 'abstain')
TWO_AXIS_OUTCOMES = ('correct', 'wrong', 'abstain', 'hedge')
LABEL_TO_OUTCOME = {'A': 'correct', 'B': 'wrong', 'C': 'abstain', 'H': 'hedge'}
ANSWERED = ('correct', 'wrong')


def load(run_dir, two_axis=False):
    tune_path = os.path.join(run_dir, 'tune.json')
    raw = json.load(open(tune_path, encoding='utf-8'))
    if not raw.get('completed'):
        raise SystemExit('generation incomplete')
    if two_axis:
        j2 = os.path.join(run_dir, 'judge2')
        meta = json.load(open(os.path.join(j2, 'meta.json'), encoding='utf-8'))
        if meta.get('generation_sha256') != hashlib.sha256(open(tune_path, 'rb').read()).hexdigest():
            raise SystemExit('judge2/meta.json was built from a different tune.json')
        finals = {}
        for line in open(os.path.join(j2, 'final.jsonl'), encoding='utf-8'):
            if line.strip():
                r = json.loads(line)
                finals[r['cell_id']] = r
        rows = []
        for r in raw['evaluations']:
            cells = {c: finals.get('%s|%s' % (r['pair_id'], c)) for c in raw['conditions']}
            if any(v is None for v in cells.values()):
                raise SystemExit('judge2/final.jsonl lacks cells for %s' % r['pair_id'])
            if any(v['gold_is_abstention'] for v in cells.values()):
                continue
            outcome = {}
            for c, v in cells.items():
                if v['label'] not in LABEL_TO_OUTCOME:
                    raise SystemExit('unresolved label for %s/%s (status %s) — finish GPT / human adjudication first'
                                     % (r['pair_id'], c, v.get('status')))
                outcome[c] = LABEL_TO_OUTCOME[v['label']]
            rows.append({**r, 'outcome': outcome})
        return raw, rows, TWO_AXIS_OUTCOMES
    scored = json.load(open(os.path.join(run_dir, 'tune_scored_items.json'), encoding='utf-8'))
    if not scored.get('grading_completed', scored.get('completed')):
        raise SystemExit('grading incomplete')
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
    return raw, rows, OUTCOMES


def boot_ci(diffs, n_boot, seed):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, dtype=float)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    sums = d[idx].sum(1)
    return float(d.sum()), [float(np.percentile(sums, 2.5)), float(np.percentile(sums, 97.5))]


def oracle_report(raw, rows, outcomes, n_boot, seed):
    base = 'baseline'
    extra = [o for o in outcomes if o not in OUTCOMES]           # 'hedge' under --two-axis
    out = {'n_questions': len(rows), 'arms': {}}
    head = ['arm', 'answered', 'correct', 'wrong', 'abstain'] + extra + ['output changed', 'wrong→correct', 'correct→wrong', 'Δcorrect vs baseline [95% CI]']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for c in raw['conditions']:
        o = [r['outcome'][c] for r in rows]
        ob = [r['outcome'][base] for r in rows]
        changed = sum(r['outputs'][c].strip() != r['outputs'][base].strip() for r in rows)
        w2c = sum(a == 'wrong' and b == 'correct' for a, b in zip(ob, o))
        c2w = sum(a == 'correct' and b == 'wrong' for a, b in zip(ob, o))
        diffs = [(b == 'correct') - (a == 'correct') for a, b in zip(ob, o)]
        d, ci = boot_ci(diffs, n_boot, seed)
        arm = {'answered': sum(x in ANSWERED for x in o), 'correct': o.count('correct'), 'wrong': o.count('wrong'),
               'abstain': o.count('abstain'), 'output_changed': changed, 'wrong_to_correct': w2c, 'correct_to_wrong': c2w,
               'delta_correct': d, 'delta_correct_ci95': ci}
        for e in extra:
            arm[e] = o.count(e)
        out['arms'][c] = arm
        cells = [c, arm['answered'], arm['correct'], arm['wrong'], arm['abstain']] + [arm[e] for e in extra] + [
            changed, w2c, c2w, '%+d [%+.0f, %+.0f]' % (d, ci[0], ci[1])]
        lines.append('| ' + ' | '.join(str(x) for x in cells) + ' |')
    return out, '\n'.join(lines)


def select_report(raw, rows, outcomes, n_boot, seed):
    cands = raw['conditions']
    extra = [o for o in outcomes if o not in OUTCOMES]
    rules = sorted({k for r in rows for k in r['picks']})
    out = {'n_questions': len(rows), 'rules': {}, 'pool': {}}
    # pool statistics
    distinct = [len({r['outputs'][c].strip() for c in cands}) for r in rows]
    any_correct = [any(r['outcome'][c] == 'correct' for c in cands) for r in rows]
    none_answered = [all(r['outcome'][c] not in ANSWERED for c in cands) for r in rows]
    out['pool'] = {'mean_distinct_candidates': float(np.mean(distinct)),
                   'questions_with_a_correct_candidate': int(sum(any_correct)),
                   'questions_without_a_substantive_candidate': int(sum(none_answered)),
                   'greedy_correct': sum(r['outcome']['greedy'] == 'correct' for r in rows)}
    # expected random pick: mean over the non-abstaining pool (fallback: all candidates)
    def pool(r):
        p = [c for c in cands if not r['rule_abstain'][c] and r['outputs'][c].strip()]
        return p or cands
    exp_random = [np.mean([r['outcome'][c] == 'correct' for c in pool(r)]) for r in rows]
    exp_random_ans = [np.mean([r['outcome'][c] in ANSWERED for c in pool(r)]) for r in rows]
    head = ['pick rule', 'answered', 'correct'] + extra + ['Δcorrect vs greedy [95% CI]', 'Δcorrect vs E[random] [95% CI]', 'Δcorrect vs max_logp [95% CI]']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    greedy_c = [r['outcome']['greedy'] == 'correct' for r in rows]
    logp_c = [r['outcome'][r['picks']['max_logp']] == 'correct' for r in rows]
    for rule in rules:
        picked = [r['picks'][rule] for r in rows]
        o = [r['outcome'][p] for r, p in zip(rows, picked)]
        corr = [x == 'correct' for x in o]
        dg, cig = boot_ci([int(a) - int(b) for a, b in zip(corr, greedy_c)], n_boot, seed)
        dr, cir = boot_ci([int(a) - b for a, b in zip(corr, exp_random)], n_boot, seed)
        dl, cil = boot_ci([int(a) - int(b) for a, b in zip(corr, logp_c)], n_boot, seed)
        res = {'answered': sum(x in ANSWERED for x in o), 'correct': sum(corr),
               'delta_vs_greedy': dg, 'delta_vs_greedy_ci95': cig, 'delta_vs_random': dr, 'delta_vs_random_ci95': cir,
               'delta_vs_max_logp': dl, 'delta_vs_max_logp_ci95': cil}
        for e in extra:
            res[e] = o.count(e)
        out['rules'][rule] = res
        cells = [rule, res['answered'], res['correct']] + [res[e] for e in extra] + [
            '%+d [%+.0f, %+.0f]' % (dg, cig[0], cig[1]), '%+.1f [%+.1f, %+.1f]' % (dr, cir[0], cir[1]),
            '%+d [%+.0f, %+.0f]' % (dl, cil[0], cil[1])]
        lines.append('| ' + ' | '.join(str(x) for x in cells) + ' |')
    out['expected_random'] = {'correct': float(np.sum(exp_random)), 'answered': float(np.sum(exp_random_ans))}
    pad = ' | '.join(['—'] * len(extra))
    pad = (' | ' + pad) if extra else ''
    lines.append('| E[random pick] | %.1f | %.1f%s | — | — | — |' % (out['expected_random']['answered'], out['expected_random']['correct'], pad))
    lines.append('| any candidate correct (upper bound) | — | %d%s | — | — | — |' % (out['pool']['questions_with_a_correct_candidate'], pad))
    return out, '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir'); ap.add_argument('--two-axis', action='store_true')
    ap.add_argument('--n-boot', type=int, default=2000); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    raw, rows, outcomes = load(a.run_dir, a.two_axis)
    phase = raw['identity']['phase']
    out, table = (oracle_report if phase == 'oracle' else select_report)(raw, rows, outcomes, a.n_boot, a.seed)
    out.update(phase=phase, model=raw['model'], n_boot=a.n_boot, bootstrap_seed=a.seed,
               grading=('two_axis_v5: judge2/final.jsonl' if a.two_axis else 'legacy single-letter judge: tune_scored_items.json'),
               excluded_abstention_only_gold=len(raw['evaluations']) - len(rows))
    suffix = '_two_axis' if a.two_axis else ''
    json.dump(out, open(os.path.join(a.run_dir, 'e42_%s_analysis%s.json' % (phase, suffix)), 'w'), indent=1, ensure_ascii=False)
    md = '# E42 %s — %s (tuning split, %d questions; %d abstention-only-gold excluded; grading: %s)\n\n%s\n' % (
        phase, raw['model'], len(rows), out['excluded_abstention_only_gold'], out['grading'], table)
    if phase == 'select':
        md += '\npool: mean distinct candidates %.2f; questions without a substantive candidate %d\n' % (
            out['pool']['mean_distinct_candidates'], out['pool']['questions_without_a_substantive_candidate'])
    open(os.path.join(a.run_dir, 'e42_%s_analysis%s.md' % (phase, suffix)), 'w', encoding='utf-8').write(md)
    print(md)


if __name__ == '__main__':
    main()
