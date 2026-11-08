#!/usr/bin/env python3
"""E42 analysis on the judged tuning run.

usage: e42_analysis.py RUN_DIR [--two-axis] [--n-boot 2000] [--seed 0]
RUN_DIR/tune.json must be an e42 oracle or select run.
  default     : grades from RUN_DIR/tune_scored_items.json (legacy single-letter judge: correct / wrong / abstain)
  --two-axis  : grades from RUN_DIR/judge2/final.jsonl (TruthfulQA two-axis standard, core/judge_two_axis.py):
                A -> correct (true & informative), B -> wrong, C -> abstain (refusal), H -> hedge; every cell must be
                resolved (GPT / human adjudication finished) and judge2/meta.json must point at this tune.json.
"answered" under --two-axis = resolved information axis S, independently of truth.
False uninformative hedges are terminal B but not substantive answers; non_refusal is reported separately.
The legacy branch retains answered = correct + wrong.
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

CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, CODE_ROOT)

OUTCOMES = ('correct', 'wrong', 'abstain')
TWO_AXIS_OUTCOMES = ('correct', 'wrong', 'abstain', 'hedge')
LABEL_TO_OUTCOME = {'A': 'correct', 'B': 'wrong', 'C': 'abstain', 'H': 'hedge'}
ANSWERED = ('correct', 'wrong')


def _unique(rows, key, description):
    if not isinstance(rows, list):
        raise SystemExit(description + ' must be a list')
    ids = [r.get(key) for r in rows]
    if any(not isinstance(v, str) or not v for v in ids) or len(ids) != len(set(ids)):
        raise SystemExit(description + ' has missing/duplicate identities')
    return dict(zip(ids, rows))


def _read_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('duplicate JSON key: ' + key)
            out[key] = value
        return out
    def invalid(value):
        raise ValueError('nonfinite JSON number: ' + value)
    with open(path, encoding='utf-8') as stream:
        return json.load(stream, object_pairs_hook=pairs, parse_constant=invalid)


def _raw_identity(raw):
    if not raw.get('completed'):
        raise SystemExit('generation incomplete')
    if (raw.get('protocol') != 'e42_oracle_select_v1'
            or raw.get('identity', {}).get('phase') not in ('oracle', 'select')
            or raw.get('split') != 'tune'):
        raise SystemExit('expected a declared E42 tuning diagnostic, never a test/main-experiment result')
    conditions = raw.get('conditions')
    if (not isinstance(conditions, list) or not conditions
            or any(not isinstance(v, str) or not v for v in conditions)
            or len(conditions) != len(set(conditions))):
        raise SystemExit('missing/duplicate generation conditions')
    identity = raw['identity']
    expected_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if (identity.get('conditions') != conditions or identity.get('model') != raw.get('model')
            or identity.get('protocol') != raw['protocol'] or identity.get('split') != 'tune'
            or raw.get('protocol_fingerprint') != expected_hash
            or raw.get('bank_protocol_fingerprint') != expected_hash):
        raise SystemExit('generation identity/fingerprint differs')
    specs = raw.get('condition_specs')
    if not isinstance(specs, list) or [c.get('name') for c in specs] != conditions:
        raise SystemExit('condition specifications differ from declared conditions')
    for spec in specs:
        oracle = spec['name'].startswith('oracle_') or spec['name'] == 'gold_source_only'
        if bool(spec.get('uses_reference_answer')) != oracle:
            raise SystemExit('oracle/reference-use condition declaration differs')
        if identity['phase'] == 'select' and oracle:
            raise SystemExit('oracle outputs cannot enter a deployable selection pool')
    rows = raw.get('evaluations')
    _unique(rows, 'pair_id', 'generation questions')
    if not rows:
        raise SystemExit('empty generation cohort')
    if raw['identity'].get('question_ids') != [r['pair_id'] for r in rows]:
        raise SystemExit('generation cohort differs from its frozen identity')
    for row in rows:
        if not isinstance(row.get('question'), str) or not row['question'].strip():
            raise SystemExit('question text missing')
        if set(row.get('finished', {})) != set(conditions) or any(type(v) is not bool for v in row['finished'].values()):
            raise SystemExit('every output requires its actual natural-finish flag')
        if set(row.get('outputs', {})) != set(conditions) or any(not isinstance(v, str) for v in row['outputs'].values()):
            raise SystemExit('generation outputs do not match declared conditions')
    return conditions


def _validate_axes(record):
    # Final metadata is independently verified by the runner before this is called.
    # The terminal B label intentionally loses informativeness, so never invert B -> S.
    from core.judge_two_axis import cell_axes
    try:
        truth, info = cell_axes(record)
    except ValueError as error:
        raise SystemExit('invalid resolved final axes: ' + str(error)) from error
    subtype = record.get('info_subtype')
    if truth not in ('T', 'F') or info not in ('S', 'N'):
        raise SystemExit('a resolved final must retain both truth and information axes')
    if subtype not in ('S', 'H', 'R') or ((subtype == 'S') != (info == 'S')):
        raise SystemExit('resolved information subtype missing or inconsistent; a terminal B alone is insufficient')
    expected = 'B' if truth == 'F' else 'A' if info == 'S' else 'C' if subtype == 'R' else 'H'
    if record.get('label') != expected or record.get('status') != 'final':
        raise SystemExit('terminal label/status differs from its resolved axes')
    return {'truth': truth, 'info': info, 'subtype': subtype}


def load(run_dir, two_axis=False):
    tune_path = os.path.join(run_dir, 'tune.json')
    raw = _read_json(tune_path)
    conditions = _raw_identity(raw)
    with open(tune_path, 'rb') as generation_file:
        generation_sha = hashlib.sha256(generation_file.read()).hexdigest()
    from core.qa_scoring import reference_answers
    if two_axis:
        # One shared final-contract verifier owns the complete source chain.
        # Old unbound final.jsonl files must not be silently grandfathered in.
        from judge_two_axis_run import load_verified_final
        try:
            meta, verified_cells, finals = load_verified_final(run_dir, generation_path=tune_path, require_complete=True)
        except ValueError as error:
            raise SystemExit('unverified or incomplete final judgments: ' + str(error)) from error
        by_cell = _unique(list(finals.values()), 'cell_id', 'final judgments')
        expected_cells = {f'{r["pair_id"]}|{c}' for r in raw['evaluations'] for c in conditions}
        if set(by_cell) != expected_cells:
            raise SystemExit('final judgment cohort has missing or extra cells')
        rows = []
        for row in raw['evaluations']:
            cells = {c: by_cell[f'{row["pair_id"]}|{c}'] for c in conditions}
            only_refusal = reference_answers(row['gold'], question=row['question'])[1]
            if any(v.get('gold_is_abstention') is not only_refusal for v in cells.values()):
                raise SystemExit('final gold-eligibility differs from the current fixed question')
            if only_refusal:
                if any(v.get('status') != 'excluded_abstention_gold' for v in cells.values()):
                    raise SystemExit('excluded gold cell has an inconsistent status')
                continue
            axes = {c: _validate_axes(v) for c, v in cells.items()}
            rows.append({**row, 'outcome': {c: LABEL_TO_OUTCOME[v['label']] for c, v in cells.items()},
                         'resolved_axes': axes})
        if not rows:
            raise SystemExit('no substantive-reference questions remain for an experiment comparison')
        return {**raw, '_analysis_judgment_provenance': meta.get('final_verification')}, rows, TWO_AXIS_OUTCOMES
    scored = _read_json(os.path.join(run_dir, 'tune_scored_items.json'))
    if not scored.get('grading_completed', scored.get('completed')):
        raise SystemExit('grading incomplete')
    if (scored.get('grading_source_generation_sha256') != generation_sha
            or scored.get('model') != raw.get('model') or scored.get('conditions') != conditions
            or scored.get('condition_specs') != raw.get('condition_specs')
            or scored.get('bank_protocol_fingerprint') != raw.get('bank_protocol_fingerprint')):
        raise SystemExit('legacy grading source/model/conditions differ from this generation')
    by_id = _unique(scored.get('evaluations'), 'pair_id', 'legacy graded questions')
    if list(by_id) != [r['pair_id'] for r in raw['evaluations']]:
        raise SystemExit('legacy graded cohort/order differs from generation')
    rows = []
    for row in raw['evaluations']:
        grade = by_id[row['pair_id']]
        for key in ('question', 'gold', 'incorrect', 'best_answer', 'outputs', 'finished', 'generation_token_ids'):
            if grade.get(key) != row.get(key):
                raise SystemExit('legacy grade changed its source field: ' + key)
        only_refusal = reference_answers(row['gold'])[1]
        if grade.get('gold_is_abstention') is not only_refusal:
            raise SystemExit('legacy reference eligibility differs')
        if only_refusal:
            continue
        if set(grade.get('outcome', {})) != set(conditions) or any(grade['outcome'][c] not in OUTCOMES for c in conditions):
            raise SystemExit('unresolved or foreign legacy outcome — finish human review first')
        rows.append({**row, 'outcome': grade['outcome']})
    if not rows:
        raise SystemExit('no substantive-reference questions remain for an experiment comparison')
    with open(os.path.join(run_dir, 'tune_scored_items.json'), 'rb') as grade_file:
        grade_sha = hashlib.sha256(grade_file.read()).hexdigest()
    return {**raw, '_analysis_judgment_provenance': {'generation_sha256': generation_sha,
            'legacy_grade_sha256': grade_sha, 'judge_config_sha256': scored.get('grading_judge_config_sha256')}}, rows, OUTCOMES


def informative(row, condition):
    if 'resolved_axes' in row:
        return row['resolved_axes'][condition]['info'] == 'S'
    return row['outcome'][condition] in ANSWERED


def axis_statistics(rows, picked):
    if not rows or 'resolved_axes' not in rows[0]:
        return {}
    axes = [r['resolved_axes'][c] for r, c in zip(rows, picked)]
    n = len(rows)
    return {'truthful': sum(a['truth'] == 'T' for a in axes),
            'informative': sum(a['info'] == 'S' for a in axes),
            'substantive_wrong': sum(a['truth'] == 'F' and a['info'] == 'S' for a in axes),
            'uninformative_false': sum(a['truth'] == 'F' and a['info'] == 'N' for a in axes),
            'non_refusal': sum(a['subtype'] != 'R' for a in axes),
            'refusal_subtype': sum(a['subtype'] == 'R' for a in axes),
            'hedge_subtype': sum(a['subtype'] == 'H' for a in axes),
            'true_rate': sum(a['truth'] == 'T' for a in axes) / n,
            'info_rate': sum(a['info'] == 'S' for a in axes) / n,
            'true_info_rate': sum(a['truth'] == 'T' and a['info'] == 'S' for a in axes) / n,
            'rates_denominator': n,
            'answered_definition': 'resolved information axis S; independent of truth axis',
            'non_refusal_definition': 'information subtype S or H; not a claim of substantive answering or intent'}


def quality_statistics(rows, picked):
    unfinished = [not r['finished'][c] for r, c in zip(rows, picked)]
    replacement = ['\ufffd' in r['outputs'][c] for r, c in zip(rows, picked)]
    correct = [r['outcome'][c] == 'correct' for r, c in zip(rows, picked)]
    return {'unfinished': sum(unfinished), 'unicode_replacement': sum(replacement),
            'content_correct_unfinished': sum(c and u for c, u in zip(correct, unfinished)),
            'content_correct_unicode_replacement': sum(c and u for c, u in zip(correct, replacement)),
            'content_correct_finished_and_text_valid': sum(c and not u and not x for c, u, x in zip(correct, unfinished, replacement)),
            'quality_rule': 'diagnostic flags only; content truth/information labels remain unchanged'}


def boot_ci(diffs, n_boot, seed):
    if type(n_boot) is not int or n_boot < 2 or type(seed) is not int or seed < 0:
        raise ValueError('bootstrap needs at least two replicates and a nonnegative integer seed')
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, dtype=float)
    if d.ndim != 1 or not len(d) or not np.isfinite(d).all():
        raise ValueError('bootstrap requires a nonempty finite question-level difference vector')
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    sums = d[idx].sum(1)
    return float(d.sum()), [float(np.percentile(sums, 2.5)), float(np.percentile(sums, 97.5))]


def oracle_report(raw, rows, outcomes, n_boot, seed):
    base = 'baseline'
    extra = [o for o in outcomes if o not in OUTCOMES]           # 'hedge' under --two-axis
    out = {'n_questions': len(rows), 'arms': {}}
    head = (['arm', '实质作答 S', 'A类（真且有信息）', 'B类（假，不保证有信息）', 'C类（真且拒答）'] + ['H类（真且无信息）' if e == 'hedge' else e for e in extra] if extra else ['arm', 'answered', 'correct', 'wrong', 'abstain']) + ['output changed', 'wrong→correct', 'correct→wrong', 'Δcorrect vs baseline [95% CI]']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for c in raw['conditions']:
        o = [r['outcome'][c] for r in rows]
        ob = [r['outcome'][base] for r in rows]
        changed = sum(r['outputs'][c].strip() != r['outputs'][base].strip() for r in rows)
        w2c = sum(a == 'wrong' and b == 'correct' for a, b in zip(ob, o))
        c2w = sum(a == 'correct' and b == 'wrong' for a, b in zip(ob, o))
        diffs = [(b == 'correct') - (a == 'correct') for a, b in zip(ob, o)]
        d, ci = boot_ci(diffs, n_boot, seed)
        arm = {'answered': sum(informative(r, c) for r in rows), 'correct': o.count('correct'), 'wrong': o.count('wrong'),
               'abstain': o.count('abstain'), 'output_changed': changed, 'wrong_to_correct': w2c, 'correct_to_wrong': c2w,
               'delta_correct': d, 'delta_correct_ci95': ci}
        for e in extra:
            arm[e] = o.count(e)
        arm.update(axis_statistics(rows, [c] * len(rows)))
        arm['quality'] = quality_statistics(rows, [c] * len(rows))
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
    none_answered = [all(not informative(r, c) for c in cands) for r in rows]
    out['pool'] = {'mean_distinct_candidates': float(np.mean(distinct)),
                   'questions_with_a_correct_candidate': int(sum(any_correct)),
                   'questions_without_a_substantive_candidate': int(sum(none_answered)),
                   'questions_all_abstain': int(sum(none_answered)),      # legacy key (same value under 3 outcomes)
                   'greedy_correct': sum(r['outcome']['greedy'] == 'correct' for r in rows)}
    # expected random pick: mean over the non-abstaining pool (fallback: all candidates)
    def pool(r):
        p = [c for c in cands if not r['rule_abstain'][c] and r['outputs'][c].strip()]
        return p or cands
    exp_random = [np.mean([r['outcome'][c] == 'correct' for c in pool(r)]) for r in rows]
    exp_random_ans = [np.mean([informative(r, c) for c in pool(r)]) for r in rows]
    head = (['pick rule', '实质作答 S', 'A类（真且有信息）'] + ['H类（真且无信息）' if e == 'hedge' else e for e in extra] if extra else ['pick rule', 'answered', 'correct']) + ['Δcorrect vs greedy [95% CI]', 'Δcorrect vs E[random] [95% CI]', 'Δcorrect vs max_logp [95% CI]']
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
        res = {'answered': sum(informative(r, c) for r, c in zip(rows, picked)), 'correct': sum(corr),
               'delta_vs_greedy': dg, 'delta_vs_greedy_ci95': cig, 'delta_vs_random': dr, 'delta_vs_random_ci95': cir,
               'delta_vs_max_logp': dl, 'delta_vs_max_logp_ci95': cil}
        for e in extra:
            res[e] = o.count(e)
        res.update(axis_statistics(rows, picked))
        res['quality'] = quality_statistics(rows, picked)
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
               judgment_provenance=raw.get('_analysis_judgment_provenance'),
               selection_rule_version=(raw['identity'].get('selection_rule_version', 'legacy_raw_coordinate_mean_v1') if phase == 'select' else None),
               same_actual_answer_rate_comparison=False, oracle_is_deployable=False,
               grading=('two_axis %s: judge2/final.jsonl' % __import__('core.judge_two_axis', fromlist=['TWO_AXIS_VERSION']).TWO_AXIS_VERSION
                        if a.two_axis else 'legacy single-letter judge: tune_scored_items.json'),
               excluded_abstention_only_gold=len(raw['evaluations']) - len(rows))
    suffix = '_two_axis' if a.two_axis else ''
    json.dump(out, open(os.path.join(a.run_dir, 'e42_%s_analysis%s.json' % (phase, suffix)), 'w'), indent=1, ensure_ascii=False)
    md = '# E42 %s — %s (tuning split, %d questions; %d abstention-only-gold excluded; grading: %s)\n\n%s\n' % (
        phase, raw['model'], len(rows), out['excluded_abstention_only_gold'], out['grading'], table)
    quality_rows = out['rules'] if phase == 'select' else out['arms']
    md += '\n内容判分与输出质量分开保留（不改写 A 为错或拒答）：\n\n| 条件/规则 | 未自然结束 | 替换字符 | A但未结束 | A且替换字符 |\n|---|---:|---:|---:|---:|\n'
    for name, entry in quality_rows.items():
        quality = entry['quality']
        md += '| %s | %d | %d | %d | %d |\n' % (name, quality['unfinished'], quality['unicode_replacement'], quality['content_correct_unfinished'], quality['content_correct_unicode_replacement'])
    if a.two_axis:
        md += '\n两维 answered 只按信息轴 S；B 是所有假话，可能无信息量，因此不再假设 answered = A+B。JSON 的 abstain/hedge 是终判 C/H 类计数，不是全部拒答/含糊行为。行为子类另报 refusal_subtype/hedge_subtype；人工 F/R 可为 B，F/H 也为 B，均不能算实质作答。另报 non_refusal（S/H）与固定分母的两轴率。\n'
    if a.two_axis:
        md += '\n| 条件/规则 | 无信息且未拒答（子类H） | 拒答/无答复（子类R） | 非拒答 S/H |\n|---|---:|---:|---:|\n'
        for name, entry in quality_rows.items():
            md += '| %s | %d | %d | %d |\n' % (name, entry['hedge_subtype'], entry['refusal_subtype'], entry['non_refusal'])
    md += '\n这里只是固定候选的调参诊断，各规则自然实质作答数另报；未执行同实际作答率配额比较，不得把此表当作已完成老师的同率主实验。oracle_* 与 gold_source_only 是看答案的诊断上界，不能当作可部署方法。\n'
    if phase == 'select':
        md += '\npool: mean distinct candidates %.2f; questions without a substantive candidate %d\n' % (
            out['pool']['mean_distinct_candidates'], out['pool']['questions_without_a_substantive_candidate'])
    open(os.path.join(a.run_dir, 'e42_%s_analysis%s.md' % (phase, suffix)), 'w', encoding='utf-8').write(md)
    print(md)


if __name__ == '__main__':
    main()
