#!/usr/bin/env python3
"""Exploratory native C gate, with the E intervention held fixed.

This diagnostic uses tuning data only. It does not replace the E26 exact-quota
table or turn an already inspected test set into a new confirmation set.
For every C dose, the online gate emits its candidate iff first-token Yes-No > 0.
Offline semantic annotations (which use references) determine how many emitted
candidates are substantive answers. Compare that observed actual count against
offline exact-count controls; do not promise a reference-free online fill-to-K.
The gate-only branch isolates the previously untested transfer of C from the
gate context into the answer context. Both branches share identical gate scores.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

import common
import e26_matched_coverage as joint
from core.config import load_experiment
from core.experiment_provenance import implementation_identity, require_unchanged
from core.joint_steering import fingerprint
from core.qa_precision import fp32_output_projection
from core.qa_scoring import reference_answers
from core.selective_eval import quota_select, validate_arms

PROTOCOL = 'native_commitment_fixed_E_tune_v1'


def plan(args, meta):
    ns = SimpleNamespace(run_dir=args.source_run, model=meta['model'])
    frozen = joint.frozen_selection(ns, meta)
    conditions = {}
    for task in joint.TASKS:
        specs = {c['name']: c for c in meta['conditions'][task]}
        if str(args.coverage) not in frozen['tasks'][task]['choices']:
            raise ValueError('fixed-E coverage was not in the original frozen selection')
        chosen = frozen['tasks'][task]['choices'][str(args.coverage)]
        conditions[task] = {}
        for method in args.methods:
            name = chosen[method + '_E']
            if name is None:
                raise ValueError('fixed E unavailable on tuning: ' + method)
            layers = meta['c_layers'][task][method]
            if len(layers) != 1:
                raise ValueError('diagnostic requires one previously estimated C layer')
            conditions[task][method] = {'e': specs[name]['e'], 'source_E_name': name, 'c_layer': layers[0]}
    identity = {'protocol': PROTOCOL, 'model': meta['model'], 'source_bank': meta['fingerprint'],
        'selection_sha256': joint.digest(args.source_run / 'selection.json'),
        'source_data_sha256': joint.digest(args.source_run / 'data.json'),
        'source_files': {f: joint.digest(args.source_run / f) for f in
                         ('tf_tune.json', 'qa_tune.json', 'qa_tune_scored.json')},
        'conditions': conditions, 'alphas': args.alphas, 'limit': args.limit,
        'split': 'tune', 'gate_rule': 'online: emit candidate iff Yes-No > 0; offline: semantic non-attempts do not count as substantive answers',
        'eligibility_scope': 'reference-informed offline semantic annotations; A/B correctness never affects ranking; exact-K controls are offline evaluations',
        'scopes': ['gate_only', 'gate_and_answer'], 'uses_test_outcomes': False,
        'source_code_sha256': joint.digest(__file__), 'implementation': implementation_identity(__file__)}
    identity['fingerprint'] = fingerprint(identity)
    return identity


def condition(task, method, e, layer, alpha, family):
    # Round-trip names prevent distinct, closely spaced doses sharing an arm.
    return {'name': '%s_%s_a%s' % (method, family, format(float(alpha), '.17g')), 'task': task, 'method': method,
            'kind': family, 'e': deepcopy(e) if family == 'EC' else None,
            'c': {'layer': layer, 'alpha': float(alpha)}}


def same_operator(a, b):
    return all(a.get(k) == b.get(k) for k in ('task', 'method', 'kind', 'e', 'c'))


def selected_source_rows(rows, ids):
    by_id = {r['id']: r for r in rows}
    if len(by_id) != len(rows) or any(i not in by_id for i in ids):
        raise ValueError('source cohort missing or duplicated')
    return [deepcopy(by_id[i]) for i in ids]


def cohort_for_plan(data, declaration, task):
    rows = data[task]['tune']
    if declaration['limit']:
        keep = {str(r['pair_id']) for r in data['qa']['tune'][:declaration['limit']]}
        rows = [r for r in rows if str(r['pair_id']) in keep]
    return rows


def expected_arm_names(declaration, task):
    names = {'baseline'}
    for method, fixed in declaration['conditions'][task].items():
        names.add(method + '_E')
        for family in ('C', 'EC'):
            for alpha in declaration['alphas']:
                c = condition(task, method, fixed['e'], fixed['c_layer'], alpha, family)
                names.update(c['name'] + '_' + scope for scope in declaration['scopes'])
    return names


def validate_payload(payload, data, raw=None, graded=False):
    """Keep every planned condition/question, and verify derived judgments.

    A fingerprint prevents accidental mixups; it is not proof against a user
    deliberately rewriting the files. Raw inference never contains gold QA.
    """
    declaration = payload['plan']
    if (declaration.get('protocol') != PROTOCOL or declaration.get('split') != 'tune'
            or declaration.get('uses_test_outcomes') is not False
            or declaration.get('fingerprint') != fingerprint({k: v for k, v in declaration.items() if k != 'fingerprint'})):
        raise ValueError('invalid native diagnostic declaration')
    if set(payload['tasks']) != set(joint.TASKS):
        raise ValueError('native diagnostic needs both declared tasks')
    if raw is not None and declaration != raw['plan']:
        raise ValueError('graded declaration differs from raw inference')
    for task in joint.TASKS:
        cohort = cohort_for_plan(data, declaration, task)
        expected = [str(q['id'] if task == 'tf' else q['pair_id']) for q in cohort]
        if len(set(expected)) != len(expected):
            raise ValueError('duplicate fixed-cohort IDs')
        arms = payload['tasks'][task]
        if set(arms) != expected_arm_names(declaration, task):
            raise ValueError('native diagnostic arm set changed')
        for name, rows in arms.items():
            if [r['id'] for r in rows] != expected:
                raise ValueError('native diagnostic fixed cohort/order changed')
            originals = raw['tasks'][task][name] if raw is not None else rows
            for row, original, q in zip(rows, originals, cohort):
                if row['pair_id'] != str(q['pair_id']):
                    raise ValueError('question group changed')
                if any(row.get(k) != original.get(k) for k in ('id', 'pair_id', 'output', 'score', 'simple_score', 'finished')):
                    raise ValueError('graded row changed its raw inference')
                if not all(np.isfinite(row[k]) and not isinstance(row[k], bool) for k in ('score', 'simple_score')):
                    raise ValueError('nonfinite/bool inference score')
                if task == 'tf':
                    if type(row.get('predicted_true')) is not bool or row['output'] != ('True' if row['predicted_true'] else 'False'):
                        raise ValueError('TF text and predicted label disagree')
                    outcome = 'correct' if row['predicted_true'] == q['label'] else 'wrong'
                    if row['outcome'] != outcome:
                        raise ValueError('TF outcome disagrees with fixed label')
                elif graded:
                    _, only_refusal = reference_answers(q['gold'])
                    if row.get('gold_is_abstention') is not bool(only_refusal):
                        raise ValueError('reference-only refusal flag changed')
                    if row['outcome'] != row.get('grade', {}).get('label'):
                        raise ValueError('QA outcome disagrees with semantic grade')
                if graded and row['outcome'] not in ('correct', 'wrong', 'abstain'):
                    raise ValueError('unresolved grade cannot enter results')
    return payload


def gate_only_rows(answer_rows, gate_rows):
    if [(r['id'], r['pair_id']) for r in answer_rows] != [(r['id'], r['pair_id']) for r in gate_rows]:
        raise ValueError('gate and answer identities/order differ')
    return [{**deepcopy(r), 'score': g['score']} for r, g in zip(answer_rows, gate_rows)]


def collect(args, meta, bank, data, bundle=None):
    declaration = plan(args, meta)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    joint.save(args.run_dir / 'plan.json', declaration, immutable=True)
    path = args.run_dir / 'native_raw.json'
    out = joint.read(path) if path.exists() else {'plan': declaration, 'completed': False, 'tasks': {}}
    if out['plan'] != declaration:
        raise ValueError('native diagnostic inputs/config/code changed; use a new directory')
    if out['completed']:
        validate_payload(out, data)
        return out
    for task in joint.TASKS:
        # TF retains both statements of each question.
        cohort = cohort_for_plan(data, declaration, task)
        ids = [str(r['id'] if task == 'tf' else r['pair_id']) for r in cohort]
        source = joint.read(args.source_run / ('%s_tune.json' % task))
        source_specs = {c['name']: c for c in source['condition_specs']}
        rows_by_arm = out['tasks'].setdefault(task, {})
        rows_by_arm['baseline'] = selected_source_rows(source['rows']['baseline'], ids)
        for method, fixed in declaration['conditions'][task].items():
            rows_by_arm[method + '_E'] = selected_source_rows(source['rows'][fixed['source_E_name']], ids)
            for family in ('C', 'EC'):
                answer_base = rows_by_arm['baseline' if family == 'C' else method + '_E']
                for alpha in args.alphas:
                    c = condition(task, method, fixed['e'], fixed['c_layer'], alpha, family)
                    both, gate = c['name'] + '_gate_and_answer', c['name'] + '_gate_only'
                    if both not in rows_by_arm:
                        reuse = next((n for n, s in source_specs.items() if same_operator(s, c)), None)
                        if alpha == 0:
                            records = deepcopy(answer_base)
                        elif reuse:
                            records = selected_source_rows(source['rows'][reuse], ids)
                        else:
                            if bundle is None:
                                bundle = joint.open_bundle(meta)
                            joint.verify_tokens(bundle, meta)
                            records = []
                            with fp32_output_projection(bundle):
                                for start in range(0, len(cohort), meta['config']['batch_size']):
                                    chunk = cohort[start:start + meta['config']['batch_size']]
                                    records.extend(joint.output_records(bundle, meta, bank, task, chunk, c))
                        rows_by_arm[both] = records
                        joint.save(path, out)
                    # Gate-only uses the *same* C-edited gate inference but the
                    # unmodified baseline/E candidate, including its refusal.
                    rows_by_arm[gate] = [{**r,
                        'answer_provenance': 'baseline' if family == 'C' else method + '_E',
                        'gate_provenance': both} for r in gate_only_rows(answer_base, rows_by_arm[both])]
                    if [r['id'] for r in rows_by_arm[gate]] != ids:
                        raise ValueError('gate/answer identities misaligned')
                    joint.save(path, out)
                    print('COLLECTED', task, both, len(rows_by_arm[both]), flush=True)
                    if args.limit and args.limit <= 3 and task == 'qa':
                        for row in rows_by_arm[both]:
                            print('PILOT', both, row['id'], repr(row['output']), 'gate=', row['score'], flush=True)
    require_unchanged(declaration['implementation'], __file__)
    out['completed'] = True
    validate_payload(out, data)
    joint.save(path, out)
    return out


def judge(args, data, judge=None):
    raw_path = args.run_dir / 'native_raw.json'
    raw = joint.read(raw_path)
    if not raw['completed']:
        raise ValueError('complete collection before grading')
    validate_payload(raw, data)
    path = args.run_dir / 'native_scored.json'
    cfg = load_experiment('E4_qa')['judge']
    if judge is None:
        from core.judge import judge_from_config
        judge = judge_from_config(cfg, cache_dir=str(args.cache_dir or args.run_dir / 'judge_cache'))
    grading_inputs = {'raw_sha256': joint.digest(raw_path), 'config': cfg, 'implementation': implementation_identity(__file__),
                      'judge': getattr(judge, 'name', None), 'rubric': getattr(judge, 'qa_rubric_id', None)}
    identity = fingerprint(grading_inputs)
    out = joint.read(path) if path.exists() else {**deepcopy(raw), 'grading_identity': identity,
                                               'grading_inputs': grading_inputs, 'completed': False}
    if out['grading_identity'] != identity or out.get('grading_inputs') != grading_inputs:
        raise ValueError('raw data or grader changed')
    validate_payload(out, data, raw=raw)
    by_id = {str(r['pair_id']): r for r in data['qa']['tune']}
    pending, last_save = 0, time.monotonic()
    try:
        for name, rows in out['tasks']['qa'].items():
            for r, original in zip(rows, raw['tasks']['qa'][name]):
                q = by_id[r['id']]
                gold, only_refusal = reference_answers(q['gold'])
                r['gold_is_abstention'] = bool(only_refusal)
                if r.get('outcome') not in ('correct', 'wrong', 'abstain'):
                    r['grade'] = judge.qa_grade(q['question'], gold, r['output'], incorrect=q.get('incorrect'))
                    from core.qa_scoring import apply_configured_outcome_policy
                    r['grade'] = apply_configured_outcome_policy(r['grade'], q['question'], r['output'], cfg)
                    r['outcome'] = r['grade']['label']
                    pending += 1
                    if pending >= 64 or time.monotonic() - last_save >= 5:
                        joint.save(path, out)
                        pending, last_save = 0, time.monotonic()
                elif r['outcome'] != r.get('grade', {}).get('label'):
                    raise ValueError('QA outcome disagrees with semantic grade')
    finally:
        # Per-call judge caching already persists API responses. This checkpoint
        # avoids rewriting the complete multi-MB artifact after every cached row,
        # while retaining all completed grades on errors and interrupted runs.
        if pending:
            joint.save(path, out)
    require_unchanged(grading_inputs['implementation'], __file__)
    out['completed'] = all(r['outcome'] in ('correct', 'wrong', 'abstain')
                           for rows in out['tasks']['qa'].values() for r in rows)
    joint.save(path, out)
    if not out['completed']:
        raise RuntimeError('unresolved grades; retry, never drop rows')
    validate_payload(out, data, raw=raw, graded=True)
    return out


def native_delivery(rows):
    clean = validate_arms({'native': rows})['native']
    selected = [r for r in clean if r['score'] > 0 and r['outcome'] != 'abstain']
    n, k = len(clean), len(selected)
    correct = sum(r['outcome'] == 'correct' for r in selected)
    return {'n': n, 'answered': k, 'actual_coverage': k / n if n else None,
            'correct': correct, 'wrong': k - correct,
            'accuracy_answered': correct / k if k else None,
            'selected_ids': [r['id'] for r in selected]}


def summarize(scored, seed=42):
    if not scored.get('completed'):
        raise ValueError('incomplete native data')
    tables = {}
    for task, raw_arms in scored['tasks'].items():
        arms = {name: [dict(r) for r in rows if task != 'qa' or not r['gold_is_abstention']]
                for name, rows in raw_arms.items()}
        validate_arms(arms)
        tables[task] = {}
        for name, rows in arms.items():
            delivery = native_delivery(rows)
            controls = {}
            sources = ['baseline'] + sorted(n for n in arms if n.endswith('_E'))
            if delivery['n']:
                q = Fraction(delivery['answered'], delivery['n'])
                for control in sources:
                    for score in ('score', 'simple_score'):
                        selected = quota_select([{**r, 'score': r[score]} for r in arms[control]], q, seed)
                        controls[control + ('_gate_threshold' if score == 'score' else '_simple_threshold')] = {
                            k: selected[k] for k in ('feasible', 'answered', 'correct', 'accuracy_answered', 'selected_ids')}
            tables[task][name] = {'native': delivery, 'same_actual_count_controls': controls,
                'quality': {'replacement_character': sum('\ufffd' in r.get('output', '') for r in rows),
                            'unfinished': sum(not r.get('finished', True) for r in rows)}}
    return {'protocol': PROTOCOL, 'plan': scored['plan'], 'tables': tables,
            'interpretation': 'Exploratory tuning diagnostic. Online positive gates emit candidates; reference-informed offline semantic labels measure substantive coverage. Exact-count controls use that offline eligibility. A/B never affects ranking. No confirmation p-values.'}


def report(args, data):
    report_implementation = implementation_identity(__file__)
    raw = joint.read(args.run_dir / 'native_raw.json')
    validate_payload(raw, data)
    scored = joint.read(args.run_dir / 'native_scored.json')
    if (scored.get('grading_identity') != fingerprint(scored.get('grading_inputs'))
            or scored.get('grading_inputs', {}).get('raw_sha256') != joint.digest(args.run_dir / 'native_raw.json')):
        raise ValueError('graded raw source/config identity changed')
    validate_payload(scored, data, raw=raw, graded=True)
    out = summarize(scored)
    out['report_implementation'] = report_implementation
    require_unchanged(report_implementation, __file__)
    joint.save(args.run_dir / 'report.json', out)
    lines = ['# 固定 E 的原生 C 门控诊断', '', '仅调参集；不能代替 E26 主表或新增测试证据。', '',
             '线上首词 Yes−No > 0 就发出候选；离线语义标注识别其中实质回答。等数量对照也使用这份离线实质回答标注，A/B对错不参与筛题，不宣称线上无参考精确填K。', '']
    for task, rows in out['tables'].items():
        lines += ['## ' + task, '', '| 方法 | 实际回答 | 答对 | 已回答准确率 | 原模型简单筛题答对 | 固定E简单筛题答对 |',
                  '|---|---:|---:|---:|---:|---:|']
        for name, row in rows.items():
            d, c = row['native'], row['same_actual_count_controls']
            es = {k: v['correct'] for k, v in c.items() if k.endswith('_E_simple_threshold')}
            lines.append('| %s | %d/%d | %d | %s | %s | %s |' % (name, d['answered'], d['n'], d['correct'],
                '—' if d['accuracy_answered'] is None else '%.4f' % d['accuracy_answered'],
                c.get('baseline_simple_threshold', {}).get('correct'), json.dumps(es)))
        lines.append('')
    (args.run_dir / 'report.md').write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['collect', 'judge', 'report'])
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--methods', nargs='+', choices=['das', 'md'], default=['das', 'md'])
    p.add_argument('--alphas', nargs='+', type=float, default=[-2., -1., -.5, 0., .5, 1., 2.])
    p.add_argument('--coverage', type=float, default=.5)
    p.add_argument('--limit', type=int)
    p.add_argument('--cache-dir', type=Path)
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error('positive limit required')
    if len(set(args.alphas)) != len(args.alphas) or not np.isfinite(args.alphas).all():
        p.error('unique finite doses required')
    if len(set(args.methods)) != len(args.methods):
        p.error('unique methods required')
    meta = joint.read(args.source_run / 'bank.json')
    ns = SimpleNamespace(run_dir=args.source_run, model=meta['model'])
    meta, bank, data = joint.load_run(ns)
    if args.phase != 'collect':
        declaration = joint.read(args.run_dir / 'plan.json')
        if (declaration['source_bank'] != meta['fingerprint']
                or declaration['source_data_sha256'] != joint.digest(args.source_run / 'data.json')
                or declaration != joint.read(args.run_dir / 'native_raw.json')['plan']):
            raise ValueError('native diagnostic/source run differs from its declared inputs')
    if args.phase == 'collect':
        collect(args, meta, bank, data)
    elif args.phase == 'judge':
        judge(args, data)
    else:
        report(args, data)


if __name__ == '__main__':
    main()
