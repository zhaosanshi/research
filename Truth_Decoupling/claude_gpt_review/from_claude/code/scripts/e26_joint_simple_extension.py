#!/usr/bin/env python3
"""Generate the missing, tuning-selected EC+simple-selector conditions.

This is explicitly an exploratory extension after inspection of the old test.
The original E26 choices, raw artifacts and primary hypothesis family remain
unchanged. A frozen external manifest binds all choices and source hashes.
"""
import argparse
from contextlib import nullcontext
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace

import common
import e26_matched_coverage as joint
from core.config import load_experiment
from core.experiment_provenance import implementation_identity, require_unchanged
from core.joint_steering import choose_conditions, fingerprint
from core.qa_precision import fp32_output_projection
from core.qa_scoring import reference_answers

PROTOCOL = 'E26_equal_simple_selector_extension_v1'


def verify(args):
    manifest = joint.read(args.manifest)
    if manifest.get('protocol') != PROTOCOL or manifest.get('exploratory') is not True:
        raise ValueError('wrong extension protocol')
    if manifest['fingerprint'] != fingerprint({k: v for k, v in manifest.items() if k != 'fingerprint'}):
        raise ValueError('extension manifest changed')
    runs = [r for r in manifest['runs'] if r['model'] == args.model]
    if len(runs) != 1:
        raise ValueError('missing/duplicate model in manifest')
    frozen = runs[0]
    for filename, expected in frozen['sources'].items():
        if joint.digest(args.source_run / filename) != expected:
            raise ValueError('frozen extension source changed: ' + filename)
    ns = SimpleNamespace(run_dir=args.source_run, model=args.model)
    meta, bank, data = joint.load_run(ns)
    if meta['fingerprint'] != frozen['bank_fingerprint']:
        raise ValueError('extension bank differs')
    joint.frozen_selection(ns, meta)
    for task in joint.TASKS:
        rows, _ = joint.scored_rows(ns, meta, task, 'tune')
        simple = {n: joint.score_variant(rs, 'simple', meta['config']['seed']) for n, rs in rows.items()}
        selected = choose_conditions(simple, meta['conditions'][task], meta['config']['coverages'], meta['config']['seed'])
        expected = {q: {m + '_EC_simple': selected[q][m + '_EC'] for m in ('das', 'md')}
                    for q in selected}
        if frozen['tasks'][task]['choices'] != expected:
            raise ValueError('extension choices are not the tuning-selected choices')
        old_name = task + '_test' + ('_scored' if task == 'qa' else '') + '.json'
        if joint.digest(args.source_run / old_name) != frozen['tasks'][task]['existing_test_source_sha256']:
            raise ValueError('original test source changed')
        old_rows, _ = joint.scored_rows(ns, meta, task, 'test')
        specs = {c['name']: c for c in meta['conditions'][task]}
        needed = {n for point in expected.values() for n in point.values() if n is not None} - set(old_rows)
        given = frozen['tasks'][task]['missing_condition_specs']
        if (len({c['name'] for c in given}) != len(given)
                or {c['name'] for c in given} != needed or any(specs[c['name']] != c for c in given)):
            raise ValueError('extension condition definitions differ from frozen E26 bank')
    return frozen, meta, bank, data


def cohort(args, data, task):
    rows = data[task]['test']
    if args.limit is not None:
        keep = {str(q['pair_id']) for q in data['qa']['test'][:args.limit]}
        rows = [r for r in rows if str(r['pair_id']) in keep]
    return rows


def artifact_identity(args, frozen, meta, task, rows):
    return {'protocol': PROTOCOL, 'model': args.model, 'bank_fingerprint': meta['fingerprint'],
        'extension_manifest_sha256': joint.digest(args.manifest),
        'condition_specs': frozen['tasks'][task]['missing_condition_specs'],
        'cohort_sha256': fingerprint(rows), 'pilot_limit': args.limit,
        'source_code_sha256': joint.digest(__file__), 'implementation': implementation_identity(__file__), 'exploratory': True}


def validate_records(records, cohort_rows, task, partial=False):
    expected = [str(r['id'] if task == 'tf' else r['pair_id']) for r in cohort_rows]
    if (len(expected) != len(set(expected)) or len(records) > len(expected)
            or [r['id'] for r in records] != (expected[:len(records)] if partial else expected)):
        raise ValueError('extension fixed cohort/order changed')
    for row, q in zip(records, cohort_rows):
        if row['pair_id'] != str(q['pair_id']):
            raise ValueError('extension question group changed')
        if any(isinstance(row[k], bool) or not math.isfinite(row[k]) for k in ('score', 'simple_score')):
            raise ValueError('extension score is not finite numeric inference')
        if task == 'tf':
            if type(row.get('predicted_true')) is not bool or row['output'] != ('True' if row['predicted_true'] else 'False'):
                raise ValueError('extension TF output differs from prediction')
            if row['outcome'] != ('correct' if row['predicted_true'] == q['label'] else 'wrong'):
                raise ValueError('extension TF outcome differs from fixed label')


def collect(args, frozen, meta, bank, data, bundle=None):
    started_implementation = implementation_identity(__file__)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    joint.save(args.run_dir / 'extension_manifest.json', joint.read(args.manifest), immutable=True)
    for task in joint.TASKS:
        rows = cohort(args, data, task)
        identity = artifact_identity(args, frozen, meta, task, rows)
        require_unchanged(started_implementation, __file__)
        path = args.run_dir / (task + '_test_extension.json')
        out = joint.read(path) if path.exists() else {**identity, 'completed': False,
            'rows': {c['name']: [] for c in identity['condition_specs']}}
        if any(out.get(k) != v for k, v in identity.items()) or set(out['rows']) != {c['name'] for c in identity['condition_specs']}:
            raise ValueError('extension resume identity changed')
        ids = [str(r['id'] if task == 'tf' else r['pair_id']) for r in rows]
        for c in identity['condition_specs']:
            saved = out['rows'][c['name']]
            validate_records(saved, rows, task, partial=True)
            if len(saved) < len(rows) and bundle is None:
                bundle = joint.open_bundle(meta)
            if bundle is not None:
                joint.verify_tokens(bundle, meta)
            with fp32_output_projection(bundle) if bundle is not None else nullcontext():
                for start in range(len(saved), len(rows), meta['config']['batch_size']):
                    chunk = rows[start:start + meta['config']['batch_size']]
                    saved.extend(joint.output_records(bundle, meta, bank, task, chunk, c))
                    joint.save(path, out)
            validate_records(saved, rows, task)
            print('COLLECTED', task, c['name'], len(saved), flush=True)
            if args.limit and args.limit <= 3 and task == 'qa':
                for r in saved:
                    print('PILOT', c['name'], r['id'], repr(r['output']), flush=True)
        require_unchanged(started_implementation, __file__)
        out['completed'] = True
        joint.save(path, out)


def judge(args, frozen, meta, data, judge=None):
    raw_path = args.run_dir / 'qa_test_extension.json'
    raw = joint.read(raw_path)
    rows = cohort(args, data, 'qa')
    identity = artifact_identity(args, frozen, meta, 'qa', rows)
    if not raw['completed'] or any(raw.get(k) != v for k, v in identity.items()):
        raise ValueError('incomplete/foreign raw extension')
    expected_ids = [str(r['pair_id']) for r in rows]
    names = {c['name'] for c in identity['condition_specs']}
    if set(raw['rows']) != names or any([r['id'] for r in rs] != expected_ids for rs in raw['rows'].values()):
        raise ValueError('raw extension conditions/cohort changed')
    for rs in raw['rows'].values():
        validate_records(rs, rows, 'qa')
    cfg = load_experiment('E4_qa')['judge']
    if judge is None:
        from core.judge import judge_from_config
        judge = judge_from_config(cfg, str(args.cache_dir or args.run_dir / 'judge_cache'))
    signature = fingerprint([joint.digest(raw_path), cfg, getattr(judge, 'name', None), getattr(judge, 'qa_rubric_id', None)])
    path = args.run_dir / 'qa_test_extension_scored.json'
    out = joint.read(path) if path.exists() else {**deepcopy(raw), 'completed': False,
        'generation_sha256': joint.digest(raw_path), 'grading_fingerprint': signature}
    if (out.get('grading_fingerprint') != signature or set(out['rows']) != names
            or out.get('generation_sha256') != joint.digest(raw_path)
            or any(out.get(k) != v for k, v in identity.items())):
        raise ValueError('extension grading identity changed')
    for name, scored in out['rows'].items():
        if [r['id'] for r in scored] != expected_ids:
            raise ValueError('extension graded cohort changed')
        for r, original, q in zip(scored, raw['rows'][name], rows):
            if any(r.get(k) != original.get(k) for k in ('id', 'pair_id', 'output', 'score', 'simple_score', 'finished', 'edits')):
                raise ValueError('extension graded row changed raw inference')
            gold, only_refusal = reference_answers(q['gold'])
            r['gold_is_abstention'] = bool(only_refusal)
            if r.get('outcome') in ('correct', 'wrong', 'abstain'):
                if r.get('grade', {}).get('label') != r['outcome']:
                    raise ValueError('extension outcome disagrees with grade')
            else:
                r['grade'] = judge.qa_grade(q['question'], gold, r['output'], incorrect=q.get('incorrect'))
                r['outcome'] = r['grade']['label']
                joint.save(path, out)
    require_unchanged(identity['implementation'], __file__)
    out['completed'] = all(r['outcome'] in ('correct', 'wrong', 'abstain') for rs in out['rows'].values() for r in rs)
    joint.save(path, out)
    if not out['completed']:
        raise RuntimeError('unresolved grades, retry without removing rows')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['collect', 'judge'])
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--cache-dir', type=Path)
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error('limit must be positive')
    frozen, meta, bank, data = verify(args)
    if args.phase == 'collect':
        collect(args, frozen, meta, bank, data)
    else:
        judge(args, frozen, meta, data)


if __name__ == '__main__':
    main()
