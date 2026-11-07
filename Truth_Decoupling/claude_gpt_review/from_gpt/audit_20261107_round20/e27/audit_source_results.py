"""Audit original five-arm/six-arm source runs; never import their metric code."""
import argparse
from collections import Counter
import hashlib
import json
from math import comb
from pathlib import Path
import statistics


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(rows, candidate, reference='baseline'):
    transitions = Counter(r['outcome'][reference] + '->' + r['outcome'][candidate] for r in rows)
    up = sum(r['outcome'][candidate] == 'correct' and r['outcome'][reference] != 'correct' for r in rows)
    down = sum(r['outcome'][candidate] != 'correct' and r['outcome'][reference] == 'correct' for r in rows)
    n = up + down
    common = [r for r in rows if r['outcome'][candidate] != 'abstain' and r['outcome'][reference] != 'abstain']
    return {'transitions': dict(transitions), 'net_correct': up - down, 'corrected': up, 'harmed': down,
            'exact_p_exploratory_only': min(1., 2 * sum(comb(n, i) for i in range(min(up, down) + 1)) / 2 ** n) if n else 1.,
            'both_answered_n': len(common),
            'both_answered_net_correct': sum((r['outcome'][candidate] == 'correct') - (r['outcome'][reference] == 'correct') for r in common)}


def audit_run(directory):
    directory = Path(directory)
    raw = json.loads((directory / 'tune.json').read_text())
    scored = json.loads((directory / 'tune_scored_items.json').read_text())
    summary = json.loads((directory / 'tune_scored.json').read_text())
    assert raw['completed'] and scored['completed'] and scored['grading_completed'] and summary['completed']
    assert raw['split'] == 'tune' and len(raw['evaluations']) == 192
    assert scored['grading_source_generation_sha256'] == summary['source_generation_sha256'] == digest(directory / 'tune.json')
    assert scored['conditions'] == raw['conditions'] and len(set(raw['conditions'])) == len(raw['conditions'])
    ids = [r['pair_id'] for r in raw['evaluations']]
    assert len(set(ids)) == len(ids) and [r['pair_id'] for r in scored['evaluations']] == ids
    assert not any(r['gold_is_abstention'] for r in scored['evaluations'])
    for a, b in zip(raw['evaluations'], scored['evaluations']):
        for key in ('question', 'gold', 'incorrect', 'outputs', 'finished'):
            assert a.get(key) == b.get(key), (a['pair_id'], key)
        same_output_grades = {}
        for condition in raw['conditions']:
            assert b['outcome'][condition] == b['judge'][condition]['label']
            assert b['outcome'][condition] in ('correct', 'wrong', 'abstain')
            answer = b['outputs'][condition]
            if answer in same_output_grades: assert same_output_grades[answer] == b['outcome'][condition]
            same_output_grades[answer] = b['outcome'][condition]
            tokens = a['generation_token_ids'][condition]
            traces = a['trace'][condition]
            assert len(tokens) == len(traces)
            for step, trace in enumerate(traces):
                assert trace['step'] == step and trace['accepted_prefix_ids'] == tokens[:step]
                assert trace['source_predictor_logical_position'] == len(a['prompt_token_ids']['source']) - 1 + step
                assert trace['base_predictor_logical_position'] == len(a['prompt_token_ids']['base']) - 1 + step
                if 'swap' in condition:
                    assert trace['cache_storage_disjoint'] and len(trace['writes']) == 1
                    for write in trace['writes']:
                        assert write['ideal_delta_norm'] <= write['source_base_distance'] + write['float32_tolerance']
                        assert write['ideal_coordinate_error'] <= write['float32_tolerance']
                        assert write['ideal_orthogonal_error'] <= write['float32_tolerance']
    result = {'model': raw['model'], 'directory': str(directory), 'n': len(ids),
              'raw_sha256': digest(directory / 'tune.json'), 'scored_sha256': digest(directory / 'tune_scored_items.json'),
              'conditions': {}, 'comparisons': {}, 'raw_score_trace_identity_checks_passed': True}
    for name in raw['conditions']:
        rows = scored['evaluations']
        rates = summary['rates'][name]['rates']
        counts = {label: sum(r['outcome'][name] == label for r in rows) for label in ('correct', 'wrong', 'abstain')}
        assert all(abs(rates[label] - count / len(rows)) < 1e-12 for label, count in counts.items())
        deltas = [w['stored_delta_over_base'] for r in raw['evaluations'] for t in r['trace'][name] for w in t['writes']]
        result['conditions'][name] = {**counts, 'answered': len(rows) - counts['abstain'],
                                      'fffd': sum('\ufffd' in r['outputs'][name] for r in rows),
                                      'unfinished': sum(not r['finished'][name] for r in rows),
                                      'max_delta_over_base': max(deltas) if deltas else 0,
                                      'median_delta_over_base': statistics.median(deltas) if deltas else 0}
        if name != 'baseline': result['comparisons'][name + '_vs_baseline'] = compare(rows, name)
    for candidate in [c for c in raw['conditions'] if c.startswith('trained_source_swap')]:
        for ref in ('source_only', 'equal_logit_mix', 'source_das_swap', 'source_random_swap'):
            result['comparisons'][candidate + '_vs_' + ref] = compare(scored['evaluations'], candidate, ref)
    return result, raw, scored


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = {'scope': 'previously exposed tuning set; counts and p-values are exploratory, not new-test confirmation', 'models': {}}
    for model in ('llama3.1-8b', 'qwen3-8b'):
        directory = args.root / 'results_r20_source_codex_20261107' / model / 'tune192'
        five, raw, scored = audit_run(directory)
        old = json.loads((args.root / 'results_qa_das_v1u' / model / 'full/tune.json').read_text())
        previous = {r['pair_id']: r for r in old['evaluations']}
        five['baseline_batch1_vs_historical_batch8'] = {
            'text_mismatch_ids': [r['pair_id'] for r in raw['evaluations'] if r['outputs']['baseline'] != previous[r['pair_id']]['outputs']['baseline']],
            'finished_mismatch_ids': [r['pair_id'] for r in raw['evaluations'] if r['finished']['baseline'] != previous[r['pair_id']]['finished']['baseline']]}
        five['source_corrects_base_wrong_examples'] = [
            {'pair_id': r['pair_id'], 'question': r['question'], 'gold': r['gold'],
             'baseline': r['outputs']['baseline'], 'source_only': r['outputs']['source_only']}
            for r in scored['evaluations'] if r['outcome']['baseline'] == 'wrong' and r['outcome']['source_only'] == 'correct']
        entry = {'five': five}
        trained_dir = args.root / 'results_r20_source_train_codex_20261107' / model / 'full92_ep3'
        if (trained_dir / 'tune_scored_items.json').exists():
            six, six_raw, six_scored = audit_run(trained_dir)
            assert six_scored['frozen_control_grade_verification']['all_identical']
            assert six_scored['frozen_control_grade_verification']['question_condition_grades_checked'] == 960
            for a, b in zip(scored['evaluations'], six_scored['evaluations']):
                assert a['pair_id'] == b['pair_id']
                for cond in scored['conditions']:
                    assert a['outputs'][cond] == b['outputs'][cond] and a['outcome'][cond] == b['outcome'][cond]
            six['all_960_original_control_grades_identical'] = True
            adapter = json.loads((trained_dir / 'source_adapter.json').read_text())
            assert len(adapter['training_pair_ids']) == 92 and not adapter['pilot']
            assert adapter['gradient_audit']['updates'] == adapter['gradient_audit']['nonzero_updates'] == 276
            six['training'] = {key: adapter[key] for key in ('training_pair_ids', 'losses', 'gradient_audit', 'causal_prefix_checks')}
            entry['six'] = six
        result['models'][model] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    for model, entry in result['models'].items():
        print(model, {stage: d['conditions'] for stage, d in entry.items()})


if __name__ == '__main__': main()
