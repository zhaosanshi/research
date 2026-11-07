"""Independent accounting of the uniform Qwen completion budget diagnostic."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent-run', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    old = json.loads((args.parent_run / 'tune.json').read_text())
    raw = json.loads((args.run / 'tune.json').read_text())
    pilot = json.loads((args.run / 'pilot.json').read_text())
    assert raw['completed'] and raw['conditions'] == old['conditions'] and len(raw['evaluations']) == 192
    assert raw['identity']['parent_generation_sha256'] == digest(args.parent_run / 'tune.json')
    assert raw['identity']['config'] == old['identity']['config']
    assert raw['identity']['config']['max_new_tokens'] == 48
    assert raw['identity']['budget_override']['max_new_tokens'] == 256
    assert pilot['completed'] and len(pilot['finished_control_checks']) == 21
    assert all(r['tokens_and_text_exact'] and r['finished'] for r in pilot['finished_control_checks'])
    assert all(r['first_48_token_ids_exact'] and r['new']['finished'] for r in pilot['extension_probes'])
    old_by = {r['pair_id']: r for r in old['evaluations']}
    actions = Counter(); changed = []; quality = {}
    for row in raw['evaluations']:
        before = old_by[row['pair_id']]
        for key in ('question', 'gold', 'incorrect', 'best_answer', 'idx', 'prompt_token_ids'):
            assert row.get(key) == before.get(key)
        for condition in raw['conditions']:
            if before['finished'][condition]:
                actions['reused_finished'] += 1
                for key in ('outputs', 'finished', 'rule_abstain', 'generation_token_ids', 'trace'):
                    assert row[key][condition] == before[key][condition]
            else:
                actions['rerun_unfinished'] += 1
                assert len(before['generation_token_ids'][condition]) == 48
                assert row['generation_token_ids'][condition][:48] == before['generation_token_ids'][condition]
                assert len(row['generation_token_ids'][condition]) <= 256
            if row['outputs'][condition] != before['outputs'][condition]:
                changed.append({'pair_id': row['pair_id'], 'condition': condition,
                                'finished': row['finished'][condition], 'new_tokens': len(row['generation_token_ids'][condition])})
            tokens = row['generation_token_ids'][condition]
            assert len(tokens) == len(row['trace'][condition])
            assert all(s['next_top1_id'] == tokens[i] and s['accepted_prefix_ids'] == tokens[:i]
                       for i, s in enumerate(row['trace'][condition]))
    assert actions == {'reused_finished': 1291, 'rerun_unfinished': 53}
    for c in raw['conditions']:
        quality[c] = {'old_unfinished': sum(not r['finished'][c] for r in old['evaluations']),
                      'new_unfinished': sum(not r['finished'][c] for r in raw['evaluations']),
                      'replacement_character_outputs': sum('\ufffd' in r['outputs'][c] for r in raw['evaluations'])}
    result = {'model': raw['model'], 'raw_sha256': digest(args.run / 'tune.json'),
              'parent_raw_sha256': digest(args.parent_run / 'tune.json'), 'all_assertions_passed': True,
              'actions': dict(actions), 'quality': quality, 'changed_whole_texts': changed,
              'completed_outputs_reused_only_after_21_exact_controls': True,
              'all_53_original_48_token_prefixes_identical': True,
              'no_original_configuration_or_generation_modified': True}
    scored_path = args.run / 'tune_scored_items.json'
    if scored_path.exists():
        graded = json.loads(scored_path.read_text())
        parent_grade = json.loads((args.parent_run / 'tune_scored_items.json').read_text())
        pg = {r['pair_id']: r for r in parent_grade['evaluations']}
        assert graded['completed'] and graded['grading_completed']
        assert graded['grading_source_generation_sha256'] == result['raw_sha256']
        assert graded['frozen_control_grade_verification']['all_identical']
        for r in graded['evaluations']:
            for c in raw['conditions']:
                if r['outputs'][c] == pg[r['pair_id']]['outputs'][c]:
                    assert r['outcome'][c] == pg[r['pair_id']]['outcome'][c]
        result['natural_outcomes'] = {c: dict(Counter(r['outcome'][c] for r in graded['evaluations'])) for c in raw['conditions']}
        result['unchanged_text_grade_count_verified'] = graded['frozen_control_grade_verification']['question_condition_grades_checked']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({'actions': result['actions'], 'quality': quality,
                      'graded': 'natural_outcomes' in result}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
