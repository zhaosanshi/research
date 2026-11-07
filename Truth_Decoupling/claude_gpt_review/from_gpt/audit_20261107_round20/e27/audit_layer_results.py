"""Independently verify the four fixed layer follow-ups and reused controls."""
import argparse
import json
from pathlib import Path

from audit_source_results import audit_run, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layers-root', type=Path, required=True)
    p.add_argument('--original-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = {'scope': 'four layers fixed before these runs; all results on previously exposed tuning questions', 'runs': {}}
    for model, layers in [('llama3.1-8b', [15, 23]), ('qwen3-8b', [17, 26])]:
        parent_dir = args.original_root / 'frozen' / model
        parent = json.loads((parent_dir / 'tune.json').read_text())
        parent_grade = json.loads((parent_dir / 'tune_scored_items.json').read_text())
        for layer in layers:
            base = args.layers_root / model / ('L' + str(layer))
            five, raw5, grade5 = audit_run(base / 'controls_full')
            six, raw6, grade6 = audit_run(base / 'train_full92_ep3')
            assert raw5['identity']['layer'] == raw6['identity']['layer'] == layer
            assert raw5['identity']['parent_control_sha256'] == digest(parent_dir / 'tune.json')
            assert raw6['identity']['control_generation_sha256'] == digest(base / 'controls_full/tune.json')
            plan = json.loads((base / 'train_full92_ep3/layer_plan.json').read_text())
            assert raw5['identity']['layer_plan']['sha256'] == digest(base / 'train_full92_ep3/layer_plan.json')
            assert layer in plan['models'][model]['additional_layers']
            assert grade5['frozen_control_grade_verification']['question_condition_grades_checked'] == 576
            assert grade6['frozen_control_grade_verification']['question_condition_grades_checked'] == 960
            reused = ('baseline', 'source_only', 'equal_logit_mix')
            for original, before, five_raw, five_grade, six_raw, six_grade in zip(
                    parent['evaluations'], parent_grade['evaluations'], raw5['evaluations'], grade5['evaluations'],
                    raw6['evaluations'], grade6['evaluations']):
                assert original['pair_id'] == five_raw['pair_id'] == six_raw['pair_id']
                assert original['prompt_token_ids'] == five_raw['prompt_token_ids'] == six_raw['prompt_token_ids']
                for condition in reused:
                    for key in ('outputs', 'finished', 'generation_token_ids', 'trace'):
                        assert original[key][condition] == five_raw[key][condition]
                    assert five_grade['outcome'][condition] == before['outcome'][condition]
                    assert five_raw['reused_control_provenance'][condition]['generated_again'] is False
                for condition in raw5['conditions']:
                    assert five_raw['outputs'][condition] == six_raw['outputs'][condition]
                    assert five_grade['outcome'][condition] == six_grade['outcome'][condition]
            adapter = json.loads((base / 'train_full92_ep3/source_adapter.json').read_text())
            assert adapter['layer'] == layer and adapter['layer_plan']['plan'] == plan
            assert not adapter['pilot'] and len(adapter['training_pair_ids']) == 92
            assert adapter['gradient_audit']['updates'] == 276
            assert adapter['train_config'] == {'epochs': 3, 'lr': .001, 'clip': 1., 'seed': 42,
                                                'optimizer': 'Adam', 'alpha': 1., 'include_eos': False, 'fixed_center': False}
            assert all(c['source_past_and_current_bitwise_equal'] and c['base_past_and_current_logits_bitwise_equal']
                       and c['same_prompt_logits_bitwise_equal'] and c['same_prompt_direction_gradient_norm'] == 0
                       for c in adapter['causal_prefix_checks'])
            report = json.loads((base / 'train_full92_ep3/tune_matched_coverage.json').read_text())
            matched = []
            for point in report['matched_coverage']['coverages']:
                matched.append({'coverage': point['target_coverage'], 'k': point['k'],
                                'correct': {name: arm.get('correct') for name, arm in point['arms'].items()},
                                'feasible': {name: arm['feasible'] for name, arm in point['arms'].items()}})
            result['runs'][model + '/L' + str(layer)] = {'five': five, 'six': six,
                    'training': {k: adapter[k] for k in ('losses', 'gradient_audit', 'causal_prefix_checks')},
                    'all_reused_raw_traces_prompts_and_grades_identical': True,
                    'frozen_plan_sha256': adapter['layer_plan']['sha256'],
                    'common_unsteered_score_matched_counts': matched}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    for name, item in result['runs'].items():
        c = item['six']['conditions']['trained_source_swap_das_init']
        comp = item['six']['comparisons']['trained_source_swap_das_init_vs_baseline']
        print(name, c, comp, flush=True)


if __name__ == '__main__': main()
