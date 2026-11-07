"""Read-only audit of the final seven-arm preference comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

from audit_source_results import audit_run, compare, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--parent-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    results = {'protocol': 'independent_preference_result_audit_v1', 'models': {}}
    for model, layer in [('llama3.1-8b', 15), ('qwen3-8b', 17)]:
        run = args.run_root / model
        parent_dir = args.parent_root / model / ('L' + str(layer)) / 'train_full92_ep3'
        summary, raw, grade = audit_run(run)
        parent_raw = json.loads((parent_dir / 'tune.json').read_text())
        parent_grade = json.loads((parent_dir / 'tune_scored_items.json').read_text())
        checkpoint = json.loads((run / 'preference_adapter.json').read_text())
        plan = json.loads((run / 'plan.json').read_text())
        assert raw['protocol'] == checkpoint['protocol'] == 'supervised_source_pair_preference_v1'
        assert raw['identity']['control_generation_sha256'] == digest(parent_dir / 'tune.json')
        assert raw['identity']['adapter_sha256'] == digest(run / 'preference_adapter.pt')
        assert plan['ce_parent']['graded_items_sha256'] == digest(parent_dir / 'tune_scored_items.json')
        assert checkpoint['fixed_plan_sha256'] == digest(run / 'plan.json')
        assert len(checkpoint['training_pair_ids']) == 92 and checkpoint['gradient_audit']['updates'] == 276
        assert not checkpoint['pilot'] and checkpoint['layer'] == layer
        assert grade['frozen_control_grade_verification']['all_identical']
        assert grade['frozen_control_grade_verification']['question_condition_grades_checked'] == 1152
        assert raw['control_identity'] == parent_raw['identity']
        for original, old_grade, current, current_grade in zip(parent_raw['evaluations'], parent_grade['evaluations'], raw['evaluations'], grade['evaluations']):
            assert original['pair_id'] == current['pair_id']
            assert original['prompt_token_ids'] == current['prompt_token_ids']
            for name in parent_raw['conditions']:
                for field in ('outputs', 'finished', 'generation_token_ids', 'trace'):
                    assert original[field][name] == current[field][name]
                assert old_grade['outcome'][name] == current_grade['outcome'][name]
        pt = torch.load(run / 'preference_adapter.pt', map_location='cpu', weights_only=True)
        initial = pt['initial_W'].contiguous().numpy().tobytes()
        assert hashlib.sha256(initial).hexdigest() == plan['initial_W']['sha256']
        assert all(t.get('shared_non_target_logit_gradient_max', 0) == 0 for t in checkpoint['loss_trace'])
        assert checkpoint['not_original_DPO_reproduction'] and not checkpoint['answer_rate_invariance_claimed']
        summary['pair_comparisons'] = {name: compare(grade['evaluations'], 'pair_preference_source_swap', name)
                                      for name in parent_raw['conditions']}
        summary['training'] = {k: checkpoint[k] for k in ('losses', 'gradient_audit', 'compute', 'loss_trace', 'pair_lengths')}
        summary['all_1152_parent_grades_and_original_fields_identical'] = True
        summary['initial_W_matches_frozen_CE_starting_bytes'] = True
        results['models'][model] = summary
        print(model, summary['conditions']['pair_preference_source_swap'], summary['pair_comparisons']['baseline'], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__': main()
