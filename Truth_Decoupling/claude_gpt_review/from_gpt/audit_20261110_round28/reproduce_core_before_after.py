"""Execute pinned 268708c core and repaired core on the same CPU counterexamples."""
import importlib.machinery
import importlib.util
import json
import math
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parents[1]
CODE = REVIEW / 'from_claude/code'
sys.path[:0] = [str(CODE), str(CODE / 'scripts')]
from core import qa_factorial_das as repaired


def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


def capture_call(function):
    try:
        value = function()
        if torch.is_tensor(value):
            return {'accepted': True, 'finite': bool(torch.isfinite(value).all()),
                    'values': value.tolist() if torch.isfinite(value).all() else None}
        return {'accepted': True, 'value': value}
    except (ValueError, RuntimeError) as exc:
        return {'accepted': False, 'error': str(exc)}


def main():
    old = load('core.e44_reviewed_268708c', HERE / 'core_reviewed_268708c.py.txt')
    upstream = load('e44_r28_reproduction_fixture', CODE / 'tests/test_qa_factorial_das.py')
    bundle = upstream.bundle.__wrapped__()
    report = {'reviewed_commit': '268708cb54f349f92de3276317d8c20b1f5ebbe2',
              'scope': 'Synthetic CPU execution only; no real Llama weights, judge calls or empirical quality claims',
              'before': {}, 'after': {}}
    for label, module in [('before', old), ('after', repaired)]:
        results = report[label]
        h = torch.zeros(1, 1, 2)
        hook = module.ShiftHook([1., 0.], amounts=[1.], mask=torch.ones(1, 1, dtype=torch.bool), decode_all=False)
        hook(None, None, h.clone())
        results['prompt_only_decode_unchanged'] = bool(torch.equal(hook(None, None, h.clone()), h))
        p, q = torch.tensor([[0., -float('inf')]]), torch.tensor([[-float('inf'), 0.]])
        results['zero_probability_jsd'] = capture_call(lambda: module.jsd(p, q))
        rows = upstream._rows(bundle)[:2]
        rows = [dict(row) for row in rows]
        rows[0]['ref_logp'], rows[1]['ref_logp'] = rows[1]['ref_logp'], rows[0]['ref_logp']
        results['same_total_length_swapped_teacher_rows'] = capture_call(lambda: module.rows_loss(bundle, 1, rows, mode='none')[0])
        prompt = upstream._prompts(bundle)['conf']
        results['truncated_reference'] = capture_call(lambda: module.score_answers(bundle, prompt,
            [' '.join('word%d' % i for i in range(20))], eot_ids=upstream.pd.turn_terminator_ids(bundle),
            max_target_tokens=4))
        results['partial_candidates'] = capture_call(lambda: module.choice_metrics([-3., -1., None], [-2., -5.]))
    assert not report['before']['prompt_only_decode_unchanged'] and report['after']['prompt_only_decode_unchanged']
    assert not report['before']['zero_probability_jsd']['finite'] and report['after']['zero_probability_jsd']['finite']
    assert report['before']['same_total_length_swapped_teacher_rows']['accepted']
    assert not report['after']['same_total_length_swapped_teacher_rows']['accepted']
    assert report['before']['truncated_reference']['value'][0]['sum_logp'] is not None
    assert report['after']['truncated_reference']['value'][0]['sum_logp'] is None
    (HERE / 'core_before_after.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
