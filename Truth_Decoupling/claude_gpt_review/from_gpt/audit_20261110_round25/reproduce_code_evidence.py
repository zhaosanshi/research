"""CPU-only checks of code premises in 820d7c9; no real-model/API calls."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

REVIEW = Path(__file__).resolve().parents[2]
CODE = REVIEW / 'from_claude/code'
sys.path[:0] = [str(CODE), str(REVIEW / 'from_gpt')]
import core.qa_source_interchange as source
import core.qa_das as qa
from core.directions import _interchange_hook_factory, train_das
from core.intervene import SteerSpec, LayerSteerHook
from core.model_io import capture_hooks, forward_logits
from test_qa_das_training_20261105 import make_bundle, CFG, RECORDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-oracle', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    report = {'scope': 'Actual production functions with synthetic CPU values; optional read-only old E42 artifact',
              'new_generation_experiments': 0, 'new_api_calls': 0}

    calls = []
    decoder = source.SourceInterchangeDecoder(SimpleNamespace(), [1, 2], [7, 8, 9],
                                               mode='source_only', use_cache=False)
    def forward(self, prompt, cache):
        calls.append(list(prompt))
        return torch.tensor([0., 5., 1.]), None
    decoder._forward = MethodType(forward, decoder)
    with patch.object(source, 'capture_source', side_effect=AssertionError('must not capture a layer')), \
         patch.object(source, 'write_base', side_effect=AssertionError('must not write a layer')):
        logits, trace = decoder.next_logits()
    assert calls == [[7, 8, 9]] and trace['writes'] == []
    report['source_only'] = {'forwarded_prompts': calls, 'base_forward_count': 0,
                             'source_forward_count': 1, 'layer_writes': len(trace['writes']),
                             'top1': int(logits.argmax())}

    bundle = make_bundle()
    pairs = qa.prepare_selfcheck_pairs(bundle, CFG, RECORDS[:1])
    states = qa._source_states(bundle, pairs, 1)
    prefixes = {}
    for side in ('positive', 'negative'):
        item = pairs[0][side]
        end = item['encoded']['prompt_length'] - 1
        with capture_hooks(bundle, [1], ('resid',), position='all') as store:
            forward_logits(bundle, item['inputs'], grad=False)
        prefixes[side] = store[('resid', 1)][:, end, :].detach().cpu()
    prompt_difference = float((prefixes['positive'] - prefixes['negative']).norm())
    actual_difference = float((states['positive'] - states['negative']).norm())
    assert prompt_difference == 0 and actual_difference > 0
    report['e25_position'] = {'prompt_state_difference': prompt_difference,
        'actual_answer_end_source_state_difference': actual_difference,
        'positions': {side: {'prompt_end': pairs[0][side]['encoded']['prompt_length'] - 1,
                             'answer_end': pairs[0][side]['answer_end'],
                             'label_predictor': pairs[0][side]['label_predictor']}
                      for side in ('positive', 'negative')},
        'interpretation': 'Same-prefix equality is true but E25 reads answer_end, where the answers already differ.'}

    h = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [9., 10., 11.]]])
    w = torch.tensor([[1.], [0.], [0.]], requires_grad=True)
    donor = torch.tensor([[7., 8., 9.]])
    train_hook = _interchange_hook_factory(lambda: w, {'h': donor}, positions='all')
    trained = train_hook(None, (), h)
    spec = SteerSpec(w.detach(), 0, mode='clamp', alpha=1., z_target=[7.], position='all')
    inference_hook = LayerSteerHook([spec])
    prompt = inference_hook(None, (), h)
    decode = inference_hook(None, (), h[:, -1:, :])
    assert torch.equal(trained, prompt)
    assert torch.equal(prompt[..., 1:], h[..., 1:])
    assert torch.equal(prompt[..., 0], torch.full((1, 3), 7.))
    assert torch.equal(decode[..., 0], torch.full((1, 1), 7.))
    report['all_position_clamp_primitive'] = {'train_equals_inference': True,
        'orthogonal_coordinates_preserved': True, 'prefill_positions_written': 3,
        'next_decode_positions_written': 1, 'applied_calls': spec.n_applied}
    try:
        train_das(SimpleNamespace(), [], 0, loss='ce', positions='all',
                  label_ids={'True': [1], 'False': [2]})
    except ValueError as exc:
        report['existing_train_das_ce_all'] = {'accepted': False, 'error': str(exc)}
    else:
        raise AssertionError('Existing train_das unexpectedly accepted ce/all')

    if args.raw_oracle:
        raw = args.raw_oracle.read_bytes()
        data = json.loads(raw)
        rows = data['evaluations']
        normalize = lambda value: ' '.join(value.lower().split())
        controls = {}
        for condition in ('truthful_source_only', 'gold_source_only'):
            controls[condition] = {
                'n': len(rows),
                'unchanged_from_baseline': sum(normalize(r['outputs'][condition]) == normalize(r['outputs']['baseline']) for r in rows),
                'all_recorded_mean_write_zero': all(r['diag'][condition]['write_over_base_mean'] == 0 for r in rows),
                'all_recorded_max_write_zero': all(r['diag'][condition]['write_over_base_max'] == 0 for r in rows)}
        report['original_e42_controls'] = {'sha256': hashlib.sha256(raw).hexdigest(), 'controls': controls}
        assert raw == args.raw_oracle.read_bytes()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
