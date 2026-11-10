"""Execute original 44b57ac E43 core and current repairs against the same CPU cases."""
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parents[1]
sys.path[:0] = [str(REVIEW / 'from_claude/code'), str(REVIEW / 'from_gpt')]
from core import qa_prompt_das as current
from test_qa_das_training_20261105 import make_bundle


def load_original():
    path = HERE / 'training_reviewed_source.py.txt'
    name = 'core.qa_prompt_das_reviewed_r27'
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


def checks(module):
    h = torch.full((1, 3, 8), 100., dtype=torch.bfloat16)
    w = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.])
    random = torch.ones(8) / 8 ** .5
    mask = torch.tensor([[False, True, True]])
    das = module.ClampHook(w, targets=100.5, mask=mask, decode_all=False)
    hd = das(None, None, h)
    control = module.ClampHook(w, targets=100.5, mask=mask, decode_all=False, random_direction=random)
    try:
        hr = control(None, None, h)
        random_result = {'accepted': True, 'actual_norm': float((hr[0, 1].float() - h[0, 1].float()).norm()),
                         'reported_relative_size': control.stats()['prefill']['rel'][0]}
    except RuntimeError as exc:
        random_result = {'accepted': False, 'error': str(exc)}
    rounded = module.ClampHook(w, targets=100.24, mask=mask, decode_all=False)
    small = rounded(None, None, h)

    g = torch.Generator().manual_seed(7)
    direction = torch.randn(4, generator=g, requires_grad=True)
    source = torch.randn(2, 4, generator=g, requires_grad=True)
    state = torch.randn(2, 3, 4, generator=g)
    module.ClampHook(direction, source_hidden=source, mask=torch.ones(2, 3, dtype=torch.bool),
                     decode_all=False)(None, None, state).square().sum().backward()

    bundle = make_bundle(); bundle.model.requires_grad_(False)
    item = {'prefix_ids': [3, 4, 5], 'answer_ids': [6, 7], 'full_ids': [3, 4, 5, 6, 7],
            'prompt_length': 3, 'answer_length': 2, 'content_start': 1}
    original_hook = module.ClampHook
    class OrthogonalPrefixCorruption(original_hook):
        def __call__(self, layer, args, out):
            edited = super().__call__(layer, args, out).clone()
            edited[:, 0, 1] += 1
            return edited
    with patch.object(module, 'ClampHook', OrthogonalPrefixCorruption):
        verification = module.verify_clamp(bundle, 1, [item], torch.zeros(1, 4), torch.tensor([1., 0., 0., 0.]))
    return {
        'bf16_random_control': {'das_stored_norm': float((hd[0, 1].float() - h[0, 1].float()).norm()), **random_result},
        'bf16_rounds_to_zero': {'actual_norm': float((small.float() - h.float()).norm()),
                               'reported_relative_size': rounded.stats()['prefill']['rel'][0]},
        'source_state_gradient': None if source.grad is None else float(source.grad.norm()),
        'direction_gradient': float(direction.grad.norm()),
        'orthogonal_unwritten_change_accepted_by_verify': verification['passed'],
    }


def main():
    before, after = checks(load_original()), checks(current)
    assert before['bf16_random_control']['actual_norm'] == 0
    assert before['bf16_random_control']['das_stored_norm'] == .5
    assert not after['bf16_random_control']['accepted']
    assert before['bf16_rounds_to_zero']['reported_relative_size'] > 0
    assert after['bf16_rounds_to_zero']['reported_relative_size'] == 0
    assert before['source_state_gradient'] > 0 and after['source_state_gradient'] is None
    assert before['orthogonal_unwritten_change_accepted_by_verify']
    assert not after['orthogonal_unwritten_change_accepted_by_verify']
    output = {'reviewed_commit': '44b57ac',
              'reviewed_source_sha256': hashlib.sha256((HERE / 'training_reviewed_source.py.txt').read_bytes()).hexdigest(),
              'scope': 'Synthetic CPU tensors / small causal model, not measured Llama effects',
              'before': before, 'after': after, 'new_gpu_model_runs': 0, 'new_api_calls': 0}
    (HERE / 'training_before_after.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(output, ensure_ascii=False))


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
