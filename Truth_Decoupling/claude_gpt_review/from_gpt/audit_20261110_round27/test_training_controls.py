"""Executable E43 training/control regressions; no downloaded model weights or API calls."""
import copy
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

REVIEW = Path(__file__).resolve().parents[2]
CODE = REVIEW / 'from_claude/code'
sys.path[:0] = [str(CODE), str(CODE / 'tests'), str(REVIEW / 'from_gpt')]
from core import qa_prompt_das as pd
from test_qa_das_training_20261105 import make_bundle


def item():
    return {'prefix_ids': [3, 4, 5], 'answer_ids': [6, 7], 'full_ids': [3, 4, 5, 6, 7],
            'prompt_length': 3, 'answer_length': 2, 'content_start': 1}


class TrainingControlTests(unittest.TestCase):
    def test_actual_bf16_write_is_not_the_intended_gap(self):
        h = torch.full((1, 3, 8), 100., dtype=torch.bfloat16)
        w = torch.zeros(8); w[0] = 1
        hook = pd.ClampHook(w, targets=100.24, mask=torch.tensor([[False, True, True]]), decode_all=False)
        changed = hook(None, None, h)
        self.assertTrue(torch.equal(changed, h))
        self.assertGreater(hook.stats()['prefill']['gap_abs'][0], 0)
        self.assertEqual(hook.stats()['prefill']['stored_norm'], [0.])
        self.assertEqual(hook.stats()['prefill']['rel'], [0.])

    def test_unattainable_random_cast_norm_is_rejected(self):
        h = torch.full((1, 3, 8), 100., dtype=torch.bfloat16)
        w = torch.zeros(8); w[0] = 1
        random = torch.ones(8) / 8 ** .5
        mask = torch.tensor([[False, True, True]])
        das = pd.ClampHook(w, targets=100.5, mask=mask, decode_all=False)(None, None, h)
        self.assertEqual(float((das[0, 1].float() - h[0, 1].float()).norm()), .5)
        with self.assertRaisesRegex(RuntimeError, 'declared cast tolerance'):
            pd.ClampHook(w, targets=100.5, mask=mask, decode_all=False, random_direction=random)(None, None, h)

    def test_random_matches_stored_reference_at_same_state(self):
        g = torch.Generator().manual_seed(1)
        h = torch.randn(2, 5, 32, generator=g).to(torch.bfloat16)
        w = pd.unit(torch.randn(32, generator=g)); random = pd.unit(torch.randn(32, generator=g))
        mask = torch.tensor([[False, False, True, True, True], [False, True, True, True, True]])
        main = pd.ClampHook(w, targets=[.7, -.3], mask=mask, decode_all=False)
        control = pd.ClampHook(w, targets=[.7, -.3], mask=mask, decode_all=False, random_direction=random)
        hm, hr = main(None, None, h), control(None, None, h)
        target = (hm.float() - h.float()).norm(dim=-1)
        actual = (hr.float() - h.float()).norm(dim=-1)
        tolerance = .01 * target + .0001 * h.float().norm(dim=-1)
        self.assertTrue(bool(((target - actual).abs()[mask] <= tolerance[mask] + 1e-7).all()))
        self.assertTrue(torch.equal(hr[~mask], h[~mask]))
        expected = [float(actual[b][mask[b]].mean()) for b in range(2)]
        self.assertEqual(control.stats()['prefill']['stored_norm'], expected)
        self.assertFalse(control.stats()['fully_information_free_random_control'])
        self.assertTrue(control.stats()['learned_direction_used_for_size_only'])

    def test_source_state_detached_but_projection_gradient_kept(self):
        g = torch.Generator().manual_seed(6)
        w = torch.randn(4, generator=g, requires_grad=True)
        source = torch.randn(2, 4, generator=g, requires_grad=True)
        h = torch.randn(2, 3, 4, generator=g)
        mask = torch.tensor([[False, True, True], [False, False, True]])
        result = pd.ClampHook(w, source_hidden=source, mask=mask, decode_all=False)(None, None, h)
        result.square().sum().backward()
        gradient = w.grad.clone()
        self.assertIsNone(source.grad)
        self.assertGreater(float(gradient.norm()), 0)
        independent = w.detach().clone().requires_grad_()
        u = independent / independent.norm()
        gaps = ((source.detach() @ u)[:, None] - h @ u) * mask
        manual = h + gaps[..., None] * u
        manual.square().sum().backward()
        self.assertTrue(torch.allclose(gradient, independent.grad, atol=1e-5))

    def test_nonfinite_target_and_zero_direction_fail_before_writing(self):
        h = torch.ones(1, 2, 4)
        for w, target in [(torch.zeros(4), 0), (torch.ones(4), float('nan'))]:
            with self.assertRaises(ValueError):
                pd.ClampHook(w, targets=target)(None, None, h)

    def test_inactive_decode_rows_are_untouched(self):
        h = torch.randn(2, 1, 4)
        hook = pd.ClampHook(torch.tensor([1., 0., 0., 0.]), targets=[7., 8.],
                            mask=torch.ones(2, 3, dtype=torch.bool), decode_all=True)
        hook.active_rows = torch.tensor([False, True])
        after = hook(None, None, h)
        self.assertTrue(torch.equal(after[0], h[0]))
        self.assertEqual(float(after[1, 0, 0]), 8.)
        self.assertEqual(hook.stats()['decode_stored_norm_steps'][0][0], 0)

    def test_verifier_checks_full_unwritten_state_not_only_projection(self):
        bundle = make_bundle(); bundle.model.requires_grad_(False)
        w = torch.tensor([1., 0., 0., 0.]); source = torch.zeros(1, 4)
        original = pd.ClampHook
        class CorruptOrthogonalPrefix(original):
            def __call__(self, module, args, out):
                changed = super().__call__(module, args, out)
                changed = changed.clone()
                changed[:, 0, 1] += 1     # The direction's coordinate stays unchanged.
                return changed
        with patch.object(pd, 'ClampHook', CorruptOrthogonalPrefix):
            result = pd.verify_clamp(bundle, 1, [item()], source, w)
        self.assertFalse(result['passed'])
        self.assertFalse(result['untouched_full_state_equal'])
        self.assertGreater(result['untouched_abs_change_max'], 0)

    def test_training_rejects_unfrozen_or_dropout_mode_model(self):
        bundle = make_bundle()
        pair = {'pair_id': 'synthetic', 'plus': {'encoded': item(), 'source_hidden': torch.ones(4)},
                'minus': {'encoded': item(), 'source_hidden': -torch.ones(4)}}
        with self.assertRaisesRegex(ValueError, 'frozen model'):
            pd.train_direction(bundle, 1, [pair], epochs=1)
        bundle.model.requires_grad_(False).train()
        with self.assertRaisesRegex(ValueError, 'eval-mode'):
            pd.train_direction(bundle, 1, [pair], epochs=1)
        bundle.model.eval()
        before = {n: p.clone() for n, p in bundle.model.named_parameters()}
        w, history = pd.train_direction(bundle, 1, [pair], epochs=1)
        self.assertAlmostEqual(float(w.norm()), 1., places=5)
        self.assertGreater(history['history'][0]['grad_norm_mean'], 0)
        self.assertTrue(all(p.grad is None and torch.equal(p, before[n]) for n, p in bundle.model.named_parameters()))


class HFGenerationTerminationTests(unittest.TestCase):
    def test_finished_rows_stop_receiving_hidden_writes(self):
        if not os.environ.get('TD_TEST_TOKENIZER'):
            self.skipTest('set TD_TEST_TOKENIZER to the locally supplied real Llama tokenizer')
        import test_qa_prompt_das as upstream
        from transformers import LogitsProcessor
        bundle = upstream.bundle.__wrapped__()
        prompts = [upstream._prompt(bundle), upstream._prompt(bundle, 'Why?')]
        traces = []
        original = pd.ClampHook
        class Observe(original):
            def __call__(self, module, args, out):
                before = out.clone()
                after = super().__call__(module, args, out)
                traces.append((before.detach(), after.detach()))
                return after
        class Stops(LogitsProcessor):
            def __init__(self, width): self.width = width
            def __call__(self, ids, scores):
                step = ids.shape[1] - self.width
                scores[:] = -torch.inf
                scores[0, 128009] = 0
                scores[1, 128009 if step == 2 else 65] = 0
                return scores
        def generate(bundle, inputs, max_new_tokens):
            out = bundle.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                logits_processor=[Stops(inputs['input_ids'].shape[1])], pad_token_id=bundle.tokenizer.pad_token_id)
            return ['', ''], out
        with patch.object(pd, 'ClampHook', Observe), patch.object(pd, 'greedy_generate', generate):
            records = pd.generate_prompts(bundle, prompts, max_new_tokens=5, batch_size=2,
                layer=1, direction=torch.ones(bundle.hidden_size), targets=.7)
        self.assertEqual([r['n_tokens'] for r in records], [0, 2])
        self.assertTrue(all(r['finished'] for r in records))
        self.assertEqual([r['write']['decode_steps_own'] for r in records], [0, 2])
        self.assertEqual(len(traces), 3)
        self.assertTrue(all(torch.equal(before[0], after[0]) for before, after in traces[1:]))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
