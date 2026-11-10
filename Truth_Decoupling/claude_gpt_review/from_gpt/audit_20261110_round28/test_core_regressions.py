"""E44 actual-function counterexamples; CPU only, no model weights or judge calls."""
import copy
import importlib.util
import math
import os
from pathlib import Path
import sys
import unittest

import torch

REVIEW = Path(__file__).resolve().parents[2]
CODE = REVIEW / 'from_claude/code'
sys.path[:0] = [str(CODE), str(CODE / 'scripts')]
from core import qa_factorial_das as fd
from core import qa_prompt_das as pd


class CoreTensorTests(unittest.TestCase):
    def test_single_token_prefill_is_not_reused_as_decode_mask(self):
        h = torch.zeros(1, 1, 2)
        hook = fd.ShiftHook([1., 0.], amounts=[1.], mask=torch.ones(1, 1, dtype=torch.bool), decode_all=False)
        self.assertFalse(torch.equal(hook(None, None, h.clone()), h))
        self.assertTrue(torch.equal(hook(None, None, h.clone()), h))
        self.assertEqual(hook.applied, 1)

    def test_single_token_finished_row_stays_unchanged(self):
        h = torch.zeros(2, 1, 2)
        hook = fd.ShiftHook([1., 0.], amounts=[1., 2.], mask=torch.ones(2, 1, dtype=torch.bool))
        hook(None, None, h.clone())
        hook.active_rows = torch.tensor([False, True])
        out = hook(None, None, h.clone())
        self.assertTrue(torch.equal(out[0], h[0]))
        self.assertEqual(out[1, 0, 0], 2.)

    def test_wrong_first_call_shape_cannot_pose_as_decode(self):
        hook = fd.ShiftHook([1., 0.], amounts=[1.], mask=torch.ones(1, 3, dtype=torch.bool))
        with self.assertRaises(RuntimeError):
            hook(None, None, torch.zeros(1, 1, 2))

    def test_jsd_supports_exact_zero_probability_without_nan_gradients(self):
        p = torch.tensor([[0., -float('inf')]], requires_grad=True)
        q = torch.tensor([[-float('inf'), 0.]])
        result = fd.jsd(p, q)
        self.assertAlmostEqual(result.item(), math.log(2), places=6)
        result.sum().backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertEqual(fd.jsd(q, q).item(), 0.)
        with self.assertRaises(ValueError):
            fd.jsd(torch.tensor([[1., 1.]]), q)

    def test_incomplete_candidates_are_not_silently_discarded(self):
        self.assertIsNone(fd.choice_metrics([-3., None], [-1.]))
        for bad in (float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                fd.choice_metrics([bad], [-1.])

    def test_best_reference_and_best_of_true_are_different_readouts(self):
        result = fd.choice_metrics([-3., -1.], [-2.], best_true_index=0)
        self.assertEqual(result['any_true_top1'], 1)
        self.assertEqual(result['standard_mc1'], 0)
        self.assertEqual(result['best_answer_margin'], -1.)
        self.assertIn('not_standard', result['mc1_definition'])

    def test_bfloat_random_control_rejects_an_unreachable_stored_norm(self):
        h = torch.full((1, 2, 8), 100., dtype=torch.bfloat16)
        ref = torch.tensor([1.] + [0.] * 7)
        rand = pd.unit(torch.ones(8))
        mask = torch.ones(1, 2, dtype=torch.bool)
        nominal = fd.ShiftHook(rand, amounts=[.5], mask=mask, decode_all=False)(None, None, h)
        reference = fd.ShiftHook(ref, amounts=[.5], mask=mask, decode_all=False)(None, None, h)
        self.assertEqual(float((nominal.float() - h.float()).norm()), 0.)
        self.assertGreater(float((reference.float() - h.float()).norm()), 0.)
        matched = fd.ShiftHook(rand, amounts=[.5], reference_direction=ref, mask=mask, decode_all=False)
        with self.assertRaisesRegex(RuntimeError, 'cannot match'):
            matched(None, None, h)

    def test_bfloat_random_control_matches_actual_stored_norm_when_reachable(self):
        torch.manual_seed(91)
        h = (torch.randn(2, 3, 32) * .1).bfloat16()
        ref, rand = pd.unit(torch.randn(32)), pd.unit(torch.randn(32))
        amount = torch.tensor([.5, -.3])
        mask = torch.tensor([[False, True, True], [True, True, False]])
        hook = fd.ShiftHook(rand, amounts=amount, reference_direction=ref, mask=mask, decode_all=False)
        result = hook(None, None, h)
        expected = (h.float() + amount[:, None, None] * ref).to(h.dtype)
        expected_norm = (expected.float() - h.float()).norm(dim=-1)[mask]
        actual_norm = (result.float() - h.float()).norm(dim=-1)[mask]
        tolerance = .01 * expected_norm + 1e-4 * h.float().norm(dim=-1)[mask]
        self.assertTrue(torch.all((actual_norm - expected_norm).abs() <= tolerance))
        self.assertTrue(torch.equal(result[~mask], h[~mask]))
        self.assertFalse(hook.stats()['fully_information_free_random_control'])

    def test_unrecognized_loss_row_cannot_be_ignored(self):
        with self.assertRaises(ValueError):
            fd.question_loss(torch.tensor([.1, .9]), ['cause', 'typo'])


class CoreModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = CODE / 'tests/test_qa_factorial_das.py'
        spec = importlib.util.spec_from_file_location('r28_upstream_core_fixture', path)
        cls.upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.upstream)
        cls.bundle = cls.upstream.bundle.__wrapped__()

    def test_reference_rows_cannot_exchange_lengths_or_target_ids(self):
        rows = self.upstream._rows(self.bundle)[:2]
        bad = [dict(row) for row in rows]
        bad[0]['ref_logp'], bad[1]['ref_logp'] = bad[1]['ref_logp'], bad[0]['ref_logp']
        self.assertEqual(sum(r['ref_logp'].shape[0] for r in bad), sum(r['encoded']['answer_length'] for r in bad))
        with self.assertRaisesRegex(RuntimeError, 'length'):
            fd.rows_loss(self.bundle, 1, bad, mode='none')
        bad = [dict(rows[0])]
        bad[0]['ref_target_ids'] = list(rows[0]['encoded']['answer_ids'])
        bad[0]['ref_target_ids'][0] += 1
        with self.assertRaisesRegex(RuntimeError, 'different target tokens'):
            fd.rows_loss(self.bundle, 1, bad, mode='none')

    def test_teacher_distribution_is_detached_from_direction_training(self):
        rows = self.upstream._rows(self.bundle)[:1]
        teacher = rows[0]['ref_logp'].clone().requires_grad_(True)
        rows[0] = dict(rows[0], ref_logp=teacher)
        w = torch.nn.Parameter(pd.unit(torch.randn(self.bundle.hidden_size)))
        values, _ = fd.rows_loss(self.bundle, 1, rows, w=w, grad=True)
        values.sum().backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(w.grad)
        self.assertTrue(torch.isfinite(w.grad).all())

    def test_truncated_reference_is_not_a_whole_answer_probability(self):
        p = self.upstream._prompts(self.bundle)['conf']
        answers = [' '.join('word%d' % i for i in range(20)), 'Paris']
        scores = fd.score_answers(self.bundle, p, answers, eot_ids=pd.turn_terminator_ids(self.bundle),
                                  max_target_tokens=4)
        self.assertEqual(scores[0]['n_tokens'], 4)
        self.assertTrue(scores[0]['truncated'])
        self.assertIsNone(scores[0]['sum_logp'])
        self.assertEqual(scores[0]['skipped'], 'truncated_reference')
        self.assertIsNotNone(scores[1]['sum_logp'])


if __name__ == '__main__':
    unittest.main()
