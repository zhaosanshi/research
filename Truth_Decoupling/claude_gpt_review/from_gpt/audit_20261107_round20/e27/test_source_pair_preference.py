"""Numerical and causal contracts for the final loss-only source comparison."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

REVIEW = Path(__file__).resolve().parents[3]
CODE = REVIEW / 'from_claude/code'
sys.path[:0] = [str(REVIEW / 'from_gpt'), str(CODE / 'scripts'), str(CODE)]
from test_qa_das_pipeline_20261105 import PipelineModel, ToyTokenizer
from core.model_io import ModelBundle, forward_logits
from core.qa_context import answer_logprobs, pack_answers
from core.qa_generation_adapter import frozen_model
from core.qa_precision import fp32_output_projection
from core import qa_source_interchange_training as source
from core import qa_source_pair_preference as pref


class PairPreferenceTests(unittest.TestCase):
    def setUp(self):
        tok = ToyTokenizer(); self.model = PipelineModel(tok)
        self.bundle = ModelBundle(self.model, tok, {'name': 'toy-pair-source', 'layers_attr': 'layers'})
        self.cfg = {'qa_system': 'Answer.', 'qa_user': 'Question: {question}\nAnswer:',
                    'max_new_tokens': 3, 'batch_size': 1}
        self.records = [{'pair_id': 'p1', 'question': 'Where?', 'positive_answer': 'Paris', 'negative_answer': 'Rome'},
                        {'pair_id': 'p2', 'question': 'Which direction?', 'positive_answer': 'North', 'negative_answer': 'Northbound'},
                        {'pair_id': 'p3', 'question': 'Can it?', 'positive_answer': 'Yes indeed', 'negative_answer': 'Yes never'}]
        self.source_system = 'Correct false premises. Be truthful.'
        self.w = torch.nn.functional.normalize(torch.arange(1., 9.), dim=0)

    def direct_sum(self, branch, w):
        if w is None:
            packed = pack_answers([branch['base']], 0, 'cpu')
            logits = forward_logits(self.bundle, packed['inputs'])
        else:
            _, logits, packed, _ = source.training_forward(self.bundle, [branch], 0, w)
        return answer_logprobs(logits, packed)[0]

    def test_symbolic_cancellation_matches_complete_sequence_logp_differences(self):
        pairs = pref.prepare_pairs(self.bundle, self.cfg, self.records, self.source_system)
        self.assertFalse(pairs[1]['has_divergence'])
        self.assertGreater(pairs[2]['shared_answer_tokens'], 0)
        with frozen_model(self.model), fp32_output_projection(self.bundle):
            for pair in pairs:
                for w in (None, torch.nn.Parameter(self.w.clone())):
                    gap = pref.content_gap(self.bundle, pair, 0, w)
                    direct = self.direct_sum(pair['positive'], w) - self.direct_sum(pair['negative'], w)
                    torch.testing.assert_close(gap, direct, atol=1e-5, rtol=1e-5)

    def test_shared_third_logit_gradient_is_exact_zero_with_unequal_lengths(self):
        pair = pref.prepare_pairs(self.bundle, self.cfg, self.records[:1], self.source_system)[0]
        self.assertEqual(pair['common_length'], 4.5)
        fake = torch.linspace(-.3, .3, 256, dtype=torch.float64, requires_grad=True)
        raw_w = torch.nn.Parameter(self.w.clone())
        with frozen_model(self.model), fp32_output_projection(self.bundle):
            with patch.object(pref, 'shared_prefix_logits', return_value=fake):
                loss = pref.preference_loss(self.bundle, pair, 0, raw_w, 0.)
            gradient = torch.autograd.grad(loss, fake)[0]
            good = pair['positive']['base']['answer_ids'][0]; bad = pair['negative']['base']['answer_ids'][0]
            others = torch.ones(256, dtype=torch.bool); others[good] = others[bad] = False
            self.assertEqual(float(gradient[others].abs().max()), 0.)
            changed = fake.detach().clone(); changed[76] += 10  # I / a possible refusal prefix, neither target here.
            with patch.object(pref, 'shared_prefix_logits', return_value=changed):
                new_loss = pref.preference_loss(self.bundle, pair, 0, raw_w, 0.)
            torch.testing.assert_close(loss, new_loss, atol=0, rtol=0)

    def test_reference_does_not_depend_on_w_and_shared_inputs_have_no_future_target(self):
        pair = pref.prepare_pairs(self.bundle, self.cfg, self.records[2:], self.source_system)[0]
        inputs = pref._prefix_pack(pair['positive']['base'], pair['shared_answer_tokens'], 'cpu')
        expected = pair['positive']['base']['prefix_ids'] + pair['positive']['base']['answer_ids'][:pair['shared_answer_tokens']]
        self.assertEqual(inputs['inputs']['input_ids'][0].tolist(), expected)
        with frozen_model(self.model), fp32_output_projection(self.bundle):
            first = pref.content_gap(self.bundle, pair, 0)
            self.assertFalse(first.requires_grad)
            w = torch.nn.Parameter(torch.flip(self.w, [0]))
            pref.preference_loss(self.bundle, pair, 0, w, float(first)).backward()
            second = pref.content_gap(self.bundle, pair, 0)
            torch.testing.assert_close(first, second, atol=0, rtol=0)
            changed = copy.deepcopy(pair)
            k = pair['shared_answer_tokens']
            for branch in ('positive', 'negative'):
                for stream in ('base', 'source'):
                    changed[branch][stream]['answer_ids'][k] += 1
                    changed[branch][stream]['full_ids'] = changed[branch][stream]['prefix_ids'] + changed[branch][stream]['answer_ids']
            a = pref.shared_prefix_logits(self.bundle, pair, 0, w)
            b = pref.shared_prefix_logits(self.bundle, changed, 0, w)
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_same_prompt_is_identity_and_preference_gradient_zero(self):
        pair = pref.prepare_pairs(self.bundle, self.cfg, self.records[:1], self.cfg['qa_system'])[0]
        raw_w = torch.nn.Parameter(self.w.clone())
        with frozen_model(self.model), fp32_output_projection(self.bundle):
            reference = pref.content_gap(self.bundle, pair, 0)
            loss = pref.preference_loss(self.bundle, pair, 0, raw_w, reference)
            loss.backward()
        self.assertEqual(float(raw_w.grad.norm()), 0.)
        self.assertAlmostEqual(float(loss.detach()), float(torch.log(torch.tensor(2.))), places=6)

    def test_real_training_preserves_initial_bytes_base_weights_and_fixed_configuration(self):
        old = [p.clone() for p in self.model.parameters()]
        initial = self.w.clone(); initial[0] = torch.nextafter(initial[0], torch.tensor(float('inf')))
        result = pref.train(self.bundle, self.records, self.cfg, initial, layer=0,
                            source_system=self.source_system, pilot=True, log_fn=None)
        self.assertTrue(torch.equal(initial, result['initial_W']))
        self.assertEqual(result['gradient_audit']['updates'], 3)
        self.assertGreater(result['gradient_audit']['nonzero_updates'], 0)
        self.assertGreater(float((result['W'] - initial).norm()), 1e-6)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(old, self.model.parameters())))
        self.assertTrue(all(not b._forward_hooks for b in self.model.layers))
        self.assertFalse(self.model._forward_pre_hooks)
        self.assertTrue(result['compute']['not_compute_matched_to_positive_CE'])
        self.assertTrue(all(t.get('shared_non_target_logit_gradient_max', 0) == 0 for t in result['loss_trace']))
        with tempfile.TemporaryDirectory() as directory:
            pref.save(result, directory); loaded = pref.load(directory)
            self.assertTrue(torch.equal(loaded['initial_W'], initial))
            self.assertTrue(torch.equal(loaded['W'], result['W']))

    def test_bfloat16_real_gradient_and_duplicate_candidate_rejection(self):
        self.model.bfloat16()
        result = pref.train(self.bundle, self.records, self.cfg, self.w, layer=0,
                            source_system=self.source_system, pilot=True, log_fn=None)
        self.assertEqual(result['gradient_audit']['nonzero_updates'], 3)
        records = copy.deepcopy(self.records); records[0]['negative_answer'] = records[0]['positive_answer']
        with self.assertRaisesRegex(ValueError, 'identical token'):
            pref.prepare_pairs(self.bundle, self.cfg, records, self.source_system)


if __name__ == '__main__': unittest.main()
