"""Causal/cache invariants on a real differentiable two-layer toy causal LM."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

REVIEW = Path(__file__).resolve().parents[3]
CODE = REVIEW / 'from_claude/code'
sys.path[:0] = [str(REVIEW / 'from_gpt'), str(CODE / 'scripts'), str(CODE)]
from test_qa_das_pipeline_20261105 import PipelineModel, ToyTokenizer
from core.model_io import ModelBundle, forward_logits
from core.qa_source_interchange import (SourceInterchangeDecoder, capture_source, interchange,
                                        greedy_source_generate)


class SourceInterchangeTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ToyTokenizer()
        self.model = PipelineModel(self.tokenizer)
        self.bundle = ModelBundle(self.model, self.tokenizer, {'name': 'toy-source', 'layers_attr': 'layers'})
        self.base = [31, 44, 57]
        self.source = [38, 39, 40, 41, 42, 43, 57]
        self.w = torch.nn.functional.normalize(torch.arange(1., 9.), dim=0)

    def test_rank_one_coordinate_and_orthogonal_complement_and_no_extrapolation(self):
        b = torch.tensor([[2., -3., 4.], [5., -2., -1.]])
        s = torch.tensor([[10., 0., -8.], [7., 2., 3.]])
        w = torch.nn.functional.normalize(torch.tensor([1., 2., 3.]), dim=0)
        out, audit = interchange(b, s, w)
        torch.testing.assert_close(out @ w, s @ w)
        delta = out - b
        torch.testing.assert_close(delta, (delta @ w).unsqueeze(-1) * w)
        self.assertTrue(all(a['ideal_delta_norm'] <= a['source_base_distance'] + 1e-5 for a in audit))
        same, _ = interchange(b, b, w)
        self.assertTrue(torch.equal(same, b))
        half, audit = interchange(b.bfloat16(), s.bfloat16(), w)
        self.assertEqual(half.dtype, torch.bfloat16)
        self.assertTrue(all(a['stored_coordinate_error'] <= a['cast_error_norm'] + a['float32_tolerance'] for a in audit))

    def test_cached_and_full_recompute_match_every_predictor_with_unequal_prompt_lengths(self):
        for mode in ('baseline', 'source_only', 'equal_logit_mix', 'source_interchange'):
            with self.subTest(mode=mode):
                cached = SourceInterchangeDecoder(self.bundle, self.base, self.source,
                                                  mode=mode, layer=0, direction=self.w, use_cache=True)
                full = SourceInterchangeDecoder(self.bundle, self.base, self.source,
                                                mode=mode, layer=0, direction=self.w, use_cache=False)
                for token in (70, 71, 72, 73):
                    a, ta = cached.next_logits(); b, tb = full.next_logits()
                    torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)
                    self.assertEqual(ta['accepted_prefix_ids'], tb['accepted_prefix_ids'])
                    if mode == 'source_interchange':
                        self.assertTrue(ta['cache_storage_disjoint'])
                        self.assertEqual(len(ta['writes']), 1)
                        self.assertEqual(len(tb['writes']), len(full.history) + 1)
                    cached.accept(token); full.accept(token)
                self.assertTrue(all(not layer._forward_hooks for layer in self.model.layers))

    def test_same_prompt_source_swap_is_identity_for_the_full_trajectory(self):
        a = SourceInterchangeDecoder(self.bundle, self.base, self.base, mode='baseline')
        b = SourceInterchangeDecoder(self.bundle, self.base, self.base,
                                     mode='source_interchange', layer=0, direction=self.w)
        for _ in range(5):
            x, _ = a.next_logits(); y, tr = b.next_logits()
            torch.testing.assert_close(x, y, atol=0, rtol=0)
            self.assertEqual(tr['writes'][0]['stored_delta_norm'], 0)
            token = int(x.argmax()); a.accept(token); b.accept(token)

    def test_source_is_causal_and_never_receives_future_answer_tokens(self):
        def capture(ids):
            x = torch.tensor([ids]); inputs = {'input_ids': x, 'attention_mask': torch.ones_like(x)}
            with capture_source(self.bundle, 0, len(self.source) - 1) as state:
                forward_logits(self.bundle, inputs)
            return state['hidden']
        past = [70, 71]
        before = capture(self.source + past)
        later_a = capture(self.source + past + [72, 73])
        later_b = capture(self.source + past + [150, 151])
        torch.testing.assert_close(before, later_a[:len(before)], atol=0, rtol=0)
        torch.testing.assert_close(before, later_b[:len(before)], atol=0, rtol=0)
        decoder = SourceInterchangeDecoder(self.bundle, self.base, self.source,
                                          mode='source_interchange', layer=0, direction=self.w)
        for i, token in enumerate(past):
            _, trace = decoder.next_logits()
            self.assertEqual(trace['accepted_prefix_ids'], past[:i])
            self.assertFalse(trace['uses_reference_answer'])
            decoder.accept(token)

    def test_hooks_clean_up_on_forward_failure_and_weights_remain_frozen(self):
        parameters = [p.clone() for p in self.model.parameters()]
        decoder = SourceInterchangeDecoder(self.bundle, self.base, self.source,
                                          mode='source_interchange', layer=0, direction=self.w)
        _, _ = decoder.next_logits()
        with self.assertRaisesRegex(RuntimeError, 'accept'):
            decoder.next_logits()
        decoder.accept(70)
        for a, b in zip(parameters, self.model.parameters()): self.assertTrue(torch.equal(a, b))
        self.assertTrue(all(not layer._forward_hooks for layer in self.model.layers))
        bad = SourceInterchangeDecoder(self.bundle, self.base, [99999],
                                       mode='source_interchange', layer=0, direction=self.w)
        with self.assertRaises(IndexError): bad.next_logits()
        self.assertTrue(all(not layer._forward_hooks for layer in self.model.layers))

    def test_raw_token_ids_are_reused_and_stop_token_ends_without_forcing(self):
        with torch.no_grad():
            self.model.head.bias.fill_(-100); self.model.head.bias[2] = 100
        result = greedy_source_generate(self.bundle, self.base, self.source, mode='source_interchange',
                                         layer=0, direction=self.w, max_new_tokens=4)
        self.assertEqual(result['token_ids'], [2]); self.assertTrue(result['finished'])
        self.assertEqual(len(result['trace']), 1)

    def test_unknown_cache_representation_cannot_claim_storage_isolation(self):
        original = self.model.forward
        def opaque_cache(*args, **kwargs):
            result = original(*args, **kwargs)
            result.past_key_values = object()
            return result
        decoder = SourceInterchangeDecoder(self.bundle, self.base, self.source,
                                          mode='source_interchange', layer=0, direction=self.w)
        with patch.object(self.model, 'forward', side_effect=opaque_cache):
            with self.assertRaisesRegex(RuntimeError, 'cannot be verified'):
                decoder.next_logits()


if __name__ == '__main__':
    unittest.main()
