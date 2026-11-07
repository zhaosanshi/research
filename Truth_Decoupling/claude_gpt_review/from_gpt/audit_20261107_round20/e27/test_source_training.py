"""Train through real causal forwards; verify masks, gradients and no future leak."""
import copy
from pathlib import Path
import sys
import tempfile
import shutil
import unittest

import torch

REVIEW = Path(__file__).resolve().parents[3]
CODE = REVIEW / 'from_claude/code'
sys.path[:0] = [str(REVIEW / 'from_gpt'), str(CODE / 'scripts'), str(CODE)]
from test_qa_das_pipeline_20261105 import PipelineModel, ToyTokenizer
from core.model_io import ModelBundle, forward_logits
from core.qa_generation_adapter import frozen_model, load_adapter as e26_centered_adapter_loader
from core.qa_precision import fp32_output_projection
from core.qa_source_interchange import SourceInterchangeDecoder
from core import qa_source_interchange_training as training


class SourceTrainingTests(unittest.TestCase):
    def setUp(self):
        tok = ToyTokenizer(); self.model = PipelineModel(tok)
        self.bundle = ModelBundle(self.model, tok, {'name': 'toy-train-source', 'layers_attr': 'layers'})
        self.cfg = {'qa_system': 'Answer.', 'qa_user': 'Question: {question}\nAnswer:', 'max_new_tokens': 3, 'batch_size': 1}
        self.records = [{'pair_id': 'p1', 'question': 'Where?', 'positive_answer': 'Paris'},
                        {'pair_id': 'p2', 'question': 'Which place?', 'positive_answer': 'A longer answer'}]
        self.source = 'Correct false premises. Be truthful.'
        self.w = torch.nn.functional.normalize(torch.arange(1., 9.), dim=0)

    def test_real_training_nonzero_gradient_only_w_changes_and_round_trip(self):
        old = [p.clone() for p in self.model.parameters()]
        result = training.train(self.bundle, self.records, self.cfg, self.w, layer=0,
                                source_system=self.source, epochs=1, log_fn=None)
        self.assertEqual(result['gradient_audit']['updates'], 2)
        self.assertEqual(result['gradient_audit']['nonzero_updates'], 2)
        self.assertGreater(float((result['W'] - self.w).norm()), 1e-6)
        self.assertAlmostEqual(float(result['W'].norm()), 1., places=6)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(old, self.model.parameters())))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.model.parameters()))
        self.assertTrue(all(not b._forward_hooks for b in self.model.layers))
        with tempfile.TemporaryDirectory() as directory:
            training.save(result, directory); loaded = training.load(directory)
            self.assertTrue(torch.equal(loaded['W'], result['W']))
            with self.assertRaises(FileExistsError): training.save(result, directory)

    def test_source_prompt_helper_is_in_closed_snapshot_and_missing_dependency_is_rejected(self):
        result = training.train(self.bundle, self.records, self.cfg, self.w, layer=0,
                                source_system=self.source, epochs=1, log_fn=None)
        self.assertIn('scripts/e29_qa_source_interchange.py', result['training_implementation']['files'])
        with tempfile.TemporaryDirectory() as directory:
            for relative in result['training_implementation']['files']:
                p = Path(directory) / 'training_code_snapshot' / relative
                p.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(CODE / relative, p)
            training.verify_training_snapshot(directory, result)
            (Path(directory) / 'training_code_snapshot/scripts/e29_qa_source_interchange.py').unlink()
            with self.assertRaisesRegex(ValueError, 'missing or changed'):
                training.verify_training_snapshot(directory, result)

    def test_source_checkpoint_is_rejected_by_the_loader_used_by_e26_centered_amplification(self):
        result = training.train(self.bundle, self.records, self.cfg, self.w, layer=0,
                                source_system=self.source, epochs=1, log_fn=None)
        with tempfile.TemporaryDirectory() as directory:
            training.save(result, directory)
            # This is the actual load_adapter imported by E26 prepare(), using
            # a valid trained checkpoint and manifest, not a mocked rejection.
            with self.assertRaisesRegex(ValueError, 'unsupported generation adapter'):
                e26_centered_adapter_loader(Path(directory) / 'source_adapter.pt')

    def test_per_question_mean_not_long_answer_weighting_and_left_pad_alignment(self):
        pairs = training.prepare_pairs(self.bundle, self.cfg, self.records, self.source)
        w = torch.nn.Parameter(self.w.clone()); trace = {}
        with frozen_model(self.model), fp32_output_projection(self.bundle):
            together, _, packed, _ = training.training_forward(self.bundle, pairs, 0, w, trace)
            separate = [training.training_forward(self.bundle, [p], 0, w)[0] for p in pairs]
            torch.testing.assert_close(together, torch.stack(separate).mean(), atol=1e-6, rtol=1e-6)
        self.assertEqual(int(packed['predictor_mask'].sum()), sum(p['base']['answer_length'] for p in pairs))
        self.assertEqual(len(trace['base_predictor_positions']), len(trace['source_predictor_positions']))
        self.assertFalse(trace['source_requires_grad']); self.assertTrue(trace['raw_direction_requires_grad'])
        for pair in pairs: self.assertEqual(pair['base']['answer_ids'], pair['source']['answer_ids'])

    def test_same_source_and_base_is_identity_with_zero_w_gradient(self):
        pairs = training.prepare_pairs(self.bundle, self.cfg, self.records[:1], self.cfg['qa_system'])
        w = torch.nn.Parameter(self.w.clone())
        with frozen_model(self.model), fp32_output_projection(self.bundle):
            loss, logits, packed, _ = training.training_forward(self.bundle, pairs, 0, w)
            baseline = forward_logits(self.bundle, packed['inputs'])
            torch.testing.assert_close(logits, baseline, atol=0, rtol=0)
            loss.backward()
        self.assertIsNotNone(w.grad); self.assertEqual(float(w.grad.norm()), 0.)

    def test_complete_training_forward_equals_cached_deployment_every_token(self):
        pair = training.prepare_pairs(self.bundle, self.cfg, self.records[:1], self.source)[0]
        w = torch.nn.Parameter(self.w.clone())
        with frozen_model(self.model), fp32_output_projection(self.bundle):
            _, logits, packed, _ = training.training_forward(self.bundle, [pair], 0, w)
            decoder = SourceInterchangeDecoder(self.bundle, pair['base']['prefix_ids'], pair['source']['prefix_ids'],
                                                mode='source_interchange', layer=0, direction=self.w)
            expected = logits[packed['predictor_mask']]
            for i, token in enumerate(pair['base']['answer_ids']):
                actual, _ = decoder.next_logits()
                torch.testing.assert_close(actual, expected[i], atol=2e-6, rtol=1e-5)
                decoder.accept(token)

    def test_changing_current_target_and_future_preserves_current_source_and_logits(self):
        pair = training.prepare_pairs(self.bundle, self.cfg, self.records[:1], self.source)[0]
        altered = copy.deepcopy(pair); t = 1
        for key in ('base', 'source'):
            altered[key]['answer_ids'][t] += 1
            altered[key]['full_ids'] = altered[key]['prefix_ids'] + altered[key]['answer_ids']
        w = torch.nn.Parameter(self.w.clone())
        with frozen_model(self.model), fp32_output_projection(self.bundle):
            _, a, packed, source_a = training.training_forward(self.bundle, [pair], 0, w)
            _, b, _, source_b = training.training_forward(self.bundle, [altered], 0, w)
        torch.testing.assert_close(source_a[:t + 1], source_b[:t + 1], atol=0, rtol=0)
        ap, bp = a[packed['predictor_mask']], b[packed['predictor_mask']]
        torch.testing.assert_close(ap[:t + 1], bp[:t + 1], atol=0, rtol=0)
        self.assertFalse(torch.equal(source_a[t + 1:], source_b[t + 1:]))
        old_label, new_label = pair['base']['answer_ids'][t], altered['base']['answer_ids'][t]
        self.assertNotEqual(float(ap[t].detach().log_softmax(-1)[old_label]), float(ap[t].detach().log_softmax(-1)[new_label]))

    def test_bfloat16_training_keeps_direction_gradient_and_random_means_trained(self):
        self.model.bfloat16()
        result = training.train(self.bundle, self.records, self.cfg, self.w, layer=0,
                                source_system=self.source, epochs=1, init='random', log_fn=None)
        self.assertEqual(result['initialization'], 'random')
        self.assertFalse(result['is_untrained_random_control'])
        self.assertEqual(result['gradient_audit']['nonzero_updates'], 2)
        self.assertGreater(float((result['W'] - result['initial_W']).norm()), 1e-6)
        self.assertTrue(all(t['residual_dtype'] == 'torch.bfloat16' for t in result['position_trace']))


if __name__ == '__main__': unittest.main()
