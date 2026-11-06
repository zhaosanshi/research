import importlib.util
from pathlib import Path
import sys
import unittest

import torch

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parents[2]
CODE = REVIEW / 'from_claude/code'
sys.path[:0] = [str(HERE), str(REVIEW/'from_gpt'), str(CODE/'scripts/review_checks'), str(CODE/'scripts'), str(CODE)]
import e27_current_dose as dose
from test_trace_corruption import ByteTokenizer
import test_qa_generation_adapter_20261106 as adapter_tests
from core.model_io import forward_logits


class ByteContinuationTests(unittest.TestCase):
    def test_pending_emoji_accepts_completing_token_not_ascii_or_new_lead(self):
        table = dose.token_byte_table(ByteTokenizer(), 8)
        pending, valid = dose.valid_continuations(b'A\xf0\x9f', table)
        self.assertEqual(pending, b'\xf0\x9f')
        self.assertIn(3, valid)
        self.assertNotIn(1, valid); self.assertNotIn(2, valid); self.assertNotIn(4, valid)
        self.assertNotIn(0, valid)
        # A single additional continuation can remain valid but incomplete.
        self.assertIn(7, valid)

    def test_complete_prefix_has_no_pending_character_and_bad_prefix_rejected(self):
        table = dose.token_byte_table(ByteTokenizer(), 8)
        self.assertEqual(dose.valid_continuations(b'A.', table), (b'', []))
        with self.assertRaises(UnicodeDecodeError): dose.valid_continuations(b'A\x80', table)


class DoseTests(unittest.TestCase):
    def setUp(self):
        self.fixture = adapter_tests.AdapterTrainingTests()
        self.fixture.setUp()
        self.bundle = self.fixture.bundle
        self.adapter = self.fixture.train(alpha=3.)
        self.bundle.tokenizer.convert_ids_to_tokens = lambda x: str(x)
        self.inputs = self.bundle.encode(['Short', 'Longer prompt'])
        self.generated = torch.tensor([[80, 81], [82, 83]])

    def test_zero_and_full_doses_match_production_controls_and_history_stays_bitwise_fixed(self):
        for step in [0, 1, 2]:
            packed = dose.trace.replay_packed(self.inputs, self.generated[:, :step])
            normal = dose.replay_reference(self.bundle, self.adapter, packed)
            history = dose.replay_reference(self.bundle, self.adapter, packed, history_only=True)
            hashes, befores = set(), set()
            for alpha in [0., .25, .5, 1., 2., 3.]:
                records = []
                with dose.dose_current(self.bundle, self.adapter, packed, alpha, records):
                    logits = forward_logits(self.bundle, packed['inputs'])[:, -1]
                self.assertEqual(len(records), 1)
                hashes.add(records[0]['history_output_sha256']);befores.add(records[0]['current_before_sha256'])
                if alpha == 0:
                    torch.testing.assert_close(logits, history, atol=1e-6, rtol=1e-6)
                    self.assertTrue(all(x['kept_current_delta_norm'] == 0 for x in records[0]['rows']))
                if alpha == 3:
                    self.assertTrue(torch.equal(logits, normal))
            self.assertEqual(len(hashes), 1); self.assertEqual(len(befores), 1)
            self.assertTrue(all(not layer._forward_hooks for layer in self.bundle.model.layers))

    def test_exception_removes_hook_and_bad_dose_rejected(self):
        packed = dose.trace.replay_packed(self.inputs, self.generated[:, :1])
        for alpha in [-1., float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                with dose.dose_current(self.bundle, self.adapter, packed, alpha, []):pass
        with self.assertRaises(RuntimeError):
            with dose.dose_current(self.bundle, self.adapter, packed, 1., []):
                forward_logits(self.bundle, packed['inputs'])
                raise RuntimeError('intentional')
        self.assertTrue(all(not layer._forward_hooks for layer in self.bundle.model.layers))

    def test_probabilities_separate_stop_tokens_valid_bytes_and_reference_ascii(self):
        tok = ByteTokenizer()
        info = dose.scored(torch.tensor([0., 1., 2., 3., 4., 5., 6., 7.]), tok, [0], 4, [3, 7])
        self.assertEqual(info['argmax_id'], 7)
        self.assertTrue(info['argmax_is_legal_continuation'])
        self.assertGreater(info['valid_continuation_probability_mass'], info['best_legal_token_probability'])
        self.assertGreater(info['best_legal_token_probability'], info['actual_reference_token_probability'])


if __name__ == '__main__':
    unittest.main()
