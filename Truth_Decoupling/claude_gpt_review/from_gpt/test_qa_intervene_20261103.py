#!/usr/bin/env python3
"""CPU-only regression checks for explicit QA predictor masks and two centres.

Uses synthetic tensors and a randomly initialized tiny Llama; no downloaded
weights, remote APIs, or real experiments. Production code is imported directly.
"""
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch


HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "from_claude/code"
sys.path.insert(0, str(CODE))
from core.intervene import LayerSteerHook, SteerSpec, steer


class TensorInterventionTests(unittest.TestCase):
    def setUp(self):
        self.w = torch.tensor([1., 0., 0.])

    def test_mask_handles_both_padding_sides_and_answer_predictors(self):
        cases = (
            # Prompt lengths are 3 and 2; answer lengths are 2 and 4.
            ([[0, 0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1]],
             [[4, 5], [2, 3, 4, 5]]),
            ([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0]],
             [[2, 3], [1, 2, 3, 4]]),
        )
        for attention, indices in cases:
            with self.subTest(attention=attention):
                attention = torch.tensor(attention, dtype=torch.bool)
                mask = torch.zeros_like(attention)
                for row, columns in enumerate(indices):
                    mask[row, columns] = True
                self.assertFalse(bool((mask & ~attention).any()))
                h = torch.arange(42, dtype=torch.float32).reshape(2, 7, 3)
                original = h.clone()
                spec = SteerSpec(self.w, 0, alpha=torch.tensor([2., 5.]),
                                 position="mask", position_mask=mask)
                result = LayerSteerHook([spec])(None, None, h)
                expected = original.clone()
                expected[0, indices[0], 0] += 2
                expected[1, indices[1], 0] += 5
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                self.assertTrue(torch.equal(h, original))
                self.assertTrue(torch.equal(result[~attention], original[~attention]))
                # First answer prediction is the last prompt position, not
                # the input position holding that first answer token.
                for row, prompt_length in enumerate((3, 2)):
                    positions = attention[row].nonzero().flatten()
                    self.assertTrue(mask[row, positions[prompt_length - 1]])
                    self.assertFalse(mask[row, positions[-1]])
                self.assertEqual(spec.n_applied, 1)

    def test_all_preserves_batch_source_and_alpha(self):
        h = torch.tensor([[[1., 2., 3.], [4., 5., 6.]],
                          [[7., 8., 9.], [10., 11., 12.]]])
        source = torch.tensor([[11., 0., 0.], [22., 0., 0.]])
        alpha = torch.tensor([1., .5])
        spec = SteerSpec(self.w, 0, mode="interchange", source_act=source,
                         alpha=alpha, position="all")
        result = LayerSteerHook([spec])(None, None, h)
        expected = h.clone()
        expected[0, :, 0] = 11
        expected[1, :, 0] = (h[1, :, 0] + 22) / 2
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        self.assertIs(spec.alpha, alpha)
        self.assertTrue(torch.equal(spec.source_act, source))

    def test_mask_clamp_keeps_rank_two_targets_with_original_rows(self):
        h = torch.zeros(2, 4, 3)
        mask = torch.tensor([[True, False, True, False], [False, True, True, True]])
        target = torch.tensor([[2., 3.], [5., 7.]])
        spec = SteerSpec(torch.eye(3)[:, :2], 0, mode="clamp", z_target=target,
                         alpha=torch.tensor([1., .5]), position="mask", position_mask=mask)
        result = LayerSteerHook([spec])(None, None, h)
        expected = h.clone()
        expected[0, [0, 2], :2] = torch.tensor([2., 3.])
        expected[1, [1, 2, 3], :2] = torch.tensor([2.5, 3.5])
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_single_active_row_does_not_take_first_source_or_alpha(self):
        mask = torch.tensor([[False, False, False], [True, False, True]])
        spec = SteerSpec(self.w, 0, mode="interchange", source_act=[[7., 0., 0.], [13., 0., 0.]],
                         alpha=torch.tensor([3., 2.]), position="mask", position_mask=mask)
        result = LayerSteerHook([spec])(None, None, torch.zeros(2, 3, 3))
        expected = torch.zeros(2, 3, 3)
        expected[1, [0, 2], 0] = 26
        self.assertTrue(torch.equal(result, expected))

    def test_zero_direction_is_exact_identity_for_every_position_mode(self):
        h = torch.arange(18, dtype=torch.bfloat16).reshape(2, 3, 3)
        for position in ("last", "all", "mask"):
            options = {"position_mask": torch.tensor([[False, True, False], [True, False, True]])} if position == "mask" else {}
            spec = SteerSpec(torch.zeros(3), 0, mode="amplify", center=[8., 9., 10.],
                             alpha=torch.tensor([2., 3.]), position=position, **options)
            self.assertTrue(torch.equal(LayerSteerHook([spec])(None, None, h), h))

    def test_empty_mask_and_max_calls_count_only_actual_nonempty_edits(self):
        h = torch.zeros(2, 3, 3)
        empty = SteerSpec(self.w, 0, position="mask", position_mask=torch.zeros(2, 3, dtype=torch.bool))
        active = SteerSpec(self.w, 0, max_calls=1)
        hook = LayerSteerHook([empty, active])
        result = hook(None, None, (h, "metadata"))
        self.assertEqual(result[1], "metadata")
        self.assertEqual((empty.n_applied, active.n_applied, hook.applied_count), (0, 1, 1))
        result = hook(None, None, h)
        self.assertIs(result, h)
        self.assertEqual((empty.n_applied, active.n_applied, hook.applied_count), (0, 1, 1))
        self.assertEqual(hook.call_count, 2)

    def test_decode_center_switches_and_restores(self):
        spec = SteerSpec(self.w, 0, mode="amplify", alpha=1.,
                         center=[10., 0., 0.], decode_center=[100., 0., 0.])
        original_center = spec.center
        original_decode = spec.decode_center
        hook = LayerSteerHook([spec])
        prefill = torch.tensor([[[3., 0., 0.], [12., 0., 0.]]])
        first = hook(None, None, prefill)
        self.assertEqual(first[0, -1, 0].item(), 14.)
        self.assertEqual(first[0, 0, 0].item(), 3.)
        self.assertIs(spec.center, original_center)
        second = hook(None, None, torch.tensor([[[12., 0., 0.]]]))
        self.assertEqual(second[0, -1, 0].item(), -76.)
        self.assertIs(spec.center, original_center)
        self.assertIs(spec.decode_center, original_decode)
        self.assertEqual((spec.n_applied, hook.applied_count), (2, 2))

    def test_center_restored_even_when_delta_raises(self):
        spec = SteerSpec(self.w, 0, mode="amplify", center=[10., 0., 0.], decode_center=[100., 0., 0.])
        hook = LayerSteerHook([spec])
        h = torch.ones(1, 1, 3)
        hook(None, None, h)
        original = spec.center
        spec.alpha = torch.tensor([1., 2.])  # invalid for this one-row batch
        with self.assertRaises(ValueError):
            hook(None, None, h)
        self.assertIs(spec.center, original)
        self.assertEqual((spec.n_applied, hook.applied_count), (1, 1))

    def test_batch_context_resets_calls_and_prompt_center(self):
        layer = torch.nn.Identity()
        bundle = types.SimpleNamespace(device=torch.device("cpu"), layer=lambda i: layer)
        spec = SteerSpec(self.w, 0, mode="amplify", center=[10., 0., 0.],
                         decode_center=[100., 0., 0.], max_calls=1)
        h = torch.tensor([[[12., 0., 0.]]])
        for batch in range(2):
            with steer(bundle, [spec]) as hooks:
                self.assertEqual(layer(h)[0, 0, 0].item(), 14.)
                self.assertTrue(torch.equal(layer(h), h))
                self.assertTrue(torch.equal(layer(h), h))
            self.assertEqual(hooks[0].applied_count, 1)
            self.assertEqual(hooks[0].call_count, 3)
            self.assertEqual(spec.n_applied, batch + 1)
            self.assertEqual(spec.center[0].item(), 10.)

    def test_invalid_position_mask_shape_and_batch_parameters_raise(self):
        with self.assertRaises(ValueError):
            SteerSpec(self.w, 0, position="typo")
        with self.assertRaises(ValueError):
            SteerSpec(self.w, 0, position="mask")
        with self.assertRaises(ValueError):
            SteerSpec(self.w, 0, position="mask", position_mask=[[1, 0]])
        with self.assertRaises(ValueError):
            SteerSpec(self.w, 0, position="all", decode_center=[0., 0., 0.])
        spec = SteerSpec(self.w, 0, position="mask", position_mask=torch.ones(1, 2, dtype=torch.bool))
        with self.assertRaises(ValueError):
            LayerSteerHook([spec])(None, None, torch.zeros(2, 3, 3))
        self.assertEqual(spec.n_applied, 0)
        spec = SteerSpec(self.w, 0)
        spec.position = "typo"
        with self.assertRaises(ValueError):
            LayerSteerHook([spec])(None, None, torch.zeros(2, 3, 3))
        spec = SteerSpec(self.w, 0, mode="interchange", source_act=torch.zeros(3, 3), position="all")
        with self.assertRaises(ValueError):
            LayerSteerHook([spec])(None, None, torch.zeros(2, 3, 3))
        spec = SteerSpec(self.w, 0, mode="clamp", z_target=[1., 2., 3.], position="all")
        with self.assertRaises(ValueError):
            LayerSteerHook([spec])(None, None, torch.zeros(2, 3, 3))


class TinyLlamaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The existing verification module disables unrelated broken optional
        # audio/image/sklearn imports locally, before importing text-only Llama.
        spec = importlib.util.spec_from_file_location("tiny_qa_reference", HERE / "verify_real_qa_path_20261102.py")
        cls.reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.reference)

    @torch.no_grad()
    def test_teacher_forced_predictor_masks_match_cached_generation(self):
        reference = self.reference
        model = reference.tiny_llama(seed=29, layers=3, max_positions=64)
        bundle = reference.ModelBundle(model, None, {"layers_attr": "model.layers"})
        prompt = torch.tensor([[1, 3, 4, 5], [0, 1, 6, 7]])
        attention = (prompt != 0).long()
        answer = torch.tensor([[8, 9], [10, 11]])
        full = torch.cat([prompt, answer], dim=1)
        full_attention = torch.cat([attention, torch.ones(2, 2, dtype=torch.long)], dim=1)
        w = torch.randn(32); w /= w.norm()
        c_prompt, c_decode = torch.randn(32) * .05, torch.randn(32) * .1
        prompt_mask = torch.zeros(2, 6, dtype=torch.bool); prompt_mask[:, 3] = True
        decode_mask = torch.zeros(2, 6, dtype=torch.bool); decode_mask[:, 4] = True
        baseline = model(input_ids=full, attention_mask=full_attention,
                         position_ids=reference.position_ids_from_mask(full_attention), use_cache=False)
        for prefill_only in (False, True):
            with self.subTest(prefill_only=prefill_only):
                tf_specs = [SteerSpec(w, 1, mode="amplify", alpha=.7, center=c_prompt,
                                      position="mask", position_mask=prompt_mask)]
                if not prefill_only:
                    tf_specs.append(SteerSpec(w, 1, mode="amplify", alpha=.7, center=c_decode,
                                              position="mask", position_mask=decode_mask))
                with steer(bundle, tf_specs):
                    teacher_forced = model(input_ids=full, attention_mask=full_attention,
                                           position_ids=reference.position_ids_from_mask(full_attention), use_cache=False)
                generated_spec = SteerSpec(w, 1, mode="amplify", alpha=.7, center=c_prompt,
                                           decode_center=c_decode, max_calls=1 if prefill_only else None)
                with steer(bundle, [generated_spec]) as hooks:
                    prefill = model(input_ids=prompt, attention_mask=attention,
                                    position_ids=reference.position_ids_from_mask(attention), use_cache=True)
                    next_attention = full_attention[:, :5]
                    decoded = model(input_ids=answer[:, :1], attention_mask=next_attention,
                                    position_ids=reference.position_ids_from_mask(next_attention)[:, -1:],
                                    past_key_values=prefill.past_key_values, use_cache=True)
                torch.testing.assert_close(teacher_forced.logits[:, 3], prefill.logits[:, -1], rtol=0, atol=1e-5)
                torch.testing.assert_close(teacher_forced.logits[:, 4], decoded.logits[:, -1], rtol=0, atol=1e-5)
                self.assertGreater(float((teacher_forced.logits[:, 3] - baseline.logits[:, 3]).abs().max()), 0)
                torch.testing.assert_close(teacher_forced.logits[:, 2], baseline.logits[:, 2], rtol=0, atol=0)
                self.assertEqual(generated_spec.n_applied, 1 if prefill_only else 2)
                self.assertEqual(hooks[1].applied_count, 1 if prefill_only else 2)
                self.assertTrue(torch.equal(generated_spec.center, c_prompt))


if __name__ == "__main__":
    unittest.main(verbosity=2)
