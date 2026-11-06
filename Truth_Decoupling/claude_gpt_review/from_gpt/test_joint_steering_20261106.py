"""E26 direction and operator tests using real forward hooks on torch blocks."""
from copy import deepcopy
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import torch
from torch import nn

CODE = Path(__file__).resolve().parent.parent / "from_claude/code"
sys.path.insert(0, str(CODE))
from core.intervene import steer
from core.joint_steering import (candidate_conditions, choose_conditions, commitment_direction,
                                 fingerprint, intervention_specs, unit)


class IdentityBlock(nn.Module):
    def forward(self, x):
        return x


class ToyModel(nn.Module):
    def __init__(self, layers=2):
        super().__init__()
        self.blocks = nn.ModuleList([IdentityBlock() for _ in range(layers)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def bundle(layers=2):
    model = ToyModel(layers)
    return types.SimpleNamespace(model=model, device=torch.device("cpu"), layer=lambda i: model.blocks[i])


def bank():
    out = {}
    for task in ["qa", "tf"]:
        for layer in [0, 1]:
            out["%s_das_E_L%d" % (task, layer)] = np.array([1., 0.])
            out["%s_das_C_L%d" % (task, layer)] = np.array([0., 1.])
            out["%s_das_C_L%d_scale" % (task, layer)] = np.array(2.)
            out["%s_prompt_center_L%d" % (task, layer)] = np.array([1., 0.])
            out["%s_gate_center_L%d" % (task, layer)] = np.array([10., 0.])
            if task == "qa":
                out["qa_decode_center_L%d" % layer] = np.array([2., 0.])
    return out


def condition(task="qa", mode="continuous", e_layer=0, c_layer=0):
    return {"name": "joint", "kind": "EC", "task": task, "method": "das",
            "e": {"layer": e_layer, "alpha": 2., "mode": mode},
            "c": {"layer": c_layer, "alpha": -.5}}


class CommitmentDirectionTests(unittest.TestCase):
    def test_continuous_positive_margins_work_without_binary_class_split(self):
        states = np.array([[0., -1.], [0., 0.], [0., 1.]])
        margins = np.array([1., 2., 9.])  # Every Yes/No sign is positive.
        w, meta = commitment_direction(states, margins, [1., 0.])
        np.testing.assert_allclose(w, [0., 1.])
        self.assertTrue(meta["available"])
        self.assertGreater(meta["score_correlation"], 0)
        self.assertAlmostEqual(meta["scale"], np.sqrt(2 / 3))
        self.assertEqual(meta["n_estimation"], 3)
        with self.assertRaises(TypeError):
            commitment_direction(states, margins, [1., 0.], correctness=[1, 0, 1])

    def test_projects_off_E_and_orients_positive_towards_yes(self):
        states = np.array([[-1., -2.], [1., 2.]])
        before = states.copy()
        w, meta = commitment_direction(states, [-1., 1.], [5., 0.])
        np.testing.assert_allclose(w, [0., 1.])
        self.assertAlmostEqual(meta["cos_after"], 0.)
        self.assertAlmostEqual(meta["scale"], 2.)
        self.assertAlmostEqual(meta["score_correlation"], 1.)
        reversed_w, rev = commitment_direction(states, [1., -1.], [5., 0.])
        np.testing.assert_allclose(reversed_w, -w)
        self.assertAlmostEqual(rev["score_correlation"], 1.)
        np.testing.assert_array_equal(states, before)

    def test_margin_scale_shift_and_hidden_scale_have_expected_units(self):
        states = np.array([[-1., -2.], [1., 2.], [.3, 1.]])
        margins = np.array([-1., 1., .6])
        w, meta = commitment_direction(states, margins, [1., 0.])
        w2, meta2 = commitment_direction(states + [100., 300.], 20 * margins + 1000, [1., 0.])
        w3, meta3 = commitment_direction(3 * states, margins, [1., 0.])
        np.testing.assert_allclose(w, w2)
        np.testing.assert_allclose(w, w3)
        self.assertAlmostEqual(meta["scale"], meta2["scale"])
        self.assertAlmostEqual(3 * meta["scale"], meta3["scale"])

    def test_absent_signal_is_explicit_and_malformed_inputs_fail(self):
        w, meta = commitment_direction([[1., 2.], [3., 4.]], [2., 2.], [1., 0.])
        self.assertIsNone(w)
        self.assertEqual(meta["reason"], "constant_yes_no_margin")
        w, meta = commitment_direction([[-1., 0.], [1., 0.]], [-1., 1.], [1., 0.])
        self.assertIsNone(w)
        self.assertEqual(meta["reason"], "no_component_orthogonal_to_E")
        for h, m, e in [([[1., 2.]], [1.], [1., 0.]), ([[1., 2.], [3., 4.]], [1.], [1., 0.]),
                        ([[1., 2.], [3., 4.]], [1., 2.], [1., 0., 0.]),
                        ([[np.nan, 2.], [3., 4.]], [1., 2.], [1., 0.]),
                        ([[1., 2.], [3., 4.]], [1., np.inf], [1., 0.])]:
            with self.subTest(h=h, m=m, e=e), self.assertRaises(ValueError):
                commitment_direction(h, m, e)
        for v in [[0., 0.], [float("inf"), 1.], [[1., 2.]]]:
            with self.assertRaises(ValueError):
                unit(v)


class ExecutedInterventionTests(unittest.TestCase):
    def test_qa_continuous_E_and_prefill_C_write_actual_different_call_counts(self):
        b, arrays = bundle(), bank()
        specs = intervention_specs(arrays, condition(), context="answer")
        prompt = torch.tensor([[[1., 5.], [2., 5.], [3., 4.]]])
        original = prompt.clone()
        with steer(b, specs) as hooks:
            result = b.model(prompt)
            decode_one = b.model(torch.tensor([[[5., 4.]]]))
            decode_two = b.model(torch.tensor([[[4., 8.]]]))
        torch.testing.assert_close(result, torch.tensor([[[1., 5.], [2., 5.], [7., 3.]]]))
        torch.testing.assert_close(decode_one, torch.tensor([[[11., 4.]]]))
        torch.testing.assert_close(decode_two, torch.tensor([[[8., 8.]]]))
        torch.testing.assert_close(prompt, original)
        self.assertEqual([s.n_applied for s in specs], [3, 1])
        self.assertEqual(hooks[0].first_shape, (1, 3, 2))
        self.assertEqual(hooks[0].call_count, 3)
        torch.testing.assert_close(specs[0].center, torch.tensor([1., 0.]))
        self.assertFalse(b.model.blocks[0]._forward_hooks)
        torch.testing.assert_close(b.model(prompt), prompt)

    def test_gate_uses_its_own_center_and_never_decode_center(self):
        b = bundle()
        specs = intervention_specs(bank(), condition(), context="gate")
        self.assertIsNone(specs[0].decode_center)
        with steer(b, specs):
            first = b.model(torch.tensor([[[3., 4.]]]))
            later = b.model(torch.tensor([[[5., 4.]]]))
        torch.testing.assert_close(first, torch.tensor([[[-11., 3.]]]))
        torch.testing.assert_close(later, torch.tensor([[[5., 4.]]]))
        self.assertEqual([s.n_applied for s in specs], [1, 1])

    def test_tf_answer_and_gate_centers_are_distinct_and_single_call(self):
        for context, expected in [("answer", [7., 3.]), ("gate", [-11., 3.])]:
            b = bundle()
            specs = intervention_specs(bank(), condition(task="tf", mode="continuous"), context=context)
            with steer(b, specs):
                first = b.model(torch.tensor([[[3., 4.]]]))
                later = b.model(torch.tensor([[[5., 4.]]]))
            torch.testing.assert_close(first, torch.tensor([[expected]]))
            torch.testing.assert_close(later, torch.tensor([[[5., 4.]]]))
            self.assertEqual([s.n_applied for s in specs], [1, 1])

    def test_same_layer_applies_E_then_C_not_reverse(self):
        # Deliberately nonorthogonal synthetic vectors make ordering observable.
        # Production C estimation is separately tested to remove this overlap.
        arrays = bank()
        arrays["tf_das_C_L0"] = np.array([1., 0.])
        arrays["tf_prompt_center_L0"] = np.array([0., 0.])
        c = condition(task="tf"); c["e"]["alpha"] = 1.; c["c"]["alpha"] = 1.
        specs = intervention_specs(arrays, c)
        self.assertEqual([s.mode for s in specs], ["amplify", "add"])
        b = bundle()
        with steer(b, specs):
            actual = b.model(torch.tensor([[[3., 4.]]]))
        # E: 3->6, C: 6->8. Reversed would give (3+2)*2=10.
        torch.testing.assert_close(actual, torch.tensor([[[8., 4.]]]))

    def test_different_layers_follow_network_depth_order(self):
        arrays = bank()
        arrays["tf_das_C_L0"] = np.array([1., 0.])
        arrays["tf_prompt_center_L1"] = np.array([0., 0.])
        c = condition(task="tf", e_layer=1, c_layer=0)
        c["e"]["alpha"] = c["c"]["alpha"] = 1.
        specs = intervention_specs(arrays, c)
        b = bundle()
        with steer(b, specs):
            actual = b.model(torch.tensor([[[3., 4.]]]))
        torch.testing.assert_close(actual, torch.tensor([[[10., 4.]]]))

    def test_prefill_multilayer_E_does_not_require_decode_centers(self):
        arrays = bank()
        del arrays["qa_decode_center_L0"], arrays["qa_decode_center_L1"]
        c = condition(mode="prefill"); c["e"] = {"layers": [0, 1], "alpha": 1., "mode": "prefill"}; c["c"] = None
        specs = intervention_specs(arrays, c)
        b = bundle()
        with steer(b, specs):
            first = b.model(torch.tensor([[[3., 4.]]]))
            later = b.model(torch.tensor([[[3., 4.]]]))
        torch.testing.assert_close(first, torch.tensor([[[9., 4.]]]))
        torch.testing.assert_close(later, torch.tensor([[[3., 4.]]]))
        self.assertEqual([s.n_applied for s in specs], [1, 1])

    def test_baseline_zero_E_and_hook_cleanup_on_error(self):
        b = bundle()
        base = {"name": "baseline", "kind": "base", "task": "qa", "method": "base", "e": None, "c": None}
        self.assertEqual(intervention_specs({}, base), [])
        c = condition(); c["e"]["alpha"] = 0.; c["c"] = None
        specs = intervention_specs(bank(), c)
        x = torch.tensor([[[3., 4.]]])
        with steer(b, specs):
            torch.testing.assert_close(b.model(x), x)
        try:
            with steer(b, intervention_specs(bank(), condition())):
                raise RuntimeError("synthetic abort")
        except RuntimeError:
            pass
        self.assertFalse(b.model.blocks[0]._forward_hooks)


class ConditionSelectionTests(unittest.TestCase):
    @staticmethod
    def rows(n_correct=0, abstain=False):
        return [{"id": str(i), "pair_id": str(i // 2), "score": float(6 - i),
                 "outcome": "abstain" if abstain else "correct" if i < n_correct else "wrong"}
                for i in range(6)]

    def test_joint_family_is_selected_independently_of_best_individuals(self):
        choices = {"das": [{"layer": 1, "alpha": 1.}, {"layer": 2, "alpha": 2.}],
                   "md": [{"layer": 0, "alpha": 1.}]}
        conditions = candidate_conditions("qa", choices, {"das": [1], "md": [0]}, [-1., 1.])
        rows = {c["name"]: self.rows(0, abstain=c["method"] == "md") for c in conditions}
        rows["baseline"] = self.rows(6)  # Baseline cannot replace a method family.
        rows["das_E_1"] = self.rows(2)
        rows["das_C_L1_a-1"] = self.rows(3)
        rows["das_C_L1_a1"] = self.rows(1)
        rows["das_EC_0_L1_a1"] = self.rows(3)
        rows["das_EC_1_L1_a-1"] = self.rows(1)
        selected = choose_conditions(rows, conditions, [.5])['0.5']
        self.assertEqual(selected["das_E"], "das_E_1")
        self.assertEqual(selected["das_C"], "das_C_L1_a-1")
        self.assertEqual(selected["das_EC"], "das_EC_0_L1_a1")
        self.assertIsNone(selected["md_E"])
        self.assertIsNone(selected["md_C"])
        self.assertIsNone(selected["md_EC"])

    def test_missing_C_candidates_remain_explicit_none_families(self):
        conditions = candidate_conditions("tf", {"das": [{"layer": 1, "alpha": 1.}]}, {"das": []}, [-1., 1.])
        rows = {c["name"]: self.rows(2) for c in conditions}
        selected = choose_conditions(rows, conditions, [.5])['0.5']
        self.assertEqual(selected["das_E"], "das_E_0")
        self.assertIn("das_C", selected)
        self.assertIn("das_EC", selected)
        self.assertIsNone(selected["das_C"])
        self.assertIsNone(selected["das_EC"])

    def test_equal_correct_counts_choose_weaker_then_fewer_shallower_layers(self):
        choices = {"das": [{"layers": [0, 1], "alpha": 1.}, {"layer": 4, "alpha": 1.},
                           {"layer": 2, "alpha": 1.}, {"layer": 5, "alpha": .5}]}
        conditions = candidate_conditions("qa", choices, {"das": []}, [-1., 1.])
        rows = {c["name"]: self.rows(2) for c in conditions}
        self.assertEqual(choose_conditions(rows, conditions, [.5])["0.5"]["das_E"], "das_E_3")
        rows["das_E_3"] = self.rows(0)
        self.assertEqual(choose_conditions(rows, conditions, [.5])["0.5"]["das_E"], "das_E_2")

    def test_condition_coverage_is_per_target_and_source_inputs_unchanged(self):
        conditions = candidate_conditions("qa", {"das": [{"layer": 1, "alpha": 1.}, {"layer": 2, "alpha": 2.}]}, {}, [])
        rows = {c["name"]: self.rows(0) for c in conditions}
        rows["das_E_1"] = self.rows(2)
        rows["das_E_0"] = [{**r, "outcome": "correct" if int(r["id"]) >= 3 else "wrong"} for r in self.rows()]
        snap = deepcopy((rows, conditions))
        selected = choose_conditions(rows, conditions, [.5, 1.])
        self.assertEqual(selected["0.5"]["das_E"], "das_E_1")
        self.assertEqual(selected["1.0"]["das_E"], "das_E_0")
        self.assertEqual((rows, conditions), snap)

    def test_bad_task_duplicates_names_missing_outputs_and_unresolved_fail(self):
        with self.assertRaises(ValueError):
            candidate_conditions("other", {}, {}, [])
        with self.assertRaises(ValueError):
            candidate_conditions("tf", {"das": []}, {"das": [1, 1]}, [1])
        for alpha in [0., np.nan, np.inf]:
            with self.assertRaises(ValueError):
                candidate_conditions("tf", {"das": []}, {"das": [1]}, [alpha])
        conditions = candidate_conditions("qa", {"das": [{"layer": 1, "alpha": 1.}]}, {}, [])
        rows = {c["name"]: self.rows(1) for c in conditions}
        with self.assertRaises(ValueError):
            choose_conditions({"baseline": rows["baseline"]}, conditions, [.5])
        rows["das_E_0"][0]["outcome"] = "unresolved"
        with self.assertRaises(ValueError):
            choose_conditions(rows, conditions, [.5])

    def test_fingerprint_is_order_independent_and_rejects_nonfinite_values(self):
        self.assertEqual(fingerprint({"b": 2, "a": 1}), fingerprint({"a": 1, "b": 2}))
        self.assertNotEqual(fingerprint({"a": 1}), fingerprint({"a": 2}))
        with self.assertRaises(ValueError):
            fingerprint({"a": float("nan")})


if __name__ == '__main__':
    unittest.main()
