"""Fair actual-coverage tests: no model inference, API calls or test-set fitting."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

CODE = Path(__file__).resolve().parent.parent / "from_claude/code"
sys.path.insert(0, str(CODE))
from core.selective_eval import (apply_threshold, calibrate_threshold, evaluate_coverage,
                                exact_paired_correctness, quota_count, quota_select,
                                random_score, validate_arms)


def row(i, outcome="correct", score=1., group=None, **extra):
    return {"id": str(i), "pair_id": str(i) if group is None else str(group),
            "outcome": outcome, "score": score, **extra}


class FixedDraws:
    def __init__(self, draws):
        self.draws = iter(draws)

    def integers(self, low, high, size):
        result = np.array(next(self.draws))
        assert len(result) == size and result.min() >= low and result.max() < high
        return result


class QuotaTests(unittest.TestCase):
    def test_exact_decimal_floor_and_invalid_arguments(self):
        for q, n, wanted in [(.58, 100, 58), (.29, 100, 29), (.3, 10, 3), (Fraction(1, 3), 3, 1),
                             (0., 5, 0), (1., 5, 5), (.5, 1, 0), (.5, 0, 0)]:
            self.assertEqual(quota_count(q, n), wanted)
        for bad in [True, -0.1, 1.1, float("nan"), float("inf"), None, "bad"]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                quota_count(bad, 10)
        for bad in [True, -1, 2.5]:
            with self.subTest(n=bad), self.assertRaises(ValueError):
                quota_count(.5, bad)

    def test_delivers_exact_k_actual_answers_on_fixed_denominator(self):
        data = [row("a", "abstain", 100), row("b", "correct", 9), row("c", "wrong", 8),
                row("d", "correct", 7), row("e", "wrong", 6)]
        got = quota_select(data, .4)
        self.assertTrue(got["feasible"])
        self.assertEqual((got["n"], got["k"], got["n_available"], got["answered"]), (5, 2, 4, 2))
        self.assertEqual(got["selected_ids"], ["b", "c"])
        self.assertEqual(got["mask"], [False, True, True, False, False])
        self.assertEqual(got["delivered_outcomes"], {"a": "abstain", "b": "correct", "c": "wrong",
                                                   "d": "abstain", "e": "abstain"})
        self.assertEqual(got["counts"], {"correct": 1, "wrong": 1, "abstain": 3})
        self.assertEqual(got["accuracy_answered"], .5)
        self.assertEqual(got["risk_answered"], .5)
        self.assertEqual(got["actual_coverage"], .4)
        self.assertEqual(got["correct_rate"], .2)

    def test_native_refusals_cannot_be_filled_or_removed(self):
        got = quota_select([row("a", "correct"), row("b", "abstain"), row("c", "abstain")], 1)
        self.assertFalse(got["feasible"])
        self.assertEqual((got["n"], got["k"], got["n_available"]), (3, 3, 1))
        self.assertEqual(got["reason"], "insufficient_actual_answers")
        for name in ["mask", "selected_ids", "delivered_outcomes", "answered", "correct", "accuracy_answered"]:
            self.assertIsNone(got[name])

    def test_ties_are_order_invariant_and_do_not_inspect_correctness(self):
        data = [row(i, "correct" if i % 2 else "wrong", 1.) for i in range(12)]
        first = quota_select(data, .5, seed=3)
        shuffled = list(reversed(data))
        relabeled = [{**r, "outcome": "wrong" if r["outcome"] == "correct" else "correct"} for r in shuffled]
        second, third = quota_select(shuffled, .5, seed=3), quota_select(relabeled, .5, seed=3)
        self.assertEqual(first["mask"], second["mask"])
        self.assertEqual(first["selected_ids"], third["selected_ids"])
        self.assertEqual(first["answered"], 6)
        self.assertEqual(first["correct"] + third["correct"], 6)

    def test_score_control_is_label_free_reproducible_and_bounded(self):
        self.assertEqual(random_score("q:1", 42), random_score("q:1", 42))
        self.assertNotEqual(random_score("q:1", 42), random_score("q:1", 43))
        self.assertNotEqual(random_score("q:1", 42), random_score("q:2", 42))
        for i in range(100):
            self.assertGreaterEqual(random_score(str(i)), 0.)
            self.assertLess(random_score(str(i)), 1.)
        with self.assertRaises(ValueError):
            random_score(1)
        for seed in [True, -1, 2**64, 1.2]:
            with self.assertRaises(ValueError):
                quota_select([row(1)], 1, seed=seed)

    def test_empty_and_singleton_do_not_fabricate_accuracy(self):
        empty = quota_select([], .5)
        self.assertTrue(empty["feasible"])
        self.assertEqual(empty["selected_ids"], [])
        self.assertIsNone(empty["actual_coverage"])
        self.assertIsNone(empty["accuracy_answered"])
        zero = quota_select([row(1)], .5)
        self.assertEqual(zero["answered"], 0)
        self.assertIsNone(zero["accuracy_answered"])
        self.assertEqual(zero["delivered_outcomes"], {"1": "abstain"})
        self.assertEqual(quota_select([row(1)], 1)["accuracy_answered"], 1.)
        json.dumps([empty, zero], allow_nan=False)

    def test_reject_unresolved_bad_schema_nonfinite_and_reference_scores(self):
        invalid = [row(1, "unresolved"), row(1, "other"), row(1, score=float("nan")),
                   row(1, score=float("inf")), row(1, "abstain", float("nan")), row(1, score=True),
                   row(1, uses_gold_labels=True), row(1, uses_reference_answer=True),
                   row(1, uses_gold_labels="no"), {**row(1), "id": ""}, {**row(1), "pair_id": None}]
        for bad in invalid:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                quota_select([bad], 0)
        with self.assertRaises(ValueError):
            quota_select([row(1), row(1)], .5)
        with self.assertRaises(ValueError):
            quota_select({"1": row(1)}, .5)

    def test_arm_cohorts_and_group_mapping_are_strict(self):
        good = [row("a", group="q"), row("b", group="q")]
        aligned = validate_arms({"b": list(reversed(good)), "a": good})
        self.assertEqual(list(aligned), ["a", "b"])
        self.assertEqual(aligned["a"], aligned["b"])
        for bad in [good[:1], [row("a", group="q"), row("c", group="q")],
                    [row("a", group="another"), good[1]]]:
            with self.assertRaises(ValueError):
                validate_arms({"a": good, "b": bad})
        for bad in [{}, {1: good}, {"": good}, []]:
            with self.assertRaises(ValueError):
                validate_arms(bad)


class BootstrapTests(unittest.TestCase):
    def test_reselects_on_replicates_instead_of_reusing_original_masks(self):
        data = [row("a", "correct", 9.), row("b", "wrong", 1.)]
        # A fixed original mask would deliver zero on [b,b]. Reselection must
        # instead deliver exactly one wrong answer, so the accuracy draw is 0.
        with patch("core.selective_eval.np.random.default_rng", return_value=FixedDraws([[0, 0], [1, 1]])):
            result = evaluate_coverage({"base": data}, [.5], n_boot=2)
        arm = result["coverages"][0]["arms"]["base"]
        self.assertEqual(arm["accuracy_answered"], 1.)
        self.assertEqual(arm["bootstrap"]["answered_count_range"], [1, 1])
        self.assertEqual(arm["bootstrap"]["actual_coverage_range"], [.5, .5])
        self.assertEqual(arm["bootstrap"]["ci95"]["accuracy_answered"], [.025, .975])

    def test_tf_pairs_stay_together_and_duplicate_instances_count(self):
        # Resample two question groups, preserving their two statements each.
        data = [row("a:t", "correct", 9, "a"), row("a:f", "wrong", 8, "a"),
                row("b:t", "correct", 7, "b"), row("b:f", "wrong", 6, "b")]
        with patch("core.selective_eval.np.random.default_rng", return_value=FixedDraws([[0, 0], [1, 1]])):
            result = evaluate_coverage({"x": data}, [.5], n_boot=2)
        arm = result["coverages"][0]["arms"]["x"]
        self.assertEqual(result["n_groups"], 2)
        self.assertEqual(result["bootstrap_unit"], "pair_id")
        self.assertEqual(arm["accuracy_answered"], .5)
        # Two copies of a group's higher-scoring true row fill the two slots.
        self.assertEqual(arm["bootstrap"]["answered_count_range"], [2, 2])
        self.assertEqual(arm["bootstrap"]["ci95"]["accuracy_answered"], [1., 1.])

    def test_all_arms_share_draws_and_input_order_does_not_change_results(self):
        data = [row(i, "wrong" if i % 3 else "correct", random_score(str(i)), group=i // 2)
                for i in range(12)]
        arms = {"one": data, "two": list(reversed(data))}
        got = evaluate_coverage(arms, [.5, .75], n_boot=30, seed=19)
        other = evaluate_coverage(dict(reversed(list(arms.items()))), [.5, .75], n_boot=30, seed=19)
        self.assertEqual(got, other)
        for entry in got["coverages"]:
            pair = entry["comparisons"][0]
            self.assertEqual(pair["bootstrap_status"], "ok")
            self.assertEqual(pair["ci95"]["accuracy_answered"], [0., 0.])
            self.assertEqual(pair["exact_fixed_deliveries"]["p_value"], 1.)

    def test_infeasible_replicates_never_get_silently_dropped(self):
        a = [row("a", "correct", 9), row("b", "abstain", 1)]
        b = [row("a", "correct", 9), row("b", "wrong", 1)]
        with patch("core.selective_eval.np.random.default_rng", return_value=FixedDraws([[0, 0], [1, 1]])):
            got = evaluate_coverage({"a": a, "b": b}, [.5], n_boot=2)["coverages"][0]
        self.assertTrue(got["arms"]["a"]["feasible"])
        self.assertEqual(got["arms"]["a"]["bootstrap"]["infeasible_resamples"], 1)
        self.assertEqual(got["arms"]["a"]["bootstrap"]["status"], "infeasible_resamples")
        self.assertIsNone(got["arms"]["a"]["bootstrap"]["ci95"])
        self.assertEqual(got["arms"]["b"]["bootstrap"]["status"], "ok")
        self.assertIsNone(got["comparisons"][0]["ci95"])
        self.assertEqual(got["comparisons"][0]["infeasible_resamples"], 1)

    def test_infeasible_point_empty_zero_quota_and_no_bootstrap_are_explicit(self):
        arms = {"a": [row(1, "abstain")], "b": [row(1, "wrong")]}
        entry = evaluate_coverage(arms, [1], n_boot=2)["coverages"][0]
        self.assertEqual(entry["arms"]["a"]["bootstrap"]["status"], "infeasible_point")
        self.assertIsNone(entry["comparisons"][0]["exact_fixed_deliveries"])
        empty = evaluate_coverage({"a": []}, [.5], n_boot=2)
        self.assertEqual(empty["coverages"][0]["arms"]["a"]["bootstrap"]["status"], "empty_cohort")
        zero = evaluate_coverage({"a": [row(1)]}, [0], n_boot=2)
        self.assertEqual(zero["coverages"][0]["arms"]["a"]["bootstrap"]["status"], "undefined_zero_answers")
        no_boot = evaluate_coverage({"a": [row(1)]}, [1], n_boot=0)
        self.assertEqual(no_boot["coverages"][0]["arms"]["a"]["bootstrap"]["status"], "not_requested")
        json.dumps([entry, empty, zero, no_boot], allow_nan=False)

    def test_unequal_group_sizes_use_shared_replicate_n_and_k(self):
        data = [row("a", group="small"), row("b", group="large"), row("c", group="large")]
        # Canonical group order: large,small. Replicate N is four, then two.
        with patch("core.selective_eval.np.random.default_rng", return_value=FixedDraws([[0, 0], [1, 1]])):
            got = evaluate_coverage({"a": data, "b": data}, [.5], n_boot=2)["coverages"][0]
        self.assertEqual(got["arms"]["a"]["bootstrap"]["answered_count_range"], [1, 2])
        self.assertEqual(got["arms"]["a"]["bootstrap"]["actual_coverage_range"], [.5, .5])
        self.assertEqual(got["comparisons"][0]["ci95"]["accuracy_answered"], [0., 0.])

    def test_bad_bootstrap_settings_rejected(self):
        for n in [-1, 1, True, 1.5]:
            with self.assertRaises(ValueError):
                evaluate_coverage({"x": [row(1)]}, [.5], n_boot=n)
        for qs in [[], [.5, "1/2"], .5, [float("nan")]]:
            with self.assertRaises(ValueError):
                evaluate_coverage({"x": [row(1)]}, qs, n_boot=0)


class ExactComparisonTests(unittest.TestCase):
    def test_four_vs_zero_singletons_is_p_point125(self):
        a = quota_select([row(i, "correct") for i in range(10)], 1)
        b = quota_select([row(i, "wrong" if i < 4 else "correct") for i in range(10)], 1)
        result = exact_paired_correctness(a, b)
        self.assertEqual(result["positive_discordant_rows"], 4)
        self.assertEqual(result["negative_discordant_rows"], 0)
        self.assertEqual(result["informative_groups"], 4)
        self.assertEqual(result["p_value"], .125)
        self.assertEqual(exact_paired_correctness(b, a)["p_value"], .125)

    def test_tf_four_statements_in_two_questions_are_not_four_independent_trials(self):
        a = quota_select([row(i, "correct", group=i // 2) for i in range(4)], 1)
        b = quota_select([row(i, "wrong", group=i // 2) for i in range(4)], 1)
        result = exact_paired_correctness(a, b)
        self.assertEqual(result["positive_discordant_rows"], 4)
        self.assertEqual(result["informative_groups"], 2)
        self.assertEqual(result["p_value"], .5)

    def test_comparison_requires_actual_equal_counts_and_same_identities(self):
        a = quota_select([row(1), row(2)], 1)
        for b in [quota_select([row(1), row(2)], .5), quota_select([row(1), row(3)], 1),
                  quota_select([row(1, group="other"), row(2)], 1),
                  quota_select([row(1, "abstain"), row(2)], 1)]:
            with self.assertRaises(ValueError):
                exact_paired_correctness(a, b)
        invalid = deepcopy(a); invalid["answered"] = 1
        with self.assertRaises(ValueError):
            exact_paired_correctness(a, invalid)


class FrozenThresholdTests(unittest.TestCase):
    def test_policy_uses_no_correctness_and_reproduces_tuning_ties(self):
        tune = [row(i, "correct" if i % 2 else "wrong", 1.) for i in range(10)]
        policy = calibrate_threshold(tune, .3, seed=9)
        swapped = [{**r, "outcome": "wrong" if r["outcome"] == "correct" else "correct"} for r in reversed(tune)]
        self.assertEqual(policy, calibrate_threshold(swapped, .3, seed=9))
        self.assertEqual(apply_threshold(tune, json.loads(json.dumps(policy)))["selected_ids"],
                         quota_select(tune, .3, seed=9)["selected_ids"])

    def test_test_batch_does_not_recalibrate_a_frozen_threshold(self):
        tune = [row(i, score=s) for i, s in enumerate([.9, .8, .7, .6])]
        policy = calibrate_threshold(tune, .5)
        self.assertEqual(policy["threshold"], .8)
        test = [row(i + 10, score=.95, outcome="abstain" if i == 0 else "correct") for i in range(4)]
        result = apply_threshold(test, policy)
        self.assertEqual(result["target_coverage"], .5)
        self.assertEqual(result["actual_coverage"], .75)
        self.assertEqual(result["answered"], 3)
        self.assertEqual(result["evaluation"], "frozen_threshold")

    def test_all_none_empty_infeasible_and_malformed_policies(self):
        tune = [row(1), row(2, "abstain")]
        all_rule = calibrate_threshold(tune, .5)
        self.assertEqual(all_rule["mode"], "all")
        self.assertEqual(apply_threshold([row(3, score=-100)], all_rule)["answered"], 1)
        none = calibrate_threshold(tune, 0)
        self.assertEqual(apply_threshold(tune, none)["answered"], 0)
        self.assertEqual(calibrate_threshold([], 0)["mode"], "none")
        bad = calibrate_threshold(tune, 1)
        self.assertFalse(bad["feasible"])
        with self.assertRaises(ValueError):
            apply_threshold(tune, bad)
        valid = calibrate_threshold([row(1, score=2), row(2, score=1)], .5)
        for change in [{"mode": "bad"}, {"threshold": float("nan")}, {"tie_cutoff": ["bad", "1"]},
                       {"seed": True}, {"feasible": False}, {"protocol": "another"}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                apply_threshold(tune, {**valid, **change})

    def test_inputs_are_not_mutated(self):
        arms = {"x": [row(1, score=2), row(2, "wrong", 1)]}
        snapshot = deepcopy(arms)
        result = evaluate_coverage(arms, [.5], n_boot=2)
        policy = calibrate_threshold(arms["x"], .5)
        before_policy = deepcopy(policy)
        apply_threshold(arms["x"], policy)
        self.assertEqual(arms, snapshot)
        self.assertEqual(policy, before_policy)
        result["coverages"][0]["arms"]["x"]["delivered_outcomes"]["1"] = "wrong"
        self.assertEqual(arms, snapshot)


if __name__ == "__main__":
    unittest.main()
