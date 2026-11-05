#!/usr/bin/env python3
"""CPU-only QA selection tests: synthetic labels, no model/API or test-set tuning."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

CODE = Path(__file__).resolve().parent.parent / "from_claude/code"
sys.path.insert(0, str(CODE))
from core.qa_selection import rank_proxy_candidates, select_real_candidates


BASE = {"name": "baseline", "kind": "baseline"}


def candidate(name, context="qa_das", layer=1, alpha=1., mode="continuous", **extra):
    return {"name": name, "kind": "E", "context": context, "mode": mode,
            "layer": layer, "alpha": alpha, **extra}


def maps(ids, **conditions):
    return {name: dict(zip(ids, values)) for name, values in conditions.items()}


class ProxySelectionTests(unittest.TestCase):
    def test_per_layer_alpha_then_three_distinct_layers_and_paired_harm(self):
        specs = [BASE, candidate("l1a1", layer=1, alpha=1.), candidate("l1a2", layer=1, alpha=2.),
                 candidate("l2", layer=2, alpha=1.), candidate("l3", layer=3, alpha=.5),
                 candidate("l4", layer=4, alpha=.5), candidate("prefill", layer=0, mode="prefill")]
        vectors = {
            "baseline": [True, True, False, False, False, False],
            "l1a1": [True, True, True, False, False, False],
            "l1a2": [True, False, True, True, False, False],
            "l2": [True, True, True, True, False, False],
            "l3": [True, False, True, True, True, False],
            "l4": [True, True, True, True, False, False],
            "prefill": [True] * 6,
        }
        # Deliberately incorrect cached summaries must not control ranking.
        rows = {name: {"correct": vector, "accuracy": 999 if name == "l1a2" else -999,
                       "harmed": -999 if name == "l1a2" else 999} for name, vector in vectors.items()}
        result = rank_proxy_candidates(specs, rows)
        self.assertEqual([s["name"] for s in result["qa_das"]], ["l4", "l2", "l3"])
        self.assertEqual(result["qa_md"], [])
        longer = rank_proxy_candidates(specs, rows, top_k=4)
        self.assertEqual([s["name"] for s in longer["qa_das"]], ["l4", "l2", "l3", "l1a1"])
        self.assertEqual(rank_proxy_candidates(specs, rows, main_mode="prefill")["qa_das"][0]["name"], "prefill")

    def test_proxy_does_not_require_improvement_over_baseline(self):
        specs = [BASE, candidate("equal", context="qa_md", layer=2), candidate("worse", context="qa_md", layer=1)]
        rows = {"baseline": {"correct": [True, True, False]},
                "equal": {"correct": [True, True, False]}, "worse": {"correct": [True, False, False]}}
        result = rank_proxy_candidates(specs, rows)
        self.assertEqual([s["name"] for s in result["qa_md"]], ["equal", "worse"])

    def test_proxy_rejects_bad_alignment_and_non_boolean_values(self):
        specs = [BASE, candidate("x")]
        for vector in ([True], [1, 0], [True, "false"], []):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                rank_proxy_candidates(specs, {"baseline": {"correct": [True, False]}, "x": {"correct": vector}})
        with self.assertRaises(ValueError):
            rank_proxy_candidates(specs, {"baseline": {"correct": [True, False]}})
        with self.assertRaises(ValueError):
            rank_proxy_candidates(specs, {"baseline": {"correct": [True, False]}, "x": {"correct": [True, False]}}, top_k=0)


class RealSelectionTests(unittest.TestCase):
    def test_192_questions_five_answer_losses_pass_six_fail(self):
        ids = ["q:%d" % i for i in range(192)]
        base = ["correct"] * 96 + ["wrong"] * 96
        five, six = base.copy(), base.copy()
        five[96] = "correct"
        six[96] = six[97] = "correct"  # larger accuracy gain cannot override answer-rate constraint
        five[-5:] = ["abstain"] * 5
        six[-6:] = ["abstain"] * 6
        result = select_real_candidates([BASE, candidate("five"), candidate("six", layer=2)],
                                        maps(ids, baseline=base, five=five, six=six), ids)
        self.assertEqual(result["selected"]["qa_das"]["name"], "five")
        self.assertEqual(result["rule"]["max_abs_answer_count_change"], 5)
        self.assertTrue(result["reports"]["five"]["eligible"])
        self.assertFalse(result["reports"]["six"]["eligible"])
        self.assertEqual(result["reports"]["five"]["delta_R"], -5 / 192)
        self.assertEqual(result["reports"]["six"]["delta_R"], -6 / 192)

    def test_192_questions_five_new_answers_pass_six_fail(self):
        ids = ["q:%d" % i for i in range(192)]
        base = ["abstain"] * 192
        five, six = base.copy(), base.copy()
        five[:5] = ["correct"] * 5
        six[:6] = ["correct"] * 6
        result = select_real_candidates([BASE, candidate("five"), candidate("six", layer=2)],
                                        maps(ids, baseline=base, five=five, six=six), ids)
        self.assertEqual(result["selected"]["qa_das"]["name"], "five")
        self.assertEqual(result["reports"]["five"]["R"], 5 / 192)
        self.assertIsNone(result["reports"]["five"]["original_correct_loss_rate"])

    def test_exact_three_percent_boundary_does_not_fail_float_subtraction(self):
        ids = [str(i) for i in range(100)]
        base = ["correct"] * 10 + ["wrong"] * 40 + ["abstain"] * 50
        after = base.copy(); after[20] = "correct"; after[50:53] = ["wrong"] * 3
        result = select_real_candidates([BASE, candidate("x")], maps(ids, baseline=base, x=after), ids)
        self.assertEqual(result["reports"]["x"]["delta_R"], .03)
        self.assertIsNotNone(result["selected"]["qa_das"])

    def test_equal_gain_prefers_fewer_original_correct_losses_including_abstention(self):
        ids = list("abcde")
        base = ["correct", "correct", "wrong", "wrong", "abstain"]
        good = ["correct", "correct", "correct", "wrong", "abstain"]
        wrong = ["wrong", "correct", "correct", "correct", "abstain"]
        abst = ["abstain", "correct", "correct", "correct", "abstain"]
        specs = [BASE, candidate("less_loss", alpha=2.), candidate("wrong_loss", layer=0, alpha=.1),
                 candidate("abstain_loss", layer=0, alpha=.1)]
        result = select_real_candidates(specs, maps(ids, baseline=base, less_loss=good, wrong_loss=wrong, abstain_loss=abst),
                                        ids, tolerance=1.)
        self.assertEqual(result["selected"]["qa_das"]["name"], "less_loss")
        self.assertEqual(result["reports"]["wrong_loss"]["lost_correct"], 1)
        self.assertEqual(result["reports"]["abstain_loss"]["lost_correct"], 1)

    def test_estimators_independent_and_ties_prefer_alpha_then_layer(self):
        ids = list("abcd")
        base = ["correct", "wrong", "wrong", "wrong"]
        after = ["correct", "correct", "wrong", "wrong"]
        specs = [BASE, candidate("das_no_gain"), candidate("md_large", "qa_md", layer=0, alpha=1.),
                 candidate("md_deep", "qa_md", layer=9, alpha=.5), candidate("md_shallow", "qa_md", layer=2, alpha=.5)]
        result = select_real_candidates(specs, maps(ids, baseline=base, das_no_gain=base,
                                                   md_large=after, md_deep=after, md_shallow=after), ids)
        self.assertIsNone(result["selected"]["qa_das"])
        self.assertEqual(result["selected"]["qa_md"]["name"], "md_shallow")

    def test_complete_three_by_three_and_all_denominators(self):
        ids = [str(i) for i in range(9)]
        base = ["correct"] * 3 + ["wrong"] * 3 + ["abstain"] * 3
        after = ["correct", "wrong", "abstain"] * 3
        result = select_real_candidates([BASE, candidate("x")], maps(ids, baseline=base, x=after), ids)
        row = result["reports"]["x"]
        self.assertEqual(len(row["transitions"]), 9)
        self.assertTrue(all(n == 1 for n in row["transitions"].values()))
        self.assertEqual(row["counts"], {"correct": 3, "wrong": 3, "abstain": 3})
        self.assertEqual((row["gained_correct"], row["lost_correct"], row["net_correct"]), (2, 2, 0))
        self.assertEqual((row["corrected"], row["damaged"], row["correct_to_abstain"]), (1, 1, 1))
        self.assertEqual((row["Acc"], row["R"], row["precision_answered"]), (1 / 3, 2 / 3, .5))
        self.assertEqual(row["denominators"], {"accuracy": 9, "answer_rate": 9, "precision_answered": 6,
                                               "original_correct_loss_rate": 3})
        self.assertIsNone(result["selected"]["qa_das"])

    def test_no_answered_items_and_baseline_only_are_valid_reports(self):
        ids = ["one", "two"]
        result = select_real_candidates([dict(BASE, eligible_for_selection=False)],
                                        {"baseline": {"one": "abstain", "two": "abstain"}}, ids)
        self.assertEqual(result["selected"], {"qa_das": None, "qa_md": None})
        self.assertIsNone(result["reports"]["baseline"]["precision_answered"])
        self.assertEqual(result["reports"]["baseline"]["R"], 0)

    def test_reference_or_ineligible_diagnostics_rejected_by_both_selectors(self):
        for flag in ({"uses_reference_answer": True}, {"eligible_for_selection": False}):
            specs = [BASE, candidate("diagnostic", **flag)]
            with self.subTest(flag=flag):
                with self.assertRaisesRegex(ValueError, "diagnostic"):
                    select_real_candidates(specs, {"baseline": {"q": "wrong"}, "diagnostic": {"q": "correct"}}, ["q"])
                with self.assertRaisesRegex(ValueError, "diagnostic"):
                    rank_proxy_candidates(specs, {"baseline": {"correct": [False]}, "diagnostic": {"correct": [True]}})

    def test_unresolved_unknown_missing_extra_or_duplicate_questions_rejected(self):
        ids = ["a", "b"]
        specs = [BASE, candidate("x")]
        good = maps(ids, baseline=["correct", "wrong"], x=["correct", "correct"])
        for value in ("unresolved", "Unknown", "C", None, True):
            data = deepcopy(good); data["x"]["a"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                select_real_candidates(specs, data, ids)
        for case in ("missing", "extra", "missing_condition", "extra_condition"):
            data = deepcopy(good)
            if case == "missing": del data["x"]["b"]
            elif case == "extra": data["x"]["c"] = "correct"
            elif case == "missing_condition": del data["x"]
            else: data["unexpected"] = data["x"].copy()
            with self.subTest(case=case), self.assertRaises(ValueError):
                select_real_candidates(specs, data, ids)
        with self.assertRaises(ValueError): select_real_candidates(specs, good, ["a", "a"])
        with self.assertRaises(ValueError): select_real_candidates(specs, good, [])

    def test_metadata_validation_and_tolerance(self):
        values = {"baseline": {"q": "wrong"}, "x": {"q": "correct"}}
        for change in ({"alpha": float("nan")}, {"alpha": float("inf")}, {"alpha": -1.},
                       {"layer": -1}, {"layer": True}, {"context": "tf"}, {"mode": "all"},
                       {"uses_reference_answer": "false"}):
            spec = candidate("x"); spec.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                select_real_candidates([BASE, spec], values, ["q"])
        for tolerance in (-.1, 1.1, float("nan"), float("inf"), True):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                select_real_candidates([BASE, candidate("x")], values, ["q"], tolerance=tolerance)

    def test_functions_do_not_mutate_inputs_or_share_output_specs(self):
        specs = [deepcopy(BASE), candidate("x", training={"id": "original"})]
        rows = {"baseline": {"correct": [False, False]}, "x": {"correct": [True, False]}}
        values = maps(["a", "b"], baseline=["wrong", "wrong"], x=["correct", "wrong"])
        before = deepcopy((specs, rows, values))
        proxy = rank_proxy_candidates(specs, rows)
        real = select_real_candidates(specs, values, ["a", "b"])
        proxy["qa_das"][0]["training"]["id"] = "changed"
        real["selected"]["qa_das"]["layer"] = 99
        real["reports"]["x"]["condition"]["training"]["id"] = "changed"
        self.assertEqual((specs, rows, values), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
