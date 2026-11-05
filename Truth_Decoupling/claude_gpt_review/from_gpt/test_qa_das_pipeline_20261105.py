#!/usr/bin/env python3
"""Offline QA-DAS integration: actual causal forwards/CE training, local judge.

Only synthetic data and the random CPU network from the older pipeline test
are used. A deliberately artificial, deterministic local grader supplies
outcomes to exercise selection; these outcomes are not evidence of model
quality. All artifacts are temporary. No pretrained weights or API calls.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "from_claude" / "code"
sys.path[:0] = [str(HERE), str(CODE / "scripts"), str(CODE)]
from test_e25_pipeline_20261103 import ToyTokenizer, ToyCausalModel
import e24_real_qa as e24
import e25_qa_das as pipeline
from core.judge import QA_GRADER_VERSION
from core.model_io import ModelBundle
from core.qa_selection import rank_proxy_candidates


class PipelineModel(ToyCausalModel):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        # Remove the old fixture's dominant X bias, allowing real intervention
        # forwards to produce distinct strings for the selection/diagnostic path.
        with torch.no_grad():
            self.head.bias.zero_()
            self.head.bias[:3] = -10
            # This toy mixes unnormalised cumulative sums. The longer v1
            # system/suffix would otherwise saturate every tanh unit and erase
            # the synthetic causal path before the label, unlike a real LM.
            for block in self.layers:
                block.linear.weight.mul_(.02)

    def get_output_embeddings(self):
        return self.head


class LocalOutcomeJudge:
    """A synthetic truth table keyed by question and actual generated text.

    For plumbing coverage, a string differing from this toy's baseline is
    assigned 'correct', otherwise 'wrong'. Identical texts receive identical
    grades; no condition name, external service, or real-world claim is used.
    """
    name = "offline-integration-truth-table"
    qa_rubric_id = QA_GRADER_VERSION
    n_cache_hits = 0

    def __init__(self, baselines):
        self.baselines = baselines
        self.n_calls = 0

    def qa_grade(self, question, gold, prediction, incorrect=None):
        self.n_calls += 1
        label = "wrong" if prediction == self.baselines[question] else "correct"
        return {"label": label, "raw": "synthetic-local-grade", "rubric_id": self.qa_rubric_id}


class QADASPipelineTests(unittest.TestCase):
    def enter_context(self, cm):
        result = cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        return result

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="qa-das-pipeline-")
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.args = SimpleNamespace(run_dir=self.run, model="tiny-qa-das", pilot=None,
                                    layers=[0, 1], all_layers=False, judge_exp="offline")
        self.cfg = pipeline.read(CODE / "configs" / "experiments" / "E25_qa_das_v1.json")
        self.cfg.update({"source_experiment": "synthetic", "tune_pairs": 3, "batch_size": 2,
                         "epochs": 2, "max_new_tokens": 2, "alphas": [1.0, 3.0]})
        pipeline.validate_config(self.cfg)
        tokenizer = ToyTokenizer()
        self.model = PipelineModel(tokenizer)
        self.model_config = {"name": "tiny-qa-das", "layers_attr": "layers", "chat": {}}
        self.bundle = ModelBundle(self.model, tokenizer, self.model_config)
        train_ids = ["train%d" % i for i in range(3)]
        eval_ids = ["eval%d" % i for i in range(6)]
        originals = [{"id": pid, "text": "Which place is item %s?" % pid,
                      "gold": ["GOLD_" + pid], "incorrect": ["WRONG_" + pid],
                      "meta": {"best_answer": "GOLD_" + pid}} for pid in train_ids + eval_ids]
        def tf(ids):
            return [{"id": pid + (":t" if truth else ":f"), "pair_id": pid, "label": truth}
                    for pid in ids for truth in (True, False)]
        data = {("truthfulqa", "all"): originals, ("synthetic_tf", "calib"): tf(train_ids[:2]),
                ("synthetic_tf", "dev"): tf(train_ids[2:]), ("synthetic_tf", "test"): tf(eval_ids)}
        self.enter_context(patch.object(pipeline.legacy, "load_experiment",
                                        return_value={"eval_datasets": ["synthetic_tf"], "eval_split": "test"}))
        self.enter_context(patch.object(pipeline.legacy, "load_split",
                                        side_effect=lambda ds, split: copy.deepcopy(data[(ds, split)])))
        self.enter_context(patch.object(pipeline, "load_model_config", return_value=self.model_config))
        self.enter_context(patch.object(pipeline, "load_model", return_value=self.bundle))
        self.enter_context(patch("socket.create_connection", side_effect=AssertionError("network forbidden")))
        self.enter_context(patch("core.judge.LLMJudge._get_client", side_effect=AssertionError("API forbidden")))
        self.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.original_parameters = [p.detach().clone() for p in self.model.parameters()]
        pipeline.estimate(self.args, self.cfg, bundle=self.bundle)
        self.bank_meta, self.bank, self.prepared = pipeline.load_bank(self.args)
        self.training = pipeline.train(self.args, self.bank_meta, self.bank, self.prepared, bundle=self.bundle)
        self.meta, self.bank = pipeline.runtime_bank(self.args, self.bank_meta, self.bank)

    def assert_question_only_generation(self):
        self.assertTrue(self.model.generated_prompts)
        for prompt in self.model.generated_prompts:
            self.assertNotIn("GOLD_", prompt)
            self.assertNotIn("WRONG_", prompt)
            self.assertNotIn("Is the answer above factually correct?", prompt)

    def grade_tuning(self, raw):
        judge = LocalOutcomeJudge({q["question"]: q["outputs"]["baseline"] for q in raw["evaluations"]})
        with patch.object(e24, "load_model_config", return_value={"name": "tiny-qa-das"}), \
                patch.object(e24, "load_experiment", return_value={"judge": {}}), \
                patch.object(e24, "out_paths", return_value=(str(self.run), "unused", "unused")), \
                patch("core.judge.judge_from_config", return_value=judge):
            pipeline.judge(self.args, self.meta, "tune.json", "tune_scored")
        self.assertGreater(judge.n_calls, 0)

    def prepare_graded_tuning(self):
        """Produce authentic pipeline artifacts before adversarial mutations."""
        pipeline.score(self.args, self.meta, self.bank, self.prepared, bundle=self.bundle)
        _, choices = pipeline.load_shortlist(self.args, self.meta, self.prepared)
        raw = pipeline.generate_questions(
            self.args, self.meta, self.bank, self.prepared["splits"]["tune"],
            choices, "tune.json", "tune", bundle=self.bundle)
        self.grade_tuning(raw)
        return choices, raw

    def test_real_end_to_end_training_selection_check_and_diagnostics(self):
        # The real estimate and training functions ran, not mocked versions.
        self.assertEqual({k: len(v) for k, v in self.prepared["splits"].items()},
                         {"train": 3, "tune": 3, "test": 3})
        self.assertEqual(self.bank_meta["das_candidate_layers"], [0])
        self.assertEqual(self.bank_meta["structural_null_layers"], [1])
        self.assertEqual(self.bank_meta["md_layers"], [0, 1])
        self.assertEqual(set(self.training["layers"]), {"0"})
        self.assertIn("E_qa_das_L0", self.bank)
        self.assertNotIn("E_qa_das_L1", self.bank)
        self.assertIn("E_qa_md_L1", self.bank)
        self.assertGreater(self.training["layers"]["0"]["per_seed"]["gradient_norm_max"], 0)
        self.assertEqual(self.training["layers"]["0"]["per_seed"]["optimizer_updates"], 6)
        for old, current in zip(self.original_parameters, self.model.parameters()):
            self.assertTrue(torch.equal(old, current))

        pipeline.check(self.args, self.meta, self.bank, self.prepared, bundle=self.bundle)
        checked = pipeline.read(self.run / "check_selfcheck.json")
        self.assertFalse(checked["eligible_for_selection"])
        self.assertEqual({r["context"] for r in checked["reports"]}, {"qa_das", "qa_md"})
        self.assertTrue(all(not r["result"]["eligible_for_selection"] for r in checked["reports"]))

        scored, shortlist = pipeline.score(self.args, self.meta, self.bank, self.prepared, bundle=self.bundle)
        self.assertTrue(scored["completed"])
        self.assertEqual(len(scored["question_ids"]), 3)
        das_specs = [c for c in scored["condition_specs"] if c.get("context") == "qa_das"]
        md_specs = [c for c in scored["condition_specs"] if c.get("context") == "qa_md"]
        self.assertEqual({c["layer"] for c in das_specs}, {0})
        self.assertEqual({c["layer"] for c in md_specs}, {0, 1})
        self.assertTrue(all(not c["uses_reference_answer"] for c in scored["condition_specs"]))
        _, choices = pipeline.load_shortlist(self.args, self.meta, self.prepared)
        self.assertTrue(all(c.get("mode") == "continuous" for c in choices[1:]))
        raw = pipeline.generate_questions(self.args, self.meta, self.bank, self.prepared["splits"]["tune"],
                                           choices, "tune.json", "tune", bundle=self.bundle)
        self.assert_question_only_generation()
        self.grade_tuning(raw)
        summary = pipeline.read(self.run / "tune_scored.json")
        self.assertEqual(summary["source_generation_sha256"], pipeline.digest(self.run / "tune.json"))
        self.assertEqual(summary["judge_rubric_id"], QA_GRADER_VERSION)
        self.assertEqual(summary["n_substantive_gold"], 3)
        frozen = pipeline.freeze(self.args, self.meta, self.prepared)
        selected = [c for c in frozen["selected"].values() if c is not None]
        self.assertTrue(selected, "toy fixture must exercise selected/reference diagnostic paths")
        self.assertTrue(all(c["eligible_for_selection"] and not c["uses_reference_answer"] for c in selected))
        test = pipeline.generate_questions(self.args, self.meta, self.bank, self.prepared["splits"]["test"],
                                            [dict(pipeline.BASELINE)] + selected, "test.json", "test", bundle=self.bundle)
        self.assertTrue(test["completed"])
        self.assertFalse(test["diagnostic_only"])
        self.assertEqual({r["pair_id"] for r in test["evaluations"]},
                         {r["pair_id"] for r in self.prepared["splits"]["test"]})
        selection_before = (self.run / "selection.json").read_bytes()
        diagnostics = pipeline.diagnose(self.args, self.meta, self.bank, self.prepared, bundle=self.bundle)
        self.assertTrue(diagnostics["diagnostic_only"])
        oracles = [c for c in diagnostics["condition_specs"] if c["uses_reference_answer"]]
        self.assertTrue(oracles)
        self.assertTrue(all(not c["eligible_for_selection"] for c in oracles))
        self.assertEqual(selection_before, (self.run / "selection.json").read_bytes())
        self.assert_question_only_generation()
        with self.assertRaisesRegex(ValueError, "reference answer|ineligible|diagnostic"):
            rank_proxy_candidates(diagnostics["condition_specs"], {})
        # Resuming completed main artifacts must not recompute model outputs.
        calls = self.model.forward_calls
        pipeline.score(self.args, self.meta, self.bank, self.prepared, bundle=self.bundle)
        pipeline.generate_questions(self.args, self.meta, self.bank, self.prepared["splits"]["tune"],
                                    choices, "tune.json", "tune", bundle=self.bundle)
        self.assertEqual(calls, self.model.forward_calls)
        pipeline.selection(self.args, self.meta)
        (self.run / "tune_scored.json").write_text((self.run / "tune_scored.json").read_text() + " ")
        with self.assertRaisesRegex(ValueError, "source changed"):
            pipeline.selection(self.args, self.meta)

    def test_bank_and_trained_checkpoint_tampering_rejected(self):
        bank_json = self.run / "bank.json"
        original = bank_json.read_bytes()
        changed = json.loads(original)
        changed["config"]["qa_system"] += " changed"
        bank_json.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            pipeline.load_bank(self.args)
        bank_json.write_bytes(original)
        checkpoint = pipeline.checkpoint_path(self.run, 0, self.cfg["seed"])
        with checkpoint.open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "checkpoint changed"):
            pipeline.runtime_bank(self.args, self.bank_meta, self.bank)

    def test_reference_condition_cannot_enter_primary_generation(self):
        oracle = {**pipeline.condition("qa_das", "continuous", 0, 1.),
                  "uses_reference_answer": True, "eligible_for_selection": False}
        before = self.model.generate_calls
        with self.assertRaisesRegex(ValueError, "forbidden"):
            pipeline.generate_questions(self.args, self.meta, self.bank, self.prepared["splits"]["test"],
                                        [dict(pipeline.BASELINE), oracle], "test.json", "test", bundle=self.bundle)
        self.assertEqual(before, self.model.generate_calls)
        self.assertFalse((self.run / "test.json").exists())

    def test_frozen_selection_rejects_missing_sources_and_forged_decisions(self):
        self.prepare_graded_tuning()
        frozen = pipeline.freeze(self.args, self.meta, self.prepared)
        chosen_method = next(k for k, c in frozen["selected"].items() if c is not None)
        selection_path = self.run / "selection.json"
        original = selection_path.read_bytes()
        # Source artifacts are untouched, so matching their hashes alone must
        # not authorize an edited winner. The decision must be recomputed.
        for mutation in ("empty_sources", "alpha999", "diagnostic"):
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(frozen)
                if mutation == "empty_sources":
                    changed["sources"] = {}
                    error = "complete source manifest"
                elif mutation == "alpha999":
                    changed["selected"][chosen_method]["alpha"] = 999
                    error = "altered|declared rule"
                else:
                    changed["selected"][chosen_method].update(
                        uses_reference_answer=True, eligible_for_selection=False)
                    error = "altered|declared rule"
                selection_path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, error):
                    pipeline.selection(self.args, self.meta)
                selection_path.write_bytes(original)
        self.assertEqual(pipeline.selection(self.args, self.meta), frozen)

    def test_completed_generation_rejects_same_id_with_changed_question(self):
        pipeline.score(self.args, self.meta, self.bank, self.prepared, bundle=self.bundle)
        _, choices = pipeline.load_shortlist(self.args, self.meta, self.prepared)
        questions = self.prepared["splits"]["tune"]
        raw = pipeline.generate_questions(self.args, self.meta, self.bank, questions,
                                          choices, "tune.json", "tune", bundle=self.bundle)
        self.assertTrue(raw["completed"])
        before = self.model.forward_calls
        original = (self.run / "tune.json").read_bytes()
        changed_questions = copy.deepcopy(questions)
        changed_questions[0]["question"] += " This is a different question."
        self.assertEqual(changed_questions[0]["pair_id"], questions[0]["pair_id"])
        with self.assertRaisesRegex(ValueError, "question identity/text/references"):
            pipeline.generate_questions(self.args, self.meta, self.bank, changed_questions,
                                        choices, "tune.json", "tune", bundle=self.bundle)
        self.assertEqual(before, self.model.forward_calls)
        self.assertEqual(original, (self.run / "tune.json").read_bytes())

    def test_incomplete_grading_fails_then_retries_without_changing_denominator(self):
        class RetryJudge:
            name = "offline-retry-judge"
            qa_rubric_id = QA_GRADER_VERSION
            n_cache_hits = 0

            def __init__(self, missing):
                self.missing, self.resolve, self.n_calls = missing, False, 0

            def qa_grade(self, question, gold, prediction, incorrect=None):
                self.n_calls += 1
                label = "unresolved" if question == self.missing and not self.resolve else "wrong"
                return {"label": label, "raw": "offline-retry", "rubric_id": self.qa_rubric_id}

        choices = [dict(pipeline.BASELINE), pipeline.condition("qa_das", "continuous", 0, 1.)]
        phases = [("check_generations.json", "check_scored", "pilot", "tune"),
                  ("tune.json", "tune_scored", "tune", "tune"),
                  ("test.json", "test_scored", "test", "test"),
                  ("tune_diagnostics.json", "diagnostics_scored", "tune_diagnostic", "tune")]
        for filename, stem, split, question_split in phases:
            with self.subTest(phase=stem):
                questions = self.prepared["splits"][question_split]
                raw = pipeline.generate_questions(self.args, self.meta, self.bank, questions,
                                                  choices, filename, split, bundle=self.bundle)
                grader = RetryJudge(questions[0]["question"])
                original = (self.run / filename).read_bytes()
                with patch.object(e24, "load_model_config", return_value={"name": "tiny-qa-das"}), \
                        patch.object(e24, "load_experiment", return_value={"judge": {}}), \
                        patch.object(e24, "out_paths", return_value=(str(self.run), "unused", "unused")), \
                        patch("core.judge.judge_from_config", return_value=grader):
                    with self.assertRaisesRegex(RuntimeError, "grading incomplete"):
                        pipeline.judge(self.args, self.meta, filename, stem)
                    partial = pipeline.read(self.run / (stem + ".json"))
                    items = pipeline.read(self.run / (stem + "_items.json"))
                    self.assertFalse(partial["completed"])
                    self.assertFalse(items["completed"])
                    self.assertEqual(partial["n_questions"], 3)
                    self.assertEqual(partial["n_substantive_gold"], 3)
                    self.assertEqual(partial["n_unresolved"], 2)
                    self.assertNotIn("rates", partial)
                    self.assertNotIn("| 条件 |", (self.run / (stem + ".md")).read_text())
                    self.assertEqual(grader.n_calls, 6)
                    grader.resolve = True
                    pipeline.judge(self.args, self.meta, filename, stem)
                final = pipeline.read(self.run / (stem + ".json"))
                items = pipeline.read(self.run / (stem + "_items.json"))
                self.assertTrue(final["completed"])
                self.assertTrue(items["completed"])
                self.assertEqual(final["n_questions"], 3)
                self.assertEqual(final["n_resolved_all_conditions"], 3)
                self.assertEqual(grader.n_calls, 8, "retry must reuse the four resolved grades")
                self.assertTrue(all(r["rates"]["n"] == 3 for r in final["rates"].values()))
                self.assertTrue(all(r["n_unresolved"] == 0 for r in final["rates"].values()))
                self.assertEqual(final["source_generation_sha256"], pipeline.digest(self.run / filename))
                self.assertEqual(items["judge_summary"], final)
                self.assertEqual((self.run / filename).read_bytes(), original)
                if stem == "tune_scored":
                    # The real freeze still consumes exactly the original
                    # generation hash, not the enriched grading checkpoint.
                    self.assertEqual(final["header"]["protocol_fingerprint"], raw["protocol_fingerprint"])

    def test_all_unresolved_grades_are_saved_before_retry(self):
        class RetryAllJudge:
            name = "offline-all-unresolved"
            qa_rubric_id = QA_GRADER_VERSION
            n_cache_hits = 0
            n_calls = 0
            resolve = False

            def qa_grade(self, question, gold, prediction, incorrect=None):
                self.n_calls += 1
                return {"label": "wrong" if self.resolve else "unresolved",
                        "raw": "offline-all-unresolved", "rubric_id": self.qa_rubric_id}

        choices = [dict(pipeline.BASELINE)]
        pipeline.generate_questions(self.args, self.meta, self.bank, self.prepared["splits"]["test"],
                                    choices, "test.json", "test", bundle=self.bundle)
        grader = RetryAllJudge()
        with patch.object(e24, "load_model_config", return_value={"name": "tiny-qa-das"}), \
                patch.object(e24, "load_experiment", return_value={"judge": {}}), \
                patch.object(e24, "out_paths", return_value=(str(self.run), "unused", "unused")), \
                patch("core.judge.judge_from_config", return_value=grader):
            with self.assertRaisesRegex(RuntimeError, "grading incomplete"):
                pipeline.judge(self.args, self.meta, "test.json", "test_scored")
            saved = pipeline.read(self.run / "test_scored_items.json")
            self.assertFalse(saved["completed"])
            self.assertEqual(len(saved["evaluations"]), 3)
            self.assertTrue(all(r["judge"]["baseline"]["label"] == "unresolved" for r in saved["evaluations"]))
            grader.resolve = True
            pipeline.judge(self.args, self.meta, "test.json", "test_scored")
        self.assertTrue(pipeline.read(self.run / "test_scored.json")["completed"])
        self.assertEqual(grader.n_calls, 6)

    def test_freeze_rejects_outcome_forged_against_semantic_label(self):
        self.prepare_graded_tuning()
        # Validate the genuine E24 output first. No selection exists yet, so
        # rejection below cannot be attributed to frozen source hashes.
        pipeline.freeze(self.args, self.meta, self.prepared, persist=False)
        self.assertFalse((self.run / "selection.json").exists())
        judged_path = self.run / "tune_scored_items.json"
        judged = pipeline.read(judged_path)
        first = judged["evaluations"][0]
        self.assertEqual(first["judge"]["baseline"]["label"], "wrong")
        self.assertEqual(first["outcome"]["baseline"], "wrong")
        first["outcome"]["baseline"] = "correct"
        judged_path.write_text(json.dumps(judged))
        with self.assertRaisesRegex(ValueError, "outcome disagrees.*judge label"):
            pipeline.freeze(self.args, self.meta, self.prepared)
        self.assertFalse((self.run / "selection.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
