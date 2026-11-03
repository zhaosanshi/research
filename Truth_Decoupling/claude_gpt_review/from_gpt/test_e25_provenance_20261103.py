#!/usr/bin/env python3
"""Synthetic-file tests of E25 score provenance and freeze validation.

Creates only temporary JSON/NPZ fixtures. Does not read experimental results,
load a model, generate answers, call a judge, or access the network.
"""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "from_claude/code"
sys.path[:0] = [str(CODE / "scripts"), str(CODE)]
_spec = importlib.util.spec_from_file_location("e25_provenance_under_test", CODE / "scripts/e25_qa_context.py")
e25 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e25)
from core.judge import QA_GRADER_VERSION


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = "synthetic-qa-no-model-loaded"
        self.args = SimpleNamespace(run_dir=self.root, model=self.model)
        self.load_model_guard = patch.object(e25, "load_model", side_effect=AssertionError("model loading is forbidden in provenance tests"))
        self.load_model_guard.start()
        self.addCleanup(self.load_model_guard.stop)
        cfg = e25.read(CODE / "configs/experiments/E25_qa_context.json")
        cfg.update(tune_pairs=3, batch_size=2, max_new_tokens=8)
        self.cfg = cfg

        def question(pid):
            return {"pair_id": pid, "question": "Synthetic question " + pid,
                    "gold": ["synthetic positive"], "incorrect": ["synthetic negative"],
                    "best_answer": "synthetic positive"}

        tune = [question("tune:%d" % i) for i in range(3)]
        pairs = [{**row, "positive_answer": "synthetic positive", "negative_answer": "synthetic negative"} for row in tune]
        self.prepared = {"splits": {"train": [question("train:0")], "tune": tune, "test": [question("test:0")]},
                         "e_train": [], "e_tune": pairs, "pilot": None}
        dump(self.root / "questions.json", self.prepared)
        np.savez(self.root / "bank.npz", synthetic_array=np.array([1., 0.]))
        self.meta = {"protocol": e25.PROTOCOL, "model": self.model,
                     "model_config": {"name": self.model, "path": "never-load-this-fixture"},
                     "config": cfg, "layers": [0, 1], "num_layers": 2, "pilot": None,
                     "bank_sha256": e25.digest(self.root / "bank.npz"),
                     "questions_sha256": e25.digest(self.root / "questions.json"),
                     "rows": {str(layer): {"C_" + context: {"gap": 1.25}
                                           for context in cfg["c_contexts"]} for layer in (0, 1)}}
        self.meta["protocol_fingerprint"] = e25.protocol_identity(self.meta)
        dump(self.root / "bank.json", self.meta)
        self.e_scores = self.make_scores("E")
        self.c_pair_scores = self.make_scores("C")
        c_specs = self.c_pair_scores["condition_specs"]
        names = [row["name"] for row in c_specs]
        generation_fp = e25.stable_hash(self.meta["protocol_fingerprint"], c_specs, "tune",
                                        [row["pair_id"] for row in tune], length=40)
        self.c_raw = {"protocol": e25.PROTOCOL, "model": self.model, "split": "tune", "completed": True,
                      "protocol_fingerprint": generation_fp,
                      "bank_protocol_fingerprint": self.meta["protocol_fingerprint"],
                      "bank_sha256": self.meta["bank_sha256"], "questions_sha256": self.meta["questions_sha256"],
                      "condition_specs": c_specs, "conditions": names,
                      "paired_scores": self.c_pair_scores,
                      "evaluations": [{**row, "idx": i, "outputs": {name: "synthetic answer" for name in names},
                                       "finished": {name: True for name in names},
                                       "rule_abstain": {name: False for name in names}} for i, row in enumerate(tune)]}
        self.c_scores = {"model": self.model, "judge_rubric_id": QA_GRADER_VERSION,
                         "n_resolved_all_conditions": 3, "n_substantive_gold": 3,
                         "header": {"protocol_fingerprint": generation_fp},
                         "rates": {name: {"rates": {"abstain": 0. if name == "baseline" else 1 / 3,
                                                    "answer_rate": 1. if name == "baseline" else 2 / 3}}
                                   for name in names}}
        self.save_all()

    def make_scores(self, target):
        specs = e25.conditions(self.meta, self.meta["layers"], target)
        ids = [row["pair_id"] for row in self.prepared["e_tune"]]
        base_correct = np.array([True, False, False])
        rows = {}
        for condition in specs:
            correct = (np.array([True, True, False])
                       if target == "E" and condition["kind"] != "baseline" else base_correct.copy())
            rows[condition["name"]] = {
                "condition": copy.deepcopy(condition), "positive_logp": np.where(correct, -1., -3.).tolist(),
                "negative_logp": [-2.] * 3, "correct": correct.tolist(),
                **e25.paired_stats(base_correct, correct)}
        return {"model": self.model, "completed": True, "metric": "paired_answer_selection_accuracy",
                "bank_sha256": self.meta["bank_sha256"], "questions_sha256": self.meta["questions_sha256"],
                "bank_protocol_fingerprint": self.meta["protocol_fingerprint"],
                "question_ids": ids, "condition_specs": specs, "conditions": rows,
                "fingerprint": e25.stable_hash(self.meta["protocol_fingerprint"], specs, ids, length=40)}

    def save_all(self, refresh_source_hash=True):
        dump(self.root / "tune_e_scores.json", self.e_scores)
        dump(self.root / "tune_c.json", self.c_raw)
        if refresh_source_hash:
            self.c_scores["source_generation_sha256"] = e25.digest(self.root / "tune_c.json")
        dump(self.root / "tune_c_scored.json", self.c_scores)

    def freeze(self):
        meta, _, _ = e25.load_bank(self.args)
        with contextlib.redirect_stdout(io.StringIO()):
            e25.freeze(self.args, meta)

    def assert_rejected(self, message):
        self.save_all()
        with self.assertRaisesRegex(ValueError, message):
            self.freeze()
        self.assertFalse((self.root / "selection.json").exists())

    def test_valid_config_and_normal_freeze(self):
        self.assertEqual(e25.validate_config(copy.deepcopy(self.cfg)), self.cfg)
        self.freeze()
        frozen = e25.read(self.root / "selection.json")
        self.assertEqual(frozen["bank_protocol_fingerprint"], self.meta["protocol_fingerprint"])
        self.assertEqual(frozen["model"], self.model)
        self.assertEqual(frozen["tune_e_sha256"], e25.digest(self.root / "tune_e_scores.json"))
        self.assertEqual(frozen["tune_c_scored_sha256"], e25.digest(self.root / "tune_c_scored.json"))
        for context in self.cfg["e_contexts"]:
            for mode in self.cfg["e_write_modes"]:
                condition = frozen["selected"]["E_%s_%s" % (context, mode)]
                self.assertEqual((condition["kind"], condition["layer"], condition["alpha"]), ("E", 0, 1.))
        for context in self.cfg["c_contexts"]:
            condition = frozen["selected"]["C_" + context]
            self.assertEqual((condition["kind"], condition["layer"], condition["alpha"]), ("C", 0, 1.))

    def test_wrong_e_fingerprint_rejected(self):
        self.e_scores["fingerprint"] = "FOREIGN_FINGERPRINT"
        self.assert_rejected("fingerprint mismatch")

    def test_foreign_e_question_id_rejected_even_with_recomputed_fingerprint(self):
        self.e_scores["question_ids"][0] = "test:0"
        self.e_scores["fingerprint"] = e25.stable_hash(
            self.meta["protocol_fingerprint"], self.e_scores["condition_specs"], self.e_scores["question_ids"], length=40)
        self.assert_rejected("different bank/questions")

    def test_wrong_e_repeated_header_hash_rejected(self):
        self.e_scores["bank_sha256"] = "FOREIGN_BANK"
        self.assert_rejected("different bank/questions")

    def test_c_test_split_rejected(self):
        self.c_raw["split"] = "test"
        self.assert_rejected("not bound to this tuning generation")

    def test_c_generation_foreign_model_rejected(self):
        self.c_raw["model"] = "foreign-model"
        self.assert_rejected("not bound to this tuning generation")

    def test_c_scores_foreign_model_rejected(self):
        self.c_scores["model"] = "foreign-model"
        self.assert_rejected("not bound to this tuning generation")

    def test_c_wrong_source_generation_sha_rejected(self):
        self.c_scores["source_generation_sha256"] = "WRONG_GENERATION"
        self.save_all(refresh_source_hash=False)
        with self.assertRaisesRegex(ValueError, "not bound to this tuning generation"):
            self.freeze()

    def test_c_wrong_header_generation_fingerprint_rejected(self):
        self.c_scores["header"]["protocol_fingerprint"] = "OTHER_GENERATION"
        self.assert_rejected("not bound to this tuning generation")

    def test_c_old_rubric_rejected(self):
        self.c_scores["judge_rubric_id"] = "legacy-keyword-rubric"
        self.assert_rejected("current semantic rubric")

    def test_c_foreign_question_id_rejected(self):
        self.c_raw["evaluations"][0]["pair_id"] = "test:0"
        self.assert_rejected("not bound to this tuning generation")

    def test_c_raw_repeated_hash_rejected(self):
        self.c_raw["bank_sha256"] = "WRONG_BANK"
        self.assert_rejected("C tuning inputs differ")

    def test_c_paired_scores_wrong_fingerprint_rejected(self):
        self.c_raw["paired_scores"]["fingerprint"] = "WRONG_PAIR_SCORING"
        self.assert_rejected("paired tuning score fingerprint mismatch")

    def test_missing_e_condition_rejected(self):
        name = self.e_scores["condition_specs"][-1]["name"]
        del self.e_scores["conditions"][name]
        self.assert_rejected("candidates differ from configured grid")

    def test_missing_c_paired_condition_rejected(self):
        name = self.c_raw["paired_scores"]["condition_specs"][-1]["name"]
        del self.c_raw["paired_scores"]["conditions"][name]
        self.assert_rejected("candidates differ from configured grid")

    def test_c_generation_grid_differs_from_paired_scores_rejected(self):
        self.c_raw["condition_specs"] = copy.deepcopy(self.c_raw["condition_specs"][:-1])
        self.assert_rejected("different candidate grids")

    def test_changed_bank_config_rejected(self):
        meta = copy.deepcopy(self.meta)
        meta["config"]["qa_system"] = "A changed scoring and generation prompt."
        dump(self.root / "bank.json", meta)
        self.assert_rejected("protocol/configuration changed")

    def test_changed_model_configuration_rejected(self):
        meta = copy.deepcopy(self.meta)
        meta["model_config"]["path"] = "different-model-weights"
        dump(self.root / "bank.json", meta)
        self.assert_rejected("protocol/configuration changed")

    def test_changed_c_gap_rejected(self):
        meta = copy.deepcopy(self.meta)
        meta["rows"]["0"]["C_qa"]["gap"] *= 10
        dump(self.root / "bank.json", meta)
        self.assert_rejected("protocol/configuration changed")

    def test_changed_direction_array_rejected(self):
        np.savez(self.root / "bank.npz", synthetic_array=np.array([0., 1.]))
        self.assert_rejected("bank/data changed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
