"""Legacy E25 E-only freeze and source-binding checks, synthetic CPU only.

Reuses the previous provenance fixture and one local toy integration fixture.
No pretrained model, network, real result or external judge is used.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_e25_provenance_20261103 as previous
import test_e25_pipeline_20261103 as integration

e25 = previous.e25


class LegacyEOnlyTests(unittest.TestCase):
    def setUp(self):
        self.fixture = previous.ProvenanceTests(methodName="test_valid_config_and_normal_freeze")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.args, self.meta, self.root = self.fixture.args, self.fixture.meta, self.fixture.root

    def freeze(self, target=None):
        if target is not None:
            self.args.target = target
        with contextlib.redirect_stdout(io.StringIO()):
            return e25.freeze(self.args, self.meta)

    def test_e_only_with_no_c_files_and_no_c_reads(self):
        for name in ("tune_c.json", "tune_c_scored.json"):
            (self.root / name).unlink()
        original_read = e25.read

        def only_used(path):
            self.assertFalse(Path(path).name.startswith("tune_c"), "E-only read a C source")
            return original_read(path)

        with patch.object(e25, "read", side_effect=only_used):
            result = self.freeze("E")
            self.assertEqual(e25.load_selection(self.args, self.meta), result)
        self.assertEqual(set(result["sources"]), {"tune_e_scores.json"})
        self.assertIsNone(result["tune_c_scored_sha256"])
        self.assertTrue(all(result["selected"]["C_" + c] is None for c in ("qa", "self")))
        self.assertTrue(all(v is not None for k, v in result["selected"].items() if k.startswith("E_")))

    def test_e_only_ignores_present_but_invalid_unused_c(self):
        for name in ("tune_c.json", "tune_c_scored.json"):
            (self.root / name).write_text("invalid unused C JSON")
        result = self.freeze("E")
        (self.root / "tune_c.json").write_text("changed unused C artifact")
        self.assertEqual(e25.load_selection(self.args, self.meta), result)

    def test_e_only_still_validates_e_provenance(self):
        scores = e25.read(self.root / "tune_e_scores.json")
        scores["fingerprint"] = "foreign"
        previous.dump(self.root / "tune_e_scores.json", scores)
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            self.freeze("E")

    def test_default_both_binds_each_actually_used_source(self):
        frozen = self.freeze()
        self.assertEqual(frozen["target"], "both")
        self.assertEqual(set(frozen["sources"]), {"tune_e_scores.json", "tune_c.json", "tune_c_scored.json"})
        self.assertEqual(e25.load_selection(self.args, self.meta), frozen)
        for name in sorted(frozen["sources"]):
            with self.subTest(name=name):
                path = self.root / name
                original = path.read_text()
                path.write_text(original + " ")
                with self.assertRaisesRegex(ValueError, "source changed"):
                    e25.load_selection(self.args, self.meta)
                path.write_text(original)

    def test_e_only_source_change_fails_before_main_loads_model(self):
        self.freeze("E")
        with (self.root / "tune_e_scores.json").open("a") as out:
            out.write(" ")
        with patch.object(sys, "argv", ["e25", "generate", "--model", self.args.model, "--run-dir", str(self.root)]):
            with self.assertRaisesRegex(ValueError, "source changed"):
                e25.main()

    def test_modified_selection_and_missing_source_refused(self):
        frozen = self.freeze("E")
        changed = copy.deepcopy(frozen)
        name = next(k for k in changed["selected"] if k.startswith("E_"))
        changed["selected"][name]["alpha"] = 999
        previous.dump(self.root / "selection.json", changed)
        with self.assertRaisesRegex(ValueError, "validated tuning decision"):
            e25.load_selection(self.args, self.meta)
        changed = copy.deepcopy(frozen)
        changed["sources"] = {}
        previous.dump(self.root / "selection.json", changed)
        with self.assertRaisesRegex(ValueError, "source set"):
            e25.load_selection(self.args, self.meta)

    def test_rejudge_c_cannot_change_frozen_point(self):
        self.freeze()
        with patch.object(integration.e24, "phase_judge", side_effect=AssertionError("no judge call")):
            with self.assertRaisesRegex(ValueError, "selection is frozen"):
                e25.judge_file(self.args, "tune")


class LegacyConfigAndTemplateTests(unittest.TestCase):
    def config(self):
        return e25.read(previous.CODE / "configs/experiments/E25_qa_context.json")

    def test_qa_prompt_is_literal_e24_prompt(self):
        shared = e25.read(previous.CODE / "configs/prompts/default_v1.json")
        self.assertEqual(self.config()["qa_system"], shared["qa"]["system_idk_ok"])

    def test_seed_and_question_template_validation(self):
        for seed in (None, True, -1, 42.0, 2**32):
            cfg = self.config()
            cfg["seed"] = seed
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "seed"):
                e25.validate_config(cfg)
        for key in ("qa_user", "self_report_user"):
            for template in (None, "", "Question missing", "{other}", "{question.x}", "{question!r}", "{question:>9}", "{question"):
                cfg = self.config()
                cfg[key] = template
                with self.subTest(key=key, template=template), self.assertRaisesRegex(ValueError, key):
                    e25.validate_config(cfg)
        cfg = self.config()
        cfg["qa_user"] = "{{literal}} Question: {question}"
        self.assertEqual(e25.validate_config(cfg), cfg)

    def test_zero_strength_remains_valid_invalid_numeric_settings_fail(self):
        cfg = self.config()
        cfg["e_alphas"] = cfg["c_alphas"] = [0.0, 1.0]
        self.assertEqual(e25.validate_config(cfg), cfg)
        for key in ("e_alphas", "c_alphas"):
            for value in ([True], [float("nan")], [float("inf")], [-1], ["1"], [1, 1], [1.0000001, 1.0000002]):
                cfg = self.config()
                cfg[key] = value
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    e25.validate_config(cfg)

    def test_tf_uses_adapter_format_and_cleaned_answers_only(self):
        raw = [{"pair_id":"q", "question":"  Where is it?  ", "best_answer":"I have no comment.",
                "gold":["I have no comment.", "Paris"], "incorrect":["I don't know", "London."]}]
        pairs, _ = e25.prepare_e_pairs(raw)
        self.assertEqual(e25.tf_statements(pairs), [
            'The answer to the question "Where is it?" is: Paris.',
            'The answer to the question "Where is it?" is: London.'])


class LegacyEOnlyToyIntegrationTests(unittest.TestCase):
    def test_estimate_score_e_freeze_generate_without_any_c_tuning(self):
        fixture = integration.E25IntegrationTests(methodName="test_split_cleaning_bank_and_full_pipeline")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        m = integration.e25
        with contextlib.redirect_stdout(io.StringIO()):
            choices = m.conditions(fixture.meta, [0], "E")
            m.score_conditions(fixture.bundle, fixture.meta, fixture.bank, fixture.prepared["e_tune"], choices,
                               fixture.directory / "tune_e_scores.json")
            fixture.args.target = "E"
            result = m.freeze(fixture.args, fixture.meta)
            self.assertFalse(any(fixture.directory.glob("tune_c*")))
            self.assertIn("re-estimated", fixture.meta["E_direction_contexts"]["tf"])
            self.assertEqual(m.load_selection(fixture.args, fixture.meta), result)
            # All alpha=0 candidate outcomes equal baseline, so no fallback is invented.
            self.assertTrue(all(value is None for value in result["selected"].values()))
            m.generate_questions(fixture.bundle, fixture.meta, fixture.bank, fixture.prepared["splits"]["test"],
                                 [{"name":"baseline","kind":"baseline"}], fixture.directory / "test.json", "test")
            generated = m.read(fixture.directory / "test.json")
            self.assertTrue(generated["completed"])
            self.assertEqual(generated["conditions"], ["baseline"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
