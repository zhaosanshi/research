#!/usr/bin/env python3
"""Offline E25 integration checks with synthetic data and a tiny causal network.

Runs the real estimate/score/generate/judge-summary/freeze functions. The model
is a randomly initialized CPU toy with an actual causal cumulative-state cache;
the semantic judge is a local stub, so scores are NOT scientific evidence.
No pretrained weights, API calls, persistent experiment outputs, or production
file edits are involved. Every run writes only inside TemporaryDirectory.

Run: python3 claude_gpt_review/from_gpt/test_e25_pipeline_20261103.py
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

import numpy as np
import torch

CODE = Path(__file__).resolve().parent.parent / "from_claude" / "code"
sys.path[:0] = [str(CODE / "scripts"), str(CODE)]
import e24_real_qa as e24
import e25_qa_context as e25
from core.intervene import steer
from core.judge import QA_GRADER_VERSION
from core.model_io import ModelBundle, forward_logits, position_ids_from_mask
from core.qa_context import encode_answer, pack_answers, answer_logprobs


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    all_special_ids = [0, 1, 2]
    padding_side = "left"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize
        text = "".join("<%s>%s\n" % (m["role"], m["content"]) for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, texts, return_tensors=None, padding=False, add_special_tokens=False):
        assert not add_special_tokens
        single = isinstance(texts, str)
        texts = [texts] if single else texts
        rows = [[ord(c) + 3 for c in text] for text in texts]
        if return_tensors is None:
            return {"input_ids": rows[0] if single else rows}
        assert return_tensors == "pt"
        width = max(map(len, rows))
        return {
            "input_ids": torch.tensor([[0] * (width - len(row)) + row for row in rows]),
            "attention_mask": torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in rows]),
        }

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i) - 3) for i in ids if int(i) >= 3)


class CausalBlock(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(self, h, previous_sum=None):
        cumulative = h.cumsum(1)
        if previous_sum is not None:
            cumulative = cumulative + previous_sum[:, None, :]
        return h + .1 * torch.tanh(self.linear(cumulative)), cumulative[:, -1]


class ToyCausalModel(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        torch.manual_seed(17)
        self.config = SimpleNamespace(hidden_size=8, num_hidden_layers=2)
        self.generation_config = SimpleNamespace(eos_token_id=2)
        self.embedding = torch.nn.Embedding(256, 8, padding_idx=0)
        self.layers = torch.nn.ModuleList([CausalBlock(8), CausalBlock(8)])
        self.head = torch.nn.Linear(8, 256)
        with torch.no_grad():
            self.head.weight.mul_(.05)
            self.head.bias.zero_()
            self.head.bias[ord("X") + 3] = 5
        self.requires_grad_(False)
        self.eval()
        self.tokenizer = tokenizer
        self.forward_calls = 0
        self.generate_calls = 0
        self.generated_prompts = []

    def forward(self, input_ids, attention_mask, position_ids=None, use_cache=False, past_key_values=None):
        self.forward_calls += 1
        h = self.embedding(input_ids) * attention_mask[:, -input_ids.shape[1]:, None]
        cache = []
        for i, layer in enumerate(self.layers):
            h, state = layer(h, None if past_key_values is None else past_key_values[i])
            cache.append(state)
        return SimpleNamespace(logits=self.head(h), past_key_values=cache if use_cache else None)

    def generate(self, input_ids, attention_mask, max_new_tokens, do_sample, num_beams, use_cache, pad_token_id):
        assert not do_sample and num_beams == 1 and use_cache and pad_token_id == 0
        self.generate_calls += 1
        self.generated_prompts.extend(self.tokenizer.decode(row) for row in input_ids)
        ids, mask, output, cache = input_ids, attention_mask, input_ids, None
        for _ in range(max_new_tokens):
            out = self(input_ids=ids, attention_mask=mask,
                       position_ids=position_ids_from_mask(mask)[:, -ids.shape[1]:],
                       past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            ids = out.logits[:, -1].argmax(-1, keepdim=True)
            output = torch.cat([output, ids], dim=1)
            mask = torch.cat([mask, torch.ones(len(mask), 1, dtype=mask.dtype)], dim=1)
        return output


class OfflineJudge:
    name = "offline-test-stub"
    qa_rubric_id = QA_GRADER_VERSION
    n_cache_hits = 0

    def __init__(self):
        self.n_calls = 0

    def qa_grade(self, question, gold, prediction, incorrect=None):
        self.n_calls += 1
        # A deliberately simple local grader for plumbing, never model quality.
        label = "correct" if prediction.strip() in gold else "wrong"
        return {"label": label, "raw": "offline", "rubric_id": self.qa_rubric_id}


class E25IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="e25-offline-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.args = SimpleNamespace(run_dir=self.directory, model="tiny-audit", pilot=None,
                                    layers=[0], all_layers=False, judge_exp="offline")
        self.cfg = {
            "source_experiment": "synthetic", "seed": 42, "tune_pairs": 4,
            "batch_size": 2, "max_new_tokens": 2, "qa_system": "Answer accurately.",
            "qa_user": "Question: {question}\nAnswer:",
            "self_report_user": "Question: {question}\nDo you know? Yes or No.",
            "e_contexts": ["tf", "qa"], "e_write_modes": ["prefill", "continuous"],
            # Zero strengths provide a deterministic end-to-end no-improvement
            # case: freeze must record None instead of inventing a winner.
            "e_alphas": [0.0], "c_contexts": ["self", "qa"], "c_alphas": [0.0],
            "c_pair_accuracy_tolerance": .02, "orthogonalize_c": True,
            "c_contrasts": [{"confident": "Answer decisively.", "cautious": "Decline to guess."}],
        }
        train_ids = ["tr0", "tr1", "tr2", "tr3"]
        eval_ids = ["ev%d" % i for i in range(8)]
        tune_ids = sorted(np.random.default_rng(42).choice(eval_ids, 4, replace=False))
        self.refusal_tune_id = tune_ids[0]
        originals = []
        for pid in train_ids + eval_ids:
            gold, incorrect = ["GOLD_" + pid], ["WRONG_" + pid]
            best = gold[0]
            if pid == "tr0":
                gold = ["Unknown", "I don't know", "GOLD_tr0"]
                best = "I have no comment."
            if pid == "tr1":
                gold, best = ["Unknown", "Uncertain"], "Unknown"
            if pid == "tr2":
                gold, best = ["The origin is unknown."], "The origin is unknown."
            if pid == self.refusal_tune_id:
                gold, best = ["I have no comment."], "I have no comment."
            originals.append({"id": pid, "text": "Where is item %s?" % pid,
                              "gold": gold, "incorrect": incorrect, "meta": {"best_answer": best}})

        def tf_records(pids):
            return [{"id": pid + (":true" if label else ":false"), "pair_id": pid, "label": label}
                    for pid in pids for label in (True, False)]

        self.dataset = {("truthfulqa", "all"): originals,
                        ("synthetic_tf", "calib"): tf_records(train_ids[:2]),
                        ("synthetic_tf", "dev"): tf_records(train_ids[2:]),
                        ("synthetic_tf", "test"): tf_records(eval_ids)}
        self.exp = {"eval_datasets": ["synthetic_tf"], "eval_split": "test",
                    "prompts": {"statement": {"system": "Judge the statement.",
                                                "user_tf": "Statement: {statement}\nTrue or False?"}}}
        tok = ToyTokenizer()
        self.bundle = ModelBundle(ToyCausalModel(tok), tok,
                                  {"name": "tiny-audit", "layers_attr": "layers", "chat": {}})
        self.enterContext(patch.object(e25, "load_split", side_effect=lambda d, s: copy.deepcopy(self.dataset[(d, s)])))
        self.enterContext(patch.object(e25, "load_experiment", return_value=self.exp))
        self.enterContext(patch.object(e25, "load_model_config", return_value=self.bundle.cfg))
        self.enterContext(patch.object(e25, "load_model", return_value=self.bundle))
        self.enterContext(patch("socket.create_connection", side_effect=AssertionError("network forbidden in this test")))
        self.enterContext(patch("core.judge.LLMJudge._get_client", side_effect=AssertionError("API forbidden in this test")))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        e25.validate_config(self.cfg)
        e25.estimate(self.args, self.cfg)
        self.meta, self.bank, self.prepared = e25.load_bank(self.args)

    # Compatibility with the workspace's Python 3.9 unittest (enterContext was
    # added to TestCase in newer Python versions).
    def enterContext(self, cm):
        result = cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        return result

    def write(self, filename, value):
        (self.directory / filename).write_text(json.dumps(value), encoding="utf-8")

    def tune_and_judge(self):
        echoices = e25.conditions(self.meta, [0], "E")
        cchoices = e25.conditions(self.meta, [0], "C")
        e25.score_conditions(self.bundle, self.meta, self.bank, self.prepared["e_tune"],
                             echoices, self.directory / "tune_e_scores.json")
        paired = e25.score_conditions(self.bundle, self.meta, self.bank, self.prepared["e_tune"],
                                      cchoices, self.directory / "tune_c_pair_scores.json")
        e25.generate_questions(self.bundle, self.meta, self.bank, self.prepared["splits"]["tune"],
                               cchoices, self.directory / "tune_c.json", "tune", paired)
        self.assertTrue(self.bundle.model.generated_prompts)
        for text in self.bundle.model.generated_prompts:
            self.assertNotIn("GOLD_", text)
            self.assertNotIn("WRONG_", text)
            self.assertNotIn("The origin is unknown.", text)
        judge = OfflineJudge()
        judge_args = SimpleNamespace(model="tiny-audit", tag="E25_tune", phase="judge",
                                     judge_exp="offline", input_file=str(self.directory / "tune_c.json"),
                                     output_prefix=str(self.directory / "tune_c_scored"),
                                     rejudge=False, n_boot=8)
        with patch.object(e24, "load_model_config", return_value={"name": "tiny-audit"}), \
                patch.object(e24, "load_experiment", return_value={"judge": {}}), \
                patch.object(e24, "out_paths", return_value=(str(self.directory), "unused", "unused")), \
                patch("core.judge.judge_from_config", return_value=judge):
            e24.phase_judge(judge_args)
        self.assertGreater(judge.n_calls, 0)
        return echoices, cchoices, paired

    def test_split_cleaning_bank_and_full_pipeline(self):
        sets = [{r["pair_id"] for r in self.prepared["splits"][s]} for s in ("train", "tune", "test")]
        self.assertEqual([len(s) for s in sets], [4, 4, 4])
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual([r["pair_id"] for r in self.prepared["e_train"]], ["tr0", "tr2", "tr3"])
        self.assertEqual(self.prepared["e_train"][0]["positive_answer"], "GOLD_tr0")
        self.assertEqual(len(self.prepared["e_tune"]), 3)
        self.assertIn(self.refusal_tune_id, self.prepared["pair_audit"]["tune"]["excluded_ids"])
        self.assertTrue(all(np.isfinite(v).all() for v in self.bank.values()))
        self.assertEqual(self.meta["protocol_fingerprint"], e25.protocol_identity(self.meta))
        self.tune_and_judge()
        scored = e25.read(self.directory / "tune_c_scored.json")
        raw = e25.read(self.directory / "tune_c.json")
        self.assertEqual(scored["judge_rubric_id"], QA_GRADER_VERSION)
        self.assertEqual(scored["source_generation_sha256"], e25.digest(self.directory / "tune_c.json"))
        self.assertEqual(scored["header"]["protocol_fingerprint"], raw["protocol_fingerprint"])
        self.assertEqual(scored["n_substantive_gold"], 3)
        # Test-output files cannot inform freeze, even if present and invalid.
        (self.directory / "test_scored.json").write_text("this must never be read")
        real_read = e25.read
        def read_tuning_only(path):
            self.assertFalse(Path(path).name.startswith("test"), "freeze read test outcomes")
            return real_read(path)
        with patch.object(e25, "read", side_effect=read_tuning_only):
            e25.freeze(self.args, self.meta)
        frozen = real_read(self.directory / "selection.json")
        self.assertTrue(all(v is None for v in frozen["selected"].values()))
        selected = [{"name": "baseline", "kind": "baseline"}] + [c for c in frozen["selected"].values() if c is not None]
        self.assertEqual(len(selected), 1)
        e25.generate_questions(self.bundle, self.meta, self.bank, self.prepared["splits"]["test"],
                               selected, self.directory / "test.json", "test")
        result = real_read(self.directory / "test.json")
        self.assertTrue(result["completed"])
        self.assertEqual(result["conditions"], ["baseline"])
        self.assertEqual({r["pair_id"] for r in result["evaluations"]}, sets[2])

    def test_paired_score_first_token_is_in_prompt_mask(self):
        item = encode_answer(self.bundle, self.cfg["qa_system"], "A different question?", "K", self.cfg["qa_user"])
        packed = pack_answers([item], 0, "cpu")
        condition = {"name": "first-token", "kind": "E", "context": "qa", "mode": "continuous", "layer": 0, "alpha": 1.0}
        specs = e25.specs_for(self.meta, self.bank, condition, packed)
        self.assertTrue(specs[0].position_mask[0, item["prompt_length"] - 1])
        self.assertFalse(specs[1].position_mask.any())
        base = forward_logits(self.bundle, packed["inputs"])
        with steer(self.bundle, specs):
            edited = forward_logits(self.bundle, packed["inputs"])
        self.assertEqual(specs[0].n_applied, 1)
        self.assertEqual(specs[1].n_applied, 0)
        self.assertGreater(float((edited[0, item["prompt_length"] - 1] - base[0, item["prompt_length"] - 1]).abs().max()), 0)
        expected = edited[0, item["prompt_length"] - 1].log_softmax(-1)[item["answer_ids"][0]]
        torch.testing.assert_close(answer_logprobs(edited, packed)[0], expected)

    def test_bank_rejects_model_config_rows_and_data_changes(self):
        with self.assertRaisesRegex(ValueError, "wrong model"):
            e25.load_bank(SimpleNamespace(run_dir=self.directory, model="another-model"))
        path = self.directory / "bank.json"
        original = path.read_text()
        for field in ("config", "model_config", "rows"):
            meta = json.loads(original)
            if field == "config":
                meta[field]["qa_system"] += " changed"
            elif field == "model_config":
                meta[field]["name"] = "changed-model"
            else:
                meta[field]["0"]["C_qa"]["gap"] += 1
            self.write("bank.json", meta)
            with self.assertRaisesRegex(ValueError, "changed"):
                e25.load_bank(self.args)
        path.write_text(original)
        with (self.directory / "questions.json").open("a") as f:
            f.write(" ")
        with self.assertRaisesRegex(ValueError, "changed"):
            e25.load_bank(self.args)

    def test_score_and_generation_resume_skip_and_reject_protocol_drift(self):
        echoices, cchoices, paired = self.tune_and_judge()
        forward_count = self.bundle.model.forward_calls
        generation_count = self.bundle.model.generate_calls
        audit_count = len(e25.read(self.directory / "tune_c.json")["audit"])
        e25.score_conditions(self.bundle, self.meta, self.bank, self.prepared["e_tune"],
                             echoices, self.directory / "tune_e_scores.json")
        e25.generate_questions(self.bundle, self.meta, self.bank, self.prepared["splits"]["tune"],
                               cchoices, self.directory / "tune_c.json", "tune", paired)
        self.assertEqual(self.bundle.model.forward_calls, forward_count)
        self.assertEqual(self.bundle.model.generate_calls, generation_count)
        self.assertEqual(len(e25.read(self.directory / "tune_c.json")["audit"]), audit_count)
        altered = copy.deepcopy(self.meta)
        altered["config"]["qa_system"] += " new"
        altered["protocol_fingerprint"] = e25.protocol_identity(altered)
        with self.assertRaisesRegex(ValueError, "resume refused"):
            e25.score_conditions(self.bundle, altered, self.bank, self.prepared["e_tune"],
                                 echoices, self.directory / "tune_e_scores.json")
        with self.assertRaisesRegex(ValueError, "resume refused"):
            e25.generate_questions(self.bundle, altered, self.bank, self.prepared["splits"]["tune"],
                                   cchoices, self.directory / "tune_c.json", "tune", paired)
        raw = e25.read(self.directory / "tune_c.json")
        raw["evaluations"][0]["outputs"]["baseline"] = "drifted answer"
        del raw["evaluations"][1]["outputs"]["baseline"]
        self.write("tune_c.json", raw)
        with self.assertRaisesRegex(RuntimeError, "different existing answer"):
            e25.generate_questions(self.bundle, self.meta, self.bank, self.prepared["splits"]["tune"],
                                   cchoices, self.directory / "tune_c.json", "tune", paired)

    def test_freeze_rejects_foreign_e_scores_old_rubric_and_wrong_generation_hash(self):
        self.tune_and_judge()
        e_original = e25.read(self.directory / "tune_e_scores.json")
        bad = copy.deepcopy(e_original)
        bad["bank_protocol_fingerprint"] = "foreign"
        self.write("tune_e_scores.json", bad)
        with self.assertRaisesRegex(ValueError, "different bank/questions"):
            e25.freeze(self.args, self.meta)
        self.write("tune_e_scores.json", e_original)
        original = e25.read(self.directory / "tune_c_scored.json")
        for key, value, message in (("judge_rubric_id", "old-rubric", "current semantic rubric"),
                                    ("source_generation_sha256", "wrong-hash", "not bound")):
            bad = copy.deepcopy(original)
            bad[key] = value
            self.write("tune_c_scored.json", bad)
            with self.assertRaisesRegex(ValueError, message):
                e25.freeze(self.args, self.meta)
        self.assertFalse((self.directory / "selection.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
