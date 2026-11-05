#!/usr/bin/env python3
"""CPU-only QA-DAS tests: frozen random causal network, no weights/API download."""
import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

CODE = Path(__file__).resolve().parent.parent / "from_claude" / "code"
sys.path.insert(0, str(CODE))
import core.qa_das as qa
from core.directions import OrthoBasis, load_das_direction
from core.model_io import ModelBundle


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    all_special_ids = [0, 1, 2]

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        # Standalone label tokenization is deliberately different. The module
        # must derive label increments in suffix context instead of using this.
        if text == " True":
            return {"input_ids": [150]}
        if text == " False":
            return {"input_ids": [151]}
        return {"input_ids": [2 if c == "§" else ord(c) + 3 for c in text]}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize and add_generation_prompt
        return "".join("<%s>%s\n" % (m["role"], m["content"]) for m in messages) + "<assistant>"


class BoundaryTokenizer(Tokenizer):
    def __call__(self, text, add_special_tokens=False):
        ids = super().__call__(text, add_special_tokens)["input_ids"]
        if text.endswith(": True"):
            ids = ids[:-6] + [155]  # consumes the final colon of the suffix
        return {"input_ids": ids}


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mix = nn.Linear(4, 4, bias=False)

    def forward(self, h):
        count = torch.arange(1, h.shape[1] + 1, device=h.device).view(1, -1, 1)
        return h + torch.tanh(self.mix(h.cumsum(1) / count))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, num_hidden_layers=3)
        self.embedding = nn.Embedding(160, 4)
        self.blocks = nn.ModuleList([Block(), Block(), Block()])
        self.head = nn.Linear(4, 160, bias=False)
        self.calls = 0

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False,
                logits_to_keep=0):
        self.calls += 1
        h = self.embedding(input_ids)
        for block in self.blocks:
            h = block(h)
        if logits_to_keep:
            h = h[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=self.head(h))


def make_bundle(tokenizer=None):
    torch.manual_seed(7)
    model = Model().cpu().eval()
    return ModelBundle(model, tokenizer or Tokenizer(), {"name": "cpu-qa-test", "layers_attr": "blocks"})


CFG = {"qa_system": "Answer briefly.", "qa_user": "Question: {question}\nAnswer:",
       "selfcheck_suffix": "\nIs the answer correct?\nAnswer:",
       "selfcheck_label_true": " True", "selfcheck_label_false": " False"}
RECORDS = [{"pair_id": "p1", "question": "Where?", "positive_answer": "Paris", "negative_answer": "New York"},
           {"pair_id": "p2", "question": "Which city?", "positive_answer": "Rome", "negative_answer": "Berlin"}]


class EncodingTests(unittest.TestCase):
    def test_fixed_answer_prefix_contextual_labels_and_indices(self):
        bundle = make_bundle()
        pairs = qa.prepare_selfcheck_pairs(bundle, CFG, RECORDS)
        self.assertEqual([p["pair_id"] for p in pairs], ["p1", "p2"])
        self.assertNotEqual(pairs[0]["positive"]["answer_end"], pairs[0]["negative"]["answer_end"])
        for pair in pairs:
            for side in ("positive", "negative"):
                item = pair[side]
                full = item["inputs"]["input_ids"][0].tolist()
                original = item["encoded"]["full_ids"]
                self.assertEqual(full, original + item["suffix_ids"])
                self.assertEqual(item["answer_end"], len(original) - 1)
                self.assertEqual(item["label_predictor"], len(full) - 1)
                self.assertGreater(item["label_predictor"], item["answer_end"])
                self.assertEqual(len(item["label_ids"]["true"]), 5)
                self.assertEqual(len(item["label_ids"]["false"]), 6)
                self.assertNotEqual(item["label_ids"]["true"], bundle.tokenizer(" True")["input_ids"])
                self.assertFalse(item["text_audit"]["labels_in_inputs"])
                self.assertFalse(item["text_audit"]["appended_eos"])

    def test_bad_suffix_label_boundaries_and_duplicate_ids_fail(self):
        args = ("s", "q", "Paris", "Q: {question}\nA:", "Check:")
        with self.assertRaisesRegex(ValueError, "retokenizes"):
            qa.encode_selfcheck(make_bundle(BoundaryTokenizer()), *args)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            qa.encode_selfcheck(make_bundle(), "s", "q", "Paris", "Q: {question}", "")
        with self.assertRaisesRegex(ValueError, "special"):
            qa.encode_selfcheck(make_bundle(), "s", "q", "Paris", "Q: {question}", "§")
        with self.assertRaisesRegex(ValueError, "identically"):
            qa.encode_selfcheck(make_bundle(), *args, label_true=" True", label_false=" True")
        with self.assertRaisesRegex(ValueError, "unique"):
            qa.prepare_selfcheck_pairs(make_bundle(), CFG, [RECORDS[0], RECORDS[0]])


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.bundle = make_bundle()
        self.bundle.model.requires_grad_(False)
        self.pairs = qa.prepare_selfcheck_pairs(self.bundle, CFG, RECORDS)

    def test_all_label_tokens_use_full_vocabulary_ce(self):
        item = self.pairs[0]["negative"]
        seen = []
        def uniform(bundle, inputs, grad=False):
            seen.append(inputs["input_ids"][0].tolist())
            return torch.zeros(1, 160, requires_grad=grad)
        trace = []
        with patch.object(qa, "_forward_last", side_effect=uniform):
            loss, count = qa._label_loss(self.bundle, item, "false", 0, grad=True, trace=trace)
        self.assertEqual(count, 6)
        self.assertAlmostEqual(float(loss.detach()), math.log(160), places=6)
        self.assertEqual(len(seen), count)
        for i, ids in enumerate(seen):
            self.assertEqual(ids, item["input_ids"] + item["label_ids"]["false"][:i])
            self.assertEqual(trace[i]["current_label_predictor"], item["label_predictor"] + i)

    def test_gradient_numeric_and_source_detach_at_different_lengths(self):
        states = qa._source_states(self.bundle, self.pairs, 0)
        source = states["positive"][0].clone().requires_grad_(True)
        item = self.pairs[0]["negative"]
        basis = OrthoBasis(4, 1, seed=42)
        edits = []
        original_factory = qa._write_hook
        def recorder(W_fn, position, **kwargs):
            hook = original_factory(W_fn, position, **kwargs)
            def wrapped(module, args, output):
                result = hook(module, args, output)
                changed = ((result - output).abs().amax(-1)[0] > 1e-7).nonzero().flatten().tolist()
                edits.append(changed)
                return result
            return wrapped
        with patch.object(qa, "_write_hook", side_effect=recorder):
            loss, count = qa._label_loss(self.bundle, item, "true", 0, basis, source=source, grad=True)
        loss.backward()
        self.assertIsNone(source.grad)
        self.assertGreater(float(basis.raw.grad.norm()), 0)
        self.assertTrue(all(e == [item["answer_end"]] for e in edits))
        # Observation must not alter either the intervention or its W gradient.
        observed_basis = OrthoBasis(4, 1, seed=42)
        trace = []
        observed_loss, _ = qa._label_loss(self.bundle, item, "true", 0, observed_basis,
                                          source=source, grad=True, trace=trace)
        observed_loss.backward()
        torch.testing.assert_close(observed_loss, loss, rtol=0, atol=0)
        torch.testing.assert_close(observed_basis.raw.grad, basis.raw.grad, rtol=0, atol=0)
        self.assertEqual(len(trace), count)
        self.assertTrue(all(row["kept_delta_norm"] > 0 for row in trace))
        index = int(basis.raw.grad.abs().argmax())
        analytic = float(basis.raw.grad.flatten()[index])
        values = []
        for sign in (1, -1):
            altered = OrthoBasis(4, 1, seed=42)
            with torch.no_grad():
                altered.raw.flatten()[index] += sign * .003
            value, _ = qa._label_loss(self.bundle, item, "true", 0, altered, source=source, grad=True)
            values.append(float(value.detach()))
        numeric = (values[0] - values[1]) / .006
        self.assertLess(abs(numeric - analytic), 3e-4)
        self.assertEqual(len(self.bundle.layer(0)._forward_hooks), 0)

    def test_training_updates_only_direction_cache_equivalence_and_identity(self):
        frozen = [p.clone() for p in self.bundle.model.parameters()]
        self.bundle.model.requires_grad_(True)  # trainer is responsible for freezing it
        cache = qa._source_states(self.bundle, self.pairs, 0)
        result = qa.train_qa_das(self.bundle, self.pairs, 0, epochs=2, log_fn=None,
                                 identity={"dataset": "synthetic", "model_revision": "unit-test"})
        again = qa.train_qa_das(self.bundle, self.pairs, 0, epochs=2, log_fn=None,
                                source_activations=cache,
                                identity={"dataset": "synthetic", "model_revision": "unit-test"})
        torch.testing.assert_close(result["W"], again["W"], rtol=0, atol=0)
        self.assertFalse(torch.allclose(result["W"], OrthoBasis(4, 1, seed=42).detached()))
        for original, current in zip(frozen, self.bundle.model.parameters()):
            self.assertTrue(torch.equal(original, current))
            self.assertFalse(current.requires_grad)
            self.assertIsNone(current.grad)
        self.assertEqual(result["training_pair_ids"], ["p1", "p2"])
        self.assertEqual(result["dataset"], "synthetic")
        self.assertEqual(len(result["selfcheck_inputs_sha256"]), 64)
        self.assertEqual(result["per_seed"]["optimizer_updates"], 4)
        self.assertGreater(result["per_seed"]["gradient_norm_max"], 0)
        trace = result["position_trace"]
        self.assertEqual(len(trace), 11)
        self.assertEqual({r["pair_id"] for r in trace}, {"p1"})
        self.assertTrue(all(r["h_norm"] > 0 and r["intended_delta_norm"] > 0 for r in trace))
        self.assertTrue(all(r["source_coordinate"] is not None and r["center_coordinate"] is None for r in trace))
        for name, base, source in (("negative_to_positive", "negative", "positive"),
                                   ("positive_to_negative", "positive", "negative")):
            rows = [r for r in trace if r["direction"] == name]
            self.assertTrue(all(r["base_answer_end"] == self.pairs[0][base]["answer_end"] for r in rows))
            self.assertTrue(all(r["source_answer_end"] == self.pairs[0][source]["answer_end"] for r in rows))
        with self.assertRaisesRegex(ValueError, "identity disagrees"):
            qa.train_qa_das(self.bundle, self.pairs, 0, epochs=1, log_fn=None, identity={"rank": 2})

    def test_last_block_has_zero_label_path_and_training_refuses(self):
        layer = self.bundle.num_layers - 1
        calls = self.bundle.model.calls
        with self.assertRaisesRegex(ValueError, "structural null"):
            qa.train_qa_das(self.bundle, self.pairs, layer, log_fn=None)
        self.assertEqual(self.bundle.model.calls, calls)
        states = qa._source_states(self.bundle, self.pairs, layer)
        basis = OrthoBasis(4, 1, seed=42)
        loss, _ = qa._label_loss(self.bundle, self.pairs[0]["negative"], "true", layer,
                                 basis, source=states["positive"][0], grad=True)
        loss.backward()
        self.assertEqual(float(basis.raw.grad.norm()), 0)

    def test_diagnostics_mark_oracle_and_self_swap_matches_baseline(self):
        w = OrthoBasis(4, 1, seed=42).detached()
        result = qa.evaluate_selfcheck(self.bundle, self.pairs, w, 0, torch.zeros(4), 0.)
        self.assertFalse(result["eligible_for_selection"])
        for pair in result["pairs"]:
            for name in ("negative_to_positive", "positive_to_negative"):
                direction = pair[name]
                for label in ("true", "false"):
                    base = direction["baseline"]["logp_sum"][label]
                    self.assertAlmostEqual(base, direction["self_swap"]["logp_sum"][label], places=6)
                    self.assertEqual(base, direction["amplify"]["logp_sum"][label])
                    expected_length = len(self.pairs[0]["positive"]["label_ids"][label])
                    for condition in ("baseline", "self_swap", "cross_swap", "amplify"):
                        audit = direction[condition]["write_audit"][label]
                        self.assertEqual(len(audit), expected_length)
                        self.assertEqual([r["label_token_index"] for r in audit], list(range(expected_length)))
                        self.assertTrue(all(r["h_norm"] > 0 for r in audit))
                        for row in audit:
                            self.assertEqual(row["residual_dtype"], "torch.float32")
                            if condition in ("baseline", "amplify"):
                                self.assertEqual(row["intended_delta_norm"], 0)
                                self.assertEqual(row["kept_delta_norm"], 0)
                                self.assertEqual(row["coordinate_before"], row["coordinate_after"])
                            elif condition == "self_swap":
                                self.assertLess(row["intended_delta_norm"], 1e-6)
                                self.assertLess(row["kept_delta_norm"], 1e-6)
                            else:
                                self.assertGreater(row["intended_delta_norm"], 0)
                                self.assertGreater(row["kept_delta_norm"], 0)
                                self.assertAlmostEqual(row["coordinate_after"], row["source_coordinate"], places=5)
                                self.assertAlmostEqual(row["kept_coordinate_delta"],
                                                       row["source_coordinate"] - row["coordinate_before"], places=5)
                self.assertNotEqual(direction["cross_swap"]["expected_label"], direction["baseline"]["expected_label"])
        terminal = qa.evaluate_selfcheck(self.bundle, self.pairs[:1], w, 2, torch.zeros(4), 3.)
        self.assertTrue(terminal["structural_null"])
        for direction in ("negative_to_positive", "positive_to_negative"):
            row = terminal["pairs"][0][direction]
            self.assertEqual(row["baseline"]["logp_sum"], row["cross_swap"]["logp_sum"])
            for audit in row["amplify"]["write_audit"].values():
                for entry in audit:
                    self.assertEqual(entry["center_coordinate"], 0)
                    self.assertAlmostEqual(entry["kept_coordinate_delta"],
                                           3 * entry["coordinate_before"], places=5)

    def test_write_audit_measures_actual_low_precision_round_trip(self):
        # BF16 rounds 1 + 0.001 back to 1: a nonzero intended edit is lost.
        h = torch.ones(1, 2, 4, dtype=torch.bfloat16)
        w = torch.tensor([[1.], [0.], [0.], [0.]])
        audit = []
        output = (h, "unchanged tuple member")
        changed = qa._write_hook(lambda: w, 0, center=torch.zeros(4), alpha=.001,
                                 audit=audit)(None, (), output)
        self.assertEqual(changed[1], output[1])
        self.assertTrue(torch.equal(changed[0], h))
        self.assertEqual(len(audit), 1)
        self.assertAlmostEqual(audit[0]["intended_delta_norm"], .001, places=7)
        self.assertEqual(audit[0]["kept_delta_norm"], 0)
        self.assertEqual(audit[0]["coordinate_before"], 1)
        self.assertEqual(audit[0]["coordinate_after"], 1)
        self.assertEqual(audit[0]["residual_dtype"], "torch.bfloat16")
        passive = []
        observed = qa._write_hook(lambda: w, 0, audit=passive, apply=False)(None, (), output)
        self.assertIs(observed, output)
        self.assertEqual(passive[0]["operation"], "baseline")
        self.assertEqual(passive[0]["kept_delta_norm"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
