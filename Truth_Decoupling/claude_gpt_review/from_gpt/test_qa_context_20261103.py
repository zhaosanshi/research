#!/usr/bin/env python3
"""CPU unit checks for E25 answer contexts. No model downloads or API calls.

Run: python3 claude_gpt_review/from_gpt/test_qa_context_20261103.py
"""
from pathlib import Path
import sys
import unittest

import torch

CODE = Path(__file__).resolve().parent.parent / "from_claude" / "code"
sys.path.insert(0, str(CODE))
from core.qa_context import answer_logprobs, encode_answer, pack_answers, prepare_e_pairs


class CharacterTokenizer:
    all_special_ids = [0, 1, 2]

    def __call__(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("automatic special tokens must be disabled")
        # A reserved character stands for an explicitly inserted special token.
        return {"input_ids": [2 if c == "§" else ord(c) + 3 for c in text]}


class FakeBundle:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or CharacterTokenizer()

    def build_chat(self, system, user, add_generation_prompt=True):
        if not add_generation_prompt:
            raise AssertionError("must render a generation prompt")
        return "S:" + system + "\nU:" + user + "\nA:"


class RetokenizingTokenizer(CharacterTokenizer):
    def __call__(self, text, add_special_tokens=False):
        ids = super().__call__(text, add_special_tokens)["input_ids"]
        if text.endswith(":x"):
            ids = ids[:-2] + [999]  # boundary-spanning token, not an intact prefix
        return {"input_ids": ids}


def question(pid, gold, incorrect, best_answer=None):
    return {"pair_id": pid, "question": "Where?", "gold": gold,
            "incorrect": incorrect, "best_answer": best_answer}


class PairPreparationTests(unittest.TestCase):
    def test_best_answer_then_gold_order_and_negative_dedup(self):
        source = question("q1", ["Paris", "Lyon"], [" PARIS. ", "London", " london. "], "Paris")
        pairs, audit = prepare_e_pairs([source])
        self.assertEqual(pairs[0]["positive_answer"], "Paris")
        self.assertEqual(pairs[0]["negative_answer"], "London")
        self.assertEqual(audit["records"][0]["positive_source"], "best_answer")
        statuses = [x["status"] for x in audit["records"][0]["negative_candidates"]]
        self.assertEqual(statuses, ["same_as_positive", "selected", "duplicate"])
        self.assertNotIn("positive_answer", source)
        self.assertEqual(pairs[0]["pair_id"], "q1")

    def test_refusal_positive_is_replaced_and_empty_questions_not_backfilled(self):
        rows = [
            question("keep", ["Unknown", "I don't know.", "Paris", "Rome"], ["London"], "I have no comment."),
            question("drop-positive", ["Unknown.", "Uncertain", "I do not know"], ["London"]),
            question("drop-negative", ["Paris"], ["Paris.", "Unknown", "I don't know"]),
        ]
        pairs, audit = prepare_e_pairs(rows)
        self.assertEqual([p["pair_id"] for p in pairs], ["keep"])
        self.assertEqual(pairs[0]["positive_answer"], "Paris")
        self.assertEqual(audit["n_input"], 3)
        self.assertEqual(audit["n_retained"], 1)
        self.assertEqual(audit["excluded_ids"], ["drop-positive", "drop-negative"])
        self.assertEqual(audit["excluded"][0]["reasons"], ["no_substantive_positive"])
        self.assertEqual(audit["excluded"][1]["reasons"], ["no_substantive_distinct_negative"])

    def test_objectively_unknown_fact_is_retained(self):
        pairs, _ = prepare_e_pairs([question("q", ["The origin is unknown."], ["It originated in Paris."])])
        self.assertEqual(pairs[0]["positive_answer"], "The origin is unknown.")

    def test_missing_and_empty_answers_are_audited(self):
        pairs, audit = prepare_e_pairs([question("q", ["  "], [])])
        self.assertEqual(pairs, [])
        self.assertEqual(audit["excluded_ids"], ["q"])
        self.assertEqual(set(audit["excluded"][0]["reasons"]),
                         {"no_substantive_positive", "no_substantive_distinct_negative"})
        self.assertEqual(prepare_e_pairs([])[1]["n_input"], 0)

    def test_duplicate_ids_fail_loudly(self):
        row = question("same", ["Paris"], ["London"])
        with self.assertRaisesRegex(ValueError, "duplicate pair_id"):
            prepare_e_pairs([row, dict(row)])


class AnswerEncodingTests(unittest.TestCase):
    def test_raw_answer_has_no_extra_template_tail(self):
        item = encode_answer(FakeBundle(), "system", "Where?", " Paris.")
        self.assertEqual(item["full_text"], item["prefix_text"] + " Paris.")
        self.assertEqual(item["answer_text"], " Paris.")
        self.assertEqual(item["full_ids"], item["prefix_ids"] + item["answer_ids"])
        self.assertEqual(item["answer_ids"], [ord(c) + 3 for c in " Paris."])
        self.assertFalse(item["text_audit"]["appended_eos"])

    def test_empty_answer_and_special_answer_tokens_rejected(self):
        for answer in ("", "   ", "\n"):
            with self.assertRaisesRegex(ValueError, "nonempty"):
                encode_answer(FakeBundle(), "s", "q", answer)
        with self.assertRaisesRegex(ValueError, "special token"):
            encode_answer(FakeBundle(), "s", "q", "§")

    def test_prefix_retokenization_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "retokenizes"):
            encode_answer(FakeBundle(RetokenizingTokenizer()), "s", "q", "x")


class PackingAndScoringTests(unittest.TestCase):
    def setUp(self):
        bundle = FakeBundle()
        self.encoded = [encode_answer(bundle, "s", "q", "x"),
                        encode_answer(bundle, "s", "longer question", "yz")]
        self.packed = pack_answers(self.encoded, pad_token_id=0, device="cpu")

    def test_left_padding_and_predictor_shift(self):
        packed = self.packed
        width = packed["inputs"]["input_ids"].shape[1]
        for row, item in enumerate(self.encoded):
            p, m = item["prompt_length"], item["answer_length"]
            start = width - p - m
            self.assertEqual(packed["inputs"]["input_ids"][row, :start].tolist(), [0] * start)
            self.assertEqual(packed["inputs"]["attention_mask"][row].tolist(), [0] * start + [1] * (p + m))
            self.assertEqual(packed["target_mask"][row].nonzero().flatten().tolist(), list(range(start + p, start + p + m)))
            self.assertEqual(packed["predictor_mask"][row].nonzero().flatten().tolist(), list(range(start + p - 1, start + p + m - 1)))
            self.assertEqual(packed["prompt_end_mask"][row].nonzero().flatten().tolist(), [start + p - 1])
            self.assertEqual(packed["answer_mask"][row].tolist(), packed["target_mask"][row].tolist())

    def test_one_and_multi_token_summed_scores_match_manual_reference(self):
        torch.manual_seed(4)
        ids = self.packed["inputs"]["input_ids"]
        logits = torch.randn(*ids.shape, 256, dtype=torch.float64, requires_grad=True)
        scores = answer_logprobs(logits, self.packed)
        expected = []
        for row in range(len(self.encoded)):
            positions = self.packed["target_mask"][row].nonzero().flatten().tolist()
            expected.append(sum(torch.log_softmax(logits[row, j - 1], dim=-1)[ids[row, j]] for j in positions))
        torch.testing.assert_close(scores, torch.stack(expected))
        scores.sum().backward()
        touched = logits.grad.abs().sum(-1) > 0
        self.assertTrue(torch.equal(touched, self.packed["predictor_mask"]))
        self.assertTrue(torch.equal(touched[0], self.packed["prompt_end_mask"][0]))

    def test_prompt_padding_and_final_answer_logits_do_not_count(self):
        torch.manual_seed(8)
        ids = self.packed["inputs"]["input_ids"]
        logits = torch.randn(*ids.shape, 256)
        changed = logits.clone()
        changed[~self.packed["predictor_mask"]] = 100 * torch.randn_like(changed[~self.packed["predictor_mask"]])
        torch.testing.assert_close(answer_logprobs(logits, self.packed), answer_logprobs(changed, self.packed))

    def test_empty_batch_and_inconsistent_spans_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            pack_answers([], 0, "cpu")
        broken = dict(self.encoded[0], answer_length=2)
        with self.assertRaisesRegex(ValueError, "disagree"):
            pack_answers([broken], 0, "cpu")


if __name__ == "__main__":
    unittest.main(verbosity=2)
