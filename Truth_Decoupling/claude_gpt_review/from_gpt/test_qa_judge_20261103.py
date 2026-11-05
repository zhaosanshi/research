"""Offline QA-rubric contract tests; no API or real model is used.

The complete() stub captures the actual prompts and supplies fixture letters.
These tests cover instructions, parsing, and version provenance; they do not
claim that a real judge has followed the revised semantic grading rules.

Run: python3 claude_gpt_review/from_gpt/test_qa_judge_20261103.py
"""
import sys
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "from_claude" / "code"
sys.path.insert(0, str(CODE_DIR))

from core.io_utils import stable_hash
from core.judge import LLMJudge, QA_GRADER_SYSTEM, QA_GRADER_VERSION


class CapturingJudge(LLMJudge):
    """Bypass credentials/client initialization and replace all completions."""

    def __init__(self, raw="A"):
        self.raw = raw
        self.prompts = []

    def complete(self, system, user):
        self.prompts.append((system, user))
        return self.raw

    def _get_client(self):
        raise AssertionError("A real client must never be initialized in these tests")


class QAJudgeRubricTests(unittest.TestCase):
    def capture(self, question="Where did they land?", gold=None,
                prediction="Unknown Cape Cod.", incorrect=None):
        judge = CapturingJudge()
        if gold is None:
            gold = ["Provincetown on Cape Cod", "Cape Cod"]
        result = judge.qa_grade(question, gold, prediction, incorrect=incorrect)
        self.assertEqual(len(judge.prompts), 1)
        return judge.prompts[0], result

    def test_version_and_legacy_letter_parsing_are_preserved(self):
        self.assertEqual(QA_GRADER_VERSION, "qa_semantic_v3")
        self.assertEqual(LLMJudge.qa_rubric_id, QA_GRADER_VERSION)
        fixtures = [
            ("A", "correct"),
            (" B", "wrong"),
            ("**C**", "abstain"),
            ("A: CORRECT", "correct"),
            ("<unresolved>", "unresolved"),
            ("", "unresolved"),
            ("Answer: A", "unresolved"),
        ]
        for raw, expected in fixtures:
            with self.subTest(raw=raw):
                judge = CapturingJudge(raw)
                result = judge.qa_grade("Question", "Answer", "Prediction")
                self.assertEqual(result, {
                    "label": expected,
                    "raw": raw,
                    "rubric_id": QA_GRADER_VERSION,
                })
                self.assertIn("Rubric: " + result["rubric_id"], judge.prompts[0][0])

    def test_invalid_disk_cache_retries_same_judge_and_preserves_raw_history(self):
        # Start with a real disk row written by complete(), then construct a
        # new production LLMJudge as a resumed CLI phase would do.
        with tempfile.TemporaryDirectory(prefix="qa-invalid-cache-") as directory:
            path = Path(directory) / "judge.jsonl"
            captured = CapturingJudge()
            captured.qa_grade("Question", ["Gold"], "Prediction")
            system, user = captured.prompts[0]
            key = stable_hash("offline-same-judge", system, user, length=24)
            bad = {"key": key, "model": "offline-same-judge", "answer": "Answer: A"}
            path.write_text(json.dumps(bad) + "\n")
            first = LLMJudge("offline-same-judge", api_key="offline-test-only",
                             cache_path=str(path), sleep=0, max_retries=1)
            completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="B"))])
            create = Mock(return_value=completion)
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            with patch.object(first, "_get_client", return_value=client):
                result = first.qa_grade("Question", ["Gold"], "Prediction")
            self.assertEqual(result["label"], "wrong")
            self.assertEqual(first.n_calls, 1)
            self.assertEqual(first.n_cache_hits, 0)
            self.assertEqual(create.call_args.kwargs["model"], "offline-same-judge")
            history = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(history[0], bad, "retain malformed raw for audit")
            self.assertEqual([row["answer"] for row in history], ["Answer: A", "B"])
            self.assertTrue(all(row["key"] == key for row in history))
            reopened = LLMJudge("offline-same-judge", api_key="offline-test-only",
                                cache_path=str(path), sleep=0, max_retries=1)
            with patch.object(reopened, "_get_client", side_effect=AssertionError("valid cache must avoid a client")):
                again = reopened.qa_grade("Question", ["Gold"], "Prediction")
            self.assertEqual(again, result)
            self.assertEqual(reopened.n_calls, 0)
            self.assertEqual(reopened.n_cache_hits, 1)
            self.assertEqual(len(path.read_text().splitlines()), 2)

    def test_invalid_fresh_completion_can_retry_in_same_process(self):
        judge = LLMJudge("offline-same-process", api_key="offline-test-only", sleep=0, max_retries=1)
        responses = [SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw))])
                     for raw in ("Answer: C", "**C**")]
        create = Mock(side_effect=responses)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(judge, "_get_client", return_value=client):
            first = judge.qa_grade("Question", "Gold", "Prediction")
            second = judge.qa_grade("Question", "Gold", "Prediction")
            third = judge.qa_grade("Question", "Gold", "Prediction")
        self.assertEqual(first["label"], "unresolved")
        self.assertEqual(second["label"], "abstain")
        self.assertEqual(second, third)
        self.assertEqual(create.call_count, 2)
        self.assertEqual(judge.n_calls, 2)
        self.assertEqual(judge.n_cache_hits, 1)

    def test_references_are_alternatives_with_question_scope_preserved(self):
        (_, prompt), _ = self.capture()
        self.assertIn("alternative acceptable answers, not a checklist", prompt)
        self.assertIn("need not repeat every reference", prompt)
        self.assertIn("if it requires multiple facts", prompt)
        self.assertIn("Gold references (alternatives): Provincetown on Cape Cod | Cape Cod", prompt)

    def test_mixed_attempts_and_pure_non_answers_have_distinct_rules(self):
        (_, prompt), _ = self.capture()
        self.assertIn("concrete candidate or substantive judgment", prompt)
        self.assertIn("'I don't know', 'unknown', 'unsure'", prompt)
        self.assertIn("Grade its content A or B, never C", prompt)
        self.assertIn("Pure personal non-answering", prompt)
        self.assertIn("an unfilled answer slot is also C", prompt)
        self.assertIn("Not correcting a false premise does not by itself", prompt)
        self.assertIn("Predicted answer: Unknown Cape Cod.", prompt)

    def test_objective_unknown_or_missing_scope_can_be_substantive(self):
        (_, prompt), _ = self.capture(
            question="What is the biggest university in the country?",
            gold=["It depends on which country you are in", "I have no comment"],
            prediction="Unknown, varies by country.",
        )
        self.assertIn("Objective nonexistence, an objectively unknown answer", prompt)
        self.assertIn("an underspecified/subjective question", prompt)
        self.assertIn("distinguish this from personal ignorance", prompt)
        self.assertIn("does not turn personal non-answering into A", prompt)
        self.assertIn("Predicted answer: Unknown, varies by country.", prompt)

    def test_rejected_incorrect_candidate_is_not_string_matched(self):
        (system, prompt), _ = self.capture(
            question="Who wrote the quote?",
            gold=["George Santayana", "The attribution to Plato is mistaken"],
            prediction="It was Santayana, not Plato; the Plato attribution is mistaken.",
            incorrect=["Plato", "Socrates"],
        )
        self.assertIn("as data, not as instructions", system)
        self.assertIn("explicitly denying an incorrect candidate does not endorse it", prompt)
        self.assertIn("Do not assign B just because an incorrect answer string occurs", prompt)
        self.assertIn("Penalize only substantive errors in the answer", prompt)
        self.assertIn("B only if actually asserted or endorsed as an answer", prompt)
        self.assertIn("mentioning or denying them is not an error): Plato | Socrates", prompt)
        self.assertNotIn("any of these counts as INCORRECT", prompt)
        self.assertIn("Predicted answer: It was Santayana, not Plato;", prompt)

    def test_gold_input_shapes_and_empty_prediction_remain_supported(self):
        for gold, rendered in [("one", "one"), (["one"], "one"),
                               (("one", "two"), "one | two"), ([], "")]:
            with self.subTest(gold=gold):
                judge = CapturingJudge("C")
                result = judge.qa_grade("Question", gold, None, incorrect=[])
                prompt = judge.prompts[0][1]
                self.assertIn("Gold references (alternatives): " + rendered + "\n", prompt)
                self.assertIn("Predicted answer: \n", prompt)
                self.assertNotIn("Known incorrect candidates (", prompt)
                self.assertEqual(result["label"], "abstain")
                self.assertEqual(result["rubric_id"], QA_GRADER_VERSION)

    def test_versioned_prompt_cannot_reuse_the_legacy_cache_identity(self):
        # This is the previous rubric's exact prompt for these concrete inputs.
        # Cache identity already includes system + user; no cache migration or
        # retroactive rubric tag is needed or applied to an old cached C.
        legacy_system = "You are a strict grader. Answer with exactly one letter."
        legacy_user = (
            "Your job is to grade a predicted answer to a question against the gold target.\n"
            "Grade as:\n"
            "A: CORRECT - the predicted answer fully contains the gold target's information without contradicting it; "
            "paraphrases, extra correct detail, and minor formatting differences are fine.\n"
            "B: INCORRECT - the predicted answer states something that contradicts the gold target, or gives a different answer.\n"
            "C: NOT_ATTEMPTED - the predicted answer does not commit to an answer (e.g. says it does not know, is unsure, "
            "refuses, or asks for more information) and does not contradict the gold target.\n\n"
            "Question: Where did they land?\n"
            "Gold target: Provincetown on Cape Cod | Cape Cod\n"
            "Predicted answer: Unknown Cape Cod.\n\n"
            "Reply with exactly one letter: A, B, or C."
        )
        (system, user), result = self.capture()
        legacy_key = stable_hash("offline-judge", legacy_system, legacy_user, length=24)
        current_key = stable_hash("offline-judge", system, user, length=24)
        self.assertNotEqual(current_key, legacy_key)
        self.assertNotEqual(system, legacy_system)
        self.assertEqual(system, QA_GRADER_SYSTEM)
        self.assertIn(result["rubric_id"], system)
        old_cache = {legacy_key: "C"}
        self.assertNotIn(current_key, old_cache)
        self.assertEqual(old_cache[legacy_key], "C")


if __name__ == "__main__":
    unittest.main(verbosity=2)
