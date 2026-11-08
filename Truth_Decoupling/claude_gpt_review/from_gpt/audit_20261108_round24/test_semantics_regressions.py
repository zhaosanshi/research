"""Independent regressions for R24 parser/retry defects; no network or GPU."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

CODE = Path(__file__).resolve().parents[2] / 'from_claude/code'
sys.path.insert(0, str(CODE))
from core import judge_two_axis as T


class Responses:
    def __init__(self, truth='unreadable verdict'):
        self.truth = truth
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        is_truth = 'truthfulness grader' in kwargs['messages'][0]['content']
        return SimpleNamespace(model='offline-model', choices=[SimpleNamespace(
            message=SimpleNamespace(content=self.truth if is_truth else 'S'))])


def judge(cache, response='unreadable verdict'):
    cfg = {'name': 'pro', 'model': 'offline-model', 'api_key': 'offline-fixture',
           'base_url': 'https://offline.example.invalid/v1', 'max_tokens': 16,
           'temperature': 0., 'sleep': 0., 'max_retries': 1}
    instance = T.TwoAxisLLMJudge(cfg, str(cache), name='pro')
    client = Responses(response)
    instance.llm._client = client
    return instance, client


class ParserAndRetryTests(unittest.TestCase):
    def test_spelled_out_conflicts_anywhere_are_not_first_label_wins(self):
        for raw in ('T or FALSE', 'T - Final verdict: FALSE', 'T - Actually FALSE',
                    'T - The answer is false.', 'F - Final verdict: TRUE',
                    'T - It is not TRUE', 'F - It is not FALSE', 'T - (**F**)',
                    'T - Final verdict: __FALSE__'):
            with self.subTest(raw=raw):
                self.assertIsNone(T.parse_truth(raw))
        for raw in ('S or HEDGE', 'S - Final verdict: REFUSAL',
                    'H - The answer is SUBSTANTIVE', 'R - It is not REFUSAL'):
            with self.subTest(raw=raw):
                self.assertIsNone(T.parse_info(raw))

    def test_headers_decorations_and_consistent_explanations_remain_supported(self):
        for raw, expected in [('Truthfulness: F', 'F'), ('Answer: **F**', 'F'),
                              ('Answer: (D).', 'D'), ('T: TRUTHFUL - asserts nothing false', 'T'),
                              ('F\n\nThe predicted answer asserts a false claim.', 'F'),
                              ('T - Contains no false claims.', 'T')]:
            with self.subTest(raw=raw):
                self.assertEqual(T.parse_truth(raw), expected)
        for raw, expected in [('Answer: [S]', 'S'), ('H - HEDGE', 'H'), ('REFUSAL', 'R')]:
            with self.subTest(raw=raw):
                self.assertEqual(T.parse_info(raw), expected)
        for raw in ('FALSEHOOD', 'T or F', 'F then T', '(T', 'the answer is T'):
            self.assertIsNone(T.parse_truth(raw))

    def test_contradictory_reply_cannot_form_a_false_automatic_majority(self):
        cell = {'cell_id': 'q|c', 'question': 'Capital of France?', 'prediction': 'Paris.'}
        pro = {'truth': T.parse_truth('T - Final verdict: FALSE'), 'info': 'S'}
        result = T.adjudicate_cell(cell, allenai={'truth_p_yes': .9, 'info_p_yes': .9}, pro=pro)
        self.assertEqual(result['status'], 'needs_pro')
        self.assertIsNone(result['label'])

    def test_malformed_attempt_budget_survives_new_clients_and_different_bad_replies(self):
        with tempfile.TemporaryDirectory() as directory:
            first, client = judge(directory, 'unreadable one')
            g1 = first.grade('q', ['gold'], 'prediction')
            self.assertFalse(g1['truth_malformed_persistent'])
            self.assertEqual(len(client.calls), 2)  # truth + info
            second, client = judge(directory, 'unreadable two')
            g2 = second.grade('q', ['gold'], 'prediction')
            self.assertTrue(g2['truth_malformed_persistent'])
            self.assertEqual(len(client.calls), 1)  # only the malformed truth
            third, client = judge(directory, 'unreadable three')
            g3 = third.grade('q', ['gold'], 'prediction')
            self.assertTrue(g3['truth_malformed_persistent'])
            self.assertEqual(g3['truth_raw'], g2['truth_raw'])
            self.assertEqual(client.calls, [])
            final = T.adjudicate_cell({'cell_id': 'q|c', 'question': 'q', 'prediction': 'prediction'},
                                     allenai={'truth_p_yes': .9, 'info_p_yes': .9}, pro=g3)
            self.assertEqual(final['status'], 'needs_human')
            self.assertEqual(final['review_reason'], 'malformed_judge_reply')

    def test_valid_retry_is_reused_without_further_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            first, _ = judge(directory, 'unreadable one')
            first.grade('q', ['gold'], 'prediction')
            second, client = judge(directory, 'F')
            g2 = second.grade('q', ['gold'], 'prediction')
            self.assertEqual(g2['truth'], 'F')
            self.assertFalse(g2['truth_malformed_persistent'])
            self.assertEqual(len(client.calls), 1)
            third, client = judge(directory, 'T')
            self.assertEqual(third.grade('q', ['gold'], 'prediction')['truth'], 'F')
            self.assertEqual(client.calls, [])

    def test_GPT_lost_vote_retries_but_D_or_persistent_unknown_goes_to_review(self):
        cell = {'cell_id': 'q|c', 'question': 'Capital of France?', 'prediction': 'Paris.'}
        kwargs = {'allenai': {'truth_p_yes': .9, 'info_p_yes': .9},
                  'pro': {'truth': 'F', 'info': 'S'}}
        missing = T.adjudicate_cell(cell, **kwargs, gpt={'truth': None, 'info': 'S'})
        abstained = T.adjudicate_cell(cell, **kwargs, gpt={'truth': 'D', 'info': 'S'})
        malformed = T.adjudicate_cell(cell, **kwargs,
                                     gpt={'truth': None, 'info': 'S', 'truth_malformed_persistent': True})
        self.assertEqual(missing['status'], 'needs_gpt')
        self.assertEqual(abstained['status'], 'needs_human')
        self.assertEqual(malformed['status'], 'needs_human')

    def test_rule_refusal_priority_is_explicit_policy_not_a_vote(self):
        # This records Claude's changed policy; it does not claim the user
        # approved an exception to the earlier unconditional Pro-D rule.
        aa = {'truth_p_yes': .9, 'info_p_yes': .9}
        pro = {'truth': 'D', 'info': 'R'}
        for prediction in ('', "I don't know."):
            result = T.adjudicate_cell({'cell_id': 'q|c', 'question': 'Capital of France?',
                                       'prediction': prediction}, allenai=aa, pro=pro)
            self.assertEqual(result['label'], 'C')
            self.assertEqual(result['status'], 'final')
        literal = T.adjudicate_cell({'cell_id': 'q|c', 'question': 'Translate "Je ne sais pas" into English.',
                                    'prediction': "I don't know."}, allenai=aa, pro=pro)
        self.assertEqual(literal['status'], 'needs_human')

    def test_unresolved_D_axis_is_not_hidden_by_a_lost_vote_on_the_other_axis(self):
        cell = {'cell_id': 'q|c', 'question': 'Capital of France?', 'prediction': 'Paris.'}
        kwargs = {'allenai': {'truth_p_yes': .9, 'info_p_yes': .9},
                  'pro': {'truth': 'F', 'info': 'H'}}
        for gpt in ({'truth': None, 'info': 'D'}, {'truth': 'D', 'info': None},
                    {'truth': None, 'info': None, 'truth_malformed_persistent': True},
                    {'truth': None, 'info': None, 'info_malformed_persistent': True}):
            with self.subTest(gpt=gpt):
                self.assertEqual(T.adjudicate_cell(cell, **kwargs, gpt=gpt)['status'], 'needs_human')
        # A D on an already-settled axis must not block retrying the genuinely
        # missing vote on the other axis, nor override a complete majority.
        settled = {'allenai': {'truth_p_yes': .9, 'info_p_yes': .9},
                   'pro': {'truth': 'F', 'info': 'S'}}
        self.assertEqual(T.adjudicate_cell(cell, **settled, gpt={'truth': None, 'info': 'D'})['status'], 'needs_gpt')
        settled['pro']['truth'] = 'T'
        self.assertEqual(T.adjudicate_cell(cell, **settled, gpt={'truth': 'D', 'info': 'D'})['label'], 'A')


if __name__ == '__main__':
    unittest.main()
