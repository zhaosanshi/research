"""Real offline pipeline tests for external sources and cross-run acceptance."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REVIEW = Path(__file__).resolve().parents[2]
CODE = REVIEW / 'from_claude/code'
sys.path[:0] = [str(CODE), str(CODE / 'scripts/review_checks'),
               str(REVIEW / 'from_gpt/audit_20261108_round22/pipeline')]
import judge_two_axis_run as R
import judge2_cross_run_check as X
from core.judge import cache_filename
from core.judge_two_axis import CACHE_PREFIX
from test_two_axis_pipeline_r22 import setup, allenai, FakeJudge


class SharedFinalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = R.load_cfg()

    def graded(self, name, complete=True):
        run = setup(self.root / name)
        allenai(run, self.cfg)
        R.run_llm(str(run), 'pro', self.cfg, judge=FakeJudge('pro', self.cfg))
        R.compute_disputes(str(run), self.cfg)
        R.run_llm(str(run), 'gpt', self.cfg, only='disputes', judge=FakeJudge('gpt', self.cfg))
        if complete:
            R.human_set(str(run), 'q2|EC', 'A', 'synthetic-test-only')
        R.adjudicate(str(run), self.cfg)
        return run

    def test_shared_human_change_invalidates_final_even_without_environment(self):
        shared = self.root / 'shared_human.jsonl'
        with patch.dict(os.environ, {'TD_JUDGE2_HUMAN': str(shared)}):
            run = self.graded('a')
            R.load_verified_final(str(run), run / 'tune.json')
            # Only the external file changes; the run's own human/vote files stay identical.
            rows = [json.loads(line) for line in shared.read_text().splitlines()]
            rows[0].update(label='B', truth='F', info='S', info_subtype='S')
            shared.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with self.assertRaisesRegex(ValueError, 'source files changed'):
            R.load_verified_final(str(run), run / 'tune.json')

    def test_new_shared_conflict_report_invalidates_old_final(self):
        shared = self.root / 'cache'
        shared.mkdir()
        with patch.dict(os.environ, {'TD_JUDGE2_CACHE': str(shared)}):
            run = self.graded('a')
            cell = R._cells(str(run))[0]
            key = R.judge_request_keys(cell, 'pro', self.cfg)[0]
            (shared / 'conflicts.json').write_text(json.dumps({'keys': {'pro': [key]}}))
            with self.assertRaisesRegex(ValueError, 'source files changed'):
                R.load_verified_final(str(run), run / 'tune.json')
            summary, _ = R.adjudicate(str(run), self.cfg)
            self.assertEqual(summary['n_needs_human'], 2)

    def test_actual_new_cache_conflict_is_seen_without_merge_report(self):
        shared = self.root / 'cache'
        shared.mkdir()
        with patch.dict(os.environ, {'TD_JUDGE2_CACHE': str(shared)}):
            run = self.graded('a')
            cell = R._cells(str(run))[0]
            key = R.judge_request_keys(cell, 'pro', self.cfg)[0]
            cache = shared / cache_filename(self.cfg['pro']['model'], prefix=CACHE_PREFIX)
            cache.write_text('\n'.join(json.dumps({'key': key, 'answer': answer}) for answer in ['T', 'F']) + '\n')
            with self.assertRaises(ValueError):
                R.load_verified_final(str(run), run / 'tune.json')
            summary, _ = R.adjudicate(str(run), self.cfg)
            self.assertEqual(summary['n_needs_human'], 2)
            final = R._rows(run / 'judge2/final.jsonl')[cell['cell_id']]
            self.assertEqual(final['review_reason'], 'inconsistent_replies:pro')

    def test_mid_aggregation_shared_human_change_cannot_be_bound_as_used(self):
        shared = self.root / 'human.jsonl'
        with patch.dict(os.environ, {'TD_JUDGE2_HUMAN': str(shared)}):
            run = self.graded('a')
            original = R.adjudicate_cell
            changed = []
            def racing(*args, **kwargs):
                if not changed:
                    shared.write_text(shared.read_text() + '\n')
                    changed.append(True)
                return original(*args, **kwargs)
            with patch.object(R, 'adjudicate_cell', side_effect=racing):
                with self.assertRaisesRegex(RuntimeError, 'changed during adjudication'):
                    R.adjudicate(str(run), self.cfg)

    def test_changed_shared_context_is_not_silently_ignored(self):
        run = self.graded('a')
        with patch.dict(os.environ, {'TD_JUDGE2_HUMAN': str(self.root / 'new_human.jsonl')}):
            with self.assertRaisesRegex(ValueError, 'context changed'):
                R.load_verified_final(str(run), run / 'tune.json')

    def test_complete_verified_runs_pass_but_duplicate_run_does_not(self):
        a, b = self.graded('a'), self.graded('b')
        report = X.check([str(a), str(b)])
        self.assertTrue(report['verified_consistent'])
        self.assertEqual(report['n_inputs_shared'], 4)
        with self.assertRaisesRegex(ValueError, 'distinct'):
            X.check([str(a), str(a)])

    def test_incomplete_runs_cannot_pass_even_in_diagnostic_mode(self):
        a, b = self.graded('a'), self.graded('b', complete=False)
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            X.check([str(a), str(b)])
        report = X.check([str(a), str(b)], allow_unresolved=True)
        self.assertFalse(report['verified_consistent'])
        self.assertEqual(report['n_unresolved_cells'], 2)
        self.assertEqual(report['n_partially_resolved'], 1)

    def test_missing_or_modified_final_does_not_become_zero_conflicts(self):
        a, b = self.graded('a'), self.graded('b')
        (b / 'judge2/final.jsonl').write_text('')
        with self.assertRaises(ValueError):
            X.check([str(a), str(b)])

    def test_complete_conflicting_verdicts_fail(self):
        a, b = self.graded('a'), self.graded('b')
        R.human_set(str(b), 'q0|EC', 'B', 'synthetic-test-only', info_subtype='S')
        R.adjudicate(str(b), self.cfg)
        report = X.check([str(a), str(b)])
        self.assertFalse(report['verified_consistent'])
        self.assertEqual(report['n_conflicting_finals'], 1)


if __name__ == '__main__':
    unittest.main()
