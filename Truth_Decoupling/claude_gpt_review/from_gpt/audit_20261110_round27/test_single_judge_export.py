"""Actual offline single-judge pipeline; synthetic votes, no model/API calls."""
import copy
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
from core.judge import cache_filename
from core.judge_two_axis import CACHE_PREFIX
from test_two_axis_pipeline_r22 import setup, generation, allenai, FakeJudge


class SingleJudgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = R.load_cfg()

    def run_fixture(self, name='run', all_rows=False, empty=False):
        raw = generation()
        if not all_rows:
            raw['evaluations'] = raw['evaluations'][:2]
        if empty:
            raw['evaluations'][0]['outputs'] = {c: '' for c in raw['conditions']}
        run = setup(self.root / name, raw)
        allenai(run, self.cfg)
        R.run_llm(str(run), 'pro', self.cfg, judge=FakeJudge('pro', self.cfg))
        return run

    def test_pro_single_executes_and_counts_explicit_information_axis(self):
        run = self.run_fixture()
        summary, _ = R.single_judge(str(run), 'pro', self.cfg)
        _, _, rows = R.load_verified_single(str(run), 'pro', run / 'tune.json')
        self.assertEqual(rows['q0|baseline']['label'], 'A')
        self.assertEqual(rows['q1|baseline']['label'], 'B')
        self.assertEqual(rows['q1|baseline']['info'], 'N')
        self.assertEqual(rows['q1|baseline']['info_subtype'], 'H')
        self.assertEqual(summary['per_condition']['baseline']['info_rate'], .5)
        self.assertEqual(summary['per_condition']['baseline']['false_noninformative_count'], 1)

    def test_allenai_preserves_binary_info_without_inventing_refusal_subtype(self):
        run = self.run_fixture(all_rows=True)
        R.single_judge(str(run), 'allenai', self.cfg)
        _, _, rows = R.load_verified_single(str(run), 'allenai')
        refusal = rows['q3|baseline']
        self.assertEqual(refusal['label'], 'N')
        self.assertEqual(refusal['info'], 'N')
        self.assertIsNone(refusal['info_subtype'])

    def test_empty_rule_placeholder_is_not_claimed_as_a_pro_vote(self):
        run = self.run_fixture(empty=True)
        summary, _ = R.single_judge(str(run), 'pro', self.cfg)
        _, _, rows = R.load_verified_single(str(run), 'pro', require_complete=False)
        self.assertEqual(rows['q0|baseline']['basis'], 'empty_output_no_judge_vote')
        self.assertIsNone(rows['q0|baseline']['label'])
        self.assertIsNone(rows['q0|baseline']['truth'])
        self.assertEqual(summary['n_no_verdict'], 2)
        self.assertFalse(summary['decisions_complete'])
        self.assertIsNone(summary['per_condition']['baseline']['true_info_rate'])
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            R.load_verified_single(str(run), 'pro')

    def test_empty_is_not_claimed_as_an_allenai_vote(self):
        run = self.run_fixture(empty=True)
        R.single_judge(str(run), 'allenai', self.cfg)
        rows = R.load_verified_single(str(run), 'allenai', require_complete=False)[2]
        self.assertIsNone(rows['q0|baseline']['label'])

    def test_pro_decline_is_not_filled_by_other_judge_or_human(self):
        run = self.run_fixture(all_rows=True)
        (run / 'judge2/gpt.jsonl').write_text('not a vote file\n')
        (run / 'judge2/human_verdicts.jsonl').write_text('not a human file\n')
        summary, _ = R.single_judge(str(run), 'pro', self.cfg)
        rows = R.load_verified_single(str(run), 'pro', require_complete=False)[2]
        self.assertEqual(rows['q2|baseline']['basis'], 'pro_declined')
        self.assertIsNone(rows['q2|baseline']['label'])
        self.assertEqual(summary['n_no_verdict'], 2)
        (run / 'judge2/gpt.jsonl').write_text('still irrelevant\n')
        R.load_verified_single(str(run), 'pro', require_complete=False)

    def test_own_conflicting_cache_replies_remain_unresolved(self):
        shared = self.root / 'shared'
        shared.mkdir()
        with patch.dict(os.environ, {'TD_JUDGE2_CACHE': str(shared)}):
            run = self.run_fixture()
            cell = R._cells(str(run))[0]
            key = R.judge_request_keys(cell, 'pro', self.cfg)[0]
            cache = shared / cache_filename(self.cfg['pro']['model'], prefix=CACHE_PREFIX)
            cache.write_text(''.join(json.dumps({'key': key, 'answer': answer}) + '\n' for answer in ['T', 'F']))
            summary, _ = R.single_judge(str(run), 'pro', self.cfg)
            rows = R.load_verified_single(str(run), 'pro', require_complete=False)[2]
            self.assertEqual(rows[cell['cell_id']]['basis'], 'inconsistent_replies')
            self.assertEqual(summary['n_no_verdict'], 2)

    def test_derived_rows_and_summary_tampering_is_rejected(self):
        run = self.run_fixture()
        R.single_judge(str(run), 'pro', self.cfg)
        path = run / 'judge2/single_pro.jsonl'
        original = path.read_bytes()
        path.write_bytes(original + b'\n')
        with self.assertRaisesRegex(ValueError, 'result or sources changed'):
            R.load_verified_single(str(run), 'pro')
        path.write_bytes(original)
        summary = run / 'judge2/single_pro_summary.json'
        summary.write_text(summary.read_text() + '\n')
        with self.assertRaises(ValueError):
            R.load_verified_single(str(run), 'pro')

    def test_new_external_conflict_invalidates_old_single_export(self):
        shared = self.root / 'shared'
        shared.mkdir()
        with patch.dict(os.environ, {'TD_JUDGE2_CACHE': str(shared)}):
            run = self.run_fixture()
            R.single_judge(str(run), 'pro', self.cfg)
            R.load_verified_single(str(run), 'pro')
            key = R.judge_request_keys(R._cells(str(run))[0], 'pro', self.cfg)[0]
            name = cache_filename(self.cfg['pro']['model'], prefix=CACHE_PREFIX)
            (shared / 'conflicts.json').write_text(json.dumps({'keys': {name: [key]}}))
            with self.assertRaises(ValueError):
                R.load_verified_single(str(run), 'pro')

    def test_another_judges_shared_conflict_does_not_invalidate_this_judge(self):
        shared = self.root / 'shared'
        shared.mkdir()
        with patch.dict(os.environ, {'TD_JUDGE2_CACHE': str(shared)}):
            run = self.run_fixture()
            R.single_judge(str(run), 'pro', self.cfg)
            other = cache_filename(self.cfg['gpt']['model'], prefix=CACHE_PREFIX)
            (shared / 'conflicts.json').write_text(json.dumps({'keys': {other: ['some-other-request']}}))
            R.load_verified_single(str(run), 'pro')

    def test_source_changes_during_export_do_not_receive_valid_manifest(self):
        run = self.run_fixture()
        original = R.judge_own_label
        changed = []
        def racing(*args, **kwargs):
            if not changed:
                path = run / 'judge2/pro.jsonl'
                path.write_text(path.read_text() + '\n')
                changed.append(True)
            return original(*args, **kwargs)
        with patch.object(R, 'judge_own_label', side_effect=racing):
            with self.assertRaisesRegex(RuntimeError, 'changed during export'):
                R.single_judge(str(run), 'pro', self.cfg)
        self.assertFalse((run / 'judge2/single_pro_manifest.json').exists())

    def test_other_generation_and_shared_context_cannot_reuse_export(self):
        run = self.run_fixture()
        R.single_judge(str(run), 'pro', self.cfg)
        other = self.root / 'other.json'
        other.write_text('{}')
        with self.assertRaises(ValueError):
            R.load_verified_single(str(run), 'pro', other)
        with patch.dict(os.environ, {'TD_JUDGE2_CACHE': str(self.root / 'elsewhere')}):
            with self.assertRaisesRegex(ValueError, 'cache context changed'):
                R.load_verified_single(str(run), 'pro')

    def test_same_input_with_different_own_votes_is_not_selected_by_condition(self):
        run = self.run_fixture()
        path = run / 'judge2/pro.jsonl'
        rows = R._rows(path)
        rows['q0|EC'].update(truth='F', truth_raw='F')
        R.write_jsonl(list(rows.values()), str(path))
        meta_path = run / 'judge2/pro_meta.json'
        meta = R._read_json(meta_path)
        meta['rows_sha256'] = R.file_sha(path)
        R.atomic_json_dump(meta, str(meta_path))
        summary, _ = R.single_judge(str(run), 'pro', self.cfg)
        output = R.load_verified_single(str(run), 'pro', require_complete=False)[2]
        self.assertEqual(summary['n_no_verdict'], 2)
        for condition in ['baseline', 'EC']:
            row = output['q0|' + condition]
            self.assertEqual(row['basis'], 'inconsistent_same_input')
            self.assertIsNone(row['label'])

    def test_missing_own_row_is_preserved_as_no_vote(self):
        run = self.run_fixture()
        path = run / 'judge2/pro.jsonl'
        rows = R._rows(path)
        rows.pop('q0|EC')
        R.write_jsonl(list(rows.values()), str(path))
        meta_path = run / 'judge2/pro_meta.json'
        meta = R._read_json(meta_path)
        meta['rows_sha256'] = R.file_sha(path)
        R.atomic_json_dump(meta, str(meta_path))
        summary, _ = R.single_judge(str(run), 'pro', self.cfg)
        output = R.load_verified_single(str(run), 'pro', require_complete=False)[2]
        self.assertEqual(summary['n_no_verdict'], 1)
        self.assertEqual(output['q0|EC']['basis'], 'missing_vote')
        self.assertEqual(output['q0|baseline']['label'], 'A')


if __name__ == '__main__':
    unittest.main()
