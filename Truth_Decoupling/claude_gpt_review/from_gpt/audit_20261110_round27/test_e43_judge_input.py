"""Real freeze/train/generate/export stages using deterministic offline model stubs."""
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parent.parent
CODE = REVIEW / 'from_claude/code'
sys.path[:0] = [str(HERE), str(CODE), str(CODE / 'scripts')]
import e43_qa_prompt_das as D
import e43_export_judge as X
from scripts.review_checks import judge_two_axis_run as R
import test_driver_bindings as driver_fixture


class JudgeInputTests(unittest.TestCase):
    def setUp(self):
        self.fixture = driver_fixture.DriverBindingsTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.train()
        f = self.fixture
        D.run_generate(f.args, f.meta, f.prepared, f.bundle, f.systems, f.ident, f.sig)
        self.path = f.run / 'judge_tune/judge_input.json'

    def export(self, **kwargs):
        return X.export(self.fixture.source, self.fixture.run, **kwargs)

    def test_complete_export_to_actual_judge_cells_is_idempotent(self):
        report = self.export()
        original = self.path.read_bytes()
        self.assertEqual(report['n_questions'], 1)
        self.assertEqual(len(report['conditions']), 6)
        self.assertEqual(self.export(), report)
        self.assertEqual(self.path.read_bytes(), original)
        meta = R.build_cells(str(self.path.parent), generation=self.path.name, audit_n=0)
        self.assertEqual(meta['n_cells'], 6)
        rows = R._cells(str(self.path.parent))
        self.assertEqual({row['prediction'] for row in rows}, {'Answer'})
        self.assertEqual({row['question'] for row in rows}, {'Question 3?'})

    def test_changed_source_question_or_reference_is_rejected(self):
        f = self.fixture
        changed = copy.deepcopy(f.prepared)
        changed['splits']['tune'][0]['question'] = 'A different question'
        (f.source / 'questions.json').write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'references changed|different question'):
            self.export()
        self.assertFalse(self.path.exists())

    def test_extra_generation_question_cannot_be_silently_dropped(self):
        path = self.fixture.run / 'gen/E_L0_main_tune.json'
        data = D.read(path)
        extra = copy.deepcopy(data['rows'][0])
        extra['pair_id'] = 'foreign'
        data['rows'].append(extra)
        D.save(path, D.seal(data))
        with self.assertRaisesRegex(ValueError, 'cohort'):
            self.export()

    def test_wrong_baseline_cannot_be_exported_as_a_paired_comparison(self):
        path = self.fixture.run / 'gen/E_L0_main_tune.json'
        data = D.read(path)
        data['rows'][0]['baseline_text'] = 'Another baseline'
        D.save(path, D.seal(data))
        with self.assertRaisesRegex(ValueError, 'comparison baseline'):
            self.export()

    def test_changed_direction_file_is_rejected(self):
        path = self.fixture.run / 'dirs/E_L0.pt'
        path.write_bytes(path.read_bytes() + b'extra')
        with self.assertRaisesRegex(ValueError, 'source file changed'):
            self.export()

    def test_pilot_is_ignored_by_default_and_rejected_when_explicit(self):
        f = self.fixture
        pilot = f.run / 'gen/E_L0_main_tune.pilot.json'
        data = D.read(f.run / 'gen/E_L0_main_tune.json')
        data.update(pilot=True, completed=False)
        D.save(pilot, D.seal(data))
        self.assertEqual(len(self.export()['conditions']), 6)
        with self.assertRaisesRegex(ValueError, 'pilot/incomplete'):
            self.export(gen_names=[pilot.name], name='pilot_attempt')

    def test_export_path_escape_and_duplicate_selection_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'simple file stem'):
            self.export(name='../outside')
        with self.assertRaisesRegex(ValueError, 'inside this run'):
            self.export(gen_names=['../other.json'])
        with self.assertRaisesRegex(ValueError, 'more than once'):
            self.export(gen_names=['E_L0_main_tune.json'] * 2)

    def test_existing_different_judge_input_is_not_overwritten(self):
        self.export()
        self.path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'existing judge input differs'):
            self.export()
        self.assertEqual(self.path.read_text(), '{}')

    def test_source_change_during_export_is_rejected_before_publication(self):
        original = R._derive_cells
        def racing(payload):
            path = self.fixture.run / 'frozen_tune.json'
            path.write_text(path.read_text() + '\n')
            return original(payload)
        with patch.object(R, '_derive_cells', side_effect=racing):
            with self.assertRaisesRegex(RuntimeError, 'changed during judge export'):
                self.export()
        self.assertFalse(self.path.exists())

    def test_frozen_and_generation_identity_strings_are_not_sufficient(self):
        path = self.fixture.run / 'frozen_tune.json'
        data = D.read(path)
        data['identity']['model'] = 'changed model'
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'unbound or changed'):
            self.export()


if __name__ == '__main__':
    unittest.main()
