"""Exercise pre-test locking and malformed quality inputs, not just happy-path parsing."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REVIEW = Path(__file__).resolve().parents[3]
CODE = REVIEW / 'from_claude/code'
sys.path[:0] = [str(REVIEW / 'from_gpt'), str(CODE / 'scripts'), str(CODE)]
import test_qa_das_pipeline_20261105 as fixtures
import e27_qa_generation as script
from core import qa_generation_adapter as ga
spec = importlib.util.spec_from_file_location('quality', CODE / 'scripts/review_checks/e27_quality_summary.py')
quality = importlib.util.module_from_spec(spec); spec.loader.exec_module(quality)


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = f = fixtures.QADASPipelineTests(methodName='runTest')
        f.setUp(); self.addCleanup(f.doCleanups)
        self.args = SimpleNamespace(run_dir=f.run / 'adapter', layers=[0], pilot=None, init='das', alpha=1.,
                                    epochs=1, lr=.001, clip=1., ranking_weight=0., seed=42, limit=None,
                                    model=f.args.model, judge_exp=script.DEFAULT_JUDGE_EXPERIMENT,
                                    primary_quality_rule='R2')
        self.ns, self.meta, self.bank, self.prepared = script.source_run(f.run)
        self.adapter = script.train(self.args, self.ns, self.meta, self.bank, self.prepared, bundle=f.bundle)

    def tune_and_grade(self):
        raw = script.generate(self.args, self.ns, self.meta, self.bank, self.prepared, self.adapter,
                              'tune', bundle=self.fixture.bundle)
        self.grade(raw, 'tune')

    def grade(self, raw, split):
        grader = fixtures.LocalOutcomeJudge({q['question']: q['outputs']['baseline'] for q in raw['evaluations']})
        with patch.object(fixtures.e24, 'load_model_config', return_value=self.fixture.model_config), \
                patch.object(fixtures.e24, 'out_paths', return_value=(str(self.args.run_dir), 'unused', 'unused')), \
                patch('core.judge.judge_from_config', return_value=grader):
            script.qa.judge(self.args, script.evaluation_meta(self.args, self.meta, self.adapter),
                            split + '.json', split + '_scored')

    def test_test_without_freeze_fails_before_model_load_or_output(self):
        with patch.object(script.qa, 'open_bundle', side_effect=AssertionError('must fail before model load')):
            with self.assertRaisesRegex(ValueError, 'requires freeze'):
                script.generate(self.args, self.ns, self.meta, self.bank, self.prepared, self.adapter, 'test')
        self.assertFalse((self.args.run_dir / 'test.json').exists())

    def test_pilot_training_and_pilot_source_cannot_be_frozen_for_test(self):
        pilot = {**self.adapter, 'pilot': True}
        with self.assertRaisesRegex(ValueError, 'pilot training'):
            script.freeze(self.args, self.meta, self.prepared, pilot)
        with self.assertRaisesRegex(ValueError, 'pilot training'):
            script.freeze(self.args, {**self.meta, 'pilot': True}, self.prepared, self.adapter)

    def test_complete_freeze_executes_then_rejects_mutation_and_posthoc_creation(self):
        self.tune_and_grade()
        frozen = script.freeze(self.args, self.meta, self.prepared, self.adapter)
        self.assertEqual(frozen['primary_quality_rule'], 'R2')
        test = script.generate(self.args, self.ns, self.meta, self.bank, self.prepared, self.adapter,
                               'test', bundle=self.fixture.bundle)
        self.assertEqual(test['pretest_freeze_fingerprint'], frozen['fingerprint'])
        self.assertEqual(len(test['evaluations']), len(self.prepared['splits']['test']))
        path = self.args.run_dir / 'adapter.json'; before = path.read_bytes()
        path.write_bytes(before + b'\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            script.frozen_test_plan(self.args, self.meta, self.prepared, self.adapter)
        path.write_bytes(before)
        (self.args.run_dir / 'selection.json').unlink()
        with self.assertRaisesRegex(ValueError, 'retrospectively'):
            script.freeze(self.args, self.meta, self.prepared, self.adapter)

    def test_bad_quality_plan_or_unresolved_grade_does_not_freeze(self):
        self.tune_and_grade()
        self.args.primary_quality_rule = None
        with self.assertRaisesRegex(ValueError, 'quality-rule'):
            script.freeze(self.args, self.meta, self.prepared, self.adapter)
        self.args.primary_quality_rule = 'R0'
        p = self.args.run_dir / 'tune_scored_items.json'; d = json.loads(p.read_text())
        d['evaluations'][0]['outcome']['baseline'] = 'unresolved'; p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            script.freeze(self.args, self.meta, self.prepared, self.adapter)
        self.assertFalse((self.args.run_dir / 'selection.json').exists())

    def test_training_code_recorded_and_legacy_is_read_without_fabrication(self):
        self.assertIn('core/qa_generation_adapter.py', self.adapter['training_implementation']['files'])
        self.assertTrue(ga.training_description(self.adapter)['uses_correct_answer_training'])
        old = copy.deepcopy(self.adapter)
        old.pop('training_implementation'); old.pop('training_kind')
        directory = self.args.run_dir / 'legacy'; ga.save_adapter(old, directory)
        before = {p.name: p.read_bytes() for p in directory.iterdir()}
        loaded = ga.load_adapter(directory)
        self.assertEqual(ga.training_description(loaded)['training_code_provenance'], 'legacy_unrecorded')
        self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})

    def test_frozen_quality_declaration_is_used_in_summary(self):
        self.tune_and_grade()
        script.freeze(self.args, self.meta, self.prepared, self.adapter)
        raw = script.generate(self.args, self.ns, self.meta, self.bank, self.prepared, self.adapter,
                              'test', bundle=self.fixture.bundle)
        self.grade(raw, 'test')
        result = quality.summarize(self.args.run_dir.parent)
        test = result['runs']['adapter']['test']
        self.assertEqual(test['primary_quality_rule'], 'R2')
        self.assertEqual(test['quality_plan_status'], 'pretest_artifact_recorded')
        for condition in test['conditions'].values():
            self.assertEqual(condition['primary_correct'], condition['R2_correct'])

    def test_missing_and_duplicate_display_indices_never_drop_questions(self):
        raw = script.generate(self.args, self.ns, self.meta, self.bank, self.prepared, self.adapter,
                              'tune', bundle=self.fixture.bundle)
        raw['evaluations'][0].pop('idx')
        for row in raw['evaluations'][1:]: row['idx'] = 0
        raw.pop('intervention_position')
        path = self.args.run_dir / 'tune.json'; path.write_text(json.dumps(raw))
        before = path.read_bytes()
        self.grade(raw, 'tune')
        scored = json.loads((self.args.run_dir / 'tune_scored_items.json').read_text())
        self.assertEqual(len(scored['evaluations']), len(raw['evaluations']))
        self.assertEqual([r['idx'] for r in scored['evaluations']], list(range(len(raw['evaluations']))))
        self.assertEqual(len(scored['grading_row_index_normalization']['changes']), len(raw['evaluations']))
        self.assertEqual(path.read_bytes(), before)
        # Resume an old failed grading copy that also lacked display indices.
        pending = self.args.run_dir / 'tune_scored_grading_input.json'
        data = json.loads(pending.read_text())
        for row in data['evaluations']: row.pop('idx')
        pending.write_text(json.dumps(data))
        self.grade(raw, 'tune')
        resumed = json.loads((self.args.run_dir / 'tune_scored_items.json').read_text())
        self.assertEqual(len(resumed['evaluations']), len(raw['evaluations']))
        self.assertEqual(path.read_bytes(), before)

    def test_duplicate_question_id_is_rejected_instead_of_deduplicated(self):
        raw = script.generate(self.args, self.ns, self.meta, self.bank, self.prepared, self.adapter,
                              'tune', bundle=self.fixture.bundle)
        raw['evaluations'][1]['pair_id'] = raw['evaluations'][0]['pair_id']
        (self.args.run_dir / 'tune.json').write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, 'unique nonempty'):
            self.grade(raw, 'tune')

    def test_reused_control_grade_cannot_silently_change(self):
        raw = script.generate(self.args, self.ns, self.meta, self.bank, self.prepared, self.adapter,
                              'tune', bundle=self.fixture.bundle)
        # This local fixture labels baseline as wrong; a forged prior correct
        # grade must fail before a complete comparison report is published.
        self.args.frozen_control_grades = {r['pair_id']: {'baseline': {'label': 'correct',
                                            'rubric_id': script.qa.QA_GRADER_VERSION}} for r in raw['evaluations']}
        with self.assertRaisesRegex(ValueError, 'different grade'):
            self.grade(raw, 'tune')
        self.assertFalse((self.args.run_dir / 'tune_scored.json').exists())


class QualityTests(unittest.TestCase):
    def test_zero_discordance_is_one_and_mismatched_ids_are_rejected(self):
        self.assertEqual(quality.sign_p(0, 0), 1.)
        with self.assertRaisesRegex(ValueError, 'identical'):
            quality.paired({'a': 'correct'}, {'a': 'wrong', 'b': 'correct'})

    def test_duplicate_missing_partition_and_incomplete_inputs_fail(self):
        row = {'pair_id': 'a', 'question': 'Q', 'gold': ['A'], 'gold_is_abstention': False,
               'outputs': {'baseline': 'A'}, 'finished': {'baseline': True},
               'outcome': {'baseline': 'correct'},
               'judge': {'baseline': {'label': 'correct', 'rubric_id': 'v'}}}
        d = {'model': 'toy', 'split': 'test', 'completed': True, 'grading_completed': True,
             'conditions': ['baseline'], 'judge_rubric_id': 'v', 'evaluations': [row]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.json'; path.write_text(json.dumps(d))
            d['grading_source_generation_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            quality.validate_scored(d, Path(directory), 'test')
            for mutate, error in ((lambda x: x['evaluations'].append(copy.deepcopy(row)), 'unique'),
                                  (lambda x: x['evaluations'][0].pop('gold_is_abstention'), 'partition'),
                                  (lambda x: x.update(grading_completed=False), 'completed')):
                bad = copy.deepcopy(d); mutate(bad)
                with self.assertRaisesRegex(ValueError, error): quality.validate_scored(bad, Path(directory), 'test')


if __name__ == '__main__':
    unittest.main()
