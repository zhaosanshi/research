"""Exploratory EC+simple supplement: immutable sources and executable collection."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / 'from_claude/code'
sys.path[:0] = [str(HERE), str(CODE / 'scripts'), str(CODE)]
import e26_joint_simple_extension as extension
import test_e26_pipeline_20261106 as fixture


class ExtensionValidationTests(unittest.TestCase):
    def setUp(self):
        self.q = [{'id': 'q:t', 'pair_id': 'q', 'label': True}, {'id': 'q:f', 'pair_id': 'q', 'label': False}]
        self.rows = [{'id': q['id'], 'pair_id': 'q', 'score': 1., 'simple_score': .5,
                      'predicted_true': q['label'], 'output': 'True' if q['label'] else 'False',
                      'outcome': 'correct'} for q in self.q]

    def test_complete_and_partial_cohort_are_distinct(self):
        extension.validate_records(self.rows[:1], self.q, 'tf', partial=True)
        with self.assertRaises(ValueError):
            extension.validate_records(self.rows[:1], self.q, 'tf')

    def test_wrong_group_rejected(self):
        self.rows[0]['pair_id'] = 'foreign'
        with self.assertRaises(ValueError):
            extension.validate_records(self.rows, self.q, 'tf')

    def test_outcome_must_follow_fixed_tf_label(self):
        self.rows[0]['outcome'] = 'wrong'
        with self.assertRaises(ValueError):
            extension.validate_records(self.rows, self.q, 'tf')

    def test_nonfinite_score_rejected(self):
        self.rows[0]['simple_score'] = float('nan')
        with self.assertRaises(ValueError):
            extension.validate_records(self.rows, self.q, 'tf')

    def test_duplicate_fixed_cohort_rejected(self):
        with self.assertRaises(ValueError):
            extension.validate_records(self.rows[:1] * 2, self.q[:1] * 2, 'tf')

    def test_real_toy_collection_grading_resume_and_identity_guards(self):
        f = fixture.E26IntegrationTests(methodName='test_complete_real_inference_judging_freeze_report_and_actual_deliveries')
        f.setUp(); self.addCleanup(f.doCleanups)
        meta, bank, data = f.prepare()
        f.tune(meta, bank, data)
        extension.joint.collect(f.args, meta, bank, data, 'test', bundle=f.bundle)
        extension.joint.judge_qa(f.args, meta, data, 'test', judge=fixture.LocalJudge())
        original_hashes = {p.name: extension.joint.digest(p) for p in f.args.run_dir.iterdir() if p.is_file()}
        frozen = {'model': f.args.model, 'sources': {n: original_hashes[n] for n in
                  ('bank.json', 'bank.npz', 'data.json', 'selection.json', 'tf_tune.json', 'qa_tune_scored.json')},
                  'bank_fingerprint': meta['fingerprint'], 'tasks': {}}
        for task in extension.joint.TASKS:
            rows, _ = extension.joint.scored_rows(f.args, meta, task, 'tune')
            scores = {n: extension.joint.score_variant(rs, 'simple', 42) for n, rs in rows.items()}
            choices = extension.choose_conditions(scores, meta['conditions'][task], meta['config']['coverages'], 42)
            old, filename = extension.joint.scored_rows(f.args, meta, task, 'test')
            needed = {v for p in choices.values() for k, v in p.items() if k.endswith('_EC') and v is not None} - set(old)
            frozen['tasks'][task] = {
                'choices': {q: {m + '_EC_simple': choices[q][m + '_EC'] for m in ('das', 'md')} for q in choices},
                'missing_condition_specs': [c for c in meta['conditions'][task] if c['name'] in needed],
                'existing_test_source_sha256': extension.joint.digest(f.args.run_dir / filename)}
        manifest = {'protocol': extension.PROTOCOL, 'exploratory': True, 'runs': [frozen]}
        manifest['fingerprint'] = extension.fingerprint(manifest)
        args = SimpleNamespace(model=f.args.model, source_run=f.args.run_dir, run_dir=f.args.run_dir / 'extension',
            manifest=f.args.run_dir / 'external_manifest.json', limit=2, cache_dir=None)
        extension.joint.save(args.manifest, manifest)
        extension.verify(args)
        wrong = deepcopy(manifest)
        wrong['runs'][0]['tasks']['qa']['choices']['0.5']['das_EC_simple'] = 'foreign'
        wrong['fingerprint'] = extension.fingerprint({k: v for k, v in wrong.items() if k != 'fingerprint'})
        extension.joint.save(args.manifest, wrong)
        with self.assertRaisesRegex(ValueError, 'tuning-selected'):
            extension.verify(args)
        extension.joint.save(args.manifest, manifest)
        # Exercise a genuinely new forward, not only the empty-supplement case.
        # This isolated component fixture supplies valid EC operators directly;
        # manifest selection verification above is tested independently.
        forward_fixture = deepcopy(frozen)
        for task in extension.joint.TASKS:
            forward_fixture['tasks'][task]['missing_condition_specs'] = [next(c for c in meta['conditions'][task] if c['kind'] == 'EC')]
        extension.collect(args, forward_fixture, meta, bank, data, bundle=f.bundle)
        with patch.object(extension, 'load_experiment', return_value={'judge': {}}):
            extension.judge(args, forward_fixture, meta, data, judge=fixture.LocalJudge())
            path = args.run_dir / 'qa_test_extension_scored.json'
            scored = extension.joint.read(path)
            self.assertTrue(scored['completed'])
            scored['bank_fingerprint'] = 'foreign'
            extension.joint.save(path, scored)
            with self.assertRaisesRegex(ValueError, 'grading identity'):
                extension.judge(args, forward_fixture, meta, data, judge=fixture.LocalJudge())
        for name, sha in original_hashes.items():
            self.assertEqual(extension.joint.digest(f.args.run_dir / name), sha)


if __name__ == '__main__':
    unittest.main()
