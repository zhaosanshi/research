import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parents[2]
CODE = REVIEW/'from_claude/code'
sys.path[:0] = [str(REVIEW/'from_gpt'), str(CODE/'scripts/review_checks'), str(CODE/'scripts'), str(CODE)]
import e27_generation_dose as dose
import e27_qa_generation as e27
import test_qa_das_pipeline_20261105 as fixtures


class GenerationDoseTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.QADASPipelineTests(methodName='runTest')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.ns, self.meta, self.bank, self.prepared = e27.source_run(f.run)
        self.adapter_args = SimpleNamespace(run_dir=f.run/'adapter', layers=[0], pilot=2, init='das', alpha=3.,
            epochs=1, lr=.001, clip=1., ranking_weight=0., seed=42, limit=None, model=f.args.model,
            judge_exp='E4_qa', include_eos=False, stop_token_id=None)
        self.adapter = e27.train(self.adapter_args, self.ns, self.meta, self.bank, self.prepared, bundle=f.bundle)
        e27.generate(self.adapter_args, self.ns, self.meta, self.bank, self.prepared, self.adapter, 'tune', bundle=f.bundle)
        ids = [r['pair_id'] for r in self.prepared['splits']['tune']]
        self.args = SimpleNamespace(run_dir=f.run/'diagnostic', adapter_run=self.adapter_args.run_dir,
            pair_ids=[ids[0], ids[-1]], all_tune=False, model=f.args.model, judge_exp='E4_qa')

    def generate(self):
        return dose.generate(self.args, self.ns, self.meta, self.bank, self.prepared, self.adapter, bundle=self.fixture.bundle)

    def test_five_arms_keep_training_identity_and_only_override_runtime_alpha(self):
        saved = copy.deepcopy(self.adapter)
        specs = dose.condition_specs(self.adapter)
        self.assertEqual(len(specs), 5)
        for c in specs:
            runtime = dose.runtime_specs(self.adapter, self.bank, c)
            if c['name'] != 'baseline':
                self.assertEqual(runtime[0].alpha, c['deployment_alpha'])
                self.assertTrue(torch.equal(runtime[0].center, saved['centers']['0']['prompt']))
                self.assertTrue(torch.equal(runtime[0].decode_center, saved['centers']['0']['decode']))
            if c.get('context') == 'generation_adapter':
                self.assertEqual(c['training_alpha'], 3.)
                self.assertTrue(torch.equal(runtime[0].W[:, 0], saved['W']['0']))
        self.assertEqual(self.adapter['alpha'], 3.)
        self.assertEqual(self.adapter['train_config'], saved['train_config'])
        self.assertTrue(torch.equal(self.adapter['W']['0'], saved['W']['0']))

    def test_real_generation_reproduces_reference_and_saves_original_batches_ids_and_actual_edits(self):
        data = self.generate()
        self.assertTrue(data['completed']); self.assertTrue(data['reference_reproduced'])
        self.assertTrue(data['diagnostic_experiment']); self.assertFalse(data['diagnostic_only'])
        self.assertEqual(len(data['evaluations']), 2)
        self.assertEqual([r['pair_id'] for r in data['evaluations']], self.args.pair_ids)
        for row in data['evaluations']:
            self.assertEqual(set(row['outputs']), set(data['conditions']))
            self.assertTrue(all(row['reference_matches'].values()))
            self.assertEqual(set(row['raw_token_ids']), set(data['conditions']))
        self.assertTrue(any(len(a['original_batch_pair_ids']) > len(a['selected_rows']) for a in data['audit']))
        self.assertEqual(self.generate(), data)
        self.fixture.assert_question_only_generation()

    def test_no_test_cohort_and_explicit_all_tune_is_separate_artifact(self):
        with self.assertRaises(ValueError):
            dose.target_questions(self.prepared, [self.prepared['splits']['test'][0]['pair_id']])
        self.generate()
        self.args.all_tune = True; self.args.pair_ids = None
        all_data = self.generate()
        self.assertEqual(len(all_data['evaluations']), len(self.prepared['splits']['tune']))
        self.assertTrue((self.args.run_dir/'pilot.json').is_file())
        self.assertTrue((self.args.run_dir/'tune.json').is_file())

    def test_true_judging_report_path_with_local_api_replacement(self):
        raw = self.generate()
        grader = fixtures.LocalOutcomeJudge({r['question']:r['outputs']['baseline'] for r in raw['evaluations']})
        with patch('core.judge.judge_from_config', return_value=grader), \
                patch.object(fixtures.e24, 'load_model_config', return_value=self.fixture.model_config), \
                patch.object(fixtures.e24, 'out_paths', return_value=(str(self.args.run_dir), 'unused', 'unused')):
            dose.judge(self.args, self.ns, self.meta, self.prepared, self.adapter)
        s=json.loads((self.args.run_dir/'pilot_scored.json').read_text())
        self.assertTrue(s['completed']);self.assertEqual(len(s['rates']), 5)
        md=(self.args.run_dir/'pilot_scored.md').read_text()
        self.assertNotIn('含金标来源的条件不是可部署性能', md)
        self.assertIn('小样本线路检查', md)


if __name__=='__main__':
    unittest.main()
