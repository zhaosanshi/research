"""E26 offline integration: real toy-network inference and fixed local grading.

Creates an actual E25 source bank/training/shortlist, imports saved synthetic TF
directions, then runs E26 prepare -> collect -> grade -> freeze -> test -> report.
The grader deliberately supplies synthetic outcomes for plumbing; none of these
numbers describe model quality. All writes are temporary; network/API forbidden.
"""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "from_claude/code"
sys.path[:0] = [str(HERE), str(CODE / "scripts"), str(CODE)]
import test_qa_das_pipeline_20261105 as source_fixture
import e26_matched_coverage as joint
import e27_qa_generation as generation
from core.config import load_experiment as actual_load_experiment
from core.judge import QA_GRADER_VERSION


class LocalJudge:
    name = "e26-offline-fixed-truth-table"
    qa_rubric_id = QA_GRADER_VERSION

    def __init__(self, refusal_questions=(), unresolved_question=None):
        self.refusals = set(refusal_questions)
        self.unresolved_question = unresolved_question
        self.calls = []

    def qa_grade(self, question, gold, prediction, incorrect=None):
        self.calls.append((question, prediction))
        label = ("unresolved" if question == self.unresolved_question else
                 "abstain" if question in self.refusals else "correct")
        return {"label": label, "raw": "synthetic offline grade", "rubric_id": self.qa_rubric_id}


class ThreeLayerPipelineModel(source_fixture.PipelineModel):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.layers.append(deepcopy(self.layers[-1]))
        self.config.num_hidden_layers = 3


class E26IntegrationTests(unittest.TestCase):
    def enter_context(self, cm):
        result = cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        return result

    def setUp(self):
        if self._testMethodName == 'test_multilayer_E27_train_export_import_collect_freeze':
            original_estimate = source_fixture.pipeline.estimate
            def estimate_all(args, cfg, bundle=None):
                args.layers, args.all_layers = None, True
                return original_estimate(args, cfg, bundle=bundle)
            self.enter_context(patch.object(source_fixture, 'PipelineModel', ThreeLayerPipelineModel))
            self.enter_context(patch.object(source_fixture.pipeline, 'estimate', side_effect=estimate_all))
        self.source = source_fixture.QADASPipelineTests(methodName="test_real_end_to_end_training_selection_check_and_diagnostics")
        self.source.setUp()
        self.addCleanup(self.source.doCleanups)
        src = self.source
        src_pipeline = source_fixture.pipeline
        src_pipeline.score(src.args, src.meta, src.bank, src.prepared, bundle=src.bundle)
        self.bundle, self.model = src.bundle, src.model
        self.args = SimpleNamespace(run_dir=src.run / "joint", model=src.args.model, qa_run=src.run,
                                    tf_das_dir=src.run / "tf_das", tf_das_sweep=src.run / "das_sweep.json",
                                    tf_md_sweep=src.run / "md_sweep.json", pilot=None,
                                    judge_exp="synthetic_joint", cache_dir=None)
        self.cfg = joint.read(CODE / "configs/experiments/E26_matched_coverage.json")
        self.cfg.update(models=[self.args.model], source_experiment="synthetic_joint", n_boot=8,
                        max_new_tokens=2, batch_size=2, e_top_k=1, c_top_k=1,
                        coverages=[.5, 1.], c_alphas=[-.5, .5])
        self.exp = {"eval_datasets": ["synthetic_tf"], "eval_split": "test", "judge": {},
                    "prompts": {"statement": {"system": "Judge factual claims.",
                        "user_tf": "Statement: {statement}\nTrue or False?",
                        "label_variants": {"True": ["True"], "False": ["False"]}}}}
        self.tf = {split: [{"id": q['pair_id'] + (":t" if label else ":f"),
                           "pair_id": q['pair_id'], "label": label,
                           "text": "Claim %s is %s." % (q['pair_id'], "valid" if label else "false")}
                          for q in src.prepared['splits'][split] for label in (True, False)]
                   for split in ('train', 'tune', 'test')}
        physical = {"calib": self.tf['train'][:4], "dev": self.tf['train'][4:],
                    "test": self.tf['tune'] + self.tf['test']}
        self.enter_context(patch.object(joint, "load_experiment", return_value=self.exp))
        self.enter_context(patch.object(joint, "load_split", side_effect=lambda ds, split: deepcopy(physical[split])))
        self.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self._make_tf_sources()

    def _make_tf_sources(self):
        self.args.tf_das_dir.mkdir()
        train_n = len(self.source.prepared['splits']['train'])
        joint.save(self.args.tf_das_dir / 'summary.json', {
            'model': self.args.model, 'num_layers': self.bundle.num_layers,
            'das': {'variant': 'tf-ce-bidir', 'loss': 'ce', 'bidirectional': True,
                    'positions': 'last', 'anchor': 'prompt',
                    'n_pairs': {'E': train_n}, 'E_trained_layers': list(range(self.bundle.num_layers))}})
        for layer in range(self.bundle.num_layers):
            w = torch.zeros(self.bundle.hidden_size, 1); w[layer, 0] = 1.
            torch.save({'W': w, 'mean_gap': [1.], 'losses': [1., .5], 'seed': 42, 'n_pairs': train_n,
                        'layer': layer, 'rank': 1, 'loss_kind': 'ce', 'bidirectional': True,
                        'positions': 'last', 'anchor': 'prompt', 'T': 1, 'epochs': 2, 'lr': .001,
                        'weighted': False, 'trim': 0.},
                       self.args.tf_das_dir / ('das_E_L%d_r1_s42_tf-ce-bidir.pt' % layer))
        eval_rows = self.tf['tune'] + self.tf['test']
        labels = np.array([r['label'] for r in eval_rows])
        tune = np.array([i < len(self.tf['tune']) for i in range(len(eval_rows))])
        ptrue = np.where(labels, .8, .2)
        yes = np.ones(len(eval_rows), bool)
        for method, path in [('das', self.args.tf_das_sweep), ('meandiff', self.args.tf_md_sweep)]:
            conds = {n: {s: {'ans': 1., 'J': 1., 'AUROC': 1., 'B': .5} for s in ['A', 'B', 'all']}
                     for n in ['baseline', 'E_L0_a1', 'E_L1_a1']}
            joint.save(path, {'model': self.args.model, 'e_estimator': method, 'conds': conds})
            arrays = {'ids': np.array([r['id'] for r in eval_rows]), 'labels': labels, 'half_a': tune}
            for name in conds:
                arrays[name + '__answer'] = yes
                arrays[name + '__ptrue'] = ptrue
            np.savez(path.with_suffix('.npz'), **arrays)

    def prepare(self, pilot=None):
        self.args.pilot = pilot
        joint.prepare(self.args, self.cfg, bundle=self.bundle)
        return joint.load_run(self.args)

    def tune(self, meta, bank, data, judge=None):
        joint.collect(self.args, meta, bank, data, 'tune', bundle=self.bundle)
        joint.judge_qa(self.args, meta, data, 'tune', judge=judge or LocalJudge())
        return joint.freeze(self.args, meta)

    def test_real_prepare_source_apis_and_pilot_do_not_shrink_source_identity(self):
        source_counts = {k: len(v) for k, v in self.source.prepared['splits'].items()}
        meta, bank, data = self.prepare(pilot=2)
        self.assertEqual(meta['pilot'], 2)
        self.assertEqual({k: len(v) for k, v in data['qa'].items()}, {'train': 2, 'tune': 2, 'test': 2})
        self.assertEqual({k: len(v) for k, v in data['tf'].items()}, {'train': 4, 'tune': 4, 'test': 4})
        self.assertEqual({k: len(v) for k, v in self.source.prepared['splits'].items()}, source_counts)
        self.assertIn('tf_das_E_L0', bank)
        self.assertIn('qa_das_E_L0', bank)
        self.assertIn('qa_md_E_L1', bank)
        self.assertTrue(meta['c_layers']['qa']['das'])
        self.assertTrue(meta['c_layers']['tf']['das'])
        before = self.model.forward_calls
        joint.prepare(self.args, self.cfg, bundle=self.bundle)
        self.assertEqual(before, self.model.forward_calls)

    def test_complete_real_inference_judging_freeze_report_and_actual_deliveries(self):
        meta, bank, data = self.prepare()
        selected = self.tune(meta, bank, data)
        before = (self.args.run_dir / 'selection.json').read_bytes()
        joint.collect(self.args, meta, bank, data, 'test', bundle=self.bundle)
        joint.judge_qa(self.args, meta, data, 'test', judge=LocalJudge())
        result = joint.report(self.args, meta)
        self.assertTrue(result['completed'])
        self.assertEqual(before, (self.args.run_dir / 'selection.json').read_bytes())
        delivered = joint.read(self.args.run_dir / 'delivered_answers.json')
        for task in ('tf', 'qa'):
            for q in self.cfg['coverages']:
                key = str(q)
                quota = result['tables'][task][key]['quota']['coverages'][0]
                self.assertEqual(quota['n'], 6 if task == 'tf' else 3)
                for name, point in quota['arms'].items():
                    if point['feasible']:
                        rows = delivered[task][key][name]
                        self.assertEqual(sum(r['delivered_outcome'] != 'abstain' for r in rows), point['k'])
                        self.assertEqual(len(rows), quota['n'])
                self.assertEqual(result['tables'][task][key]['quota']['n_groups'], 3)
                self.assertIn('baseline_simple', quota['arms'])
                self.assertIn('baseline_random', quota['arms'])
                self.assertTrue(any(c.endswith('_EC') for c in selected['tasks'][task]['choices'][key]))
        for prompt in self.model.generated_prompts:
            self.assertNotIn('GOLD_', prompt)
            self.assertNotIn('WRONG_', prompt)
        calls = self.model.forward_calls
        joint.collect(self.args, meta, bank, data, 'test', bundle=self.bundle)
        self.assertEqual(calls, self.model.forward_calls)
        for old, current in zip(self.source.original_parameters, self.model.parameters()):
            self.assertTrue(torch.equal(old, current))

    def test_reference_changes_do_not_change_inference_scores_or_outputs(self):
        meta, bank, data = self.prepare()
        c = meta['conditions']['qa'][0]
        row = data['qa']['tune'][0]
        changed = {**row, 'gold': ['SECRET_REPLACEMENT'], 'incorrect': ['OTHER_SECRET'],
                   'positive_answer': 'REFERENCE_LEAK'}
        first = joint.output_records(self.bundle, meta, bank, 'qa', [row], c)
        second = joint.output_records(self.bundle, meta, bank, 'qa', [changed], c)
        self.assertEqual(first, second)
        tfc = meta['conditions']['tf'][0]
        t = data['tf']['tune'][0]
        a = joint.output_records(self.bundle, meta, bank, 'tf', [t], tfc)[0]
        b = joint.output_records(self.bundle, meta, bank, 'tf', [{**t, 'label': not t['label']}], tfc)[0]
        self.assertEqual(a['output'], b['output'])
        self.assertEqual(a['score'], b['score'])
        self.assertEqual(a['simple_score'], b['simple_score'])
        self.assertNotEqual(a['outcome'], b['outcome'])

    def test_partial_judging_keeps_cohort_and_retries_only_unresolved(self):
        meta, bank, data = self.prepare()
        joint.collect(self.args, meta, bank, data, 'tune', bundle=self.bundle)
        judge = LocalJudge(unresolved_question=data['qa']['tune'][0]['question'])
        with self.assertRaisesRegex(RuntimeError, 'unresolved'):
            joint.judge_qa(self.args, meta, data, 'tune', judge=judge)
        partial = joint.read(self.args.run_dir / 'qa_tune_scored.json')
        self.assertFalse(partial['completed'])
        self.assertTrue(all(len(v) == 3 for v in partial['rows'].values()))
        with self.assertRaises(ValueError):
            joint.freeze(self.args, meta)
        calls = len(judge.calls)
        expected_retry = partial['n_unresolved']
        judge.unresolved_question = None
        joint.judge_qa(self.args, meta, data, 'tune', judge=judge)
        self.assertEqual(len(judge.calls) - calls, expected_retry)
        self.assertTrue(joint.read(self.args.run_dir / 'qa_tune_scored.json')['completed'])
        joint.freeze(self.args, meta)

    def test_default_cli_judge_uses_actual_E4_config_and_factory_contract(self):
        meta, bank, data = self.prepare()
        joint.collect(self.args, meta, bank, data, 'tune', bundle=self.bundle)
        local = LocalJudge()
        argv = ['e26_matched_coverage.py', 'judge-tune', '--model', self.args.model,
                '--run-dir', str(self.args.run_dir)]  # Deliberately omit --judge-exp.
        with patch.object(joint, 'load_experiment', wraps=actual_load_experiment) as config_loader, \
                patch('core.judge.judge_from_config', autospec=True, return_value=local) as factory, \
                patch.object(sys, 'argv', argv):
            joint.main()
        config_loader.assert_called_once_with('E4_qa')
        self.assertEqual(factory.call_args.args[0], actual_load_experiment('E4_qa')['judge'])
        self.assertEqual(factory.call_args.kwargs['cache_dir'], str(self.args.run_dir / 'judge_cache'))
        self.assertEqual(len(local.calls), len(meta['conditions']['qa']) * len(data['qa']['tune']))
        self.assertTrue(joint.read(self.args.run_dir / 'qa_tune_scored.json')['completed'])

    def test_insufficient_actual_answers_preserves_infeasible_rows_and_report(self):
        meta, bank, data = self.prepare()
        tune_judge = LocalJudge(refusal_questions=[data['qa']['tune'][0]['question']])
        self.tune(meta, bank, data, judge=tune_judge)
        joint.collect(self.args, meta, bank, data, 'test', bundle=self.bundle)
        test_judge = LocalJudge(refusal_questions=[data['qa']['test'][0]['question']])
        joint.judge_qa(self.args, meta, data, 'test', judge=test_judge)
        result = joint.report(self.args, meta)
        full = result['tables']['qa']['1.0']
        self.assertFalse(full['quota']['coverages'][0]['arms']['baseline_commitment']['feasible'])
        self.assertIn('das_E', full['unavailable_families'])
        self.assertIn('das_C', full['unavailable_families'])
        self.assertIn('das_EC', full['unavailable_families'])
        self.assertEqual(full['native']['baseline_commitment']['n'], 3)
        self.assertFalse(full['frozen_threshold']['baseline_commitment']['feasible'])
        half = result['tables']['qa']['0.5']['quota']['coverages'][0]['arms']['baseline_commitment']
        self.assertEqual(half['answered'], 1)

    def test_frozen_choice_cannot_be_edited_or_rejudged(self):
        meta, bank, data = self.prepare()
        selected = self.tune(meta, bank, data)
        with self.assertRaises(ValueError):
            joint.judge_qa(self.args, meta, data, 'tune', judge=LocalJudge())
        selected['tasks']['qa']['choices']['0.5']['das_E'] = 'baseline'
        joint.save(self.args.run_dir / 'selection.json', selected)
        with self.assertRaisesRegex(ValueError, 'frozen'):
            joint.frozen_selection(self.args, meta)
        with self.assertRaises(ValueError):
            joint.collect(self.args, meta, bank, data, 'test', bundle=self.bundle)

    def test_all_arms_missing_same_tuning_row_is_rejected(self):
        meta, bank, data = self.prepare()
        joint.collect(self.args, meta, bank, data, 'tune', bundle=self.bundle)
        joint.judge_qa(self.args, meta, data, 'tune', judge=LocalJudge())
        p = self.args.run_dir / 'tf_tune.json'; raw = joint.read(p)
        for rows in raw['rows'].values():
            rows.pop()
        joint.save(p, raw)
        with self.assertRaises(ValueError):
            joint.freeze(self.args, meta)

    def test_changed_derived_grade_or_reference_eligibility_is_rejected(self):
        meta, bank, data = self.prepare()
        joint.collect(self.args, meta, bank, data, 'tune', bundle=self.bundle)
        joint.judge_qa(self.args, meta, data, 'tune', judge=LocalJudge())
        p = self.args.run_dir / 'qa_tune_scored.json'; original = joint.read(p)
        for field, value in [('outcome', 'wrong'), ('gold_is_abstention', True), ('score', 999.)]:
            altered = deepcopy(original)
            altered['rows']['baseline'][0][field] = value
            joint.save(p, altered)
            with self.subTest(field=field), self.assertRaises(ValueError):
                joint.freeze(self.args, meta)
        joint.save(p, original)
        joint.freeze(self.args, meta)

    def test_test_cohort_cannot_shrink_after_selection(self):
        meta, bank, data = self.prepare()
        self.tune(meta, bank, data)
        joint.collect(self.args, meta, bank, data, 'test', bundle=self.bundle)
        joint.judge_qa(self.args, meta, data, 'test', judge=LocalJudge())
        p = self.args.run_dir / 'tf_test.json'; raw = joint.read(p)
        for rows in raw['rows'].values():
            rows.pop()
        joint.save(p, raw)
        with self.assertRaises(ValueError):
            joint.report(self.args, meta)

    def test_simple_E_control_has_its_own_tuning_winner(self):
        # Isolate the selector integration: model collection is covered above.
        # The same two E candidates reverse ranking under the two score rules.
        meta, _, _ = self.prepare()
        def candidate(name, alpha):
            return {'name': name, 'kind': 'E', 'method': 'das', 'task': 'qa',
                    'e': {'layer': 0, 'alpha': alpha}, 'c': None}
        conditions = [{'name': 'baseline', 'kind': 'base', 'method': 'base', 'task': 'qa', 'e': None, 'c': None},
                      candidate('a', 1.), candidate('b', 2.),
                      {'name': 'ca', 'kind': 'C', 'method': 'das', 'task': 'qa', 'e': None, 'c': {'layer': 0, 'alpha': -.5}},
                      {'name': 'cb', 'kind': 'C', 'method': 'das', 'task': 'qa', 'e': None, 'c': {'layer': 0, 'alpha': .5}}]
        def records(correct_index, simple_order):
            return [{'id': str(i), 'pair_id': str(i), 'outcome': 'correct' if i == correct_index else 'wrong',
                     'score': float(3 - i), 'simple_score': float(simple_order[i]), 'output': str(i)}
                    for i in range(3)]
        rows = {'baseline': records(-1, [3, 2, 1]), 'a': records(0, [0, 3, 2]), 'b': records(1, [0, 3, 2]),
                'ca': records(0, [0, 3, 2]), 'cb': records(1, [0, 3, 2])}
        altered = deepcopy(meta)
        altered['conditions'] = {task: deepcopy(conditions) for task in joint.TASKS}
        altered['config']['coverages'] = [.5]
        for task in joint.TASKS:
            joint.save(self.args.run_dir / (task + '_selector_fixture.json'), {'fixture': task})
        with patch.object(joint, 'scored_rows', side_effect=lambda args, m, task, split:
                          (deepcopy(rows), task + '_selector_fixture.json')):
            selected = joint.freeze(self.args, altered, persist=False)
        for task in joint.TASKS:
            chosen = selected['tasks'][task]['choices']['0.5']
            self.assertEqual(chosen['das_E'], 'a')
            self.assertEqual(chosen['das_E_simple'], 'b')
            self.assertEqual(chosen['das_C'], 'ca')
            self.assertEqual(chosen['das_C_simple'], 'cb')
            comparison = joint.comparison_arms(rows, chosen, altered['config'])
            self.assertEqual(comparison['das_E'][0]['output'], '0')
            self.assertEqual(comparison['das_E'][0]['score'], 3.)
            self.assertEqual(comparison['das_E_simple'][0]['score'], 0.)
            self.assertEqual(comparison['das_C'][0]['score'], 3.)
            self.assertEqual(comparison['das_C_simple'][0]['score'], 0.)

    def test_multilayer_E27_train_export_import_collect_freeze(self):
        self.assertEqual(self.bundle.num_layers, 3)
        self.assertEqual(self.source.meta['das_candidate_layers'], [0, 1])
        source_ns, source_meta, source_bank, source_data = generation.source_run(self.source.run)
        adapt_args = SimpleNamespace(run_dir=self.source.run / 'generation_adapter', layers=[0, 1],
            pilot=None, init='das', alpha=1.25, epochs=1, lr=.001, clip=1., ranking_weight=0., seed=42)
        adapter = generation.train(adapt_args, source_ns, source_meta, source_bank, source_data, bundle=self.bundle)
        self.assertEqual(adapter['layers'], [0, 1])
        self.assertTrue(all(adapter['gradient_audit'][str(l)]['maximum'] > 0 for l in [0, 1]))
        self.assertTrue((adapt_args.run_dir / 'adapter.pt').exists())
        self.assertTrue((adapt_args.run_dir / 'adapter.json').exists())
        self.args.qa_adapters = [adapt_args.run_dir]
        meta, bank, data = self.prepare()
        method = 'das_generation_adapter'
        self.assertEqual(meta['e_choices']['qa'][method], [{'layers': [0, 1], 'alpha': 1.25, 'mode': 'continuous'}])
        self.assertTrue(meta['c_layers']['qa'][method])
        for l in [0, 1]:
            np.testing.assert_array_equal(bank['qa_%s_E_L%d' % (method, l)], adapter['W'][str(l)].numpy())
        adapter_conditions = [c for c in meta['conditions']['qa'] if c['method'] == method and c['e']]
        self.assertTrue(adapter_conditions)
        for c in adapter_conditions:
            self.assertEqual(c['e']['layers'], [0, 1])
            self.assertEqual(c['e']['alpha'], 1.25)
        selected = self.tune(meta, bank, data)
        for point in selected['tasks']['qa']['choices'].values():
            self.assertIsNotNone(point[method + '_E'])
            self.assertIsNotNone(point[method + '_EC'])
        joint.collect(self.args, meta, bank, data, 'test', bundle=self.bundle)
        joint.judge_qa(self.args, meta, data, 'test', judge=LocalJudge())
        result = joint.report(self.args, meta)
        self.assertIn(method + '_EC', result['tables']['qa']['0.5']['quota']['coverages'][0]['arms'])
        raw = joint.read(self.args.run_dir / 'qa_test.json')
        chosen = selected['tasks']['qa']['choices']['0.5'][method + '_EC']
        edits = raw['rows'][chosen][0]['edits']
        self.assertEqual(sum('_E_L' in name for name in edits), 2)
        self.assertEqual(sorted(n for name, n in edits.items() if '_E_L' in name), [2, 2])
        self.assertEqual([n for name, n in edits.items() if '_C_L' in name], [1])


if __name__ == '__main__':
    unittest.main()
