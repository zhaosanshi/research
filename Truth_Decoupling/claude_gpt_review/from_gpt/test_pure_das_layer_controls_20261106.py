"""Execute frozen layer/timing controls with a real causal CPU toy network."""
import contextlib
import copy
import io
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

HERE=Path(__file__).resolve().parent
CODE=HERE.parent/'from_claude/code'
sys.path[:0]=[str(HERE),str(CODE/'scripts'),str(CODE)]
import test_qa_das_pipeline_20261105 as fixture
import e24_real_qa as e24
from core.judge import QA_GRADER_VERSION

spec=importlib.util.spec_from_file_location('pure_layer_controls',CODE/'scripts/review_checks/e27_pure_das_layer_controls.py')
control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)


class ThreeLayerModel(fixture.PipelineModel):
    def __init__(self,tokenizer):
        super().__init__(tokenizer)
        self.layers.append(copy.deepcopy(self.layers[-1]))
        self.config.num_hidden_layers=3


class LocalJudge:
    name='pure-layer-local-plumbing'
    qa_rubric_id=QA_GRADER_VERSION
    n_cache_hits=0
    def __init__(self):self.n_calls=0
    def qa_grade(self,question,gold,prediction,incorrect=None):
        self.n_calls+=1
        return {'label':'wrong','raw':'B','rubric_id':self.qa_rubric_id}


class PureLayerControlTests(unittest.TestCase):
    def enter(self,cm):
        result=cm.__enter__();self.addCleanup(cm.__exit__,None,None,None);return result
    def setUp(self):
        self.enter(patch.object(fixture,'PipelineModel',ThreeLayerModel))
        old=fixture.pipeline.estimate
        def estimate(args,cfg,bundle=None):
            args.layers,args.all_layers=None,True
            return old(args,cfg,bundle=bundle)
        self.enter(patch.object(fixture.pipeline,'estimate',side_effect=estimate))
        self.src=fixture.QADASPipelineTests('test_real_end_to_end_training_selection_check_and_diagnostics')
        self.src.setUp();self.addCleanup(self.src.doCleanups)
        self.enter(contextlib.redirect_stdout(io.StringIO()))
        choices=[dict(fixture.pipeline.BASELINE)]+[fixture.pipeline.condition('qa_das',mode,l,3)
            for l in [0,1] for mode in ['prefill','continuous']]
        fixture.pipeline.generate_questions(self.src.args,self.src.meta,self.src.bank,
            self.src.prepared['splits']['tune'],choices,'tune.json','tune',bundle=self.src.bundle)
        self.args=SimpleNamespace(source_run=self.src.run,run_dir=self.src.run/'control',model=self.src.args.model,
            layers=[0,1],limit=1,pair_ids=[self.src.prepared['splits']['tune'][1]['pair_id']],all_questions=False,
            judge_exp='E4_qa')
        self.state=control.inputs(self.args)
        self.before=[p.detach().clone() for p in self.src.model.parameters()]

    def generate(self):
        with patch('torch.optim.Adam',side_effect=AssertionError('no optimizer allowed')), \
             patch.object(fixture.pipeline,'train',side_effect=AssertionError('no training allowed')):
            return control.generate(self.args,self.state,bundle=self.src.bundle)

    def test_real_five_arm_generation_keeps_original_batches_and_frozen_parameters(self):
        raw=self.generate()
        self.assertEqual(raw['selected_questions'],1)
        self.assertEqual(raw['generation_context_questions'],2)
        self.assertEqual(len(raw['conditions']),5)
        self.assertTrue(raw['completed'] and raw['pilot'])
        self.assertEqual((self.args.run_dir/'questions.json').read_bytes(),(self.src.run/'questions.json').read_bytes())
        self.assertTrue(all(x['exact_text_equal'] and x['finished_equal'] for x in raw['source_reproduction']))
        self.assertEqual(raw['batch_context'][0]['source_batch_ids'],[q['pair_id'] for q in self.src.prepared['splits']['tune'][:2]])
        context=control.qa.read(self.args.run_dir/raw['generation_context_file'])
        self.assertTrue(all(not c.get('uses_reference_answer') and not c['eligible_for_selection'] for c in raw['condition_specs']))
        for audit in context['audit']:
            if audit['condition']=='baseline':continue
            expected=1 if 'prefill' in audit['condition'] else audit['new_tokens']
            self.assertTrue(all(n==expected for n in audit['edits'].values()))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(self.before,self.src.model.parameters())))
        self.src.assert_question_only_generation()

    def test_resume_is_idempotent_and_keeps_source_bytes(self):
        self.generate();before=(self.args.run_dir/'tune.json').read_bytes()
        n=self.src.model.forward_calls;self.generate()
        self.assertEqual(self.src.model.forward_calls,n)
        self.assertEqual((self.args.run_dir/'tune.json').read_bytes(),before)

    def test_incomplete_or_changed_source_reference_is_rejected(self):
        p=self.src.run/'tune.json';d=control.qa.read(p);d['completed']=False;control.qa.save(p,d)
        with self.assertRaisesRegex(ValueError,'source tuning'):control.inputs(self.args)

    def test_reference_mismatch_is_saved_but_not_published_complete(self):
        p=self.src.run/'tune.json';d=control.qa.read(p);pid=self.args.pair_ids[0]
        next(r for r in d['evaluations'] if r['pair_id']==pid)['outputs'][self.state[-1]['source_reference_condition']]='WRONG REFERENCE'
        control.qa.save(p,d);self.state=control.inputs(self.args)
        with self.assertRaisesRegex(RuntimeError,'did not reproduce'):self.generate()
        self.assertFalse(control.qa.read(self.args.run_dir/'tune.json')['completed'])

    def test_nonprefix_ids_preserve_batches_and_test_ids_are_rejected(self):
        prepared=self.src.prepared
        q,context,batches=control.selected_context(prepared,2,[r['pair_id'] for r in prepared['splits']['tune'][1:]],2)
        self.assertEqual([b['source_batch_start'] for b in batches],[0,2])
        self.assertEqual(len(q),2);self.assertEqual(len(context),3)
        with self.assertRaisesRegex(ValueError,'not in the tuning'):
            control.selected_context(prepared,1,[prepared['splits']['test'][0]['pair_id']],2)
        with self.assertRaises(ValueError):control.selected_context(prepared,2,[prepared['splits']['tune'][0]['pair_id']],2)

    def test_actual_judge_report_path_only_scores_requested_ids(self):
        raw=self.generate();grader=LocalJudge()
        with patch('core.judge.judge_from_config',return_value=grader), \
             patch.object(e24,'load_model_config',return_value={'name':self.args.model}), \
             patch.object(e24,'out_paths',return_value=(str(self.args.run_dir),'unused','unused')):
            control.judge(self.args,self.state)
        summary=control.qa.read(self.args.run_dir/'tune_scored.json')
        items=control.qa.read(self.args.run_dir/'tune_scored_items.json')
        self.assertTrue(summary['completed']);self.assertEqual(summary['n_questions'],1)
        self.assertEqual(len(items['evaluations']),1)
        self.assertEqual(grader.n_calls,5)
        self.assertEqual(summary['source_generation_sha256'],control.qa.digest(self.args.run_dir/'tune.json'))
        self.assertTrue((self.args.run_dir/'tune_scored.md').exists())


if __name__=='__main__':unittest.main()
