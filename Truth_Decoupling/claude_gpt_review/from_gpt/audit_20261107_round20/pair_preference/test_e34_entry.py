"""Real toy forwards/updates through the E34 entry; no network or model weights."""
import copy,json,hashlib,re
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[3];CODE=ROOT/'from_claude/code'
sys.path[:0]=[str(ROOT/'from_gpt'),str(CODE/'scripts'),str(CODE)]
import test_qa_das_pipeline_20261105 as source_fixture
import e34_qa_source_pair_preference as entry
import e24_real_qa as e24
from core.judge import QA_GRADER_VERSION,qa_cache_key
from core.qa_scoring import reference_answers

class SyntheticJudge:
    name='synthetic-entry-grader';qa_rubric_id=QA_GRADER_VERSION;n_cache_hits=0;n_calls=0
    def qa_grade(self,question,gold,prediction,incorrect=None):
        self.n_calls+=1
        return {'label':'correct','raw':'A','rubric_id':QA_GRADER_VERSION}

class E34EntryTests(unittest.TestCase):
    def context(self,cm):
        value=cm.__enter__();self.addCleanup(cm.__exit__,None,None,None);return value

    def setUp(self):
        self.fixture=source_fixture.QADASPipelineTests(methodName='test_real_end_to_end_training_selection_check_and_diagnostics')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        f=self.fixture;self.context(patch.object(entry,'FIXED_LAYERS',{f.meta['model']:0}))
        self.context(patch.object(entry,'FULL_PAIRS',3));self.context(patch.object(entry,'TUNE_PAIRS',3))
        self.args=SimpleNamespace(source_run=f.run,ce_run=f.run/'ce_parent',run_dir=f.run/'pair',pilot=3,phase='train')
        source=entry.legacy.source_identity(f.args,f.meta,[0]);self.source=source
        ce=entry.ce_training.train(f.bundle,f.prepared['e_train'],f.meta['config'],f.bank['E_qa_das_L0'],layer=0,
             source_system=entry.source_eval.SOURCE_SYSTEM,epochs=3,identity={'source':source,'pilot':False},log_fn=None)
        entry.ce_training.save(ce,self.args.ce_run);entry.snapshot(self.args.ce_run/'training_code_snapshot',ce['training_implementation'])
        self.ce=ce;rows=[]
        rw=torch.nn.functional.normalize(torch.arange(1.,f.bundle.hidden_size+1),dim=0)
        with entry.fp32_output_projection(f.bundle):
            for i,q in enumerate(f.prepared['splits']['tune']):
                base,src=entry.source_eval.prompt_tokens(f.bundle,ce['config'],q['question'])
                row={**copy.deepcopy(q),'idx':i,'outputs':{},'finished':{},'rule_abstain':{},'generation_token_ids':{},'trace':{},
                     'cache_reference_checks':{},'prompt_token_ids':{'base':base,'source':src}}
                for method in entry.PARENT_CONDITIONS:
                    mode=method if method in ('baseline','source_only','equal_logit_mix') else 'source_interchange'
                    w=rw if method=='source_random_swap' else ce['W'] if method=='trained_source_swap_das_init' else ce['initial_W']
                    got=entry.greedy_source_generate(f.bundle,base,src,mode=mode,layer=0,direction=w,max_new_tokens=ce['config']['max_new_tokens'])
                    row['outputs'][method]=got['output'];row['finished'][method]=got['finished'];row['rule_abstain'][method]=entry.rule_abstains(got['output'])
                    row['generation_token_ids'][method]=got['token_ids'];row['trace'][method]=got['trace']
                rows.append(row)
        identity={'protocol':entry.ce_training.PROTOCOL,'model':f.meta['model'],'source':source,'layer':0,
                  'adapter_sha256':entry.qa.digest(self.args.ce_run/'source_adapter.pt'),'config':ce['config'],
                  'source_system':entry.source_eval.SOURCE_SYSTEM,'batch_size':1,'alpha':1.,
                  'condition':'trained_source_swap_das_init','same_prefix_tokens':True,'gold_used_at_inference':False,
                  'question_ids':[q['pair_id'] for q in f.prepared['splits']['tune']]}
        raw={'protocol':entry.ce_training.PROTOCOL,'model':f.meta['model'],'identity':identity,'completed':True,
             'conditions':list(entry.PARENT_CONDITIONS),'condition_specs':[{'name':n,'kind':'E','uses_reference_answer':False,'eligible_for_selection':False} for n in entry.PARENT_CONDITIONS],
             'protocol_fingerprint':entry.qa.fingerprint(identity),'bank_protocol_fingerprint':entry.qa.fingerprint(identity),
             'control_identity':{'old_five_arm':'preserved'},'evaluations':rows,'split':'tune','pilot':False}
        entry.qa.save(self.args.ce_run/'tune.json',raw);self.parent=raw
        cfg=entry.legacy.validate_judge_experiment(entry.LEGACY_JUDGE);self.judge_cfg=cfg
        scored=copy.deepcopy(raw);cache={}
        for r in scored['evaluations']:
            refs,abst=reference_answers(r['gold']);r['judge_reference_answers']=refs;r['gold_is_abstention']=abst
            r['judge']={n:{'label':'correct','raw':'A','rubric_id':QA_GRADER_VERSION} for n in entry.PARENT_CONDITIONS}
            r['outcome']={n:'correct' for n in entry.PARENT_CONDITIONS}
            for n in entry.PARENT_CONDITIONS:
                key=qa_cache_key(cfg['model'],r['question'],refs,r['outputs'][n],r.get('incorrect') or None)
                cache[key]={'key':key,'answer':'A'}
        scored.update(grading_completed=True,grading_source_generation_sha256=entry.qa.digest(self.args.ce_run/'tune.json'),
                      grading_judge_config_sha256=entry.qa.fingerprint(cfg))
        entry.qa.save(self.args.ce_run/'tune_scored_items.json',scored)
        cache_dir=self.args.ce_run/'judge_cache';cache_dir.mkdir()
        fname='judge_cache_%s.jsonl'%re.sub(r'[^A-Za-z0-9_.-]','_',cfg['model'])
        (cache_dir/fname).write_text(''.join(json.dumps(x)+'\n' for x in cache.values()))

    def test_actual_entry_training_generation_and_full_legacy_grade_inheritance(self):
        f=self.fixture;state=entry.source_state(self.args);before=[p.clone() for p in f.model.parameters()]
        result=entry.train(self.args,state,bundle=f.bundle)
        self.assertTrue(torch.equal(result['initial_W'],self.ce['initial_W']))
        self.assertFalse(torch.equal(result['initial_W'],self.ce['W']))
        self.assertGreater(result['gradient_audit']['nonzero_updates'],0)
        adapter=entry.verified_adapter(self.args,state)
        generated=entry.generate(self.args,state,adapter,True,bundle=f.bundle)
        self.assertEqual(generated['conditions'],[*entry.PARENT_CONDITIONS,entry.preference.METHOD])
        entry.validate_inherited(self.parent['evaluations'],generated['evaluations'])
        self.assertEqual(generated['control_identity'],self.parent['identity'])
        self.assertEqual(generated['identity']['control_generation_sha256'],entry.qa.digest(self.args.ce_run/'tune.json'))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,f.model.parameters())))
        for prompt in f.model.generated_prompts:
            self.assertNotIn('GOLD_',prompt);self.assertNotIn('WRONG_',prompt)
        self.assertEqual(entry.freeze_control_grades(self.args,state,adapter,'check'),18)
        self.assertNotEqual(Path(self.args.judge_cache_dir).resolve(),(self.args.ce_run/'judge_cache').resolve())
        self.args.model=f.meta['model'];self.args.judge_exp=entry.LEGACY_JUDGE
        with patch('core.judge.judge_from_config',return_value=SyntheticJudge()),\
             patch.object(e24,'load_experiment',return_value={'judge':self.judge_cfg}),\
             patch.object(e24,'load_model_config',return_value=f.model_config),\
             patch.object(e24,'out_paths',return_value=(str(self.args.run_dir),'unused','unused')):
            entry.qa.judge(self.args,{**f.meta,'protocol_fingerprint':generated['bank_protocol_fingerprint']},'check.json','check_scored')
        scored=entry.qa.read(self.args.run_dir/'check_scored_items.json')
        self.assertTrue(scored['completed']);self.assertEqual(scored['frozen_control_grade_verification']['question_condition_grades_checked'],18)
        self.assertEqual(len(scored['evaluations']),3)
        with self.assertRaisesRegex(ValueError,'pilot'):entry.generate(self.args,state,adapter,False,bundle=f.bundle)

    def test_changed_copied_tokens_or_prompt_or_trace_are_rejected(self):
        for field in ('generation_token_ids','trace','outputs','finished'):
            a=copy.deepcopy(self.parent['evaluations']);name=entry.PARENT_CONDITIONS[0]
            a[0][field][name]='changed'
            with self.assertRaises(ValueError):entry.validate_inherited(self.parent['evaluations'],a)
        a=copy.deepcopy(self.parent['evaluations']);a[0]['prompt_token_ids']['base'].append(99)
        with self.assertRaises(ValueError):entry.validate_inherited(self.parent['evaluations'],a)

    def test_plan_change_is_rejected_and_no_final_ce_weight_initialization(self):
        state=entry.source_state(self.args);plan=entry.freeze_or_verify(self.args,state,True)
        self.assertEqual(plan['initial_W'],entry.tensor_fingerprint(self.ce['initial_W']))
        plan['layer']=1;entry.qa.save(self.args.run_dir/'plan.json',plan)
        with self.assertRaisesRegex(ValueError,'plan'):entry.freeze_or_verify(self.args,state)

    def test_parent_grades_are_frozen_before_training(self):
        state=entry.source_state(self.args);plan=entry.freeze_or_verify(self.args,state,True)
        path=self.args.ce_run/'tune_scored_items.json'
        self.assertEqual(plan['ce_parent']['graded_items_sha256'],entry.qa.digest(path))
        graded=entry.qa.read(path);name=entry.PARENT_CONDITIONS[0]
        graded['evaluations'][0]['judge'][name]['label']='wrong';graded['evaluations'][0]['outcome'][name]='wrong'
        entry.qa.save(path,graded)
        with self.assertRaisesRegex(ValueError,'frozen E34 plan'):entry.freeze_or_verify(self.args,state)

if __name__=='__main__':unittest.main()
