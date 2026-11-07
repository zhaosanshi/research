"""Budget/reuse bookkeeping with a deterministic decoder fixture, no GPU/API."""
import contextlib,copy,importlib.util,json,re,tempfile
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[3];CODE=ROOT/'from_claude/code'
sys.path[:0]=[str(CODE/'scripts'),str(CODE)]
spec=importlib.util.spec_from_file_location('completion_under_test',CODE/'scripts/review_checks/e34_natural_completion.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
from core.judge import QA_GRADER_VERSION,qa_cache_key

class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);p=Path(self.temp.name)
        self.args=SimpleNamespace(run_dir=p/'new',parent_run=p/'old',ce_run=p/'ce',source_run=p/'source',pilot_reviewed=True)
        self.args.run_dir.mkdir();self.args.parent_run.mkdir();self.calls=[]
        self.weights={n:torch.tensor([i+1.]) for i,n in enumerate(c.CONDITIONS) if n not in ('baseline','source_only','equal_logit_mix')}
        rows=[]
        for i in range(6):
            r={'pair_id':str(i),'question':str(i),'gold':['answer'],'incorrect':['wrong'],'best_answer':'answer','idx':i,
               'prompt_token_ids':{'base':[i],'source':[i+10]},'outputs':{},'finished':{},'rule_abstain':{},'generation_token_ids':{},'trace':{}}
            for name in c.CONDITIONS:
                end=not(i>=3 and name==c.preference.METHOD)
                r['outputs'][name]=name+' '+str(i);r['finished'][name]=end;r['rule_abstain'][name]=False
                r['generation_token_ids'][name]=[i,99] if end else list(range(48));r['trace'][name]=[{'original':True}]
            rows.append(r)
        parent={'protocol':c.preference.PROTOCOL,'conditions':list(c.CONDITIONS),'condition_specs':[{'name':n} for n in c.CONDITIONS],
                'identity':{'parent':'identity'},'evaluations':rows,'E':{},'C':{'enabled':False}}
        c.qa.save(self.args.parent_run/'tune.json',parent)
        grade=copy.deepcopy(parent);cfg=c.ga.validate_judge_experiment('E4_qa_legacy_r20')
        grade.update(grading_judge_config_sha256=c.qa.fingerprint(cfg),grading_source_generation_sha256=c.qa.digest(self.args.parent_run/'tune.json'))
        entries={}
        for r in grade['evaluations']:
            r['judge_reference_answers']=r['gold'];r['judge']={n:{'label':'correct','raw':'A','rubric_id':QA_GRADER_VERSION} for n in c.CONDITIONS}
            for name in c.CONDITIONS:
                key=qa_cache_key(cfg['model'],r['question'],r['gold'],r['outputs'][name],r['incorrect']);entries[key]={'key':key,'answer':'A'}
        c.qa.save(self.args.parent_run/'tune_scored_items.json',grade)
        folder=self.args.parent_run/'judge_cache';folder.mkdir();filename='judge_cache_%s.jsonl'%re.sub(r'[^A-Za-z0-9_.-]','_',cfg['model'])
        (folder/filename).write_text(''.join(json.dumps(x)+'\n' for x in entries.values()))
        plan={'protocol':c.PROTOCOL,'model':'toy','ended_control_ids':{n:['0','1','2'] for n in c.CONDITIONS},
              'unfinished_pair_pilot_ids':['3','4','5'],'decoder_code':{n:c.qa.digest(CODE/n) for n in c.CRITICAL_CODE},
              'entry_code_sha256':c.qa.digest(CODE/'scripts/review_checks/e34_natural_completion.py'),
              'parent_generation_sha256':c.qa.digest(self.args.parent_run/'tune.json'),
              'parent_scored_sha256':c.qa.digest(self.args.parent_run/'tune_scored_items.json')}
        self.state={'plan':plan,'parent':parent,'grade':grade,'pref':{'config':{},'layer':0},'weights':self.weights,
                    'meta':{'model':'toy'},'prepared':{}}
        self.bundle=SimpleNamespace(model=torch.nn.Linear(1,1))
        self.patches=[patch.object(c.qa,'verify_bundle'),patch.object(c,'fp32_output_projection',side_effect=lambda b:contextlib.nullcontext()),
                      patch.object(c.source_eval,'prompt_tokens',side_effect=lambda b,cfg,q:([int(q)],[int(q)+10])),
                      patch.object(c,'greedy_source_generate',side_effect=self.decoder)]
        for p in self.patches:p.start();self.addCleanup(p.stop)

    def decoder(self,bundle,base,source,*,mode,layer,direction,max_new_tokens):
        self.assertEqual(max_new_tokens,256);self.assertEqual(source,[base[0]+10])
        if mode=='source_interchange':
            names=[n for n,w in self.weights.items() if direction is w];self.assertEqual(len(names),1);name=names[0]
        else:self.assertIsNone(direction);name=mode
        i=base[0];old=self.state['parent']['evaluations'][i];self.calls.append((str(i),name))
        if old['finished'][name]:return {'token_ids':old['generation_token_ids'][name][:],'output':old['outputs'][name], 'finished':True,'trace':[{'new_probe':True}]}
        return {'token_ids':old['generation_token_ids'][name]+[100+i,99],'output':old['outputs'][name]+' tail','finished':True,'trace':[{'extended':True}]}

    def test_all_ended_controls_verified_before_reuse_and_only_unfinished_are_extended(self):
        original=copy.deepcopy(self.state['parent'])
        pilot=c.generate(self.args,self.state,True,bundle=self.bundle)
        self.assertEqual(len(pilot['finished_control_checks']),21);self.assertEqual(len(pilot['extension_probes']),3)
        self.assertEqual(len(self.calls),24)
        full=c.generate(self.args,self.state,False,bundle=self.bundle)
        self.assertEqual(len(self.calls),27);self.assertEqual(len(full['completion_audit']),42)
        self.assertEqual(sum(r['action']=='reused_original_naturally_finished' for r in full['completion_audit']),39)
        self.assertEqual(self.state['parent'],original)
        for r in full['evaluations'][3:]:self.assertEqual(r['generation_token_ids'][c.preference.METHOD][:48],list(range(48)))
        with patch.object(c.qa,'judge') as j:
            counts=c.judge(self.args,self.state)
        self.assertEqual(counts,{'frozen':39,'changed':3});self.assertEqual(sum(len(v) for v in self.args.frozen_control_grades.values()),39)
        self.assertTrue(j.called);self.assertNotEqual(Path(self.args.judge_cache_dir).resolve(),(self.args.parent_run/'judge_cache').resolve())

    def test_prefix_or_completed_text_change_hard_fails(self):
        old=self.state['parent']['evaluations'][3]
        with patch.object(c,'greedy_source_generate',return_value={'token_ids':[-1]+list(range(1,48))+[99],'output':'changed','finished':True}):
            with self.assertRaisesRegex(RuntimeError,'48-token'):c.one(self.args,self.state,self.bundle,old,c.preference.METHOD)
        old=self.state['parent']['evaluations'][0]
        with patch.object(c,'greedy_source_generate',return_value={'token_ids':old['generation_token_ids']['baseline'],'output':'changed','finished':True}):
            with self.assertRaisesRegex(RuntimeError,'already-finished'):c.one(self.args,self.state,self.bundle,old,'baseline')

    def test_failed_pilot_blocks_full_and_no_review_ack_blocks_full(self):
        pilot=c.generate(self.args,self.state,True,bundle=self.bundle)
        self.args.pilot_reviewed=False
        with self.assertRaisesRegex(ValueError,'pilot-reviewed'):c.generate(self.args,self.state,False,bundle=self.bundle)
        self.args.pilot_reviewed=True;pilot['extension_probes'][0]['new']['finished']=False;c.qa.save(self.args.run_dir/'pilot.json',pilot)
        with self.assertRaisesRegex(ValueError,'unfinished'):c.generate(self.args,self.state,False,bundle=self.bundle)

if __name__=='__main__':unittest.main()
