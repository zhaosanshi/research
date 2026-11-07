"""Actual E25 grading: fewer full-file writes, durable interruption, exact resume."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

CODE=Path(__file__).resolve().parents[1]/'from_claude/code'
sys.path[:0]=[str(CODE),str(CODE/'scripts')]
import e25_qa_das as qa
import e24_real_qa as e24
from core.judge import QA_GRADER_VERSION


class Grader:
    name='checkpoint-toy';qa_rubric_id=QA_GRADER_VERSION
    def __init__(self,fail=None,clock=None,observe=None,unresolved=False):
        self.calls=0;self.fail=fail;self.clock=clock;self.observe=observe;self.unresolved=unresolved
    def qa_grade(self,q,refs,prediction,incorrect=None):
        self.calls+=1
        if self.observe:self.observe(self.calls)
        if self.fail==self.calls:raise RuntimeError('synthetic interruption')
        if self.clock is not None:self.clock[0]+=6.
        label='unresolved' if self.unresolved else 'correct' if prediction=='right' else 'wrong'
        return {'label':label,'raw':label,'rubric_id':self.qa_rubric_id}


class GradingCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.args=SimpleNamespace(run_dir=self.root,model='toy',judge_exp='unused')
        self.cfg={'judge':{'model':'toy','name':'checkpoint-toy'}}
    def raw(self,n):
        data={'completed':True,'model':'toy','split':'tune','bank_protocol_fingerprint':'toy-bank',
            'conditions':['baseline'],'condition_specs':[{'name':'baseline'}],
            'evaluations':[{'pair_id':'q%d'%i,'idx':i,'question':'Question %d?'%i,'gold':['right'],'incorrect':['wrong'],
                'outputs':{'baseline':'right' if i%2 else 'wrong'},'finished':{'baseline':True},'rule_abstain':{'baseline':False}}
                for i in range(n)]}
        qa.save(self.root/'tune.json',data);return data
    def report_stub(self,args):
        data=qa.read(Path(args.input_file))
        self.assertTrue(data['grading_completed'])
        qa.save(Path(args.output_prefix+'.json'),{})
        qa.save(Path(args.output_prefix+'_items.json'),data)
    def run_grade(self,grader,clock=None):
        with patch.object(e24,'load_experiment',return_value=self.cfg),patch('core.judge.judge_from_config',return_value=grader),\
                patch.object(e24,'phase_judge',side_effect=self.report_stub),patch.object(qa.time,'monotonic',side_effect=lambda:0. if clock is None else clock[0]):
            qa.judge(self.args,{'protocol_fingerprint':'toy-bank'},'tune.json','tune_scored')
    def test_many_cached_or_fast_labels_do_not_rewrite_whole_artifact_per_item(self):
        raw=self.raw(200);before=(self.root/'tune.json').read_bytes();writes=[];save=qa.save
        def spy(path,data):
            if str(path).endswith('_grading_input.json'):writes.append(copy.deepcopy(data))
            return save(path,data)
        with patch.object(qa,'save',side_effect=spy):self.run_grade(Grader())
        self.assertLess(len(writes),20)
        self.assertEqual((self.root/'tune.json').read_bytes(),before)
        result=qa.read(self.root/'tune_scored_items.json')
        self.assertEqual([r['outcome']['baseline'] for r in result['evaluations']],['wrong' if i%2==0 else 'correct' for i in range(200)])
        self.assertTrue(qa.read(self.root/'tune_scored_grading_input.json')['grading_completed'])
    def test_exception_flushes_then_resume_reuses_resolved_labels_without_new_calls(self):
        self.raw(5)
        with self.assertRaisesRegex(RuntimeError,'synthetic interruption'):self.run_grade(Grader(fail=3))
        pending=qa.read(self.root/'tune_scored_grading_input.json')
        self.assertFalse(pending['grading_completed'])
        self.assertEqual([r.get('outcome',{}).get('baseline') for r in pending['evaluations']],['wrong','correct',None,None,None])
        grader=Grader();self.run_grade(grader);self.assertEqual(grader.calls,3)
        self.assertTrue(qa.read(self.root/'tune_scored_items.json')['completed'])
    def test_slow_calls_checkpoint_before_count_threshold_and_unresolved_stays_explicit(self):
        self.raw(2);clock=[0.];observed=[]
        def observe(n):
            if n==2:observed.append(qa.read(self.root/'tune_scored_grading_input.json')['evaluations'][0]['outcome']['baseline'])
        self.run_grade(Grader(clock=clock,observe=observe),clock)
        self.assertEqual(observed,['wrong'])
        for p in self.root.glob('tune_scored*'):p.unlink()
        with self.assertRaisesRegex(RuntimeError,'grading incomplete'):self.run_grade(Grader(unresolved=True))
        result=qa.read(self.root/'tune_scored.json')
        self.assertFalse(result['completed']);self.assertEqual(result['n_unresolved'],2)

if __name__=='__main__':unittest.main()
