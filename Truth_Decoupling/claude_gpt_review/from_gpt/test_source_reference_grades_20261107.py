"""Frozen same-text grades, policy conversion, and the real E37 judge entry."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent;CODE=HERE.parent/'from_claude/code'
sys.path[:0]=[str(CODE),str(CODE/'scripts')]
import e37_source_commitment_order as entry
from core.judge import QA_GRADER_VERSION,qa_cache_key,cache_filename
from core.qa_generation_adapter import fingerprint
from core.qa_reference_grades import reference_index,verify_matching_grades,semantic_key
from core.qa_scoring import apply_configured_outcome_policy


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.parent=self.root/'reference256';self.parent.mkdir();self.run=self.root/'new';self.run.mkdir()
        self.cfg=json.loads((CODE/'configs/experiments/E4_qa.json').read_text())['judge']
        self.oldcfg=json.loads((CODE/'configs/experiments/E4_qa_legacy_r20.json').read_text())['judge']
        self.parent_names=['baseline','old_E','old_T'];self.names=['baseline','E_source','EC_before_p1']
        rows=[]
        for i,text in enumerate(('a complete long answer','I don\'t know.','a wrong answer')):
            q={'idx':i,'pair_id':'q%d'%i,'question':'Question %d?'%i,'gold':['gold'],
               'incorrect':['bad'],'best_answer':'gold','outputs':{n:text for n in self.parent_names},
               'finished':{n:True for n in self.parent_names}}
            rows.append(q)
        self.raw={'model':'toy','identity':{'source':'same'},'conditions':self.parent_names,
                  'condition_specs':[{'name':n} for n in self.parent_names],'evaluations':rows,'completed':True}
        entry.qa.save(self.parent/'tune.json',self.raw)
        self.graded=copy.deepcopy(self.raw)
        self.graded.update(grading_completed=True,grading_source_generation_sha256=entry.qa.digest(self.parent/'tune.json'),
                           grading_judge_config_sha256=fingerprint(self.oldcfg),grading_judge_name=self.oldcfg['name'])
        for i,row in enumerate(self.graded['evaluations']):
            label='wrong' if i==2 else 'correct';raw='B' if i==2 else 'A'
            row.update(judge_reference_answers=['gold'],gold_is_abstention=False,
                       judge={n:{'label':label,'raw':raw,'rubric_id':QA_GRADER_VERSION} for n in self.parent_names},
                       outcome={n:label for n in self.parent_names})
        entry.qa.save(self.parent/'tune_scored_items.json',self.graded)

    def reference(self,graded=None):
        return reference_index(self.raw,self.graded if graded is None else graded,
            raw_sha256=entry.qa.digest(self.parent/'tune.json'),reference_config=self.oldcfg,target_config=self.cfg)

    def current(self):
        reference=self.reference();raw=copy.deepcopy(self.raw);raw['conditions']=self.names
        raw['condition_specs']=[{'name':n} for n in self.names]
        raw.update(split='tune',pilot=True,E={},C={},bank_protocol_fingerprint=fingerprint(raw['identity']))
        for row in raw['evaluations']:
            text=row['outputs']['baseline'];row.update(outputs={n:text for n in self.names},finished={n:True for n in self.names},
                rule_abstain={n:False for n in self.names},generation_token_ids={n:[2] for n in self.names},
                trace={n:[{'step':0,'accepted_prefix_ids':[],'next_top1_id':2}] for n in self.names},gate_readout={n:{} for n in self.names})
        entry.qa.save(self.run/'check.json',raw)
        return raw,reference

    def test_policy_conversion_is_uniform_without_changing_parent(self):
        value=self.reference();key=semantic_key(self.cfg,self.graded['evaluations'][1],"I don't know.")
        self.assertEqual(value['index'][key]['original_label'],'correct')
        self.assertEqual(value['index'][key]['grade']['label'],'abstain')
        self.assertEqual(self.graded['evaluations'][1]['judge']['baseline']['label'],'correct')

    def test_parent_conflict_and_foreign_judge_fail(self):
        bad=copy.deepcopy(self.graded);bad['evaluations'][1]['judge']['old_E']['label']='abstain';bad['evaluations'][1]['outcome']['old_E']='abstain'
        with self.assertRaisesRegex(ValueError,'conflicting'):self.reference(bad)
        with self.assertRaisesRegex(ValueError,'requests differ'):
            reference_index(self.raw,self.graded,raw_sha256=entry.qa.digest(self.parent/'tune.json'),
                            reference_config=self.oldcfg,target_config={**self.cfg,'model':'foreign'})
        with self.assertRaisesRegex(ValueError,'temperature'):
            reference_index(self.raw,self.graded,raw_sha256=entry.qa.digest(self.parent/'tune.json'),
                            reference_config=self.oldcfg,target_config={**self.cfg,'temperature':.5})

    def test_real_entry_seeds_same_budget_reference_and_freezes_renamed_EC(self):
        raw,reference=self.current();cache=self.parent/'judge_cache';cache.mkdir()
        lines=[]
        for row in self.graded['evaluations']:
            lines.append({'key':qa_cache_key(self.cfg['model'],row['question'],['gold'],row['outputs']['baseline'],['bad'],config=self.cfg),
                          'model':self.cfg['model'],'answer':row['judge']['baseline']['raw']})
        (cache/cache_filename(self.cfg['model'])).write_text(''.join(json.dumps(v)+'\n' for v in lines))
        cfg=self.cfg
        class CacheOnlyJudge:
            name=cfg['name'];qa_rubric_id=QA_GRADER_VERSION;n_calls=0;n_cache_hits=0
            def __init__(self,cache_dir):
                self.entries={v['key']:v['answer'] for v in map(json.loads,(Path(cache_dir)/cache_filename(cfg['model'])).read_text().splitlines())}
            def qa_grade(self,question,gold,prediction,incorrect=None):
                key=qa_cache_key(cfg['model'],question,gold,prediction,incorrect,config=cfg)
                if key not in self.entries:raise AssertionError('unexpected API request')
                letter=self.entries[key];self.n_cache_hits+=1
                return {'raw':letter,'label':{'A':'correct','B':'wrong','C':'abstain'}[letter],'rubric_id':QA_GRADER_VERSION}
        args=SimpleNamespace(run_dir=self.run,reference_run=self.parent,e_run=self.root/'absent_wrong_48_cache',pilot=True)
        identity={**raw['identity'],'reference_generation_sha256':entry.qa.digest(self.parent/'tune.json'),
                  'reference_grading_sha256':entry.qa.digest(self.parent/'tune_scored_items.json')}
        raw['identity']=identity;raw['bank_protocol_fingerprint']=fingerprint(identity);entry.qa.save(self.run/'check.json',raw)
        state=(None,{'model':'toy'},{'splits':{'tune':raw['evaluations']}},None,None,None,self.raw,self.cfg)
        with patch.object(entry,'frozen',return_value=identity),patch('core.judge.judge_from_config',side_effect=lambda config,cache_dir:CacheOnlyJudge(cache_dir)),\
             patch('e24_real_qa.load_model_config',return_value={'name':'toy'}),contextlib.redirect_stdout(io.StringIO()):
            entry.judge(args,state)
        verification=entry.qa.read(self.run/'check_reference_grade_verification.json')
        self.assertEqual(verification['matched_question_conditions'],9)
        graded=entry.qa.read(self.run/'check_scored_items.json')
        self.assertEqual(graded['evaluations'][1]['outcome']['EC_before_p1'],'abstain')
        altered=copy.deepcopy(graded);altered['evaluations'][0]['judge_reference_answers']=['foreign reference']
        with self.assertRaisesRegex(ValueError,'reference inputs'):
            verify_matching_grades(raw,altered,reference,raw_sha256=entry.qa.digest(self.run/'check.json'),target_config=self.cfg)
        altered=copy.deepcopy(graded);altered['condition_specs'][0]['name']='foreign'
        with self.assertRaisesRegex(ValueError,'identity differs'):
            verify_matching_grades(raw,altered,reference,raw_sha256=entry.qa.digest(self.run/'check.json'),target_config=self.cfg)
        graded['evaluations'][0]['judge']['EC_before_p1']['label']='wrong';graded['evaluations'][0]['outcome']['EC_before_p1']='wrong'
        with self.assertRaisesRegex(ValueError,'changed its frozen'):
            verify_matching_grades(raw,graded,reference,raw_sha256=entry.qa.digest(self.run/'check.json'),target_config=self.cfg)


if __name__=='__main__':unittest.main()
