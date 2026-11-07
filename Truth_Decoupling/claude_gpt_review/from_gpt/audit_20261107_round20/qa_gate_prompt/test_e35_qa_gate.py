import copy,json,tempfile
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[3];CODE=ROOT/'from_claude/code'
sys.path[:0]=[str(ROOT/'from_gpt'),str(CODE/'scripts'),str(CODE)]
import test_qa_das_pipeline_20261105 as fixtures
from core.model_io import ModelBundle
import e35_qa_gate_prompt_diagnostic as diag

class CompatibleToyTokenizer(fixtures.ToyTokenizer):
    def decode(self,ids,skip_special_tokens=True,**kwargs):
        return super().decode(ids,skip_special_tokens=skip_special_tokens)

class E35Tests(unittest.TestCase):
    def test_real_toy_original_answers_and_both_gate_readouts_without_gold(self):
        tok=CompatibleToyTokenizer();model=fixtures.PipelineModel(tok)
        bundle=ModelBundle(model,tok,{'name':'tiny-ability','layers_attr':'layers'})
        meta={'model':'tiny-ability','fingerprint':'fixture','token_audit':[],
              'config':{'batch_size':2,'max_new_tokens':2,'seed':42,'qa_gate_user':'Question: {question}\nCan you answer? Reply Yes or No.',
                        'gate_labels':{'yes':'Yes','no':'No'}},
              'qa_config':{'qa_system':'Answer.','qa_user':'Question: {question}\nAnswer:'}}
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);src=root/'source';src.mkdir()
            for name in diag.SOURCE_FILES:(src/name).write_text('{}')
            args=SimpleNamespace(model='tiny-ability',source_run=src,run_dir=root/'run',pilot=True)
            with patch.object(diag,'sources',return_value=(meta,{},[])):
                diag.prepare(args);raw=diag.collect(args,bundle=bundle);result=diag.report(args)
            self.assertTrue(raw['completed']);self.assertEqual(len(raw['baseline_original_answers']),6)
            self.assertEqual({n:len(rs) for n,rs in raw['rows'].items()},{'legacy_question_gate':6,'explicit_substantive_ability':6})
            for rows in raw['rows'].values():
                for row in rows:self.assertNotIn('expected_original_answer',row['prompt'])
            self.assertTrue((args.run_dir/'source_snapshot/scripts/e35_qa_gate_prompt_diagnostic.py').exists())

    def test_full_report_keeps_correct_no_and_applies_same_guard_to_both_readouts(self):
        rows=[{'id':str(i),'pair_id':str(i),'question':'Question '+str(i)} for i in range(3)]
        base=[{'id':str(i),'pair_id':str(i),'output':text,'score':s,'simple_score':simple,'outcome':label,
               'grade':{'label':label,'raw':'A' if label=='correct' else 'B','rubric_id':'qa_semantic_v3'},'gold_is_abstention':False}
              for i,(text,s,simple,label) in enumerate([('No, it does not.',-20.,-.01,'correct'),('Yes, definitely.',10.,-.8,'wrong'),("I don't know.",-30.,-.1,'correct')])]
        class Tok:
            def decode(self,ids,**kw):return 'Yes' if ids[0]==0 else 'No'
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'plan.json').write_text('{}')
            plan={'model':'fixture','pilot':False,'split':'tune','implementation':{},'seed':42,'question_rows':rows,
                  'conditions':[{'name':'legacy_question_gate'},{'name':'explicit_substantive_ability'}]}
            raw={'protocol':diag.PROTOCOL,'model':'fixture','pilot':False,'completed':True,
                 'plan_sha256':diag.joint.digest(p/'plan.json'),'implementation':{},'legacy_source_score_max_abs_diff':0.,'rows':{}}
            for name,scores in [('legacy_question_gate',[-20.,10.,-30.]),('explicit_substantive_ability',[20.,10.,-30.])]:
                raw['rows'][name]=[{**row,'prompt':'fixture','input_ids':[1],'attention_mask':[1],
                    **diag.probability_record(np.array([score,0.,-10.],dtype=np.float32),0,1,Tok())} for row,score in zip(rows,scores)]
            diag.joint.save(p/'readouts.json',raw)
            args=SimpleNamespace(model='fixture',run_dir=p,pilot=False)
            with patch.object(diag,'load_plan',return_value=plan),patch.object(diag,'sources',return_value=({}, {'qa':{'tune':rows}},base)):
                r=diag.report(args)
            view=r['policies']['full_personal_refusal_guard']['conditions']
            self.assertEqual(view['legacy_question_gate']['n_available'],2)
            self.assertEqual(view['explicit_substantive_ability']['n_available'],2)
            self.assertEqual(view['legacy_question_gate']['correctness_auc_among_answers'],0.)
            self.assertEqual(view['explicit_substantive_ability']['correctness_auc_among_answers'],1.)
            self.assertEqual(view['explicit_substantive_ability']['quotas']['0.5']['correct'],1)
            self.assertEqual(r['policies']['legacy']['conditions']['legacy_question_gate']['n_available'],3)

    def test_synthetic_expected_polarities_never_enter_question_records(self):
        rows=diag.synthetic_questions();self.assertEqual(len(rows),6);self.assertEqual(len({r['pair_id'] for r in rows}),3)
        self.assertTrue(all(set(r)=={'id','pair_id','question'} for r in rows))
        self.assertTrue(all('expected' not in r['question'] for r in rows))

if __name__=='__main__':unittest.main()
