import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[3];CODE=ROOT/'from_claude/code'
sys.path[:0]=[str(CODE/'scripts'),str(CODE)]
spec=importlib.util.spec_from_file_location('e31_under_test',CODE/'scripts/e31_causal_gate_layers.py')
e31=importlib.util.module_from_spec(spec);spec.loader.exec_module(e31)


class E31Tests(unittest.TestCase):
    def fixture(self):
        meta={'num_layers':32,'conditions':{},'c_layers':{}}
        selection={'tasks':{}};bank={}
        for task in ('tf','qa'):
            meta['conditions'][task]=[];meta['c_layers'][task]={};selection['tasks'][task]={'choices':{'0.5':{}}}
            for method in ('das','md'):
                name=method+'_E_fixed';meta['conditions'][task].append({'name':name,'kind':'E','method':method,
                    'e':{'layer':10,'alpha':2.,'mode':'continuous' if task=='qa' else 'prefill'}})
                selection['tasks'][task]['choices']['0.5'][method+'_E']=name
                meta['c_layers'][task][method]=[26]
                for layer in (7,15,23,26):
                    key=task+'_'+method+'_C_L'+str(layer);bank[key]=np.array([1.,0.]);bank[key+'_scale']=np.array(2.)
        return meta,selection,bank

    def test_default_layer_grid_is_declared_not_data_selected(self):
        self.assertEqual(e31.requested_layers(32,[26]),[7,15,23,26])
        self.assertEqual(e31.requested_layers(36,[31]),[8,17,26,31])
        self.assertEqual(e31.requested_layers(32,[26],True),list(range(32)))

    def test_unavailable_layer_is_recorded_without_substitution(self):
        meta,selection,bank=self.fixture();del bank['qa_das_C_L15']
        p=e31.task_plan('qa',meta,bank,selection,[],False)
        self.assertEqual(p['unavailable_requested_layers'],[{'method':'das','layer':15,
            'reason':'no_available_estimation_C_in_original_bank','original_read_layer':False}])
        das_layers={c['c']['layer'] for c in p['conditions'] if c['method']=='das' and c['c']}
        self.assertEqual(das_layers,{7,23,26})

    def test_fixed_e_and_frozen_answer_sources_stay_distinct(self):
        meta,selection,bank=self.fixture();p=e31.task_plan('qa',meta,bank,selection,[],False)
        self.assertEqual(len(p['conditions']),35)
        for c in p['conditions']:
            if c['kind']=='C':self.assertIsNone(c['e']);self.assertEqual(c['answer_condition'],'baseline')
            if c['kind']=='EC':
                self.assertEqual(c['e'],{'layer':10,'alpha':2.,'mode':'continuous'})
                self.assertEqual(c['answer_condition'],c['method']+'_E_fixed')
            if c['c']:self.assertIn(c['c']['alpha'],(-1.,1.))

    def test_model_inputs_remove_all_grade_and_reference_fields(self):
        r={'pair_id':'x','question':'Q?','gold':['secret answer'],'outcome':'correct','label':True,'output':'secret answer'}
        self.assertEqual(e31.clean_input(r,'qa'),{'id':'x','pair_id':'x','question':'Q?'})
        r.update(id='x:T',text='Statement')
        self.assertEqual(e31.clean_input(r,'tf'),{'id':'x:T','pair_id':'x','text':'Statement'})

    def test_join_changes_only_gate_score(self):
        old=[{'id':'x','pair_id':'x','output':'fixed answer','outcome':'correct','finished':False,'score':1.,'simple_score':-2.}]
        new=e31.join_answers(old,[{'id':'x','pair_id':'x','score':7.}])
        self.assertEqual(new,[{**old[0],'score':7.}]);self.assertEqual(old[0]['score'],1.)
        with self.assertRaises(ValueError):e31.join_answers(old,[{'id':'x','pair_id':'other','score':7.}])

    def test_native_gate_zero_is_no_and_refusal_never_counts(self):
        rows=[{'id':str(i),'pair_id':str(i),'outcome':outcome,'score':score}
              for i,(outcome,score) in enumerate([('abstain',100.),('wrong',0.),('correct',.1),('wrong',.2)])]
        d=e31.delivery_summary(rows);self.assertEqual((d['answered'],d['correct'],d['wrong']),(2,1,1))
        self.assertEqual(d['actual_coverage'],.5)
        self.assertEqual(e31.delivery_summary(rows,False)['answered'],3)

    def test_score_artifact_rejects_labels_missing_rows_duplicates_and_nonfinite(self):
        inputs=[{'id':'a','pair_id':'a'},{'id':'b','pair_id':'b'}]
        good=[{'id':'a','pair_id':'a','score':1.},{'id':'b','pair_id':'b','score':2.}]
        e31.validate_condition_rows(good,inputs)
        for bad in ([good[0]], [good[0],good[0]], [{**good[0],'label':True},good[1]], [{**good[0],'score':float('nan')},good[1]]):
            with self.assertRaises(ValueError):e31.validate_condition_rows(bad,inputs)

    def test_quota_summarization_counts_actual_answers(self):
        rows=[{'id':str(i),'pair_id':str(i),'outcome':o,'score':float(3-i)} for i,o in enumerate(('abstain','correct','wrong'))]
        out=e31.summarize_condition(rows,[.5,1.],42)
        self.assertEqual(out['quotas']['0.5']['answered'],1)
        self.assertEqual(out['quotas']['0.5']['correct'],1)
        self.assertFalse(out['quotas']['1.0']['feasible'])

    def test_report_matches_every_native_point_to_exact_control_count(self):
        meta,selection,bank=self.fixture()
        with tempfile.TemporaryDirectory() as td:
            import json
            p=Path(td);(p/'plan.json').write_text('{}')
            inputs=[{'id':str(i),'pair_id':str(i),'question':'q'} for i in range(3)]
            plan={'model':'llama3.1-8b','pilot_limit':3,'implementation':{},'coverages':[.25,.5,.75],'seed':42,
                  'tasks':{t:e31.task_plan(t,meta,bank,selection,inputs,False) for t in ('tf','qa')}}
            tune={};tasks={}
            for task,tp in plan['tasks'].items():
                base=[{'id':str(i),'pair_id':str(i),'output':'fixed'+str(i),'outcome':o,'finished':True,
                       'score':float(i+1),'simple_score':float(i+1)} for i,o in enumerate(('correct','wrong','abstain'))]
                tune[task]={'baseline':base,'das_E_fixed':base,'md_E_fixed':base}
                tasks[task]={'rows':{},'control_source_score_max_abs_diff':{}}
                for c in tp['conditions']:
                    tasks[task]['rows'][c['name']]=[{'id':str(i),'pair_id':str(i),'score':float(i+1) if c['control'] else [5.,-5.,5.][i]} for i in range(3)]
                    if c['control']:tasks[task]['control_source_score_max_abs_diff'][c['name']]=0.
            artifact={'protocol':e31.PROTOCOL,'model':'llama3.1-8b','split':'tune','completed':True,'pilot_limit':3,
                      'plan_sha256':e31.joint.digest(p/'plan.json'),'implementation':{},'tasks':tasks}
            (p/'gate_scores.json').write_text(json.dumps(artifact))
            args=SimpleNamespace(run_dir=p,source_run=p,model='llama3.1-8b')
            with patch.object(e31,'load_plan',return_value=plan),patch.object(e31,'source',return_value=(meta,bank,{},selection,tune)),patch.object(e31,'require_unchanged'):
                result=e31.report(args)
                rec=result['tasks']['qa']['conditions']['qa_das_C_L7_a1']
                self.assertEqual(rec['native_gate']['answered'],1)
                self.assertEqual(rec['same_actual_count_controls']['base_simple']['answered'],1)
                self.assertEqual(rec['same_actual_count_controls']['base_simple']['correct'],0)
                self.assertEqual(rec['same_actual_count_controls']['base_simple']['delta_correct_native_minus_control'],1)
                artifact['model']='foreign';(p/'gate_scores.json').write_text(json.dumps(artifact))
                with self.assertRaisesRegex(ValueError,'foreign'):e31.report(args)


if __name__=='__main__':unittest.main()
