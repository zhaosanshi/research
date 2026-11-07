"""Regression of comparison direction and incomplete statistical families."""
from copy import deepcopy
from pathlib import Path
import sys,unittest,json,tempfile
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analyze_source_order as analysis

class SourceOrderAnalysisTests(unittest.TestCase):
    def test_swapped_model_and_wrong_quota_are_rejected_before_tabulation(self):
        point={'n':192,'k':0,'arms':{n:{'feasible':True,'answered':48} for n in analysis.NAMES}}
        table={'coverages':[{**deepcopy(point),'target_coverage':q} for q in (.25,.5,.75)]}
        data={'model':'llama3.1-8b','conditions':list(analysis.NAMES),'n_boot':2000,
              'original_generation_protocol':'source_e_commitment_layer_order_v1','source_files':{},
              'common_unmodified_confidence_corrected':table,'native_policy_confidence_corrected':table,
              'quality_sensitivity':{'common_unmodified_model':table,'native_policy':table}}
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'report.json';p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'registered E37'):
                analysis.summarize({'qwen3-8b':p})
            with self.assertRaisesRegex(ValueError,'wrong quota'):
                analysis.summarize({'llama3.1-8b':p})

    def test_reversed_comparison_reverses_interval_not_probability(self):
        point={'arms':{'E':{'feasible':True},'EC':{'feasible':True}},'comparisons':[
            {'arm_a':'E','arm_b':'EC','exact_fixed_deliveries':{'p_value':.125},
             'ci95':{'accuracy_answered':[-.2,-.05]},'bootstrap_status':'ok','infeasible_resamples':0}]}
        got=analysis.comparison(point,'EC','E')
        self.assertEqual(got['ci95'],[.05,.2]);self.assertEqual(got['p'],.125)
        self.assertEqual(analysis.comparison(point,'E','EC')['ci95'],[-.2,-.05])

    def test_feasible_missing_duplicate_or_unresolved_test_is_not_filled_as_p_one(self):
        point={'arms':{'E':{'feasible':True},'EC':{'feasible':True}},'comparisons':[]}
        with self.assertRaises(ValueError):analysis.comparison(point,'EC','E')
        row={'arm_a':'E','arm_b':'EC','exact_fixed_deliveries':{'p_value':None}}
        for comps in ([row],[row,row]):
            point['comparisons']=comps
            with self.assertRaises(ValueError):analysis.comparison(point,'EC','E')
        point['arms']['EC']['feasible']=False
        with self.assertRaises(ValueError):analysis.comparison(point,'EC','E')
        point['comparisons']=[{'arm_a':'E','arm_b':'EC','exact_fixed_deliveries':None,
                              'ci95':None,'bootstrap_status':'infeasible_point'}]
        got=analysis.comparison(point,'EC','E');self.assertIsNone(got['p']);self.assertIsNone(got['ci95'])

    def test_planned_family_size_is_retained_without_fabricating_infeasible_p_values(self):
        rows=[{'p':p} for p in (.01,.04,.2,None)]
        self.assertEqual(analysis.holm(rows,'p'),[.04,.12,.4,None])
        self.assertEqual(analysis.holm([{'p':None}]*4,'p'),[None]*4)
        for p in (float('nan'),-.1,1.1):
            with self.assertRaises(ValueError):analysis.holm([{'p':p}],'p')

if __name__=='__main__':unittest.main()
