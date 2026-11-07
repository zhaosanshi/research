"""R21 metadata prevents confounding order, dose and inherited answer budgets."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parent.parent;CODE=ROOT/'from_claude/code'
sys.path[:0]=[str(CODE/'scripts'),str(CODE)]
import e32_source_matched_coverage as scorer
from e37_source_commitment_order import conditions

class OrderCoverageTests(unittest.TestCase):
    def fixture(self,model='qwen3-8b',pilot=False):
        qids=['q%d'%i for i in range(3 if pilot else 192)]
        cfg={'qa_system':'question only','qa_user':'{question}','max_new_tokens':48,'batch_size':8}
        layer,before,after,budget,nlayers=(17,8,26,256,36) if model=='qwen3-8b' else (15,7,23,48,32)
        specs=conditions(model)
        ident={'protocol':scorer.SOURCE_PROTOCOLS[7],'model':model,'config':cfg,
            'layer':layer,'e_layer':layer,'c_layers':{'before':before,'after':after},
            'max_new_tokens':budget,'batch_size':1,'question_ids':qids,'conditions':scorer.ORDER_NAMES,
            'alpha':1.,'c_alphas':[-1.,1.],'c_position':'base_prompt_end_only','same_prefix_tokens':True,
            'source_has_C':False,'training_called':False,'all_arms_freshly_generated':True,'pilot':pilot,
            'gold_used_at_inference':False,
            'calibration_mode':'train_relative_norm_matched','c_calibration_sha256':'c'*64,
            'c_coefficients':{'before':.2,'after':.7},'condition_specs':specs}
        for key in ('adapter_sha256','adapter_manifest_sha256','c_bank_meta_sha256','c_bank_sha256','c_data_sha256','reference_generation_sha256'):ident[key]='a'*64
        return {'identity':ident,'condition_specs':deepcopy(specs)}, {'model':model,'config':cfg,'num_layers':nlayers},qids

    def test_both_fixed_model_plans_and_full_calibrated_arms_are_accepted(self):
        for model in ('llama3.1-8b','qwen3-8b'):
            raw,meta,ids=self.fixture(model)
            self.assertEqual(scorer.order_source_names(raw,meta,ids,False),scorer.ORDER_NAMES)

    def test_wrong_physical_order_budget_source_or_training_is_rejected(self):
        raw,meta,ids=self.fixture()
        for key,value in [('e_layer',26),('c_layers',{'before':26,'after':8}),('max_new_tokens',48),
                          ('source_has_C',True),('all_arms_freshly_generated',False),('training_called',True),
                          ('c_bank_sha256','missing'),('c_coefficients',{'before':.2,'after':float('nan')})]:
            bad=deepcopy(raw);bad['identity'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):scorer.order_source_names(bad,meta,ids,False)

    def test_original_sigma_only_allowed_for_three_question_diagnostic(self):
        raw,meta,ids=self.fixture(pilot=True);raw['identity']['calibration_mode']='original_sigma_diagnostic';raw['identity']['c_calibration_sha256']=None
        self.assertEqual(scorer.order_source_names(raw,meta,ids,True),scorer.ORDER_NAMES)
        with self.assertRaises(ValueError):scorer.order_source_names(raw,meta,ids,False)
        raw['identity']['calibration_mode']='train_relative_norm_matched'
        with self.assertRaises(ValueError):scorer.order_source_names(raw,meta,ids,True)

    def test_mutating_both_copies_of_condition_cannot_swap_oracle_or_c_sign(self):
        raw,meta,ids=self.fixture()
        for field in ('sign','gold','E'):
            bad=deepcopy(raw)
            if field=='sign':bad['condition_specs'][2]['c']['alpha']=1.
            elif field=='gold':bad['condition_specs'][2]['uses_reference_answer']=True
            else:bad['condition_specs'][3]['e']['layer']=25
            bad['identity']['condition_specs']=deepcopy(bad['condition_specs'])
            with self.subTest(field=field),self.assertRaises(ValueError):scorer.order_source_names(bad,meta,ids,False)

    def test_neighbor_extension_preserves_complete_parent_and_exact_inherited_specs(self):
        raw,meta,ids=self.fixture(model='llama3.1-8b')
        raw['identity']['source']={'frozen':'source'};raw['identity']['source_system']='fixed source'
        full=deepcopy(raw['identity']);child=deepcopy(full)
        child.update(protocol=scorer.SOURCE_PROTOCOLS[8],neighbor_c_layer=14,neighbor_c_coefficient=.4,
                     all_arms_freshly_generated=False,all_controls_inherited_exactly=True,
                     c_layers={**full['c_layers'],'neighbor':14},
                     c_coefficients={**full['c_coefficients'],'neighbor':.4},conditions=scorer.ORDER_NAMES+scorer.NEIGHBOR_NAMES)
        for k in ('parent_generation_sha256','parent_scored_sha256','parent_plan_sha256',
                  'e39_plan_sha256','e39_report_sha256','e39_calibration_sha256','e39_gate_sha256'):child[k]='f'*64
        specs=deepcopy(full['condition_specs'])
        for name in scorer.NEIGHBOR_NAMES:
            enabled=name.startswith('EC_')
            specs.append({'name':name,'e_enabled':enabled,'e':{'layer':15,'alpha':1.} if enabled else None,
                          'uses_reference_answer':False,'c':{'side':'neighbor','layer':14,
                          'alpha':-1. if name.endswith('_m1') else 1.,'position':'base_prompt_end_only'}})
        child['condition_specs']=specs
        value={'identity':child,'control_identity':full,'condition_specs':deepcopy(specs)}
        self.assertEqual(len(scorer.neighbor_source_names(value,meta,ids,ids,False)),14)
        trial=deepcopy(value);trial['identity']['question_ids']=ids[:3]
        self.assertEqual(len(scorer.neighbor_source_names(trial,meta,ids[:3],ids,True)),14)
        for case in ('old_arm','parent','layer','coefficient','gold'):
            bad=deepcopy(value)
            if case=='old_arm':
                bad['condition_specs'][2]['c']['alpha']=1.
                bad['identity']['condition_specs']=deepcopy(bad['condition_specs'])
            elif case=='parent':bad['control_identity']['question_ids']=ids[:-1]
            elif case=='layer':bad['identity']['neighbor_c_layer']=13
            elif case=='coefficient':bad['identity']['neighbor_c_coefficient']=.8
            else:bad['control_identity']['gold_used_at_inference']=True
            with self.subTest(case=case),self.assertRaises(ValueError):
                scorer.neighbor_source_names(bad,meta,ids,ids,False)

if __name__=='__main__':unittest.main()
