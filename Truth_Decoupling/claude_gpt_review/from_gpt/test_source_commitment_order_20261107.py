"""Executed base/source isolation and causal replay tests for R21."""
import copy
from pathlib import Path
import sys
import tempfile
import io
import contextlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / 'from_claude/code'
sys.path[:0] = [str(HERE), str(CODE/'scripts'), str(CODE)]
from test_qa_das_pipeline_20261105 import PipelineModel, ToyTokenizer
from core.model_io import ModelBundle
from core.qa_source_interchange import SourceInterchangeDecoder, _cache_tensors
from core.qa_source_commitment import SourceCommitmentDecoder, generate,calibrate_coefficients,relative_write_profile
from core.intervene import SteerSpec, steer


class OrderTests(unittest.TestCase):
    def setUp(self):
        tokenizer = ToyTokenizer(); model = PipelineModel(tokenizer)
        model.layers.append(copy.deepcopy(model.layers[0])); model.config.num_hidden_layers = 3
        self.bundle = ModelBundle(model, tokenizer, {'name':'toy-order', 'layers_attr':'layers'})
        self.base, self.source = [31,44,57], [38,39,40,41,57]
        self.e = torch.nn.functional.normalize(torch.arange(1.,9.),dim=0)
        self.c = torch.nn.functional.normalize(torch.arange(8.,0.,-1),dim=0)

    def decoder(self, layer, cache=True, source=None, e=True, alpha=1.):
        return SourceCommitmentDecoder(self.bundle,self.base,self.source if source is None else source,
            e_layer=1 if e else None,e_direction=self.e if e else None,c_layer=layer,
            c_direction=self.c,c_scale=.4,c_alpha=alpha,use_cache=cache)

    def test_both_actual_orders_cached_equal_full_replay_all_past_writes(self):
        for layer in (0,2):
            for alpha in (-1.,1.):
                a=self.decoder(layer,alpha=alpha); b=self.decoder(layer,False,alpha=alpha)
                for step,token in enumerate((70,71,72)):
                    x,tx=a.next_logits(); y,ty=b.next_logits()
                    torch.testing.assert_close(x,y,atol=2e-6,rtol=1e-5)
                    self.assertEqual(len(tx['c_writes']),int(step==0));self.assertEqual(len(ty['c_writes']),1)
                    self.assertEqual(len(ty['writes']),step+1)
                    self.assertEqual(ty['c_writes'][0]['physical_position'],len(self.base)-1)
                    expected=['C','E'] if layer==0 else ['E','C']
                    self.assertEqual([e['operator'] for e in ty['base_operation_events']],expected)
                    self.assertEqual([e['operator'] for e in tx['base_operation_events']],expected if step==0 else ['E'])
                    a.accept(token);b.accept(token)

    def test_source_stays_unmodified_when_token_values_and_lengths_equal(self):
        for source in (self.source,[38,44,57],list(self.base)):
            for layer in (0,2):
                plain=SourceInterchangeDecoder(self.bundle,self.base,source,mode='source_interchange',layer=1,direction=self.e)
                edited=self.decoder(layer,source=source)
                for token in (70,71,72):
                    plain.next_logits(); _,tr=edited.next_logits()
                    self.assertEqual(tr['source_c_writes'],0)
                    for a,b in zip(_cache_tensors(plain.source_cache),_cache_tensors(edited.source_cache)):
                        torch.testing.assert_close(a,b,rtol=0,atol=0)
                    plain.accept(token);edited.accept(token)

    def test_c_only_matches_existing_main_operator_and_only_prompt_end(self):
        for layer in (0,2):
            dec=self.decoder(layer,e=False)
            x=torch.tensor([self.base]);mask=torch.ones_like(x)
            spec=SteerSpec(self.c,layer,mode='add',alpha=1.,coef=[.4],max_calls=1)
            with steer(self.bundle,[spec]):
                ref=self.bundle.model(input_ids=x,attention_mask=mask,use_cache=True)
            observed,tr=dec.next_logits()
            torch.testing.assert_close(observed,ref.logits[0,-1],rtol=0,atol=0)
            self.assertEqual(tr['source_forward_calls'],0)
            self.assertAlmostEqual(tr['c_writes'][0]['ideal_delta_norm'],.4,places=6)
            dec.accept(70);_,tr=dec.next_logits();self.assertEqual(tr['c_writes'],[])

    def test_zero_c_controls_are_identical_to_existing_decoder(self):
        for e in (False,True):
            old=SourceInterchangeDecoder(self.bundle,self.base,self.source,mode='source_interchange' if e else 'baseline',
                                        layer=1 if e else None,direction=self.e if e else None)
            new=SourceCommitmentDecoder(self.bundle,self.base,self.source,e_layer=1 if e else None,e_direction=self.e if e else None)
            for token in (70,71,72):
                x,_=old.next_logits();y,_=new.next_logits();torch.testing.assert_close(x,y,rtol=0,atol=0)
                old.accept(token);new.accept(token)

    def test_failure_cleans_hooks_and_generation_uses_natural_stop(self):
        dec=self.decoder(0,e=False)
        with patch.object(self.bundle.model.layers[2],'forward',side_effect=RuntimeError('injected failure')):
            with self.assertRaisesRegex(RuntimeError,'injected failure'):dec.next_logits()
        self.assertTrue(all(not l._forward_hooks for l in self.bundle.model.layers))
        with torch.no_grad():
            self.bundle.model.head.bias.fill_(-100);self.bundle.model.head.bias[2]=100
        value=generate(self.bundle,self.base,self.source,e_layer=1,e_direction=self.e,c_layer=2,
                       c_direction=self.c,c_scale=.4,c_alpha=-1.,max_new_tokens=5)
        self.assertEqual(value['token_ids'],[2]);self.assertTrue(value['finished'])
        self.assertEqual(len(value['trace']),1)

    def test_same_layer_unfixed_dose_and_unknown_stream_fail(self):
        with self.assertRaises(ValueError):self.decoder(1)
        with self.assertRaises(ValueError):self.decoder(0,alpha=2.)
        with self.assertRaisesRegex(RuntimeError,'unknown forward-stream'):self.decoder(0)._forward(list(self.base),None)

    def test_training_only_calibration_matches_actual_bfloat_stored_norm(self):
        generator=torch.Generator().manual_seed(77)
        states={'before':torch.randn(12,8,generator=generator).bfloat16(),
                'after':(4*torch.randn(12,8,generator=generator)).bfloat16()}
        directions={'before':self.c,'after':self.e};scales={'before':.2,'after':8.}
        old={k:v.clone() for k,v in states.items()}
        result=calibrate_coefficients(states,directions,scales)
        self.assertEqual(result['n_signed_measurements_per_layer'],24)
        self.assertLess(result['layers']['after']['coefficient'],scales['after'])
        for side in states:
            point=result['layers'][side]
            profile=relative_write_profile(states[side],directions[side],point['coefficient'])
            self.assertLessEqual(abs(profile['median']-result['rho']),result['tolerance'])
            self.assertTrue(torch.equal(states[side],old[side]))
        with self.assertRaises(ValueError):calibrate_coefficients(states,directions,{'before':0.,'after':8.})

    def test_entry_really_generates_all_ten_arms_and_gates(self):
        import e37_source_commitment_order as entry
        cfg={'qa_system':'Be accurate.','qa_user':'Question: {question}','max_new_tokens':4,'batch_size':1}
        adapter={'layer':1,'W':self.e,'config':cfg}
        questions=[{'idx':i,'pair_id':'q%d'%i,'question':'What is %d?'%i,'gold':['A'],'incorrect':['B'],'best_answer':'A'} for i in range(3)]
        parent=[]
        for q in questions:
            base,source=entry.source_eval.prompt_tokens(self.bundle,cfg,q['question'])
            row={**q,'outputs':{},'finished':{},'generation_token_ids':{},'prompt_token_ids':{'base':base,'source':source}}
            for name,kw in [('baseline',{}),(entry.preference.METHOD,{'e_layer':1,'e_direction':self.e})]:
                value=generate(self.bundle,base,source,max_new_tokens=4,**kw)
                row['outputs'][name]=value['output'];row['finished'][name]=value['finished'];row['generation_token_ids'][name]=value['token_ids']
            parent.append(row)
        plan={'toy':{'e_layer':1,'before':0,'after':2,'budget':4}}
        with tempfile.TemporaryDirectory() as tmp,patch.dict(entry.MODEL_PLAN,plan,clear=True):
            specs=entry.conditions('toy');identity={'conditions':[c['name'] for c in specs],'condition_specs':specs,
                'implementation':{},'max_new_tokens':4,'c_coefficients':{'before':.2,'after':.2}}
            values={side:{'direction':self.c,'scale':.2} for side in ('before','after')}
            state=(None,{'model':'toy'},{'splits':{'tune':questions}},adapter,{'config':{}},values,{'evaluations':parent},{})
            args=SimpleNamespace(run_dir=Path(tmp),pilot=True)
            with patch.object(entry,'frozen',return_value=identity),patch.object(entry,'implementation_identity',return_value={}),\
                 patch.object(entry.qa,'verify_bundle'),patch.object(entry.joint,'gate_map',return_value={'yes':92,'no':81}),\
                 contextlib.redirect_stdout(io.StringIO()):
                raw=entry.generate(args,state,bundle=self.bundle)
            self.assertTrue(raw['completed']);self.assertEqual(len(raw['evaluations']),3)
            self.assertEqual(sum(len(r['outputs']) for r in raw['evaluations']),30)
            entry.validate_rows(raw,questions,True)
            for row in raw['evaluations']:
                self.assertEqual(len(row['gate_readout']),10)
                for side in ('before','after'):
                    for sign in ('m1','p1'):
                        self.assertEqual(row['trace']['C_'+side+'_'+sign][0]['c_writes'][0]['scale'],
                                         row['trace']['EC_'+side+'_'+sign][0]['c_writes'][0]['scale'])
            raw['evaluations'][0]['trace']['EC_before_m1'][0]['next_top1_id']=-1
            with self.assertRaises(ValueError):entry.validate_rows(raw,questions,True)


if __name__=='__main__':unittest.main()
