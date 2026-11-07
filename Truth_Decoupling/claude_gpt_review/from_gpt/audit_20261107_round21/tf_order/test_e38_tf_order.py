import importlib.util
from pathlib import Path
import sys
import os
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
CODE=ROOT/'from_claude/code'
sys.path[:0]=[str(CODE),str(CODE/'scripts')]
import e38_tf_commitment_order as m
from core.intervene import SteerSpec

class Bundle:
    device=torch.device('cpu')
    def __init__(self):
        self.layers=[torch.nn.Identity() for _ in range(32)]
    def layer(self,i):return self.layers[i]
    def encode(self,texts):
        return {'input_ids':torch.tensor([[1,2]]*len(texts)), 'attention_mask':torch.ones((len(texts),2),dtype=torch.long)}
    def logits(self,inputs,**kw):
        h=torch.tensor([1.,2.,3.,4.]).expand(len(inputs['input_ids']),2,4).clone()
        for layer in self.layers:h=layer(h)
        return h

class E38Test(unittest.TestCase):
    @classmethod
    def real_source_root(cls):
        configured=os.environ.get('TD_R21_TF_FIXTURES')
        if configured:
            base=Path(configured)
            if not base.is_dir():raise FileNotFoundError('TD_R21_TF_FIXTURES directory does not exist')
            return base
        if not hasattr(cls,'_source_fixture'):
            archive=ROOT/'from_gpt/audit_20261107_round20/tf_ability/artifacts.tar.gz'
            cls._source_fixture=tempfile.TemporaryDirectory()
            cls.addClassCleanup(cls._source_fixture.cleanup)
            base=Path(cls._source_fixture.name)
            with tarfile.open(archive) as tf:
                for model in m.EXPECTED_E:
                    for filename in m.SOURCE_FILES:
                        rel=Path(model)/'tf_ability_full'/filename
                        member=tf.getmember(str(rel))
                        if not member.isfile():raise ValueError('real fixture must be a regular archived file')
                        dest=base/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                        dest.write_bytes(tf.extractfile(member).read())
        return Path(cls._source_fixture.name)

    def test_actual_layer_order_and_norms_not_list_order(self):
        b=Bundle()
        specs=[SteerSpec([1.,0.,0.,0.],11,mode='amplify',alpha=3.,max_calls=1,name='E'),
               SteerSpec([0.,1.,0.,0.],7,mode='add',coef=[2.],alpha=-1.,max_calls=1,name='C')]
        with patch.object(m,'forward_logits',lambda bundle,inputs,**kw:bundle.logits(inputs)):
            logits,trace,tokens=m.traced_forward(b,['x','y'],specs)
        self.assertEqual([v['layer'] for v in trace[0]],[7,11])
        self.assertEqual(trace[0][0]['requested_delta_l2'],2.)
        self.assertEqual(trace[0][0]['actual_delta_l2'],2.)
        np.testing.assert_array_equal(logits,np.array([[4,0,3,4]]*2))
        self.assertEqual(len(tokens),2)
        self.assertTrue(all(len(l._forward_hooks)==0 for l in b.layers))

    def test_same_layer_orthogonal_operations_commute_in_fp32(self):
        from core.intervene import LayerSteerHook
        h=torch.tensor([[[1.,2.,3.,4.]]])
        def spec():
            return [SteerSpec([1.,0.,0.,0.],2,mode='amplify',alpha=3.,name='E'),
                    SteerSpec([0.,1.,0.,0.],2,mode='add',coef=[2.],alpha=-1.,name='C')]
        forward=LayerSteerHook(spec())(None,None,h)
        reverse=LayerSteerHook(list(reversed(spec())))(None,None,h)
        self.assertTrue(torch.equal(forward,reverse))

    def test_fixed_E_and_C_pair_controls_real_sources(self):
        base=self.real_source_root()
        for model in m.EXPECTED_E:
            a=SimpleNamespace(model=model,source_run=base/model/'tf_ability_full',limit=3)
            p=m.make_plan(a)
            self.assertEqual(len(p['conditions']),19);self.assertEqual(len(p['input_rows']),6)
            for c in p['conditions']:
                if c['kind']=='EC':
                    self.assertEqual(c['e'],p['fixed_E'][c['method']]['e'])
                    matching=[x for x in p['conditions'] if x['kind']=='C' and x['method']==c['method'] and x['c']==c['c']]
                    self.assertEqual(len(matching),1)
            self.assertTrue(all(set(r)=={'id','pair_id','text'} for r in p['input_rows']))
            a.limit=2
            with self.assertRaises(ValueError):m.make_plan(a)

    def test_calibration_uses_all_train_pairs_and_preserves_bank(self):
        base=self.real_source_root()/'llama3.1-8b/tf_ability_full'
        import json
        data=json.loads((base/'data.json').read_text())
        rows=m.calibration_rows(data)
        self.assertEqual(len(rows),192);self.assertEqual(len({r['pair_id'] for r in rows}),96)
        self.assertTrue(all(set(r)=={'id','pair_id','text'} for r in rows))
        bank={'tf_das_C_L7_scale':np.array(2.),'tf_das_C_L23_scale':np.array(4.),'tf_das_C_L7':np.array([1.,0.])}
        plan={'dose_mode':'train_relative','c_layers':[7,23],'calibration':{'methods':{'das':{'layers':{'before':{'coefficient':.5},'after':{'coefficient':1.}}}}}}
        applied=m.effective_bank(bank,plan)
        self.assertEqual(float(applied['tf_das_C_L7_scale']),.5)
        self.assertEqual(float(bank['tf_das_C_L7_scale']),2.)
        self.assertIs(applied['tf_das_C_L7'],bank['tf_das_C_L7'])
        with self.assertRaises(ValueError):m.load_calibration(SimpleNamespace(dose_mode='original_sigma',calibration_run=None,limit=None),{},{},{},[])

    def test_zero_doses_reduce_to_existing_single_direction(self):
        b=Bundle()
        def specs(e,c):
            return [SteerSpec([1.,0.,0.,0.],11,mode='amplify',alpha=e,max_calls=1,name='E'),
                    SteerSpec([0.,1.,0.,0.],7,mode='add',coef=[2.],alpha=c,max_calls=1,name='C')]
        with patch.object(m,'forward_logits',lambda bundle,inputs,**kw:bundle.logits(inputs)):
            ec0=m.traced_forward(b,['x'],specs(3.,0.))[0]
            e=m.traced_forward(b,['x'],specs(3.,0.)[:1])[0]
            e0c=m.traced_forward(b,['x'],specs(0.,-1.))[0]
            c=m.traced_forward(b,['x'],specs(0.,-1.)[1:])[0]
        np.testing.assert_array_equal(ec0,e);np.testing.assert_array_equal(e0c,c)

    def test_inference_rejects_gold_before_model_call(self):
        with self.assertRaises(ValueError):m.infer_chunk(None,None,None,[dict(id='x',pair_id='p',text='x',label=True)],{})

if __name__=='__main__':unittest.main()
