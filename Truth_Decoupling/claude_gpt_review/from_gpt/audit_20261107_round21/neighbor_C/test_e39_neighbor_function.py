from pathlib import Path
import sys
import unittest
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'from_claude/code'),str(ROOT/'from_claude/code/scripts')]
import e39_neighbor_c_function as m

class E39Test(unittest.TestCase):
 def test_BF16_fit_keeps_external_rho(self):
  g=torch.Generator().manual_seed(42)
  h=torch.randn(92,64,generator=g).bfloat16();w=torch.randn(64,generator=g);w=w/w.norm()
  result=m.match_fixed_target(h,w,.5,.018418746069073677)
  self.assertEqual(result['rho'],.018418746069073677)
  self.assertLessEqual(result['absolute_error'],result['tolerance'])
  self.assertEqual(len(result['search']),48)
  self.assertEqual(len(result['matched_profile']['negative']),92)
  with self.assertRaises(ValueError):m.match_fixed_target(h.float(),w,.5,.018)
  with self.assertRaises(ValueError):m.match_fixed_target(h,w,.0001,.018)

 def fake(self,positive_correct=83,amplitude=.2):
  q=[{'pair_id':str(i),'question':str(i)} for i in range(92)];rows=[]
  for i,r in enumerate(q):
   cond={}
   for name,alpha in m.NAMES.items():
    value=0 if alpha is None else (-amplitude if alpha<0 else amplitude*(1 if i<positive_correct else -1))
    cond[name]={'yes_minus_no':value,'p_yes':.6,'p_no':.4,'top1_is_yes_no':True,
      'trace':{'source_forward_calls':0,'c_writes':[] if alpha is None else [{'layer':14,'alpha':alpha,'stored_delta_over_base':.018}]}}
   rows.append({**r,'conditions':cond})
  return {'completed':True,'check_first3':False,'rows':rows},{'questions':q,'c_layer':14}

 def test_thresholds_require_both_sign_and_magnitude(self):
  raw,plan=self.fake();self.assertTrue(m.summarize(raw,plan)['qualifies_training_function'])
  raw,plan=self.fake(82);self.assertFalse(m.summarize(raw,plan)['qualifies_training_function'])
  raw,plan=self.fake(92,.09);self.assertFalse(m.summarize(raw,plan)['qualifies_training_function'])

 def test_nonfinite_or_wrong_layer_fails(self):
  raw,plan=self.fake();raw['rows'][0]['conditions']['C_plus']['yes_minus_no']=float('nan')
  with self.assertRaises(ValueError):m.summarize(raw,plan)
  raw,plan=self.fake();raw['rows'][0]['conditions']['C_plus']['trace']['c_writes'][0]['layer']=7
  with self.assertRaises(ValueError):m.summarize(raw,plan)

if __name__=='__main__':unittest.main()
