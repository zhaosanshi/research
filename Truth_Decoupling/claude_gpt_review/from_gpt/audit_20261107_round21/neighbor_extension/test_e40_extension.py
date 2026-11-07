from pathlib import Path
import copy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'from_claude/code'),str(ROOT/'from_claude/code/scripts'),str(ROOT/'from_gpt')]
import e40_neighbor_c_extension as m
from test_qa_das_pipeline_20261105 import PipelineModel,ToyTokenizer
from core.model_io import ModelBundle

class ExtensionTests(unittest.TestCase):
 def fixture(self):
  specs=m.parent_entry.conditions(m.MODEL);names=[s['name'] for s in specs]
  q={'pair_id':'p','question':'what?','gold':['answer'],'incorrect':['wrong'],'best_answer':'answer'}
  row={**q,'idx':0,'prompt_token_ids':{'base':[3,4],'source':[5,6]},'gate_prompt_ids':[7,8]}
  for field in m.ARM_FIELDS:
   row[field]={n:({'output':'answer','finished':True,'rule_abstain':False,'generation_token_ids':[2],
     'trace':[{'accepted_prefix_ids':[],'next_top1_id':2}],'gate_readout':{'yes_minus_no':1}}.get(field)) for n in names}
  row['outputs']={n:'answer' for n in names}
  identity={'c_layers':{'before':7,'after':23},'c_coefficients':{'before':.1,'after':.2},'c_directions':{},'fingerprint':'old'}
  parent={'identity':identity,'protocol':m.operation.PROTOCOL,'condition_specs':specs,'conditions':names,'evaluations':[row]}
  child={'control_identity':copy.deepcopy(identity),'parent_protocol':parent['protocol'],'conditions':names+[c['name'] for c in m.new_conditions()],
         'evaluations':[copy.deepcopy(row)]}
  return q,parent,child

 def test_all_ten_parent_fields_must_be_exact(self):
  q,parent,child=self.fixture();m.validate_inheritance(child,parent,[q])
  child['evaluations'][0]['trace'][parent['conditions'][8]][0]['next_top1_id']=3
  with self.assertRaises(ValueError):m.validate_inheritance(child,parent,[q])
  q,parent,child=self.fixture();child['evaluations'][0]['outputs'][parent['conditions'][9]]='changed'
  with self.assertRaises(ValueError):m.validate_inheritance(child,parent,[q])

 def test_plan_deepcopies_parent_and_binds_all_required_sources(self):
  q,parent,_=self.fixture();old=copy.deepcopy(parent)
  state={'parent':parent,'prepared':{'splits':{'tune':[q]}},'coefficient':.149,'rho':.0184,
         'C':torch.tensor([1.,0.]),'e39_plan':{'C_original_scale':.61}}
  a=SimpleNamespace(pilot=True,parent_run=Path('/synthetic'),parent_grades=Path('/synthetic/grades.json'),e39_run=Path('/synthetic/e39'))
  with patch.object(m.qa,'digest',return_value='1'*64),patch.object(m,'implementation_identity',return_value={}):
   identity=m.plan(a,state)
  self.assertEqual(parent,old);self.assertEqual(len(identity['conditions']),14)
  self.assertEqual(identity['c_layers']['neighbor'],14);self.assertEqual(identity['c_coefficients']['neighbor'],.149)
  for key in ('parent_generation_sha256','parent_scored_sha256','e39_report_sha256','e39_calibration_sha256'):
   self.assertEqual(identity[key],'1'*64)
  self.assertFalse(identity['all_arms_freshly_generated'])

 def test_actual_neighbor_decoder_E_and_source_isolation(self):
  tok=ToyTokenizer();model=PipelineModel(tok)
  while len(model.layers)<16:model.layers.append(copy.deepcopy(model.layers[0]))
  model.config.num_hidden_layers=16;b=ModelBundle(model,tok,{'name':'toy','layers_attr':'layers'})
  e=torch.nn.functional.normalize(torch.arange(1.,9.),dim=0);c=torch.nn.functional.normalize(torch.arange(8.,0.,-1),dim=0)
  state={'adapter':{'W':e},'C':c,'coefficient':.149}
  for condition in m.new_conditions():
   dec=m.operation.SourceCommitmentDecoder(b,[31,44,57],[38,39,40,41,57],**m.parameters(condition,state))
   _,trace=dec.next_logits()
   self.assertEqual([x['layer'] for x in trace['base_operation_events']],[14,15] if condition['e_enabled'] else [14])
   self.assertEqual(trace['source_c_writes'],0);self.assertEqual(trace['source_forward_calls'],int(condition['e_enabled']))
   dec.accept(70);_,later=dec.next_logits();self.assertEqual(later['c_writes'],[])

if __name__=='__main__':unittest.main()
