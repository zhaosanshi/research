import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import unittest

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
CODE=ROOT/'from_claude/code'
sys.path[:0]=[str(CODE),str(CODE/'scripts'),str(CODE/'scripts/review_checks')]
import e42_analysis as A
import e42_qa_oracle_and_select as E
import judge_two_axis_run as R
from core.qa_generation_adapter import fingerprint


def fixture(phase='oracle',conditions=('baseline',)):
 rows=[{'pair_id':'q0','question':'What is the capital of France?','gold':['Paris'],'incorrect':['Berlin'],
        'outputs':dict.fromkeys(conditions,'Paris'),'finished':dict.fromkeys(conditions,True),
        'generation_token_ids':{c:[4,2] for c in conditions},'rule_abstain':dict.fromkeys(conditions,False)}]
 identity={'protocol':'e42_oracle_select_v1','model':'fixture','phase':phase,'split':'tune',
           'conditions':list(conditions),'question_ids':['q0']}
 return {'protocol':identity['protocol'],'model':'fixture','identity':identity,'split':'tune','completed':True,
         'conditions':list(conditions),'condition_specs':[{'name':c,'uses_reference_answer':False} for c in conditions],
         'protocol_fingerprint':fingerprint(identity),'bank_protocol_fingerprint':fingerprint(identity),'evaluations':rows}


def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False))
def final(c='baseline',truth='T',info='S',subtype='S',label='A'):
 return {'cell_id':'q0|'+c,'pair_id':'q0','condition':c,'truth':truth,'info':info,'info_subtype':subtype,
         'label':label,'status':'final','source':'judges:fixture','gold_is_abstention':False}

class MetricsTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.path=Path(self.temp.name)
 def load_two(self,raw,records):
  write(self.path/'tune.json',raw);items={r['cell_id']:r for r in records}
  with patch.object(R,'load_verified_final',return_value=({},items,items),create=True) as verifier:
   result=A.load(str(self.path),True)
   verifier.assert_called_once_with(str(self.path),generation_path=str(self.path/'tune.json'),require_complete=True)
   return result
 def legacy(self,raw):
  write(self.path/'tune.json',raw);grade=copy.deepcopy(raw)
  grade['grading_completed']=True;grade['grading_source_generation_sha256']=hashlib.sha256((self.path/'tune.json').read_bytes()).hexdigest()
  for row in grade['evaluations']:row.update(gold_is_abstention=False,outcome=dict.fromkeys(raw['conditions'],'correct'))
  write(self.path/'tune_scored_items.json',grade);return grade
 def test_false_hedge_is_wrong_but_not_substantive(self):
  raw,rows,outcomes=self.load_two(fixture(),[final(truth='F',info='N',subtype='H',label='B')])
  table,_=A.oracle_report(raw,rows,outcomes,40,0);arm=table['arms']['baseline']
  self.assertEqual((arm['wrong'],arm['answered'],arm['uninformative_false'],arm['non_refusal']),(1,0,1,1))
  self.assertEqual(arm['info_rate'],0)
 def test_select_and_random_expectation_use_info_axis(self):
  raw=fixture('select',('greedy','sample_0'));raw['evaluations'][0]['picks']={'greedy':'greedy','random':'greedy','max_logp':'sample_0'}
  raw,rows,outcomes=self.load_two(raw,[final('greedy','F','N','H','B'),final('sample_0')])
  table,_=A.select_report(raw,rows,outcomes,40,0)
  self.assertEqual(table['rules']['greedy']['answered'],0);self.assertEqual(table['rules']['max_logp']['answered'],1)
  self.assertEqual(table['expected_random']['answered'],.5)
 def test_unfinished_A_stays_content_A_and_has_quality_flag(self):
  raw=fixture();raw['evaluations'][0]['finished']['baseline']=False
  raw,rows,outcomes=self.load_two(raw,[final()]);table,_=A.oracle_report(raw,rows,outcomes,40,0)
  self.assertEqual(table['arms']['baseline']['correct'],1)
  self.assertEqual(table['arms']['baseline']['quality']['content_correct_unfinished'],1)
 def test_unresolved_and_missing_human_axis_cannot_complete(self):
  unresolved=final();unresolved.update(status='needs_human',label=None)
  with self.assertRaises(SystemExit):self.load_two(fixture(),[unresolved])
  incomplete=final(truth='F',label='B');incomplete.update(source='human',info_subtype=None)
  with self.assertRaises(SystemExit):self.load_two(fixture(),[incomplete])
 def test_explicit_human_false_hedge_preserves_information_axis(self):
  record=final(truth='F',info='N',subtype='H',label='B');record['source']='human'
  raw,rows,outcomes=self.load_two(fixture(),[record]);table,_=A.oracle_report(raw,rows,outcomes,40,0)
  self.assertEqual(table['arms']['baseline']['answered'],0)
  self.assertEqual(table['arms']['baseline']['non_refusal'],1)
 def test_wrong_gold_eligibility_is_rejected(self):
  record=final();record['gold_is_abstention']=True
  with self.assertRaises(SystemExit):self.load_two(fixture(),[record])
 def test_duplicate_finals_and_extra_cohort_rejected(self):
  write(self.path/'tune.json',fixture())
  record=final()
  with patch.object(R,'load_verified_final',return_value=({}, {}, {'x':record,'y':dict(record)}),create=True):
   with self.assertRaises(SystemExit):A.load(str(self.path),True)
 def test_valid_legacy_counts_retained(self):
  self.legacy(fixture());raw,rows,outcomes=A.load(str(self.path));table,_=A.oracle_report(raw,rows,outcomes,40,0)
  self.assertEqual((table['arms']['baseline']['correct'],table['arms']['baseline']['answered']),(1,1))
 def test_legacy_stale_text_hash_and_duplicate_ids_fail(self):
  for kind in ('text','hash','duplicate'):
   grade=self.legacy(fixture())
   if kind=='text':grade['evaluations'][0]['outputs']['baseline']='Berlin'
   elif kind=='hash':grade['grading_source_generation_sha256']='0'*64
   else:grade['evaluations'].append(copy.deepcopy(grade['evaluations'][0]))
   write(self.path/'tune_scored_items.json',grade)
   with self.assertRaises(SystemExit):A.load(str(self.path))
 def test_test_split_and_gold_oracle_cannot_enter_select_pool(self):
  raw=fixture();raw['split']='test';write(self.path/'tune.json',raw)
  with self.assertRaises(SystemExit):A.load(str(self.path))
  raw=fixture('select',('greedy','gold_source_only'));raw['condition_specs'][1]['uses_reference_answer']=True
  write(self.path/'tune.json',raw)
  with self.assertRaises(SystemExit):A.load(str(self.path))
 def test_zscore_mean_is_scale_invariant_and_constant_layers_zero(self):
  names=['a','b','c'];v=[[100.,0.],[90.,10.],[0.,1.]]
  def choose(scale=1.,constant=False):
   r={'outputs':dict.fromkeys(names,'answer'),'rule_abstain':dict.fromkeys(names,False),'readout':{n:{
      'mean_token_logp':0.,'verdict_margin':0.,'coord_dasce_L1':v[i][0]*scale,
      'coord_dasce_L2':v[i][1],'coord_dasce_L3':77.} for i,n in enumerate(names)}}
   return E.picks(r,names,{'dasce':{1:None,2:None,3:None}},[1,2,3],0)['coord_dasce_mean']
  self.assertEqual(choose(),'b');self.assertEqual(choose(1000.),'b')
 def test_select_prompt_path_never_reads_gold_and_oracle_still_requires_it(self):
  import torch
  class Bundle:
   def __init__(self):self.texts=[]
   def build_chat(self,system,user):return system+'\n'+user
   def encode(self,text):
    self.texts.append(text)
    return {'input_ids':torch.tensor([[ord(c) for c in text]]),'attention_mask':torch.ones((1,len(text)),dtype=torch.long)}
  class Question(dict):
   def get(self,key,default=None):
    if key in ('gold','best_answer','incorrect'):raise AssertionError('deployable prompt read a reference')
    return super().get(key,default)
  cfg={'qa_system':'Answer briefly.','qa_user':'Question: {question}\nAnswer:'}
  bundle=Bundle();q=Question(question='How does drinking coffee affect your growth?')
  current=E.prompts_for(bundle,cfg,q,include_gold=False)
  self.assertEqual(set(current),{'base','truthful'});self.assertEqual(len(bundle.texts),2)
  with_gold=E.prompts_for(Bundle(),cfg,{'question':q['question'],'gold':['It does not stunt growth.']})
  self.assertEqual(current['base'],with_gold['base']);self.assertEqual(current['truthful'],with_gold['truthful'])
  with self.assertRaises(ValueError):E.prompts_for(Bundle(),cfg,dict(q))
 def test_both_signs_align_identically_and_near_zero_is_uninformative(self):
  import torch
  w=torch.tensor([-.6,.8]);md=[1.,0.]
  a,ma=E.orient_readout_direction(w,md);b,mb=E.orient_readout_direction(-w,md)
  self.assertTrue(torch.equal(a,b));self.assertTrue(ma['flipped']);self.assertFalse(mb['flipped'])
  _,weak=E.orient_readout_direction(torch.tensor([0.,1.]),md)
  self.assertEqual(weak['status'],'orientation_uninformative');self.assertFalse(weak['eligible_for_readout_selection'])
 def test_qadas_select_aligns_but_oracle_keeps_saved_axis(self):
  import torch
  from types import SimpleNamespace
  w=torch.tensor([-.6,.8]);args=SimpleNamespace(families=['qadas'])
  bank={'E_qa_das_L1':w.numpy()}
  oracle=E.directions(args,bank,{},[1])['qadas'][1]
  select,meta=E.directions(args,bank,{},[1],wmd={1:[1.,0.]},return_orientation=True)
  self.assertTrue(torch.equal(oracle,w));self.assertTrue(torch.equal(select['qadas'][1],-w))
  self.assertTrue(meta['qadas']['1']['flipped'])
 def test_real_source_swap_projection_is_sign_invariant(self):
  import torch
  sys.path.insert(0,str(ROOT/'from_gpt'))
  from test_qa_das_pipeline_20261105 import PipelineModel,ToyTokenizer
  from core.model_io import ModelBundle
  from core.qa_source_interchange import SourceInterchangeDecoder
  tok=ToyTokenizer();model=PipelineModel(tok);bundle=ModelBundle(model,tok,{'name':'sign-toy','layers_attr':'layers'})
  w=torch.nn.functional.normalize(torch.arange(1.,9.),dim=0)
  positive=SourceInterchangeDecoder(bundle,[31,44,57],[38,39,40,41,57],mode='source_interchange',layer=0,direction=w)
  negative=SourceInterchangeDecoder(bundle,[31,44,57],[38,39,40,41,57],mode='source_interchange',layer=0,direction=-w)
  for token in (70,71,72):
   a,_=positive.next_logits();b,_=negative.next_logits();torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
   positive.accept(token);negative.accept(token)
 def test_actual_runner_to_analysis_contract_and_tamper_failures(self):
  from types import SimpleNamespace
  from core.judge_two_axis_contract import canonical_sha,cell_sha,file_sha,judge_contract
  from core.judge_two_axis import TWO_AXIS_VERSION,qa_sha
  from core.io_utils import write_jsonl,atomic_json_dump
  cfg={'version':TWO_AXIS_VERSION,'pro':{'model':'fixture-pro','temperature':0},
       'gpt':{'model':'fixture-gpt'},'allenai':{'threshold':.5,'truth_model':'fixture-truth','info_model':'fixture-info'}}
  write(self.path/'tune.json',fixture());R.build_cells(str(self.path),audit_n=0)
  cell=R._cells(str(self.path))[0];directory=self.path/'judge2'
  runtime={'fixture_only':'no real model used in this test'};runtime_sha=canonical_sha(runtime)
  contract=judge_contract('allenai',cfg)
  vote={'cell_id':cell['cell_id'],'qa_sha':qa_sha(cell['question'],cell['prediction']),
    'input_sha':cell_sha(cell),'judge_contract_sha':canonical_sha(contract),'runtime_sha':runtime_sha,
    'truth_p_yes':.9,'truth_p_no':.1,'info_p_yes':.9,'info_p_no':.1}
  write_jsonl([vote],str(directory/'allenai.jsonl'))
  atomic_json_dump({'judge_contract_sha':canonical_sha(contract),'judge_contract':contract,'runtime_identity':runtime,
    'runtime_sha':runtime_sha,'rows_sha256':file_sha(directory/'allenai.jsonl'),
    'cells_sha256':file_sha(directory/'cells.jsonl'),'generation_sha256':file_sha(self.path/'tune.json')},str(directory/'allenai_meta.json'))
  class Judge:
   judge_contract_sha=canonical_sha(judge_contract('pro',cfg))
   llm=SimpleNamespace(n_calls=0,n_cache_hits=0)
   def grade(self,*args):
    return {'judge':'pro','model':'fixture-pro','rubric':TWO_AXIS_VERSION,'judge_contract_sha':self.judge_contract_sha,'qa_sha':qa_sha(args[0],args[2]),'truth':'D','truth_raw':'D','info':'S','info_raw':'S'}
  R.run_llm(str(self.path),'pro',cfg,judge=Judge());R.adjudicate(str(self.path),cfg)
  with self.assertRaises(SystemExit):A.load(str(self.path),True)
  R.human_set(str(self.path),'q0|baseline','A',by='synthetic-test-reviewer')
  R.adjudicate(str(self.path),cfg);raw,rows,_=A.load(str(self.path),True)
  self.assertEqual(rows[0]['outcome']['baseline'],'correct');self.assertIsNotNone(raw['_analysis_judgment_provenance'])
  original=(directory/'final.jsonl').read_bytes()
  damaged=json.loads(original);damaged.update(label='B',truth='F')
  write_jsonl([damaged],str(directory/'final.jsonl'))
  with self.assertRaises(SystemExit):A.load(str(self.path),True)
  (directory/'final.jsonl').write_bytes(original)
  changed=fixture();changed['evaluations'][0]['outputs']['baseline']='Berlin';write(self.path/'tune.json',changed)
  with self.assertRaises(SystemExit):A.load(str(self.path),True)
 def test_orientation_rejects_train_eval_overlap_before_model_forward(self):
  records=[{'id':'q:T','pair_id':'q','text':'yes','label':True},{'id':'q:F','pair_id':'q','text':'no','label':False}]
  with patch.object(E,'load_experiment',return_value={'eval_datasets':['fixture'],'prompts':{}}),patch.object(E,'load_split',side_effect=lambda dataset,split: records if split=='calib' else []),patch.object(E,'capture_last_token') as forward:
   with self.assertRaises(ValueError):E.tf_estimation(None,[1],expected_train_ids=['q'],evaluation_ids=['q'])
   forward.assert_not_called()
   with self.assertRaises(ValueError):E.tf_estimation(None,[1],expected_train_ids=['different'])
   forward.assert_not_called()
 def test_literal_translation_gold_is_kept_in_actual_new_pipeline(self):
  from types import SimpleNamespace
  from core.judge_two_axis_contract import canonical_sha,cell_sha,file_sha,judge_contract
  from core.judge_two_axis import TWO_AXIS_VERSION,qa_sha
  from core.io_utils import write_jsonl,atomic_json_dump
  raw=fixture();row=raw['evaluations'][0]
  row.update(question="Translate 'no sé' into English.",gold=["I don't know"],outputs={'baseline':"I don't know"})
  write(self.path/'tune.json',raw);R.build_cells(str(self.path),audit_n=0)
  cell=R._cells(str(self.path))[0];self.assertFalse(cell['gold_is_abstention'])
  cfg={'version':TWO_AXIS_VERSION,'pro':{'model':'fixture-pro'},'gpt':{'model':'fixture-gpt'},
       'allenai':{'threshold':.5,'truth_model':'fixture-truth','info_model':'fixture-info'}}
  directory=self.path/'judge2';runtime={'fixture':'translation'};contract=judge_contract('allenai',cfg)
  vote={'cell_id':cell['cell_id'],'qa_sha':qa_sha(cell['question'],cell['prediction']),'input_sha':cell_sha(cell),
        'judge_contract_sha':canonical_sha(contract),'runtime_sha':canonical_sha(runtime),
        'truth_p_yes':.9,'truth_p_no':.1,'info_p_yes':.9,'info_p_no':.1}
  write_jsonl([vote],str(directory/'allenai.jsonl'))
  atomic_json_dump({'judge_contract_sha':canonical_sha(contract),'runtime_identity':runtime,'runtime_sha':canonical_sha(runtime),
     'rows_sha256':file_sha(directory/'allenai.jsonl'),'cells_sha256':file_sha(directory/'cells.jsonl'),
     'generation_sha256':file_sha(self.path/'tune.json')},str(directory/'allenai_meta.json'))
  class Judge:
   judge_contract_sha=canonical_sha(judge_contract('pro',cfg))
   llm=SimpleNamespace(n_calls=0,n_cache_hits=0)
   def grade(self,*args):return {'judge':'pro','model':'fixture-pro','rubric':TWO_AXIS_VERSION,'judge_contract_sha':self.judge_contract_sha,'qa_sha':qa_sha(args[0],args[2]),'truth':'T','truth_raw':'T','info':'S','info_raw':'S'}
  R.run_llm(str(self.path),'pro',cfg,judge=Judge());R.adjudicate(str(self.path),cfg)
  loaded,rows,outcomes=A.load(str(self.path),True)
  self.assertEqual(len(rows),1);self.assertEqual(rows[0]['outcome']['baseline'],'correct')
 def test_bootstrap_empty_or_nonfinite_does_not_publish_interval(self):
  for values in ([],[float('nan')]):
   with self.assertRaises(ValueError):A.boot_ci(values,20,0)

if __name__=='__main__':unittest.main()
