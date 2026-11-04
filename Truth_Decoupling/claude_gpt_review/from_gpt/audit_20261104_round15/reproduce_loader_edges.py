"""Independent edge checks for d267f7a DAS loader/cache; CPU only, no model downloads."""
import argparse, importlib.util, json, pathlib, tempfile, sys, types, copy, warnings
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--code-root',type=pathlib.Path,required=True)
parser.add_argument('--old-directions',type=pathlib.Path,required=True)
parser.add_argument('--out',type=pathlib.Path,required=True)
args=parser.parse_args()
if args.out.exists():parser.error('output exists; choose a new path')
NEW=args.code_root
OLD=args.old_directions
sys.path.insert(0,str(NEW))
import core.directions as d
spec=importlib.util.spec_from_file_location('old_das_review15',OLD);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
REP={}
BASE={'rank':1,'n_pairs':96,'T':2,'epochs':10,'lr':.001,'loss_kind':'kl','positions':'last','bidirectional':False,'weighted':False,'trim':0.}
def save(folder,seed,w,gap=1.,**fields):
 row={**BASE,'W':torch.as_tensor(w,dtype=torch.float32).reshape(-1,1),'mean_gap':[gap],'losses':[.1],'seed':seed,**fields}
 torch.save(row,pathlib.Path(folder)/d.das_file_name('E',0,1,seed))
def load(files,**kwargs):
 with tempfile.TemporaryDirectory() as td:
  for f in files:save(td,**f)
  try:
   with warnings.catch_warnings():
    warnings.simplefilter('ignore');w,m=d.load_das_direction(td,'E',0,**kwargs)
   return {'accepted':True,'W':w.tolist(),'gap':m['mean_gap'],'protocol':m['protocol']}
  except Exception as e:return {'accepted':False,'type':type(e).__name__,'message':str(e)}
# Missing first-seed key must not hide a conflict between the next two files.
REP['missing_first_key_hides_anchor_conflict']=load([
 {'seed':42,'w':[1,0]}, {'seed':43,'w':[1,0],'anchor':'prompt'}, {'seed':44,'w':[1,0],'anchor':'current'}])
REP['full_protocol_conflict_rejected']=load([
 {'seed':42,'w':[1,0],'anchor':'prompt'}, {'seed':43,'w':[1,0],'anchor':'current'}])
assert REP['missing_first_key_hides_anchor_conflict']['accepted']
assert not REP['full_protocol_conflict_rejected']['accepted']
# Model/data fields present in checkpoints but not part of the checked protocol.
REP['different_models_and_data_accepted']=load([
 {'seed':42,'w':[1,0],'anchor':'prompt','model_name':'llama','training_pair_ids':['a']},
 {'seed':43,'w':[1,0],'anchor':'prompt','model_name':'qwen','training_pair_ids':['b']}])
# Non-finite values are not caught by norm < eps comparisons.
REP['nan_direction_accepted']=load([{'seed':42,'w':[float('nan'),0],'anchor':'prompt'}])
REP['infinite_gap_accepted']=load([{'seed':42,'w':[1,0],'gap':float('inf'),'anchor':'prompt'}])
assert REP['nan_direction_accepted']['accepted'] and REP['infinite_gap_accepted']['accepted']
# Trimming is nonlinear: aligned per-seed trimmed gaps cannot be linearly merged.
X=np.array([[10,0],[0,20],[-5,-5]]+[[1,1]]*7,dtype=np.float64)
w1=np.array([1.,0.],dtype=np.float32).astype(float);w2=np.array([.8,.6],dtype=np.float32).astype(float)
w1/=np.linalg.norm(w1);w2/=np.linalg.norm(w2)
g1=float(d.trimmed_mean(X@w1,.1));g2=float(d.trimmed_mean(X@w2,.1))
r=load([{'seed':42,'w':w1,'gap':g1,'trim':.1,'anchor':'prompt'}, {'seed':43,'w':w2,'gap':g2,'trim':.1,'anchor':'prompt'}])
expected=float(d.trimmed_mean(X@np.array(r['W']),.1))
REP['multi_seed_trimmed_gap_not_exact']={**r,'per_seed_gaps':[g1,g2],'actual_trimmed_gap_on_merged_W':expected,'difference':r['gap']-expected}
assert abs(r['gap']-expected)>.1
# Single valid stored directions: compare actual outputs, not only a small allclose test.
rng=np.random.default_rng(15);count=200;unequal_w=unequal_gap=0;maxgap=0.
with tempfile.TemporaryDirectory() as td:
 for k in range(count):
  w=rng.normal(size=16);w/=np.linalg.norm(w);gap=float(rng.normal());ref=None if k%3==0 else rng.normal(size=16)
  save(td,42,w,gap,anchor='prompt')
  wn,mn=d.load_das_direction(td,'E',0,align_to=ref);wo,mo=old.load_das_direction(td,'E',0,align_to=ref)
  unequal_w+=not np.array_equal(wn,wo);unequal_gap+=mn['mean_gap']!=mo['mean_gap'];maxgap=max(maxgap,abs(mn['mean_gap']-mo['mean_gap']))
REP['single_seed_compatibility_200']={'cases':count,'direction_not_bit_equal':unequal_w,'gap_not_bit_equal':unequal_gap,'max_gap_absolute_difference':maxgap}
assert unequal_w==0 and maxgap<1e-12
# Cache: T invalidates both directions; input or model changes at the same T still do not.
class Block(nn.Module):
 def __init__(self):super().__init__();self.mix=nn.Linear(4,4,bias=False)
 def forward(self,h):return h+torch.tanh(self.mix(h.cumsum(1)/torch.arange(1,h.shape[1]+1).view(1,-1,1)))
class Tiny(nn.Module):
 def __init__(self):super().__init__();self.embedding=nn.Embedding(17,4);self.blocks=nn.ModuleList([Block(),Block()]);self.head=nn.Linear(4,17,bias=False)
 def forward(self,input_ids,**kw):
  h=self.embedding(input_ids)
  for block in self.blocks:h=block(h)
  return types.SimpleNamespace(logits=self.head(h))
def enc(v):
 x=torch.tensor([v]);return {'input_ids':x,'attention_mask':torch.ones_like(x)}
def bundle(seed):
 torch.manual_seed(seed);m=Tiny().eval();m.requires_grad_(False)
 return types.SimpleNamespace(model=m,device=torch.device('cpu'),hidden_size=4,layer=lambda i:m.blocks[i],cache_tag=seed)
b=bundle(7);pairs=[{'base':enc([1,2,3]),'source':enc([4,5,6,3])},{'base':enc([2,1,3]),'source':enc([6,5,4,3])}];req=[]
def teacher(bun,inp,T):
 req.append({'model_object':id(bun.model),'ids':inp['input_ids'].tolist(),'T':T})
 return torch.arange(T)+(int(inp['input_ids'].sum())+bun.cache_tag)%10+1
with patch.object(d,'teacher_tokens',teacher):
 d.train_das(b,pairs,0,T=1,epochs=1,anchor='prompt',bidirectional=True,log_fn=None)
 d.train_das(b,pairs,0,T=2,epochs=1,anchor='prompt',bidirectional=False,log_fn=None)
 after_t_change=len(req);base_cleared=all('y_base' not in p for p in pairs)
 d.train_das(b,pairs,0,T=2,epochs=1,anchor='prompt',bidirectional=True,log_fn=None)
 after_bidir=len(req)
 pairs[0]['source']=enc([9,10,3])
 d.train_das(b,pairs,0,T=2,epochs=1,anchor='prompt',bidirectional=True,log_fn=None)
 after_input_change=len(req);cached_after_input=pairs[0]['y_src'].tolist();expected_after_input=(torch.arange(2)+(int(pairs[0]['source']['input_ids'].sum())+7)%10+1).tolist()
 d.train_das(bundle(8),pairs,0,T=2,epochs=1,anchor='prompt',bidirectional=True,log_fn=None)
 after_model_change=len(req)
REP['teacher_cache_identity']={'requests_after_T_change':after_t_change,'base_cache_cleared':base_cleared,'requests_after_bidir_restored':after_bidir,'requests_after_source_input_change':after_input_change,'requests_after_model_change':after_model_change,'teacher_T_sequence':[r['T'] for r in req],'cached_source_after_input_change':cached_after_input,'expected_source_after_input_change':expected_after_input}
assert after_t_change==6 and base_cleared and after_bidir==8 and after_input_change==after_bidir==after_model_change
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return str(x)
 return x
REP=clean(REP)
args.out.write_text(json.dumps(REP,ensure_ascii=False,indent=2,allow_nan=False))
print(json.dumps(REP,ensure_ascii=False,indent=2))
print('PASS: fixes and remaining edge cases reproduced; no production changes or real model weights')
