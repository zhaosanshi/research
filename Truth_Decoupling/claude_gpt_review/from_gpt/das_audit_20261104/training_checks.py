"""Read-only audit: real pinned DAS functions, tiny frozen CPU network, temporary checkpoints."""
import argparse, json, math, pathlib, sys, tempfile, types, itertools, copy, subprocess
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--code-root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[2]/'from_claude/code')
parser.add_argument('--out',type=pathlib.Path,required=True)
args=parser.parse_args()
if args.out.exists():parser.error('output exists; use a new audit output file')
CODE=args.code_root
sys.path.insert(0,str(CODE))
import core.directions as d

class Block(nn.Module):
 def __init__(self):
  super().__init__(); self.mix=nn.Linear(4,4,bias=False)
 def forward(self,h):
  n=torch.arange(1,h.shape[1]+1).view(1,-1,1)
  return h+torch.tanh(self.mix(h.cumsum(1)/n))
class Tiny(nn.Module):
 def __init__(self):
  super().__init__();self.embedding=nn.Embedding(17,4);self.blocks=nn.ModuleList([Block(),Block()]);self.head=nn.Linear(4,17,bias=False)
 def forward(self,input_ids,**kw):
  h=self.embedding(input_ids)
  for block in self.blocks:h=block(h)
  return types.SimpleNamespace(logits=self.head(h))
def enc(ids):
 x=torch.tensor([ids]);return {'input_ids':x,'attention_mask':torch.ones_like(x)}
torch.manual_seed(7);model=Tiny().eval();model.requires_grad_(False)
b=types.SimpleNamespace(model=model,device=torch.device('cpu'),hidden_size=4,layer=lambda i:model.blocks[i])
pair={'base':enc([1,2,3]),'source':enc([4,5,6,3])};teacher=torch.tensor([8,9,10]);rep={}
checks=[]
for rank,kind,anchor,pos in itertools.product([1,2],['kl','jsd'],['prompt','current'],['last','all']):
 basis=d.OrthoBasis(4,rank,seed=42);tr=[]
 loss,n=d.multistep_kl_loss(b,basis,pair,0,teacher,3,loss=kind,anchor=anchor,positions=pos,trace=tr)
 loss.backward();assert n==3 and basis.raw.grad.norm()>0 and torch.isfinite(basis.raw.grad).all()
 k=int(basis.raw.grad.abs().argmax());analytical=float(basis.raw.grad.flatten()[k]);values=[]
 for sign in [1,-1]:
  other=d.OrthoBasis(4,rank,seed=42)
  with torch.no_grad():other.raw.flatten()[k]+=sign*1e-3
  v,_=d.multistep_kl_loss(b,other,pair,0,teacher,3,loss=kind,anchor=anchor,positions=pos)
  values.append(float(v.detach()))
 numerical=(values[0]-values[1])/.002;assert abs(numerical-analytical)<4e-4,(rank,kind,anchor,pos,numerical,analytical)
 assert not model.blocks[0]._forward_hooks
 orth=float((basis.detached().T@basis.detached()-torch.eye(rank)).abs().max());assert orth<1e-5
 checks.append({'rank':rank,'loss':kind,'anchor':anchor,'positions':pos,'gradient_error':abs(numerical-analytical),'orthogonality_error':orth,'trace':tr})
rep['gradient_branch_checks']=checks
# Bidirectional backward before one step equals the sum objective's backward.
pairs=[dict(pair,y_src=teacher,y_base=teacher.flip(0)),{'base':enc([2,1,3]),'source':enc([6,5,4,3]),'y_src':teacher,'y_base':teacher.flip(0)}]
original=[p.clone() for p in model.parameters()]
r=d.train_das(b,pairs,0,rank=2,T=3,epochs=1,seed=42,loss='jsd',bidirectional=True,positions='all',anchor='prompt',log_fn=None)
basis=d.OrthoBasis(4,2,seed=42);opt=torch.optim.Adam(basis.parameters(),lr=.001)
for pair in pairs:
 opt.zero_grad();lf,_=d.multistep_kl_loss(b,basis,pair,0,pair['y_src'],3,loss='jsd',positions='all',anchor='prompt');lr,_=d.multistep_kl_loss(b,basis,pair,0,pair['y_base'],3,loss='jsd',positions='all',anchor='prompt',base_key='source',source_key='base');(.5*lf+.5*lr).backward();torch.nn.utils.clip_grad_norm_(basis.parameters(),1.);opt.step()
rep['bidir_vs_joint_backward_max_W_diff']=float((basis.detached()-r['W']).abs().max());assert rep['bidir_vs_joint_backward_max_W_diff']<1e-6
rep['model_parameters_unchanged']=all(torch.equal(x,y) and y.grad is None for x,y in zip(original,model.parameters()));assert rep['model_parameters_unchanged']
# Current metadata losses are online epoch averages, not the final checkpoint loss.
final_loss=np.mean([float(d.multistep_kl_loss(b,lambda:r['W'],p,0,p['y_src'],3,loss='jsd',positions='all',anchor='prompt')[0])*.5+float(d.multistep_kl_loss(b,lambda:r['W'],p,0,p['y_base'],3,loss='jsd',positions='all',anchor='prompt',base_key='source',source_key='base')[0])*.5 for p in pairs])
rep['online_vs_final_checkpoint']={'reported_final_loss':r['losses'][-1],'fixed_checkpoint_mean_loss':final_loss}
# Teacher cache ignores a changed T when a caller reuses pair dictionaries.
fresh=[{'base':enc([1,2,3]),'source':enc([4,5,6,3])},{'base':enc([2,1,3]),'source':enc([6,5,4,3])}];requests=[]
def fake_teacher(bundle,inputs,T):requests.append(T);return torch.arange(T)+8
with patch.object(d,'teacher_tokens',fake_teacher):
 r1=d.train_das(b,fresh,0,T=1,epochs=1,log_fn=None,anchor='prompt')
 r3=d.train_das(b,fresh,0,T=3,epochs=1,log_fn=None,anchor='prompt')
rep['teacher_cache_changed_T']={'teacher_requests':requests,'second_metadata_T':r3['T'],'second_actual_trace_steps':len(r3['position_trace'])}

def checkpoint(folder,seed,W,gaps,rank=1,**extra):
 torch.save({'W':torch.tensor(W,dtype=torch.float32),'mean_gap':gaps,'losses':[.1],'seed':seed,'rank':rank,'n_pairs':96,**extra},pathlib.Path(folder)/d.das_file_name('E',0,rank,seed))
with tempfile.TemporaryDirectory() as td:
 checkpoint(td,42,[[1.],[0.]],[1.]);checkpoint(td,43,[[-1.],[0.]],[-1.])
 w,meta=d.load_das_direction(td,'E',0,align_to=[0.,1.]);rep['loader_identical_subspaces_cancel']={'W':w.tolist(),'norm':float(np.linalg.norm(w))}
with tempfile.TemporaryDirectory() as td:
 eps=.001;a=np.array([1.,eps]);a/=np.linalg.norm(a);c=np.array([1.,-eps]);c/=np.linalg.norm(c)
 checkpoint(td,42,a[:,None],[1.]);checkpoint(td,43,c[:,None],[1.]);w,meta=d.load_das_direction(td,'E',0,align_to=[0.,1.]);rep['loader_nearly_identical_subspaces_rotate']={'seed_cos':float(a@c),'merged_W':w.tolist(),'cos_seed_vs_merged':float(abs(a@w))}
with tempfile.TemporaryDirectory() as td:
 checkpoint(td,42,[[1.],[0.]],[1.]);checkpoint(td,43,[[0.],[1.]],[1.]);w,meta=d.load_das_direction(td,'E',0,align_to=[1.,1.]);rep['loader_mean_gap_scale']={'W':w.tolist(),'reported_gap':meta['mean_gap'],'actual_gap_for_delta_1_1':float(np.array([1.,1.])@w)}
with tempfile.TemporaryDirectory() as td:
 checkpoint(td,42,[[1.,0.],[0.,1.],[0.,0.]],[1.,2.],rank=2);w,meta=d.load_das_direction(td,'E',0,rank=2);rep['loader_rank2_truncated']={'loaded_shape':list(w.shape),'W':w.tolist(),'project_e2':float(np.dot(w,[0.,1.,0.]))}
with tempfile.TemporaryDirectory() as td:
 checkpoint(td,42,[[1.],[0.]],[1.],anchor='prompt',loss_kind='kl');checkpoint(td,43,[[0.],[1.]],[1.],anchor='current',loss_kind='jsd');w,meta=d.load_das_direction(td,'E',0,align_to=[1.,1.]);rep['loader_incompatible_metadata_accepted']={'n_seeds':meta['n_seeds'],'W':w.tolist()}
rep['shell_syntax']={f.name:subprocess.run(['bash','-n',str(f)],capture_output=True,text=True).returncode for f in sorted((CODE/'scripts').glob('das_*.sh'))}
assert all(v==0 for v in rep['shell_syntax'].values())
args.out.parent.mkdir(parents=True,exist_ok=True)
args.out.write_text(json.dumps(rep,indent=2))
print(json.dumps({k:v for k,v in rep.items() if k!='gradient_branch_checks'},indent=2));print('PASS gradient branch count',len(checks),'max finite-difference error',max(x['gradient_error'] for x in checks))
