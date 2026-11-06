from pathlib import Path
import torch,json,hashlib,numpy as np
root=Path('/workspace/Truth_Decoupling');out=[]
for model in ['llama3.1-8b','qwen3-8b']:
 for init in ['das','md','random']:
  old=root/'results_e27_search_codex_20261106'/model/'ce_alpha3_ep3'/init
  new=root/'results_e27_release_20261106'/model/'ce_alpha3_ep3'/(init+'_s42')
  a=torch.load(old/'adapter.pt',map_location='cpu',weights_only=True);b=torch.load(new/'adapter.pt',map_location='cpu',weights_only=True)
  r={'model':model,'init':init,'old_checkpoint_sha256':hashlib.sha256((old/'adapter.pt').read_bytes()).hexdigest(),'new_checkpoint_sha256':hashlib.sha256((new/'adapter.pt').read_bytes()).hexdigest(),'layers':a['layers'],'W':{},'initial_W':{},'losses_equal':a['losses']==b['losses'],'metadata_key_changes':sorted(set(a['train_config'])^set(b['train_config'])),'outputs':{}}
  for k in a['W']:
   x=a['W'][k].float();y=b['W'][k].float();r['W'][k]={'equal':bool(torch.equal(x,y)),'max_abs_diff':float((x-y).abs().max()),'cosine':float(torch.nn.functional.cosine_similarity(x,y,dim=0))};r['initial_W'][k]=bool(torch.equal(a['initial_W'][k],b['initial_W'][k]))
  for phase in ['tune','test']:
   pa,pb=old/(phase+'.json'),new/(phase+'.json')
   if pa.exists() and pb.exists():
    da=json.loads(pa.read_text());db=json.loads(pb.read_text());assert [x['pair_id'] for x in da['evaluations']]==[x['pair_id'] for x in db['evaluations']]
    r['outputs'][phase]={c:sum(x['outputs'][c]!=y['outputs'][c] for x,y in zip(da['evaluations'],db['evaluations'])) for c in da['conditions']}
  out.append(r)
path=Path('/workspace/Truth_Decoupling/results_r19_codex_20261106/release_reproduction.json');path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
