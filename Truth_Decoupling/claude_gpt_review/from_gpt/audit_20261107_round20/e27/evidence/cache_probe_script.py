import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path[:0]=['scripts','.']
import e29_qa_source_interchange as run
from core.qa_source_interchange import SourceInterchangeDecoder
from core.qa_precision import fp32_output_projection
p=argparse.ArgumentParser();p.add_argument('--model',choices=('llama3.1-8b','qwen3-8b'),required=True);a=p.parse_args()
args=SimpleNamespace(source_run=Path('/workspace/Truth_Decoupling/results_qa_das_v1u')/a.model/'full',phase='check',pair_ids=None,seed=42)
ns,meta,bank,prepared,questions,layer,w,random_w,identity=run.inputs(args)
bundle=run.qa.open_bundle(meta,prepared)
base,source=run.prompt_tokens(bundle,meta['config'],questions[0]['question'])
result={'question':questions[0]['question'],'pair_id':questions[0]['pair_id'],'model':meta['model'],'comparisons':[]}
for dtype in ('bfloat16','float32'):
 if dtype=='float32':bundle.model.float()
 with fp32_output_projection(bundle):
  for name,mode,vec in [('baseline','baseline',w),('source_only','source_only',w),('mix','equal_logit_mix',w),('das','source_interchange',w),('random','source_interchange',random_w)]:
   cached=SourceInterchangeDecoder(bundle,base,source,mode=mode,layer=layer,direction=vec,use_cache=True)
   b=SourceInterchangeDecoder(bundle,base,source,mode=mode,layer=layer,direction=vec,use_cache=False)
   steps=[]
   for i in range(3):
    x,ta=cached.next_logits();y,tb=b.next_logits(); token=int(x.argmax());steps.append({'step':i,'max_diff':float((x-y).abs().max()),'rms_diff':float((x-y).square().mean().sqrt()),'top1_equal':int(x.argmax())==int(y.argmax()),'cached_top1':int(x.argmax()),'full_top1':int(y.argmax()),'writes_cached':ta['writes'],'writes_full':tb['writes']});cached.accept(token);b.accept(token)
   result['comparisons'].append({'dtype':dtype,'condition':name,'steps':steps});print(dtype,name,[(s['max_diff'],s['top1_equal']) for s in steps],flush=True)
   (Path('/workspace/Truth_Decoupling/results_r20_source_codex_20261107')/('cache_probe_'+a.model+'.json')).write_text(json.dumps(result,indent=2))
