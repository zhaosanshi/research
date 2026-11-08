import importlib.util,json,sys,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
CODE=Path('/workspace/Truth_Decoupling/review_r23_claude/claude_gpt_review/from_claude/code')
sys.path[:0]=[str(CODE),str(CODE/'scripts')]
import torch
from transformers import AutoTokenizer
from core.model_io import ModelBundle
from core.config import load_model_config
p=Path('/workspace/Truth_Decoupling/review_r22_metrics_check/e42_future.py')
spec=importlib.util.spec_from_file_location('future_e42_for_nogold_audit',p);future=importlib.util.module_from_spec(spec);spec.loader.exec_module(future)
rawpath=Path('/workspace/Truth_Decoupling/results_e42_20261108/llama3.1-8b/select-tune/tune.json')
rawbytes=rawpath.read_bytes();raw=json.loads(rawbytes)
assert raw['completed'] and len(raw['evaluations'])==192
cfg=load_model_config('llama3.1-8b');tok=AutoTokenizer.from_pretrained(cfg['path'],trust_remote_code=True,local_files_only=True)
if tok.pad_token is None:tok.pad_token=tok.eos_token
tok.padding_side='left'
bundle=object.__new__(ModelBundle);bundle.tokenizer=tok;bundle.cfg=cfg;bundle.device=torch.device('cpu')
qa_cfg={'qa_system':raw['identity']['base_system'],'qa_user':raw['identity']['user_template']}
class NoGold(dict):
 def get(self,key,default=None):
  if key in ('gold','best_answer','incorrect'):raise AssertionError('reference was read')
  return super().get(key,default)
checks=[]
for row in raw['evaluations']:
 q=NoGold(question=row['question'],pair_id=row['pair_id'])
 value=future.prompts_for(bundle,qa_cfg,q,include_gold=False)
 assert set(value)=={'base','truthful'}
 assert value==row['prompt_token_ids']
 old_shaped=future.prompts_for(bundle,qa_cfg,row,include_gold=True)
 assert all(value[k]==old_shaped[k] for k in value)
 checks.append({'pair_id':row['pair_id'],'base_tokens':len(value['base']),'truthful_tokens':len(value['truthful'])})
try:future.prompts_for(bundle,qa_cfg,dict(NoGold(question=raw['evaluations'][0]['question'])))
except ValueError:oracle_requires_gold=True
else:raise AssertionError('oracle lost its required gold source')
exp=future.load_experiment('E2b_tqa');est=[]
for dataset in exp['eval_datasets']:
 est+=future.load_split(dataset,'calib')+future.load_split(dataset,'dev')
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
expected=json.loads(Path('/workspace/Truth_Decoupling/review_r22_metrics_check/tf_source_expected.json').read_text())
actual={'TF_train_identity_sha256':canonical([{k:r.get(k) for k in ('id','pair_id','text','label')} for r in est]),
        'statement_prompt_sha256':canonical(exp['prompts']['statement'])}
assert actual==expected,'current TF training source differs from preserved direction-bank proof'
assert not {r['pair_id'] for r in est}&{r['pair_id'] for r in raw['evaluations']}
assert rawpath.read_bytes()==rawbytes
out={'protocol':'r22_real_tokenizer_gold_independence_audit_v1','n_questions':len(checks),'model_loaded':False,'gpu_used':False,
 'api_calls':0,'model':raw['model'],'current_TF_training_source_exact_preserved_bank':actual,
 'all_base_truthful_token_ids_exact_original':True,'gold_access_guard_passed':True,'oracle_still_requires_gold':oracle_requires_gold,
 'source_generation_sha256':hashlib.sha256(rawbytes).hexdigest(),'future_entry_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
 'model_io_source_matches_generation':hashlib.sha256((CODE/'core/model_io.py').read_bytes()).hexdigest()==raw['identity']['implementation']['files']['core/model_io.py'],
 'questions':checks}
Path('/workspace/Truth_Decoupling/review_r22_metrics_check/nogold_check.json').write_text(json.dumps(out,indent=2)+'\n')
print('REAL TOKENIZER PASS: 192 no-gold questions, both existing prompt token arrays exact; no model/GPU/API calls')
