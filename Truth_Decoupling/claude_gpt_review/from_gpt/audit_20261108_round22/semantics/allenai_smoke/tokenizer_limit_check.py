import sys,json
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent/'code/scripts'))
import allenai_tqa_judge as A
from transformers import AutoTokenizer
root=Path(__file__).resolve().parent
models=Path('/workspace/Truth_Decoupling/models')
class NeverForward:
 config=SimpleNamespace(max_position_embeddings=4096)
 def __call__(self,**kw):raise AssertionError('over-context prompt reached model')
report={}
for kind in ['truth','info']:
 tok=AutoTokenizer.from_pretrained(str(models/('allenai__truthfulqa-'+kind+'-judge-llama2-7B')),use_fast=False)
 tail='The Earth is flat and Richard Feynman never won a Nobel Prize.'
 answer=('Richard Feynman won a Nobel Prize. ' * 200)+tail
 prompt=A.build_prompt('Who won a Nobel Prize?',answer,kind)
 assert tail in prompt and prompt.endswith(A.CUES[kind])
 n=len(tok(prompt,add_special_tokens=True)['input_ids']);assert n>1024
 yi,ni=A.yes_no_ids(tok,prompt)
 try:A.score(NeverForward(),tok,prompt,yi,ni,'cpu',1024)
 except A.UnscorablePrompt as exc:result={'status':'unscorable','reason':exc.reason,'actual':exc.actual,'limit':exc.limit,'forward_calls':0}
 else:raise AssertionError('long prompt was not rejected')
 report[kind]={'actual_tokenizer':type(tok).__name__,'answer_characters':len(answer),'prompt_tokens':n,'false_tail_preserved':True,'complete_prompt_sha256':A.hashlib.sha256(prompt.encode()).hexdigest(),'score':result}
(root/'tokenizer_limit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
