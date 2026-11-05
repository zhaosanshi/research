"""Execute the submitted cache-building loop and specs_for with local fakes; no model/API."""
from pathlib import Path
import os
import ast,json
ROOT=Path(os.environ.get('TD_AUDIT_CLAUDE_ROOT', str(Path(__file__).resolve().parents[3] / 'from_claude')))
SRC=ROOT/'code/scripts/review_checks/verify_own_answers.py'
source=SRC.read_text(); tree=ast.parse(source)
loop=next(n for n in tree.body if isinstance(n,ast.For) and '"das" not in dirs[l]' in ast.get_source_segment(source,n))
spec=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='specs_for')
old={'estimator':'das','layers':[11],'alpha':3,'das_dir':'old','das_variant':'tf'}
new={**old,'das_dir':'new','das_variant':'tf-ce-bidir'}
calls=[]
def load(d,*args,**kwargs):
 calls.append((d,kwargs['variant']))
 return ([1.,0.] if d=='old' else [0.,1.]),{}
ns={'configs':{'das_tf_full':old,'das_tfce_full':new},'dirs':{11:{'md':[1.,1.],'cB':[0.,0.]}},'load_das_direction':load,'SteerSpec':lambda w,l,**kw:{'W':w,'layer':l,**kw}}
exec(compile(ast.Module(body=[loop,spec],type_ignores=[]),str(SRC),'exec'),ns)
result={'source':str(SRC),'loop_line':loop.lineno,'specs_line':spec.lineno,'load_calls':calls,'old_condition':ns['specs_for'](old),'new_condition':ns['specs_for'](new),'expected_new_W':[0.,1.],'current_experiments':{}}
base=SRC.parents[3]/'results/02_问答DAS_v1/第16轮_DAS正结果搜索/measurements'
for f in sorted(base.glob('verify_own*.json')):
 d=json.loads(f.read_text()); paths={(v['das_dir'],v['das_variant']) for v in d['configs'].values() if v and v['estimator']=='das'}
 result['current_experiments'][f.name]={'distinct_das_sources':len(paths),'mixture_present':len(paths)>1}
assert calls==[('old','tf')]
assert result['new_condition'][0]['W']==[1.,0.]
assert all(not v['mixture_present'] for v in result['current_experiments'].values())
Path(os.environ.get('TD_AUDIT_OUT', str(Path(__file__).resolve().parent))).joinpath('direction_cache_reproduction.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
