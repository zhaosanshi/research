"""CPU regression evidence only; no production changes.
撰写：赵磊（GPT-6 代笔）。
变更记录：2026-11-16 初建，赵磊（GPT-6 代笔）。
"""
import ast,json,math,hashlib,types
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[2]/'code'
def extract(path,names,ns):
 tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),ns)
ns={'torch':torch,'math':math};extract(root/'core/qa_prompt_das.py',{'unit','_stored_norm_matched_random'},ns)
ns['pd']=types.SimpleNamespace(unit=ns['unit'],_stored_norm_matched_random=ns['_stored_norm_matched_random'])
extract(root/'core/qa_factorial_das.py',{'ShiftHook'},ns)
before=torch.tensor([[[0.,0.],[128.,0.]]],dtype=torch.bfloat16)
u=torch.tensor([0.,1.]);ref=torch.tensor([1.,0.]);wc=u;amount=.1
hook=ns['ShiftHook'](u,amounts=torch.tensor([amount]),mask=torch.ones((1,2),dtype=torch.bool),decode_all=False,reference_direction=ref)
after=hook(None,(),before);delta=after.float()-before.float()
actual=(delta@wc)[0];nominal=amount*float(u@wc)
out={'scope':'CPU synthetic bfloat16 counterexample; exact mirrored ShiftHook and matching helper, no model/GPU','state_before':before.float().tolist(),'actual_stored_delta':delta.tolist(),'nominal_direct':nominal,'actual_direct_per_position':actual.tolist(),'last_nominal_subtraction_residual_with_identity_downstream':float(actual[-1])-nominal,'mean_nominal_subtraction_residual_with_identity_downstream':float(actual.mean())-nominal}
assert actual[-1]==0 and nominal>.09
print(json.dumps(out))
# Actual load_grid function, no heavy import.
g={'Path':Path,'json':json,'hashlib':hashlib};extract(root/'scripts/review_checks/ff_ec_probe.py',{'tok','e_alone','load_grid'},g)
import tempfile
fixture=tempfile.TemporaryDirectory(prefix='gpt_probe_cpu_');grid=Path(fixture.name)
base={'completed':True,'split':'tune','conditions':['baseline_x','E_L22_x-w-p1'],'question_ids':['q1'],'evaluations':[{'pair_id':'q1','outputs':{'baseline_x':'answer','E_L22_x-w-p1':'answer'},'rule_abstain':{'baseline_x':False,'E_L22_x-w-p1':False}},{'pair_id':'q1','outputs':{'baseline_x':'answer','E_L22_x-w-p1':'I do not know'},'rule_abstain':{'baseline_x':False,'E_L22_x-w-p1':True}}]}
(grid/'judge_input.json').write_text(json.dumps(base));flags,_,_=g['load_grid'](grid,22,['1']);print('duplicate accepted:',flags)
assert flags['q1']['E1'] is True

fixture.cleanup()
