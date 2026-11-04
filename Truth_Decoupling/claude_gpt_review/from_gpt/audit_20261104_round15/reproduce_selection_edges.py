#!/usr/bin/env python3
"""Independent offline checks for review 15; synthetic reports, never model inference."""
import argparse
import ast
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--code-root',type=Path,required=True)
parser.add_argument('--data-root',type=Path,required=True)
args=parser.parse_args()
CODE = args.code_root
DATA = args.data_root
OUT = Path(tempfile.mkdtemp(prefix='td_review15_selection_checks_', dir='/tmp'))
OUT.mkdir(exist_ok=True)
sys.path[:0] = [str(CODE/'scripts'), str(CODE)]
from core.metrics import auroc, sel_key, spearman_rho


def module(name):
    spec = importlib.util.spec_from_file_location('audit15_'+name, CODE/'scripts'/(name+'.py'))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


cmp = module('e23_compare_estimators')
two = module('report_2x2')
merge = module('e23_merge')
rw = module('report_read_write_layers')
est = module('report_est_size')
results = {}
results['run_directory'] = str(OUT)


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2)+'\n')


def run(script, args, log):
    proc = subprocess.run([sys.executable, str(CODE/'scripts'/(script+'.py'))] + [str(x) for x in args], capture_output=True, text=True)
    (OUT/log).write_text(proc.stdout+'\nSTDERR:\n'+proc.stderr)
    return proc


IDS = np.array(['q:%d:%s' % (i, label) for i in range(4) for label in ('T','F')])
LABELS = np.tile([True, False], 4)
HALF = np.array([True]*4+[False]*4)
BASE_A = np.array([.8,.2,.2,.8]*2)
BASE_B = np.array([.2,.8,.8,.2]*2)  # Same counts/AUC, opposite per-question correctness.


def fixture(root, tag, name, *, base=BASE_A, candidate=None, infeasible=False, n_est=192,
            model='fixture', estimator='das', labels=LABELS, merge_path=False):
    root = Path(root)
    directory = root / ('results_tqa/E23' if merge_path else 'E23') / 'fixture'
    directory.mkdir(parents=True, exist_ok=True)
    p = base.copy() if candidate is None else np.array(candidate, float)
    answer = np.zeros(8, bool) if infeasible else np.ones(8, bool)
    arrays = {'baseline': (np.ones(8, bool), np.array(base, float)), name: (answer, p)}
    conds = {}
    for condition, (a, probability) in arrays.items():
        conds[condition] = {split: cmp.metrics(a[mask], probability[mask], np.asarray(labels)[mask])
                           for split, mask in [('A',HALF),('B',~HALF),('all',np.ones(8,bool)),('est',HALF)]}
    record = {'model':model,'n':8,'n_est':n_est,'num_layers':1,'layers':[0],
              'tune_pairs':2,'teacher_split':False,'wording':'pik','direction_prompt_allow_idk':False,
              'e_alphas':[2.],'e_estimator':estimator,'das':None,'conds':conds}
    stem = directory / ('knob_sweep_'+tag)
    dump(stem.with_suffix('.json'), record)
    np.savez(stem.with_suffix('.npz'), ids=IDS, labels=labels, half_a=HALF,
             **{c+'__answer':a for c,(a,p) in arrays.items()}, **{c+'__ptrue':p for c,(a,p) in arrays.items()})
    return record


# 1. The exact key agrees with true integer confusion counts, even on unbalanced classes.
for np_, nn in [(2,3),(17,23),(192,192),(337,337)]:
    rng = np.random.default_rng(89)
    tp0, fp0 = int(rng.integers(np_+1)), int(rng.integers(nn+1))
    j0 = tp0/np_ - fp0/nn
    for _ in range(1000):
        tp, fp = int(rng.integers(np_+1)), int(rng.integers(nn+1))
        assert sel_key(tp/np_ - fp/nn - j0, np_*nn) == (tp-tp0)*nn - (fp-fp0)*np_
assert sel_key(.05208333333333326,192**2)==sel_key(.05208333333333337,192**2)
results['integer_key_balanced_and_unbalanced'] = 'PASS'

# 2. All exposed selectors choose the known real-cache tied condition correctly.
p = DATA/'results_das_tf_r288/E23/qwen3-8b/knob_sweep_meandiff_full'
j = json.loads(p.with_suffix('.json').read_text())
names = [n for n in j['conds'] if n.startswith('E_')]
assert cmp.rule_pick(j['conds'],names,192**2)=='E_L34_a3'
assert two.pick_e(j['conds'],192**2)[0]=='E_L34_a3'
summary = json.loads((DATA/'results_das_tf_r288/E1/qwen3-8b/summary.json').read_text())
profile = {'E':{'rows':{k:{'sep':v['sep_E']} for k,v in summary['rows'].items()}}}
assert rw.summarize(summary,profile,j)['write_candidate']['condition']=='E_L34_a3'
results['real_cache_exposed_selectors_agree'] = 'PASS'

# 3. Main E23 selection block itself, extracted unchanged, handles no feasible E.
tree=ast.parse((CODE/'scripts/e23_knob_sweep.py').read_text())
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
start=next(i for i,n in enumerate(main.body) if isinstance(n,ast.FunctionDef) and n.name=='delta')
stop=next(i for i,n in enumerate(main.body) if i>start and isinstance(n,ast.Assign)
          and any(isinstance(t,ast.Name) and t.id=='out_dir' for t in n.targets))
none_root=OUT/'none'
ja=fixture(none_root,'a','E_L0_a2',infeasible=True)
jb=fixture(none_root,'b','E_L0_a2',infeasible=True,estimator='meandiff')
ns={'np':np,'sel_key':sel_key,'res':{'conds':copy.deepcopy(ja['conds'])},'labels':LABELS,'half_a':HALF,
    'y_est':LABELS,'y_dev':LABELS,'args':SimpleNamespace(teacher_split=False)}
exec(compile(ast.Module(body=main.body[start:stop],type_ignores=[]),'unmodified_e23_selection','exec'),ns)
assert ns['res']['selection']['E']['chosen'] is None
assert ns['res']['selection']['E']['B_half'] is None
assert ns['res']['selection_on_est']['E']['chosen'] is None
assert cmp.rule_pick(ja['conds'],['E_L0_a2'],4) is None
assert two.pick_e(ja['conds'],4)[0] is None
simple_summary={'model':'fixture','num_layers':1,'direction_prompt_allow_idk':False,'n_E':[96,96]}
simple_profile={'E':{'rows':{'0':{'sep':.8}}}}
assert rw.summarize(simple_summary,simple_profile,jb)['write_candidate'] is None
results['e23_and_readwrite_none'] = 'PASS'

# 4. Real CLI paths for no-feasible E, including missing test oracle.
proc=run('e23_compare_estimators',['--root',none_root,'--model','fixture','--tag-a','a','--tag-b','b',
                                 '--n-boot','20','--out',none_root/'compare.md'],'none_compare.log')
assert proc.returncode==0 and 'no paired difference' in proc.stdout,proc.stderr
fixture(none_root,'c','Cp_L0_abst_a1',infeasible=True)
proc=run('report_2x2',['--e-root',none_root,'--e-tag','a','--c-root',none_root,'--c-tag','c','--model','fixture',
                      '--n-boot','20','--out',none_root/'two.md'],'none_2x2.log')
assert proc.returncode==0 and 'none feasible' in proc.stdout,proc.stderr
fixture(none_root,'m1','E_L0_a2',infeasible=True,merge_path=True)
fixture(none_root,'m2','E_L1_a2',infeasible=True,merge_path=True)
proc=run('e23_merge',['--root',none_root,'--model','fixture','--tags','m1','m2','--out-tag','merged',
                     '--n-boot','20'],'none_merge.log')
assert proc.returncode==0,proc.stderr
merged=json.loads((none_root/'results_tqa/E23/fixture/knob_sweep_merged.json').read_text())
assert merged['selection']['E']['chosen'] is None and merged['selection']['E']['test'] is None
for root,n_est in [(none_root/'old',192),(none_root/'new',576)]:
    fixture(root,'das_tf_full','E_L0_a2',infeasible=True,n_est=n_est)
    fixture(root,'meandiff_full','E_L0_a2',infeasible=True,n_est=n_est,estimator='meandiff')
proc=run('report_est_size',['--root-old',none_root/'old','--root-new',none_root/'new','--model','fixture',
                          '--n-boot','20','--out',none_root/'est.md'],'none_est.log')
assert proc.returncode==0,proc.stderr
est_result=json.loads((none_root/'est.json').read_text())
assert all(v is None for v in est_result['chosen'].values()) and not est_result['paired_differences']
results['four_none_feasible_cli_reports'] = 'PASS'

# Feasible on tune but no feasible oracle on test, formerly a formatting/crash edge.
oracle_root=OUT/'no_test_oracle'
for tag in ['a','b']:
    fixture(oracle_root,tag,'E_L0_a2')
    stem=oracle_root/'E23/fixture'/('knob_sweep_'+tag)
    with np.load(stem.with_suffix('.npz')) as archive:z={k:archive[k] for k in archive.files}
    z['E_L0_a2__answer'][~HALF]=False
    np.savez(stem.with_suffix('.npz'),**z)
    rec=json.loads(stem.with_suffix('.json').read_text());rec['conds']['E_L0_a2']['B']['ans']=0.
    dump(stem.with_suffix('.json'),rec)
proc=run('e23_compare_estimators',['--root',oracle_root,'--model','fixture','--tag-a','a','--tag-b','b',
                                 '--n-boot','20','--out',oracle_root/'compare.md'],'no_test_oracle.log')
assert proc.returncode==0 and 'none feasible' in proc.stdout,proc.stderr
results['feasible_tune_no_test_oracle'] = 'PASS'

# 5. Merge must not silently substitute another run's per-item baseline.
bad_root=OUT/'baseline_guard_repro'
candidate=np.array([.8,.2,.8,.2,.8,.2,.2,.8])  # Tune improved; test exactly equals first chunk baseline.
fixture(bad_root,'one','E_L0_a2',base=BASE_A,candidate=candidate,merge_path=True)
fixture(bad_root,'two','E_L1_a2',base=BASE_B,candidate=BASE_B,merge_path=True)
proc=run('e23_merge',['--root',bad_root,'--model','fixture','--tags','one','two','--out-tag','bad_merge',
                     '--n-boot','200'],'baseline_guard_repro.log')
assert proc.returncode==0,proc.stderr
merged=json.loads((bad_root/'results_tqa/E23/fixture/knob_sweep_bad_merge.json').read_text())
assert merged['selection']['E']['chosen']=='E_L0_a2'
assert merged['selection']['E']['ci']['J']!=[0.,0.]
results['remaining_bug_merge_baseline_substitution']={
    'expected_own_baseline_delta_J_CI':[0.,0.],
    'actual_merged_delta_J_CI':merged['selection']['E']['ci']['J'],
    'fixture_root':str(bad_root)}

# The same merge guard accepts conflicting labels and method metadata.
wrong_root=OUT/'label_metadata_guard_repro'
fixture(wrong_root,'one','E_L0_a2',infeasible=True,merge_path=True)
fixture(wrong_root,'two','E_L1_a2',infeasible=True,model='OTHER_MODEL',estimator='meandiff',labels=~LABELS,merge_path=True)
proc=run('e23_merge',['--root',wrong_root,'--model','fixture','--tags','one','two','--out-tag','bad_merge',
                     '--n-boot','20'],'label_metadata_guard_repro.log')
results['remaining_bug_merge_label_model_estimator_guard']={'accepted':proc.returncode==0,'fixture_root':str(wrong_root)}
assert proc.returncode==0,proc.stderr

# Existing Spearman implementation in the merge report still disagrees with core metrics for ties.
bad_rho=merge.spearman([1.,1.,1.],[1.,2.,3.])
assert bad_rho==1. and np.isnan(spearman_rho([1.,1.,1.],[1.,2.,3.]))
results['remaining_bug_merge_spearman_constant']={'merge_value':bad_rho,'correct_value':'undefined/NaN'}

# 6. The actual plan-split main entry hard-fails wrong training counts, with all model work replaced by stubs.
sweep=module('e23_knob_sweep')
prompts=json.loads((CODE/'configs/prompts/default_v1.json').read_text())
datasets={s:[json.loads(line) for line in (DATA/'code/data/processed/truthfulqa_tf'/(s+'.jsonl')).read_text().splitlines()]
          for s in ('calib','dev','test')}
das_dir=OUT/'fake_das';das_dir.mkdir(exist_ok=True);dump(das_dir/'summary.json',{'est_extra':None})
bundle=SimpleNamespace(num_layers=1,device='cpu',tokenizer=None,build_chat=lambda *args,**kwargs:'mock text')
def capture(bundle,texts,layers,components=('resid',),batch_size=8,return_logits=False):
    h=torch.stack([torch.tensor([float(i%2),1.]) for i in range(len(texts))])
    return {('resid',l):h for l in layers},torch.zeros(len(texts),2) if return_logits else None
argv=['e23','--model','fixture','--exp','fixture','--layers','0','--plan-split',
      '--e-estimator','das','--das-dir',str(das_dir),'--no-cs']
with patch.object(sys,'argv',argv), patch.object(sweep,'load_experiment',return_value={'seed':42,'eval_datasets':['tf'],'prompts':prompts}), \
     patch.object(sweep,'load_model_config',return_value={'name':'fixture'}), patch.object(sweep,'load_split',side_effect=lambda ds,s:datasets[s]), \
     patch.object(sweep,'load_model',return_value=bundle) as load, patch.object(sweep,'tf_prompt',return_value='mock text'), \
     patch.object(sweep,'capture_last_token',side_effect=capture), \
     patch.object(sweep,'build_label_token_map',return_value={'Answer':[0],'Abstain':[1]}), \
     patch.object(sweep,'load_das_direction',return_value=(np.array([1.,0.]),{'n_pairs':[96]})), \
     patch.object(sweep,'next_token_logits',side_effect=AssertionError('evaluation must not run')), contextlib.redirect_stdout(io.StringIO()):
    try:sweep.main()
    except SystemExit as ex:
        assert '--plan-split' in str(ex) and '96' in str(ex) and '32' in str(ex)
        results['plan_split_wrong_training_count_hard_fail']={'passed':True,'error':str(ex),'load_model_stub_calls_before_guard':load.call_count}
    else:raise AssertionError('plan-split accepted 96-pair directions')

dump(OUT/'results.json',results)
print(json.dumps(results,indent=2))
