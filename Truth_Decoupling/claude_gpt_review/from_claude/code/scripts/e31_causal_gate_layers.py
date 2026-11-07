#!/usr/bin/env python3
"""E31 exploratory intervention-layer diagnosis with frozen candidate answers.

C writes only the separate Yes/No gate prompt. E is fixed to the original E26
50%-quota tuning choice, and candidate answers stay exactly the saved base/E
answers. Default C layers are predeclared 25/50/75% depths plus each method's
original correlation-selected layer. --all-layers is explicit and non-default.
No test outputs are used; all effectiveness summaries are tuning diagnostics.
"""
import argparse
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import e26_matched_coverage as joint
from core.experiment_provenance import implementation_identity, require_unchanged
from core.joint_steering import fingerprint
from core.selective_eval import quota_select

PROTOCOL = 'e31_causal_gate_layers_v1'
METHODS = ('das', 'md')
COVERAGES = (.25, .5, .75)
ALPHAS = (-1., 1.)
SOURCE_FILES = ('bank.json','bank.npz','data.json','selection.json',
                'tf_tune.json','qa_tune.json','qa_tune_scored.json')


def read(path):
    return joint.read(path)


def source(args):
    ns = SimpleNamespace(run_dir=args.source_run, model=args.model)
    meta, bank, data = joint.load_run(ns)
    selection = joint.frozen_selection(ns, meta)
    if meta['pilot']:
        raise ValueError('source must be a complete original E26 run')
    if meta.get('generation_adapters'):
        raise ValueError('this diagnostic excludes supervised E27 generation adapters')
    tune = {task: joint.scored_rows(ns, meta, task, 'tune')[0] for task in joint.TASKS}
    return meta, bank, data, selection, tune


def requested_layers(num_layers, original, all_layers=False):
    if type(num_layers) is not int or num_layers < 1:
        raise ValueError('invalid number of layers')
    preset = list(range(num_layers)) if all_layers else [math.ceil(num_layers*f)-1 for f in (.25,.5,.75)]
    if any(type(x) is not int or not 0 <= x < num_layers for x in original):
        raise ValueError('original layer outside model')
    return sorted(set(preset + list(original)))


def clean_input(row, task):
    # The forward pass never receives labels, references, outcomes or answers.
    if task == 'tf':
        return {'id': str(row['id']), 'pair_id': str(row['pair_id']), 'text': row['text']}
    return {'id': str(row['pair_id']), 'pair_id': str(row['pair_id']), 'question': row['question']}


def task_plan(task, meta, bank, selection, inputs, all_layers):
    control = {'name': task+'_base_gate', 'kind':'base', 'task':task, 'method':'base','e':None,'c':None,
               'answer_condition':'baseline', 'control':True, 'original_read_layer':False}
    conditions = [control]; unavailable=[]; specs={c['name']:c for c in meta['conditions'][task]}
    fixed={}
    for method in METHODS:
        name=selection['tasks'][task]['choices']['0.5'][method+'_E']
        if name is None:
            raise ValueError('original 50% tuning E choice unavailable: '+task+'/'+method)
        selected=specs[name]
        if selected['kind']!='E' or selected['method']!=method:
            raise ValueError('source E choice identity changed')
        e=deepcopy(selected['e']);fixed[method]={'condition':name,'e':e}
        conditions.append({'name':task+'_'+method+'_E_gate','kind':'E','task':task,'method':method,'e':e,'c':None,
                           'answer_condition':name,'control':True,'original_read_layer':False})
        old_layers=meta['c_layers'][task][method]
        for layer in requested_layers(meta['num_layers'],old_layers,all_layers):
            key='%s_%s_C_L%d' % (task,method,layer)
            if key not in bank or key+'_scale' not in bank:
                unavailable.append({'method':method,'layer':layer,'reason':'no_available_estimation_C_in_original_bank',
                                    'original_read_layer':layer in old_layers})
                continue
            if not np.isfinite(bank[key]).all() or not math.isfinite(float(bank[key+'_scale'])) or float(bank[key+'_scale'])<=0:
                raise ValueError('invalid existing C direction/scale')
            for kind,answer,e_spec in [('C','baseline',None),('EC',name,e)]:
                for alpha in ALPHAS:
                    conditions.append({'name':'%s_%s_%s_L%d_a%g' % (task,method,kind,layer,alpha),
                        'kind':kind,'task':task,'method':method,'e':e_spec,'c':{'layer':layer,'alpha':alpha},
                        'answer_condition':answer,'control':False,'original_read_layer':layer in old_layers})
    if len({c['name'] for c in conditions})!=len(conditions):
        raise ValueError('duplicate gate condition')
    return {'input_rows':inputs,'fixed_E':fixed,'conditions':conditions,'unavailable_requested_layers':unavailable,
            'original_C_layers':meta['c_layers'][task]}


def make_plan(args):
    if args.limit is not None and args.limit<1:
        raise ValueError('limit must be a positive number of tuning questions')
    meta, bank, data, selection, tune=source(args)
    keep={str(r['pair_id']) for r in data['qa']['tune'][:args.limit]}
    tasks={}
    for task in joint.TASKS:
        inputs=[clean_input(r,task) for r in data[task]['tune'] if str(r['pair_id']) in keep]
        if not inputs:
            raise ValueError('empty tuning input cohort')
        tasks[task]=task_plan(task,meta,bank,selection,inputs,args.all_layers)
    plan={'protocol':PROTOCOL,'model':args.model,'exploratory':True,'split':'tune',
          'bank_fingerprint':meta['fingerprint'],'source_sha256':{n:joint.digest(args.source_run/n) for n in SOURCE_FILES},
          'implementation':implementation_identity(__file__), 'pilot_limit':args.limit,
          'all_layers':args.all_layers,'layer_rule':'ceil(depth_fraction * num_layers) - 1, fractions .25/.50/.75, union original method-specific C layers; unavailable layers stay explicit',
          'alphas':list(ALPHAS),'zero_alpha':'reuse base/E gate controls without adding a zero-write hook',
          'coverages':list(COVERAGES),'batch_size':meta['config']['batch_size'],'seed':meta['config']['seed'],
          'native_rule':'deliver frozen candidate iff intervened first Yes-No score > 0 and candidate is a substantive answer',
          'candidate_answer_rule':'original baseline for C, original tuning-50%-selected E for EC; no new answer generation',
          'precision':'float32 output projection, native backbone, original E26 batch size',
          'claims_limit':'Exploratory gate-only writing-layer diagnosis; no test selection and no claim about all answer-context C effects.',
          'tasks':tasks}
    plan['fingerprint']=fingerprint(plan)
    return plan


def prepare(args):
    plan=make_plan(args);args.run_dir.mkdir(parents=True,exist_ok=True)
    path=args.run_dir/'plan.json'
    if path.exists() and read(path)!=plan:
        raise ValueError('prepared plan/source/code changed')
    root=Path(__file__).resolve().parents[1]
    for rel,h in plan['implementation']['files'].items():
        target=args.run_dir/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and joint.digest(target)!=h:
            raise ValueError('existing source snapshot changed')
        shutil.copyfile(root/rel,target)
        if joint.digest(target)!=h:
            raise ValueError('source snapshot copied inconsistent bytes')
    joint.save(path,plan,immutable=True)
    print('E31 PREPARED',args.model,{t:len(x['conditions']) for t,x in plan['tasks'].items()},'pilot',args.limit,flush=True)
    return plan


def load_plan(args):
    plan=read(args.run_dir/'plan.json')
    if plan['fingerprint']!=fingerprint({k:v for k,v in plan.items() if k!='fingerprint'}):
        raise ValueError('plan fingerprint changed')
    if plan.get('protocol')!=PROTOCOL or plan.get('model')!=args.model or plan.get('split')!='tune' or plan.get('exploratory') is not True:
        raise ValueError('foreign/unsupported plan')
    require_unchanged(plan['implementation'],__file__)
    if args.limit is not None and args.limit != plan['pilot_limit']:
        raise ValueError('CLI limit differs from prepared plan')
    if args.all_layers and not plan['all_layers']:
        raise ValueError('CLI all-layers differs from prepared plan')
    # Reconstruct the same predeclared plan from source, not current test results.
    ns=SimpleNamespace(**vars(args));ns.limit=plan['pilot_limit'];ns.all_layers=plan['all_layers']
    if make_plan(ns)!=plan:
        raise ValueError('frozen plan no longer matches sources or preset choices')
    for rel,h in plan['implementation']['files'].items():
        if joint.digest(args.run_dir/'source_snapshot'/rel)!=h:
            raise ValueError('saved source snapshot changed')
    return plan


def validate_condition_rows(rows, inputs, partial=False):
    ids=[r['id'] for r in inputs]
    if len(rows)>len(ids) or [r['id'] for r in rows]!=(ids[:len(rows)] if partial else ids):
        raise ValueError('gate row cohort/order incomplete or changed')
    for r,q in zip(rows,inputs):
        if set(r)!={'id','pair_id','score'} or r['pair_id']!=q['pair_id'] or isinstance(r['score'],bool) or not math.isfinite(r['score']):
            raise ValueError('gate row identity/score invalid')


def collect(args,bundle=None):
    plan=load_plan(args);meta,bank,data,selection,tune=source(args)
    bundle=joint.open_bundle(meta) if bundle is None else bundle
    joint.verify_tokens(bundle,meta)
    identity={'protocol':PROTOCOL,'model':args.model,'split':'tune','plan_sha256':joint.digest(args.run_dir/'plan.json'),
              'pilot_limit':plan['pilot_limit'],'implementation':plan['implementation']}
    path=args.run_dir/'gate_scores.json'
    out=read(path) if path.exists() else {**identity,'completed':False,'tasks':{task:{'rows':{},'control_source_score_max_abs_diff':{}} for task in joint.TASKS}}
    if any(out.get(k)!=v for k,v in identity.items()) or set(out['tasks'])!=set(joint.TASKS):
        raise ValueError('gate resume identity changed')
    with joint.fp32_output_projection(bundle):
        for task,tp in plan['tasks'].items():
            names={c['name'] for c in tp['conditions']}
            if not set(out['tasks'][task]['rows']).issubset(names):
                raise ValueError('unexpected gate condition in resume')
            inputs=tp['input_rows']
            for c in tp['conditions']:
                rows=out['tasks'][task]['rows'].setdefault(c['name'],[])
                validate_condition_rows(rows,inputs,partial=True)
                for start in range(len(rows),len(inputs),plan['batch_size']):
                    chunk=inputs[start:start+plan['batch_size']]
                    texts=[joint.gate_text(bundle,meta,task,r) for r in chunk]
                    lm=joint.gate_map(bundle,texts[0],meta['config'])
                    logits=joint.one_step(bundle,texts,joint.intervention_specs(bank,c,'gate'))
                    margins=logits[:,lm['yes']]-logits[:,lm['no']]
                    rows.extend({'id':q['id'],'pair_id':q['pair_id'],'score':float(m)} for q,m in zip(chunk,margins))
                    validate_condition_rows(rows,inputs,partial=True)
                    joint.save(path,out)
                validate_condition_rows(rows,inputs)
                if c['control']:
                    old={r['id']:r for r in tune[task][c['answer_condition']]}
                    delta=max(abs(r['score']-old[r['id']]['score']) for r in rows)
                    out['tasks'][task]['control_source_score_max_abs_diff'][c['name']]=delta
                    # The complete cohort restores the original exact batching; a
                    # 3-question pilot may have a different final-batch padding.
                    if plan['pilot_limit'] is None and delta>1e-5:
                        raise RuntimeError('full control gate scores fail production reproduction: '+c['name']+' diff='+str(delta))
                print('E31 CONDITION',args.model,task,c['name'],len(rows),flush=True)
    require_unchanged(plan['implementation'],__file__)
    out['completed']=True;joint.save(path,out)
    return out


def join_answers(source_rows, scores):
    lookup={r['id']:r for r in source_rows}
    if len(lookup)!=len(source_rows) or any(r['id'] not in lookup for r in scores):
        raise ValueError('frozen candidate answer cohort mismatch')
    out=[]
    for r in scores:
        old=lookup[r['id']]
        if old['pair_id']!=r['pair_id']:
            raise ValueError('candidate answer group differs')
        out.append({**old,'score':r['score']})
    return out


def delivery_summary(rows,positive_gate=True):
    use=[r for r in rows if r['outcome']!='abstain' and (not positive_gate or r['score']>0)]
    counts=Counter(r['outcome'] for r in use);n=len(rows);a=len(use)
    return {'n':n,'answered':a,'correct':counts['correct'],'wrong':counts['wrong'],
            'actual_coverage':a/n if n else None,'accuracy_answered':counts['correct']/a if a else None,
            'selected_ids':[r['id'] for r in use]}


def summarize_condition(rows,coverages,seed):
    result={'native_gate':delivery_summary(rows),'candidate_unfiltered':delivery_summary(rows,False),'quotas':{}}
    for q in coverages:
        v=quota_select(rows,q,seed=seed)
        result['quotas'][str(q)]={k:v[k] for k in ('feasible','reason','n','k','n_available','answered','correct','wrong','actual_coverage','accuracy_answered','selected_ids')}
    return result


def report(args):
    plan=load_plan(args);meta,bank,data,selection,tune=source(args);scores=read(args.run_dir/'gate_scores.json')
    if (scores.get('protocol')!=PROTOCOL or scores.get('model')!=args.model or scores.get('split')!='tune'
            or set(scores.get('tasks',{}))!=set(joint.TASKS)
            or not scores.get('completed') or scores.get('plan_sha256')!=joint.digest(args.run_dir/'plan.json')
            or scores.get('implementation')!=plan['implementation'] or scores.get('pilot_limit')!=plan['pilot_limit']):
        raise ValueError('incomplete/foreign collected gate scores')
    result={'protocol':PROTOCOL,'model':args.model,'exploratory':True,'split':'tune','pilot_limit':plan['pilot_limit'],
            'plan_sha256':joint.digest(args.run_dir/'plan.json'),'gate_scores_sha256':joint.digest(args.run_dir/'gate_scores.json'),
            'interpretation':'All layer points retained. Best-on-tune values are in-sample diagnostics, not held-out gains or significance.',
            'tasks':{}}
    for task,tp in plan['tasks'].items():
        source_rows=tune[task];names={c['name'] for c in tp['conditions']}
        if set(scores['tasks'][task]['rows'])!=names:
            raise ValueError('missing/extra gate conditions')
        records={};simple={};current={};joined_rows={};recomputed_control_diffs={}
        for c in tp['conditions']:
            raw=scores['tasks'][task]['rows'][c['name']];validate_condition_rows(raw,tp['input_rows'])
            joined=join_answers(source_rows[c['answer_condition']],raw)
            joined_rows[c['name']]=joined
            records[c['name']]={'condition':c,**summarize_condition(joined,plan['coverages'],plan['seed'])}
            if c['control']:
                old={r['id']:r for r in source_rows[c['answer_condition']]}
                delta=max(abs(r['score']-old[r['id']]['score']) for r in raw)
                recomputed_control_diffs[c['name']]=delta
                if plan['pilot_limit'] is None and delta>1e-5:
                    raise ValueError('report control scores fail original production reproduction')
                simple_rows=[{**r,'score':r['simple_score']} for r in joined]
                simple[c['name']+'_simple']=summarize_condition(simple_rows,plan['coverages'],plan['seed'])['quotas']
            if c['original_read_layer']:
                current[c['name']]=records[c['name']]
        if recomputed_control_diffs!=scores['tasks'][task]['control_source_score_max_abs_diff']:
            raise ValueError('stored control reproduction diagnostic differs from raw scores')
        base_rows=joined_rows[task+'_base_gate']
        for name,rec in records.items():
            method=rec['condition']['method'];controls={'base_gate':base_rows,
                'base_simple':[{**r,'score':r['simple_score']} for r in base_rows]}
            if method in METHODS:
                erows=joined_rows[task+'_'+method+'_E_gate']
                controls.update(fixed_E_gate=erows,fixed_E_simple=[{**r,'score':r['simple_score']} for r in erows])
            k=rec['native_gate']['answered'];n=rec['native_gate']['n'];matched={}
            for label,rs in controls.items():
                v=quota_select(rs,Fraction(k,n),seed=plan['seed'])
                matched[label]={key:v[key] for key in ('feasible','reason','n','k','answered','correct','accuracy_answered')}
                matched[label]['delta_correct_native_minus_control']=(rec['native_gate']['correct']-v['correct']) if v['feasible'] else None
            rec['same_actual_count_controls']=matched
        best={}
        for method in METHODS:
            for kind in ('C','EC'):
                cand=[v for v in records.values() if v['condition']['method']==method and v['condition']['kind']==kind]
                family=method+'_'+kind;best[family]={}
                for q in map(str,plan['coverages']):
                    eligible=[v for v in cand if v['quotas'][q]['feasible']]
                    chosen=min(eligible,key=lambda v:(-v['quotas'][q]['correct'],abs(v['condition']['c']['alpha']),
                        v['condition']['c']['layer'],v['condition']['c']['alpha'],v['condition']['name'])) if eligible else None
                    best[family][q]=None if chosen is None else {'condition':chosen['condition']['name'],'correct':chosen['quotas'][q]['correct'],
                        'warning':'selected on these same tuning answers; optimistic diagnostic only'}
        result['tasks'][task]={'conditions':records,'simple_controls':simple,'original_read_layer_conditions':list(current),
                              'best_on_tune_only':best,'unavailable_requested_layers':tp['unavailable_requested_layers'],
                              'control_source_score_max_abs_diff':scores['tasks'][task]['control_source_score_max_abs_diff']}
    require_unchanged(plan['implementation'],__file__)
    joint.save(args.run_dir/'report.json',result)
    print('E31 REPORT',args.model,'pilot',plan['pilot_limit'],flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','collect','report']);p.add_argument('--model',required=True,choices=['llama3.1-8b','qwen3-8b'])
    p.add_argument('--source-run',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--limit',type=int);p.add_argument('--all-layers',action='store_true');a=p.parse_args()
    {'prepare':prepare,'collect':collect,'report':report}[a.phase](a)


if __name__=='__main__':main()
