#!/usr/bin/env python3
"""Independent rank/count audit of the four predeclared E33 prompt conditions."""
from collections import Counter,defaultdict
from fractions import Fraction
import json,math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'from_claude/code/scripts/review_checks'))
from e26_independent_recount import read,digest,fingerprint,quota,require,tie


def auc_rank(scores,labels):
    # Count earlier negatives in ascending score order, with half credit for ties.
    bins=defaultdict(lambda:[0,0])
    for score,label in zip(scores,labels):bins[score][int(bool(label))]+=1
    pos=sum(v[1] for v in bins.values());neg=sum(v[0] for v in bins.values())
    if not pos or not neg:return None
    below=0;numerator=0.
    for score,(n,p) in sorted(bins.items()):numerator+=p*(below+n/2);below+=n
    return numerator/(pos*neg)


def check_quota(rows,q,seed,reported):
    d=quota(rows,q,'score',seed);require(d is not None,'TF quota unexpectedly infeasible')
    kept={i for i,v in d.items() if v!='abstain'};correct=sum(v=='correct' for v in d.values())
    require(reported['answered']==len(kept) and reported['correct']==correct and set(reported['selected_ids'])==kept,'quota result differs')
    return correct


def run(model,artifact_root,source_root):
    folder=artifact_root/model/'full192';source=source_root/model
    plan=read(folder/'plan.json');raw=read(folder/'readouts.json');report=read(folder/'report.json')
    require(plan['pilot_limit'] is None and plan['split']=='tune','not a full tuning run')
    require(plan['fingerprint']==fingerprint({k:v for k,v in plan.items() if k!='fingerprint'}),'plan changed')
    require(raw['completed'] and report['readouts_sha256']==digest(folder/'readouts.json'),'incomplete/changed raw')
    require(raw['plan_sha256']==report['plan_sha256']==digest(folder/'plan.json'),'plan source changed')
    for rel,h in plan['source_sha256'].items():require(digest(source/rel)==h,'original source changed')
    for rel,h in plan['implementation']['files'].items():require(digest(folder/'source_snapshot'/rel)==h,'executed source snapshot changed')
    data=read(source/'data.json');base=read(source/'tf_tune.json')['rows']['baseline'];truth={r['id']:r['label'] for r in data['tf']['tune']}
    require([r['id'] for r in base]==[r['id'] for r in plan['input_rows']['statement']],'TF cohort changed')
    require(len(plan['input_rows']['question'])==192 and len(base)==384,'not full original cohort')
    names=[c['name'] for c in plan['conditions']];require(set(names)==set(raw['rows'])==set(report['conditions']),'condition set changed')
    defs=plan['conditions'];require(defs[0]['user']==defs[1]['user'] and defs[1]['system']==defs[2]['system'],'single-variable prompt comparisons changed')
    seed=plan['seed'];controls={'base_simple':[{**r,'score':r['simple_score']} for r in base]}
    for key,identity in [('id_random','id'),('pair_random','pair_id')]:
        controls[key]=[{**r,'score':(int(tie(r[identity],seed)[:14],16)>>3)/float(2**53)} for r in base]
    summary=[];n_inputs=0;n_quota=0
    for c in defs:
        rows=raw['rows'][c['name']];inputs=plan['input_rows'][c['input_kind']];require([r['id'] for r in rows]==[r['id'] for r in inputs],'input identity changed')
        for r,inp in zip(rows,inputs):
            require(c['system'] in r['prompt'] and c['user'].format(**inp) in r['prompt'],'rendered prompt content changed')
            require(len(r['input_ids'])==len(r['attention_mask']) and r['attention_mask'][-1]==1,'token trace malformed')
            require(0<=r['p_yes']<=1 and 0<=r['p_no']<=1 and math.isclose(r['p_yes']+r['p_no'],r['yes_no_probability_mass'],abs_tol=1e-12),'full probability fields differ')
            require(math.isclose(r['p_yes']/r['yes_no_probability_mass'],r['p_yes_conditional_on_yes_no'],rel_tol=1e-8,abs_tol=1e-12),'conditional odds differ')
            require(r['top1_is_canonical_yes_or_no']==(r['top1_token_id'] in (r['yes_token_id'],r['no_token_id'])),'top1 status differs')
            n_inputs+=1
        lookup={r['id']:r for r in rows};expanded=[{**r,'score':lookup[r['pair_id'] if c['input_kind']=='question' else r['id']]['score']} for r in base]
        margins=[r['score'] for r in expanded];labels=[truth[r['id']] for r in base];actual=report['conditions'][c['name']]
        ta=auc_rank(margins,labels);ca=auc_rank(margins,[r['outcome']=='correct' for r in base])
        require(ta==actual['truth_label_auc'] and ca==actual['baseline_correctness_auc'],'rank-sum AUC differs')
        pairs=defaultdict(dict)
        for r in expanded:pairs[r['pair_id']][truth[r['id']]]=r['score']
        diff={k:v[True]-v[False] for k,v in pairs.items()};require(diff==actual['paired_truth_minus_false']['by_pair'],'pair difference changed')
        if c['input_kind']=='question':require(set(diff.values())=={0.},'question score not shared exactly')
        yes=[r for r in expanded if r['score']>0];native=actual['native_gate']
        require(native['answered']==len(yes) and native['correct']==sum(r['outcome']=='correct' for r in yes),'native count differs')
        for name,rs in controls.items():check_quota(rs,Fraction(len(yes),len(base)),seed,native['same_K_controls'][name]);n_quota+=1
        qs={}
        for q,entry in actual['quotas'].items():qs[q]=check_quota(expanded,q,seed,entry);n_quota+=1
        mass=[r['yes_no_probability_mass'] for r in rows];pa=actual['probability_audit']
        require(math.isclose(sum(mass)/len(mass),pa['mean_yes_no_mass'],rel_tol=1e-12),'mass mean differs')
        require(min(mass)==pa['min_yes_no_mass'] and sum(not r['top1_is_canonical_yes_or_no'] for r in rows)==pa['top1_outside_canonical_yes_no'],'mass/top1 audit differs')
        summary.append({'condition':c['name'],'truth_auc':ta,'correctness_auc':ca,'mean_true_false_margin':sum(diff.values())/len(diff),
                        'mean_yes_no_mass':pa['mean_yes_no_mass'],'min_yes_no_mass':min(mass),'top1_outside':pa['top1_outside_canonical_yes_no'],
                        'native_answered':native['answered'],'native_correct':native['correct'],
                        'native_same_K_correct':{name:r['correct'] for name,r in native['same_K_controls'].items()},'quota_correct':qs})
    for name,rs in controls.items():
        require(auc_rank([r['score'] for r in rs],[r['outcome']=='correct' for r in rs])==report['controls'][name]['correctness_auc'],'control AUC differs')
        for q,entry in report['controls'][name]['quotas'].items():check_quota(rs,q,seed,entry);n_quota+=1
    require(raw['legacy_source_score_max_abs_diff']==0,'legacy control not exactly reproduced')
    return {'model':model,'all_match':True,'raw_inputs_checked':n_inputs,'quota_cells_checked':n_quota,'conditions':summary,'controls':report['controls'],
            'artifact_sha256':{name:digest(folder/name) for name in ('plan.json','readouts.json','report.json')}}


def main():
    here=Path(__file__).resolve().parent;source=ROOT/'from_claude/results/03_E26同作答率_E27生成适配/E26_main'
    out={'all_match':True,'runs':[run(m,here/'results',source) for m in ('llama3.1-8b','qwen3-8b')]}
    (here/'independent_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    for r in out['runs']:print(r['model'],'inputs',r['raw_inputs_checked'],'quota cells',r['quota_cells_checked'])
if __name__=='__main__':main()
