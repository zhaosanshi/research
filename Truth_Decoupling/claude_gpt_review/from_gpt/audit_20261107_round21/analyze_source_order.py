"""Summarize every predeclared E37 combination; never select a winning layer."""
import argparse,hashlib,json,math
from pathlib import Path

MODELS=('llama3.1-8b','qwen3-8b')
COMBOS=('EC_before_m1','EC_before_p1','EC_after_m1','EC_after_p1')
NAMES={'baseline','E_source',*COMBOS,*(x[1:] for x in COMBOS)}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def holm(rows,key):
    valid=[(r[key],i) for i,r in enumerate(rows) if r[key] is not None]
    if any(not isinstance(p,(int,float)) or not math.isfinite(p) or not 0<=p<=1 for p,_ in valid):raise ValueError('invalid p value')
    running=0.;out=[None]*len(rows)
    for rank,(p,i) in enumerate(sorted(valid)):
        running=max(running,min(1.,(len(rows)-rank)*p));out[i]=running
    return out

def comparison(point,a,b):
    matches=[r for r in point['comparisons'] if {r['arm_a'],r['arm_b']}=={a,b}]
    if len(matches)!=1:raise ValueError('predeclared comparison missing/duplicated')
    r=matches[0]
    if not point['arms'][a]['feasible'] or not point['arms'][b]['feasible']:
        if (r.get('exact_fixed_deliveries') is not None or r.get('ci95') is not None
                or r.get('bootstrap_status') != 'infeasible_point'):
            raise ValueError('infeasible comparison contains an effect/test interval')
        return {'p':None,'ci95':None,'reason':'infeasible_actual_coverage','infeasible_resamples':None}
    same=r['arm_a']==a
    exact=r.get('exact_fixed_deliveries')
    if not isinstance(exact,dict) or exact.get('p_value') is None:raise ValueError('missing paired test')
    interval=r.get('ci95')
    ci=None if interval is None else interval['accuracy_answered']
    if ci is not None and not same:ci=[-ci[1],-ci[0]]
    return {'p':exact['p_value'],'ci95':ci,'reason':r['bootstrap_status'],'infeasible_resamples':r['infeasible_resamples']}

def summarize(inputs):
    rows=[];sources={}
    for model,path in inputs.items():
        d=read(path)
        if d.get('model')!=model or set(d['conditions'])!=NAMES or d.get('n_boot')!=2000 or d.get('original_generation_protocol')!='source_e_commitment_layer_order_v1':
            raise ValueError('requires full registered E37 ten-arm uniform sidecar')
        sources[model]={'path':str(path),'sha256':sha(path),'generation_and_grade':d['source_files']}
        views={'R1_common':d['common_unmodified_confidence_corrected'],
               'R1_native':d['native_policy_confidence_corrected'],
               'R2_common':d['quality_sensitivity']['common_unmodified_model'],
               'R2_native':d['quality_sensitivity']['native_policy']}
        for view,data in views.items():
            if [p['target_coverage'] for p in data['coverages']]!=[.25,.5,.75]:raise ValueError('fixed coverage set changed')
            for point in data['coverages']:
                if (point['n']!=192 or point['k']!=int(point['target_coverage']*192)
                        or set(point['arms'])!=NAMES
                        or any(a['answered']!=point['k'] for a in point['arms'].values() if a['feasible'])):
                    raise ValueError('pilot, wrong quota or arm subset cannot enter full report')
                for name in COMBOS:
                    control=name[1:];cmp=comparison(point,name,'E_source');vs_c=comparison(point,name,control)
                    counts={n:point['arms'][n]['correct'] for n in ('baseline','E_source',control,name)}
                    available=all(point['arms'][n]['feasible'] for n in ('baseline','E_source',control,name))
                    delta=None if counts[name] is None or counts['E_source'] is None else counts[name]-counts['E_source']
                    rows.append({'model':model,'view':view,'coverage':point['target_coverage'],'k':point['k'],
                        'condition':name,'C_control':control,'counts':counts,'all_four_feasible':available,
                        'delta_vs_fixed_E':delta,'p_E':cmp['p'],'ci95_accuracy_vs_E':cmp['ci95'],
                        'bootstrap_status':cmp['reason'],'infeasible_resamples':cmp['infeasible_resamples'],
                        'p_C':vs_c['p']})
    if set(inputs)!=set(MODELS):raise ValueError('both models required; no partial family correction')
    for view in ('R1_common','R1_native','R2_common','R2_native'):
        group=[r for r in rows if r['view']==view]
        if len(group)!=24:raise ValueError('primary family must retain all 24 predeclared cells')
        for r,p in zip(group,holm(group,'p_E')):r['holm_p_E']=p
    return {'protocol':'r21_fixed_E_source_order_analysis_v1','primary_view':'R2_common','primary_family_size':24,
            'hypothesis':'each fixed C position/sign combination versus the exact same E, across both models and all three quotas',
            'exploratory_tuning_only':True,'infeasible_policy':'No fabricated effects or p values; planned family size retained in adjustment',
            'sources':sources,'rows':rows}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--llama',type=Path,required=True);p.add_argument('--qwen',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    result=summarize(dict(zip(MODELS,(a.llama,a.qwen))));a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 自由问答：固定E、不同C层序的完整同率表','','本轮为已暴露调参题探索。主要口径为全臂完整拒答守卫＋未自然完成/乱码的正确计失败，使用共同原模型平均词元对数概率排序。其余口径全部另列。','','']
    for view in ('R2_common','R1_common','R2_native','R1_native'):
        lines += ['## '+view,'','|模型|作答率/数量|C位置/符号|原模型|E|C|E+C|组合−E|校正p|区间状态|','|---|---|---|---:|---:|---:|---:|---:|---:|---|']
        for r in result['rows']:
            if r['view']!=view:continue
            c=r['counts'];fmt=lambda x:'不可行' if x is None else str(x)
            lines.append('| %s | %g%% / %d | %s | %s | %s | %s | %s | %s | %s | %s |'%(r['model'],100*r['coverage'],r['k'],r['condition'],fmt(c['baseline']),fmt(c['E_source']),fmt(c[r['C_control']]),fmt(c[r['condition']]),fmt(r['delta_vs_fixed_E']),fmt(r['holm_p_E']),r['bootstrap_status']))
        lines.append('')
    lines += ['主要24格分别校正；C位置同时伴随该层方向改变。训练状态中位写入预算已对齐，功能位移和评估实际剂量仍需看原始诊断，不能把该表解释为所有层/顺序的普遍结论。']
    (a.output_dir/'analysis.md').write_text('\n'.join(lines)+'\n')
    print('all 24 primary comparisons retained; auxiliary views retained',len(result['rows']))
if __name__=='__main__':main()
