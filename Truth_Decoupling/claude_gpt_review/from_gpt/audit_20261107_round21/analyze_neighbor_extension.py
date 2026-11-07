"""Separate six-cell E40 family; never replaces the original 24-cell analysis."""
import argparse,json
from pathlib import Path
from analyze_source_order import read,sha,comparison,holm,NAMES

EXTRA={'C_neighbor_m1','EC_neighbor_m1','C_neighbor_p1','EC_neighbor_p1'}

def summarize(path):
    d=read(path)
    if (d.get('model')!='llama3.1-8b' or d.get('original_generation_protocol')!='source_e_commitment_neighbor_extension_v1'
            or d.get('n_boot')!=2000 or set(d['conditions'])!=NAMES|EXTRA):raise ValueError('complete Llama E40 fourteen-arm sidecar required')
    views={'R2_common':d['quality_sensitivity']['common_unmodified_model'],
           'R1_common':d['common_unmodified_confidence_corrected'],
           'R2_native':d['quality_sensitivity']['native_policy'],
           'R1_native':d['native_policy_confidence_corrected']}
    rows=[]
    for view,table in views.items():
        if [p['target_coverage'] for p in table['coverages']]!=[.25,.5,.75]:raise ValueError('fixed coverages changed')
        group=[]
        for p in table['coverages']:
            if (p['n']!=192 or p['k']!=int(192*p['target_coverage']) or set(p['arms'])!=NAMES|EXTRA
                    or any(a['answered']!=p['k'] for a in p['arms'].values() if a['feasible'])):raise ValueError('wrong cohort/quota/arms')
            for name in ('EC_neighbor_m1','EC_neighbor_p1'):
                c=name[1:];test=comparison(p,name,'E_source')
                counts={n:p['arms'][n]['correct'] for n in ('baseline','E_source',c,name)}
                delta=None if counts[name] is None or counts['E_source'] is None else counts[name]-counts['E_source']
                group.append({'view':view,'condition':name,'coverage':p['target_coverage'],'k':p['k'],'counts':counts,
                    'delta_vs_E':delta,'p_E':test['p'],'ci95_accuracy_vs_E':test['ci95'],'bootstrap_status':test['reason'],
                    'infeasible_resamples':test['infeasible_resamples'],'p_C':comparison(p,name,c)['p']})
        if len(group)!=6:raise ValueError('preserve all six extension hypotheses')
        for r,v in zip(group,holm(group,'p_E')):r['holm_p_E']=v
        rows+=group
    return {'protocol':'r21_neighbor_extension_analysis_v1','model':'llama3.1-8b','primary_view':'R2_common',
            'family_size':6,'primary_24_cell_family_unchanged':True,'exploratory_tuning_only':True,
            'source':{'path':str(path),'sha256':sha(path),'raw_sources':d['source_files']},'rows':rows}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sidecar',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();d=summarize(a.sidecar);a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'analysis.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    lines=['# 合格的前层C14：Llama补充结果','','仅通过训练功能验收的Llama C14进入此项；固定E15、相同已冻结写入预算。原192题十臂父控制与全部判分不变，新增四臂。主要口径为R2_common，六项比较单独校正，不替换原24项。','']
    for view in ('R2_common','R1_common','R2_native','R1_native'):
        lines+=['## '+view,'','|实际交付|组合|原模型|E|对应C|E+C|组合−E|校正p|区间状态|','|---|---|---:|---:|---:|---:|---:|---:|---|']
        for r in d['rows']:
            if r['view']!=view:continue
            c=r['counts'];name=r['condition'];f=lambda v:'不可行' if v is None else str(v)
            lines.append('| %d | %s | %s | %s | %s | %s | %s | %s | %s |'%(r['k'],name,f(c['baseline']),f(c['E_source']),f(c[name[1:]]),f(c[name]),f(r['delta_vs_E']),f(r['holm_p_E']),r['bootstrap_status']))
        lines.append('')
    lines+=['C14在训练与本轮调参的自报能力门控上都有明确方向性；这不自动证明实际回答行为独立，或所有层/力度/任务都适用。全部结果仍为已暴露题集上的探索。']
    (a.output_dir/'analysis.md').write_text('\n'.join(lines)+'\n');print('six primary cells; all four views retained')
