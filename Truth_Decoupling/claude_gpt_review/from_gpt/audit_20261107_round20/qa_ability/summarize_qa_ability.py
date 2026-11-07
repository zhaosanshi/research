#!/usr/bin/env python3
"""Both fixed selectors and both estimators for the repaired QA-only run."""
import argparse,json,math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'from_claude/code/scripts/review_checks'),str(ROOT/'from_claude/code')]
from e26_independent_recount import audit_run,read,digest,require
from core.qa_scoring import FULL_PERSONAL_REFUSAL_POLICY


def holm(ps):
    out=[None]*len(ps);running=0.
    for rank,i in enumerate(sorted(range(len(ps)),key=lambda i:ps[i])):
        running=max(running,min(1.,(len(ps)-rank)*ps[i]));out[i]=running
    return out


def summarize(results_root,output):
    cells=[];families={};audits=[];checks=[];sources=[]
    original=ROOT/'from_claude/results/03_E26同作答率_E27生成适配/E26_main'
    frozen=ROOT/'from_gpt/audit_20261107_round20/qa_gate_prompt/results'
    for model in ('llama3.1-8b','qwen3-8b'):
        run=results_root/model/'qa_ability_full';bank=read(run/'bank.json');rep=read(run/'report.json');sel=read(run/'selection.json')
        template=read(frozen/model/'full192/plan.json')['conditions'][1]
        require(bank['config']['evaluation_tasks']==['qa'] and rep['evaluation_tasks']==['qa'] and set(rep['tables'])=={'qa'},'not a QA-only run')
        require(bank['config']['qa_gate_system']==template['system'] and bank['config']['qa_gate_user']==template['user'],'QA template differs from predeclared diagnostic')
        require(not (run/'tf_tune.json').exists() and not (run/'tf_test.json').exists(),'unexpected TF output in QA-only run')
        require((run/'data.json').read_bytes()==(original/model/'data.json').read_bytes(),'fixed full dataset changed')
        audits.append(audit_run(run,bootstrap=True))
        comparisons=[]
        for split in ('tune','test'):
            raw=read(run/('qa_'+split+'.json'));scored=read(run/('qa_'+split+'_scored.json'))
            old=read(original/model/('qa_'+split+'_scored.json'))['rows']['baseline'];new=scored['rows']['baseline']
            require([r['id'] for r in old]==[r['id'] for r in new],'baseline cohort changed')
            changed=[a['id'] for a,b in zip(old,new) if a['output']!=b['output'] or a['finished']!=b['finished']]
            max_score=max(abs(a['simple_score']-b['simple_score']) for a,b in zip(old,new))
            policy_count=0
            for rows in scored['rows'].values():
                for r in rows:
                    require(r['grade'].get('outcome_policy')==FULL_PERSONAL_REFUSAL_POLICY,'guard policy not uniform across all methods')
                    policy_count+=1
            comparisons.append({'split':split,'n':len(new),'baseline_changed_output_or_completion_ids':changed,
                                'baseline_simple_score_max_abs_diff':max_score,
                                'baseline_label_changes':[a['id'] for a,b in zip(old,new) if a['outcome']!=b['outcome']],
                                'all_method_guarded_grade_records_checked':policy_count})
        cosines={}
        with np.load(original/model/'bank.npz') as old,np.load(run/'bank.npz') as new:
            for method,layers in bank['c_layers']['qa'].items():
                for layer in layers:
                    key='qa_'+method+'_C_L'+str(layer);x,y=old[key].tolist(),new[key].tolist()
                    cosines[key]=math.fsum(a*b for a,b in zip(x,y))/math.sqrt(math.fsum(a*a for a in x)*math.fsum(b*b for b in y))
        checks.append({'model':model,'data_byte_identical':True,'new_C_layers':bank['c_layers']['qa'],
                       'baseline_and_policy_checks':comparisons,'old_new_same_layer_C_cosines':cosines})
        sources.append({'model':model,'sha256':{f:digest(run/f) for f in ('bank.json','bank.npz','data.json','selection.json','qa_tune.json','qa_tune_scored.json','qa_test.json','qa_test_scored.json','report.json','delivered_answers.json')}})
        for q,table in rep['tables']['qa'].items():
            entry=table['quota']['coverages'][0];arms=entry['arms']
            for method in ('das','md'):
                for selector,suffix,bname in [('commitment','','baseline_commitment'),('simple','_simple','baseline_simple')]:
                    mapping={'base':bname,'E':method+'_E'+suffix,'C':method+'_C'+suffix,'EC':method+'_EC'+suffix}
                    row={'model':model,'method':method,'selector':selector,'coverage':float(q),'n':entry['n'],'k':entry['k'],
                         'conditions':{k:'baseline' if k=='base' else sel['tasks']['qa']['choices'][q].get(v) for k,v in mapping.items()},
                         'arms':{k:{'feasible':True,'correct':arms[v]['correct'],'answered':arms[v]['answered'],'accuracy_answered':arms[v]['accuracy_answered']}
                                  if arms.get(v,{}).get('feasible') else {'feasible':False} for k,v in mapping.items()}}
                    cells.append(row)
                    for control in ('base','E','C'):
                        family=method+'/'+selector+': EC - '+control;a,b=mapping['EC'],mapping[control]
                        r={k:row[k] for k in ('model','method','selector','coverage','n','k')};r.update(a=a,b=b)
                        if not(row['arms']['EC']['feasible'] and row['arms'][control]['feasible']):
                            r.update(delta_correct=None,p=1.,note='infeasible; retain in predeclared six-QA-cell correction family')
                        else:
                            hits=[x for x in entry['comparisons'] if {x['arm_a'],x['arm_b']}=={a,b}];require(len(hits)==1,'missing/duplicate comparison')
                            c=hits[0];sign=1 if c['arm_a']==a else -1;delta=row['arms']['EC']['correct']-row['arms'][control]['correct']
                            require(delta==sign*c['exact_fixed_deliveries']['delta_correct_count'],'comparison sign differs')
                            ci=(c.get('ci95') or {}).get('accuracy_answered')
                            r.update(delta_correct=delta,p=c['exact_fixed_deliveries']['p_value'],delta_accuracy=delta/entry['k'],
                                     ci95_accuracy=None if ci is None else sorted(sign*x for x in ci),
                                     bootstrap_status=c['bootstrap_status'],infeasible_resamples=c['infeasible_resamples'])
                        families.setdefault(family,[]).append(r)
    for family,rs in families.items():
        require(len(rs)==6,'QA family incomplete')
        for r,h in zip(rs,holm([r['p'] for r in rs])):r['p_holm']=h
    result={'protocol':'E26_qa_ability_repair_summary_v1','exploratory':True,'evaluated_tasks':['qa'],
       'scope':'Fixed explicit substantive-ability gate, newly estimated C and tuning-frozen selections, full test. Uniform full-personal-refusal guard for every arm. Six QA cells per correction family; not the historical twelve-cell preregistration.',
       'audits':audits,'checks':checks,'sources':sources,'cells':cells,'families':families}
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# 修正 QA 门控语义后的完整四臂表','',
      '只评估真实问答；新旧模型回答都按同一完整个人拒答规则处理。原题正确答案是 No、纠正错误前提或准确解释客观未知，均可被视为能提供实质回答。新模板先冻结，未按完整结果改词。', '',
      '该修复是探索性复验。最小控制中 Llama 仍有 1/6 的能力判断偏差，不能称新测量完美。实际配额资格来自离线语义判分，不宣称无需裁判即可在线精确达到作答率。','']
    for method,label in [('das','DAS'),('md','配对均值差')]:
        for selector,sl in [('commitment','承诺分排序'),('simple','简单置信筛题')]:
            family=method+'/'+selector+': EC - E';index={(r['model'],r['coverage']):r for r in families[family]}
            lines+=['## '+label+' / '+sl,'','| 模型 | 目标 | 实际配额/总数 | 原模型 | E | C | E+C | 组合减 E | 95%区间（百分点） | Holm p |',
                    '|---|---:|---:|---:|---:|---:|---:|---:|---|---:|']
            for c in cells:
                if c['method']!=method or c['selector']!=selector:continue
                r=index[(c['model'],c['coverage'])];ci=r.get('ci95_accuracy')
                values=[str(c['arms'][a]['correct']) if c['arms'][a]['feasible'] else '不可行' for a in ('base','E','C','EC')]
                lines.append('| %s | %.0f%% | %d/%d | %s | %s | %s | %s | %s | %s | %.4g |'%
                  ('Llama' if c['model'].startswith('llama') else 'Qwen',100*c['coverage'],c['k'],c['n'],*values,
                   '—' if r['delta_correct'] is None else '%+d'%r['delta_correct'],
                   '—' if ci is None else '[%+.2f, %+.2f]'%tuple(100*x for x in ci),r['p_holm']))
            significant=sum(r['delta_correct'] is not None and r['delta_correct']>0 and r['p_holm']<.05 for r in families[family])
            lines+=['','组合对 E 经校正显著为正：%d/6。'%significant,'']
    lines+=['完整方法对原模型、对 C 的比较、固定门槛和逐题交付均保留在原始 JSON。无法填满配额明确报不可行；存在不可行自举时不丢弃抽样补出区间。','',
            '每类六个 QA 格子独立做 Holm（多重比较校正 · 同时检验多个格子时收紧显著门槛）。不与旧 TF/QA 原表混成新的预注册结论。','']
    output.with_suffix('.md').write_text('\n'.join(lines))
    for family,rs in families.items():
        if family.endswith('EC - E'):print(family,[(r['model'],r['coverage'],r['delta_correct'],r['p_holm']) for r in rs])
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--results-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();summarize(a.results_root,a.out)
if __name__=='__main__':main()
