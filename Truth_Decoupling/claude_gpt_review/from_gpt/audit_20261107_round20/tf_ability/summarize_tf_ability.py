#!/usr/bin/env python3
"""Exploratory six-cell TF repair analysis; never merge this into old 12-cell plan.
Both estimators and both selectors are retained. New frozen E/C directions and
choices are evaluated; unchanged QA results are explicitly outside this report.
"""
import argparse,json,math,hashlib
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'from_claude/code/scripts/review_checks'))
from e26_independent_recount import audit_run,read,digest,require


def holm(ps):
    result=[None]*len(ps);running=0.
    for rank,i in enumerate(sorted(range(len(ps)),key=lambda i:ps[i])):
        running=max(running,min(1.,(len(ps)-rank)*ps[i]));result[i]=running
    return result


def run(results_root,output):
    families={};cells=[];audits=[];invariants=[];sources=[]
    original=ROOT/'from_claude/results/03_E26同作答率_E27生成适配/E26_main'
    prompt_root=ROOT/'from_gpt/audit_20261107_round20/gate_prompt/results'
    for model in ('llama3.1-8b','qwen3-8b'):
        folder=results_root/model/'tf_ability_full';bank=read(folder/'bank.json');rep=read(folder/'report.json');sel=read(folder/'selection.json')
        require(bank['config']['evaluation_tasks']==['tf'] and rep['evaluation_tasks']==['tf'] and set(rep['tables'])=={'tf'},'not a TF-only repair')
        require(not list(folder.glob('qa_*')),'unexpected QA outputs in TF-only run')
        frozen=read(prompt_root/model/'full192/plan.json')['conditions'][2]
        require(bank['config']['tf_gate_system']==frozen['system'] and bank['config']['tf_gate_user']==frozen['user'],
                'TF prompt differs from predeclared E33 explicit condition')
        audits.append(audit_run(folder,bootstrap=True))
        sources.append({'model':model,'files':{n:digest(folder/n) for n in ('bank.json','bank.npz','data.json','selection.json','tf_tune.json','tf_test.json','report.json','delivered_answers.json')}})
        require((original/model/'data.json').read_bytes()==(folder/'data.json').read_bytes(),'full dataset changed')
        baseline_checks=[]
        for split in ('tune','test'):
            a=read(original/model/('tf_'+split+'.json'))['rows']['baseline'];b=read(folder/('tf_'+split+'.json'))['rows']['baseline']
            require([r['id'] for r in a]==[r['id'] for r in b],'baseline cohort changed')
            changed=sum(any(x[k]!=y[k] for k in ('output','predicted_true','outcome')) for x,y in zip(a,b))
            require(changed==0,'unsteered TF answer predictions changed')
            delta=max(abs(x['simple_score']-y['simple_score']) for x,y in zip(a,b))
            require(delta<=1e-5,'unsteered TF confidence changed beyond numerical tolerance')
            baseline_checks.append({'split':split,'n':len(a),'prediction_changes':changed,'simple_score_max_abs_diff':delta})
        cosines={}
        with np.load(original/model/'bank.npz',allow_pickle=False) as old,np.load(folder/'bank.npz',allow_pickle=False) as new:
            for method,layers in bank['c_layers']['tf'].items():
                for layer in layers:
                    key='tf_'+method+'_C_L'+str(layer)
                    if key in old:
                        x,y=old[key].tolist(),new[key].tolist()
                        cosines[key]=math.fsum(a*b for a,b in zip(x,y))/math.sqrt(math.fsum(a*a for a in x)*math.fsum(b*b for b in y))
        invariants.append({'model':model,'baseline_checks':baseline_checks,'full_data_json_byte_identical_to_original':True,'data_sha256':digest(folder/'data.json'),'new_c_layers':bank['c_layers']['tf'],
                           'old_vs_new_C_cosine_at_new_layer':cosines})
        for q,table in rep['tables']['tf'].items():
            entry=table['quota']['coverages'][0];arms=entry['arms'];comp=entry['comparisons']
            for method in ('das','md'):
                for selector,suffix,base in [('commitment','', 'baseline_commitment'),('simple','_simple','baseline_simple')]:
                    mapping={'base':base,'E':method+'_E'+suffix,'C':method+'_C'+suffix,'EC':method+'_EC'+suffix}
                    row={'model':model,'coverage':float(q),'n':entry['n'],'k':entry['k'],'method':method,'selector':selector,
                         'conditions':{k:('baseline' if k=='base' else sel['tasks']['tf']['choices'][q].get(v)) for k,v in mapping.items()},
                         'arms':{k:({'feasible':True,'correct':arms[v]['correct'],'answered':arms[v]['answered'],
                                    'accuracy_answered':arms[v]['accuracy_answered']} if arms.get(v,{}).get('feasible') else {'feasible':False}) for k,v in mapping.items()}}
                    cells.append(row)
                    for control in ('base','E','C'):
                        family=method+'/'+selector+': EC - '+control;a,b=mapping['EC'],mapping[control]
                        rec={key:row[key] for key in ('model','coverage','n','k','method','selector')};rec.update(a=a,b=b)
                        if not(row['arms']['EC']['feasible'] and row['arms'][control]['feasible']):
                            rec.update(delta_correct=None,p=1.,note='infeasible; retain in six-TF-cell family')
                        else:
                            hit=[r for r in comp if {r['arm_a'],r['arm_b']}=={a,b}];require(len(hit)==1,'comparison missing/duplicated')
                            c=hit[0];sign=1 if c['arm_a']==a else -1;ex=c['exact_fixed_deliveries']
                            delta=row['arms']['EC']['correct']-row['arms'][control]['correct']
                            require(delta==sign*ex['delta_correct_count'],'comparison direction mismatch')
                            ci=(c.get('ci95') or {}).get('accuracy_answered')
                            rec.update(delta_correct=delta,p=ex['p_value'],delta_accuracy=delta/entry['k'],
                                ci95_accuracy=None if ci is None else sorted([sign*x for x in ci]),
                                bootstrap_status=c['bootstrap_status'],infeasible_resamples=c['infeasible_resamples'])
                        families.setdefault(family,[]).append(rec)
    for family,recs in families.items():
        require(len(recs)==6,'TF comparison family incomplete')
        for r,p in zip(recs,holm([r['p'] for r in recs])):r['p_holm']=p
    result={'protocol':'E26_tf_ability_repair_summary_v1','exploratory':True,'evaluated_tasks':['tf'],
      'scope':'New ability TF gate, re-estimated C, tuning-refrozen choices, full test comparison. QA was not regenerated or regraded. Six-cell TF corrections are not the old twelve-cell primary plan.',
      'sources':sources,'audits':audits,'invariants':invariants,'cells':cells,'families':families}
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# 修正 C 门控含义后的真假判断完整表','',
      '两模型均完成新方向估计、调参、冻结与完整测试。只评估真假判断；自由问答没有在此运行重生成或判分，必须另引旧 QA 原件。', '',
      '明确版使用 E33 测试前冻结的可答性系统与“能判真/判假都回答 Yes”用户提示，没有按完整结果继续改词。原模型的真假预测保持不变。结果为探索性修复验证，不能重新包装成旧主分析的确认结果。','']
    for method,title in [('das','DAS'),('md','配对均值差')]:
        for selector,subtitle in [('commitment','按自身承诺分排序'),('simple','统一使用简单置信筛题')]:
            family=method+'/'+selector+': EC - E';cmp={(r['model'],r['coverage']):r for r in families[family]}
            lines += ['## '+title+'：'+subtitle,'','| 模型 | 目标作答率 | 实际交付/总数 | 原模型 | E | C | E+C | 组合减 E | 95%区间（百分点） | Holm p |',
                      '|---|---:|---:|---:|---:|---:|---:|---:|---|---:|']
            for c in cells:
                if c['method']!=method or c['selector']!=selector:continue
                r=cmp[(c['model'],c['coverage'])];ci=r.get('ci95_accuracy')
                nums=[str(c['arms'][k]['correct']) if c['arms'][k]['feasible'] else '不可行' for k in ('base','E','C','EC')]
                lines.append('| %s | %.0f%% | %d/%d | %s | %s | %s | %s | %s | %s | %.4g |' %
                  ('Llama' if c['model'].startswith('llama') else 'Qwen',100*c['coverage'],c['k'],c['n'],*nums,
                   '—' if r['delta_correct'] is None else '%+d'%r['delta_correct'],
                   '—' if ci is None else '[%+.2f, %+.2f]'%tuple(100*x for x in ci),r['p_holm']))
            sig=sum(r['delta_correct'] is not None and r['delta_correct']>0 and r['p_holm']<.05 for r in families[family])
            lines += ['', '本比较六个真假格子中，经校正显著为正：%d/6。'%sig,'']
    lines += ['各 E/C/E+C 家族都在调参集独立选配置。组合对 C、对原模型及全部逐题交付/冻结门槛结果见 JSON 与原始运行；不能把方法家族差值全归因于 C 的单独作用。','',
              '每类只在本次六个真假格子内做 Holm（多重比较校正 · 同时做多个检验时收紧显著门槛）；不与旧 QA 拼成一份新的预注册十二格结论。','']
    output.with_suffix('.md').write_text('\n'.join(lines))
    for family,rs in families.items():
        if family.endswith('EC - E'):print(family,[(r['model'],r['coverage'],r['delta_correct'],r['p_holm']) for r in rs])
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--results-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.results_root,a.out)
if __name__=='__main__':main()
