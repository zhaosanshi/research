"""Summarize a fallible, partly unresolved blind audit without inventing labels."""
from collections import Counter
from copy import deepcopy
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from audit_and_sidecars import read,save,digest,eval_arms,correct_arms
from native_confidence import native_score,validate_generation
from core.selective_eval import quota_select
from predicate_blind_judge import parse,digest as object_digest

def summaries(rows,raw_rows,name,quality,score_fields=('native','common')):
    unknown=[i for i,r in enumerate(rows) if r['outcome']=='unresolved'];scenarios=[]
    attempted={('source_only','tqa:324'),('equal_logit_mix','tqa:324'),
               ('pair_preference_source_swap','tqa:401'),('pair_preference_source_swap','tqa:449')}
    if any((name,rows[i]['pair_id']) not in attempted for i in unknown):
        raise ValueError('unresolved answer eligibility has not been independently adjudicated')
    for labels in product(('correct','wrong'),repeat=len(unknown)):
        rr=deepcopy(rows)
        for i,label in zip(unknown,labels):rr[i]['outcome']=label
        if quality:
            for r in rr:
                raw=raw_rows[r['pair_id']]
                if r['outcome']=='correct' and (not raw['finished'][name] or '\ufffd' in raw['outputs'][name]):r['outcome']='wrong'
        n=Counter(r['outcome'] for r in rr)
        point={'natural_correct':n['correct'],'natural_answered':n['correct']+n['wrong']}
        for key in score_fields:
            point[key]=[quota_select([{**r,'score':r[key]} for r in rr],q,42) for q in (.25,.5,.75)]
        scenarios.append(point)
    out={'unresolved_records':len(unknown),'possible_labels_per_unresolved':['correct','wrong'],
         'eligibility_adjudication':'These four unresolved records are explicit country/city lists, hence attempted under the fixed rubric. Only A/B remains unknown; four records yield 16 joint assignments.',
         'natural_correct_range':[min(s['natural_correct'] for s in scenarios),max(s['natural_correct'] for s in scenarios)],
         'natural_answered_range':[min(s['natural_answered'] for s in scenarios),max(s['natural_answered'] for s in scenarios)]}
    for key in score_fields:
        out[key]=[]
        for i,q in enumerate((.25,.5,.75)):
            ps=[s[key][i] for s in scenarios];feasible=[p for p in ps if p['feasible']]
            out[key].append({'coverage':q,'all_scenarios_feasible':len(feasible)==len(ps),'any_scenario_feasible':bool(feasible),
                'correct_range':[min(p['correct'] for p in feasible),max(p['correct'] for p in feasible)] if feasible else None,
                'k':ps[0]['k']})
    return out

def main():
    results=read(HERE/'blind_labels.json');tasks={r['id']:r for r in read(HERE/'tasks.json')}
    by_id={r['id']:r for r in results};mapping=read(HERE/'mapping.json')
    if len(by_id)!=len(results) or set(by_id)!=set(tasks):raise ValueError('blind audit cohort differs')
    for key,row in by_id.items():
        if key!=object_digest({k:tasks[key][k] for k in ('question','references','prediction')}):raise ValueError('task hash differs')
        if row['completed'] and parse(row['raw'],tasks[key]['prediction'])!=row['parsed']:raise ValueError('evidence differs')
    report={'completed_unique':sum(r['completed'] for r in results),'total_unique':len(results),
        'unresolved_unique':[tasks[k] for k,r in by_id.items() if not r['completed']],
        'scope':'same-provider condition-blind evidence audit; calibration and JSON checks do not establish factual correctness',
        'models':{},'input_sha256':{n:digest(HERE/n) for n in ('blind_labels.json','tasks.json','mapping.json')}}
    lines=['# 带原文断言的盲审：校准通过，但真实数据未全部通过','',
        '这是同一个 DeepSeek 模型的独立提示与缓存审计，不是独立模型复现。12 个预设校准例全部通过；真实七组两模型的 2688 条记录去重为 1028 个问答，1024 个得到可验证原文引用的 JSON，4 个未通过，不伪造标签。',
        '四条未决都是具体国家/城市名单，按既有规则明确属于实质尝试，仅A/B未定；全局枚举16种A/B组合，绝不把它们改成拒答而改变交付资格。下面范围不是置信区间；其余标签仍可能有语义错误，不将此表当最终真值或确认性结论。各组采用相同拒答守卫。','']
    labelmap={'A':'correct','B':'wrong','C':'abstain'}
    for model in ('llama3.1-8b','qwen3-8b'):
        run=Path('/tmp/td_r20_e34_inputs')/model;raw,grade,_,gp=validate_generation(run,'tune')
        if digest(gp)!=mapping['source_grade_sha256'][model]:raise ValueError('graded source changed')
        scores=read(run/'tune_simple_scores.json');rawrows={r['pair_id']:r for r in raw['evaluations']}
        common={r['pair_id']:r['scores'] for r in scores['rows']};arms,q=eval_arms(grade);changes=[]
        for i,r in enumerate(grade['evaluations']):
            for name in grade['conditions']:
                key=object_digest({'question':r['question'],'references':r['judge_reference_answers'],'prediction':r['outputs'][name]})
                b=by_id[key];label=labelmap[b['parsed']['label']] if b['completed'] else 'unresolved'
                if label!=r['outcome'][name]:changes.append({'pair_id':r['pair_id'],'condition':name,'original':r['outcome'][name],
                    'audited':label,'question':r['question'],'prediction':r['outputs'][name],'evidence':b.get('parsed'),'task_id':key})
                arms[name][i].update(outcome=label,common=common[r['pair_id']][name],native=native_score(rawrows[r['pair_id']],name)['score'])
        # Derive the narrow reference-free guard without converting unresolved items.
        from core.qa_scoring import full_personal_refusal_decision
        for name,rows in arms.items():
            for r in rows:
                if full_personal_refusal_decision(q[r['pair_id']],r['output'])['apply']:r['outcome']='abstain'
        out={'changes_from_original':changes,'semantic':{},'strict_complete':{}}
        for quality,view in [(False,'semantic'),(True,'strict_complete')]:
            out[view]={n:summaries(rs,rawrows,n,quality) for n,rs in arms.items()}
        report['models'][model]=out
        lines+=['## '+model,'','| 视图 | 方法 | 自然答对范围 | 自然实答范围 | 共同分25/50/75% | 各自置信25/50/75% |','|---|---|---|---|---|---|']
        def span(x):return str(x[0]) if x[0]==x[1] else '%s–%s'%tuple(x)
        def point(x):return '不可行' if not x['any_scenario_feasible'] else span(x['correct_range'])+('（可行性未定）' if not x['all_scenarios_feasible'] else '')
        for view in ('semantic','strict_complete'):
            for name,v in out[view].items():lines.append('| %s | %s | %s | %s | %s | %s |'%(view,name,span(v['natural_correct_range']),span(v['natural_answered_range']),'/'.join(map(point,v['common'])),'/'.join(map(point,v['native']))))
    save(HERE/'summary.json',report);(HERE/'summary.md').write_text('\n'.join(lines)+'\n')
    print('valid',report['completed_unique'],'unresolved',len(report['unresolved_unique']))

if __name__=='__main__':main()
