"""Qwen uniform-256 audit only; no Llama full-256 artifact is invented."""
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE.parent.parent)]
from summarize import summaries
from audit_and_sidecars import read,save,digest,eval_arms,correct_arms,natural
from native_confidence import native_score,validate_generation
from core.selective_eval import evaluate_coverage
from core.qa_scoring import full_personal_refusal_decision
from e34_sidecars import quality_view
from predicate_blind_judge import parse,digest as object_digest

def main():
    run=Path('/tmp/td_r20_completion_inputs/qwen3-8b/uniform256')
    raw,grade,rp,gp=validate_generation(run,'tune');mapping=read(HERE/'mapping256.json')
    if digest(rp)!=mapping['generation_sha256']:raise ValueError('256 source generation changed')
    labels={r['id']:r for r in read(HERE/'blind_labels.json')}
    for row in labels.values():
        if row['completed'] and parse(row['raw'],row['payload']['prediction'])!=row['parsed']:raise ValueError('invalid evidence')
    arms,questions=eval_arms(grade);guarded,audit=correct_arms(arms,questions);strict,changes=quality_view(guarded,raw)
    by_id={r['pair_id']:r for r in raw['evaluations']}
    result={'scope':'Qwen seven-arm uniform 256-token diagnostic; Llama full-256 was not run because its pilot failed',
        'model':raw['model'],'source_files':{rp.name:digest(rp),gp.name:digest(gp),'blind_labels.json':digest(HERE/'blind_labels.json')},
        'guard_audit':audit,'legacy_guard':{},'legacy_guard_complete':{},'blind_evidence':{},'blind_evidence_complete':{},
        'blind_unique_completed':sum(r['completed'] for r in labels.values()),'blind_unique_total':len(labels),
        'unresolved_occurrences':[],'grading_warning':'Same provider remains fallible; JSON evidence checks do not establish semantic ground truth'}
    for name,rs in arms.items():
        for i,r in enumerate(rs):
            score=native_score(by_id[r['pair_id']],name)['score']
            guarded[name][i]['score']=strict[name][i]['score']=score
            g=grade['evaluations'][i]
            key=object_digest({'question':g['question'],'references':g['judge_reference_answers'],'prediction':g['outputs'][name]})
            b=labels[key];r['outcome']={'A':'correct','B':'wrong','C':'abstain'}[b['parsed']['label']] if b['completed'] else 'unresolved'
            if not b['completed']:result['unresolved_occurrences'].append({'pair_id':r['pair_id'],'condition':name,'finished':g['finished'][name]})
            if full_personal_refusal_decision(g['question'],r['output'])['apply']:r['outcome']='abstain'
            r['native']=score
        result['blind_evidence'][name]=summaries(rs,by_id,name,False,('native',))
        result['blind_evidence_complete'][name]=summaries(rs,by_id,name,True,('native',))
    result['legacy_guard']={'natural':{n:natural(rs) for n,rs in guarded.items()},'native':evaluate_coverage(guarded,(.25,.5,.75),n_boot=2000,seed=42)}
    result['legacy_guard_complete']={'natural':{n:natural(rs) for n,rs in strict.items()},'changes':changes,
        'native':evaluate_coverage(strict,(.25,.5,.75),n_boot=2000,seed=42)}
    save(HERE/'summary.json',result)
    lines=['# Qwen 全七组统一 256 预算：守卫、质量和证据审计','',result['scope'],
        '原48预算结果保留；新完整尾巴逐条重判。这里先报实际生成置信，统一原模型分须另有真实重算产物，不能沿用截断答案分。',
        '证据审计612/614合规，两条完整列表仍引用改写而未决。它们明确是实质尝试，仅枚举A/B的4种组合，不把它们当拒答改交付资格。下列范围不是置信区间；裁判本身仍可能错。','',
        '| 视图 | 方法 | 自然答对/实答 | 各自生成置信25/50/75%答对 |','|---|---|---|---|']
    def span(x):return str(x[0]) if x[0]==x[1] else '%s–%s'%tuple(x)
    for view in ('legacy_guard','legacy_guard_complete'):
        for n,v in result[view]['natural'].items():
            p=[x['arms'][n] for x in result[view]['native']['coverages']]
            lines.append('| %s | %s | %s/%s | %s |'%(view,n,v['correct'],v['answered'],'/'.join(str(x['correct']) if x['feasible'] else '不可行' for x in p)))
    for view in ('blind_evidence','blind_evidence_complete'):
        for n,v in result[view].items():
            point=lambda x:'不可行' if not x['any_scenario_feasible'] else span(x['correct_range'])
            lines.append('| %s | %s | %s/%s | %s |'%(view,n,span(v['natural_correct_range']),span(v['natural_answered_range']),'/'.join(map(point,v['native']))))
    (HERE/'summary.md').write_text('\n'.join(lines)+'\n')
    print(result['blind_unique_completed'],result['blind_unique_total'],result['unresolved_occurrences'])

if __name__=='__main__':main()
