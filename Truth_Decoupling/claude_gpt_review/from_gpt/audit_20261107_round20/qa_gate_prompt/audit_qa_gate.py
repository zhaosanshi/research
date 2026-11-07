#!/usr/bin/env python3
"""Independent fixed-cohort count/rank verification of E35 readouts."""
from pathlib import Path
import json,sys,math
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'from_claude/code/scripts/review_checks'),str(ROOT/'from_claude/code'),
              str(ROOT/'from_gpt/audit_20261107_round20/gate_prompt')]
from e26_independent_recount import read,digest,fingerprint,quota,require,tie
from audit_prompt_readouts import auc_rank
from core.qa_scoring import apply_full_personal_refusal_policy
from fractions import Fraction


def count(rows,q,seed,reported):
    d=quota(rows,q,'score',seed)
    require(reported['feasible']==(d is not None),'quota feasibility differs')
    if d is None:return
    correct=sum(x=='correct' for x in d.values());ids={i for i,x in d.items() if x!='abstain'}
    require(reported['correct']==correct and reported['answered']==len(ids) and set(reported['selected_ids'])==ids,'quota count/mask differs')


def run(model):
    here=Path(__file__).resolve().parent;folder=here/'results'/model/'full192'
    plan=read(folder/'plan.json');raw=read(folder/'readouts.json');rep=read(folder/'report.json')
    oldroot=ROOT/'from_claude/results/03_E26同作答率_E27生成适配/E26_main'/model
    base=read(oldroot/'qa_tune_scored.json')['rows']['baseline'];data=read(oldroot/'data.json');questions={q['pair_id']:q['question'] for q in data['qa']['tune']}
    require(len(base)==192 and raw['completed'] and not plan['pilot'],'not a full diagnostic')
    require(raw['legacy_source_score_max_abs_diff']==0,'legacy gate not exact')
    require(plan['fingerprint']==fingerprint({k:v for k,v in plan.items() if k!='fingerprint'}),'plan changed')
    require(raw['plan_sha256']==rep['plan_sha256']==digest(folder/'plan.json') and rep['readouts_sha256']==digest(folder/'readouts.json'),'artifact linkage differs')
    for rel,h in plan['source_sha256'].items():require(digest(oldroot/rel)==h,'source changed')
    for rel,h in plan['implementation']['files'].items():require(digest(folder/'source_snapshot'/rel)==h,'snapshot changed')
    maps={};records=0
    for c in plan['conditions']:
        rows=raw['rows'][c['name']];require([r['id'] for r in rows]==[r['id'] for r in base],'readout cohort differs');maps[c['name']]={r['id']:r['score'] for r in rows}
        for row in rows:
            require(c['system'] in row['prompt'] and c['user'].format(question=questions[row['id']]) in row['prompt'],'prompt text differs')
            require(math.isclose(row['p_yes']+row['p_no'],row['yes_no_probability_mass'],abs_tol=1e-12),'mass differs')
            require(row['top1_is_canonical_yes_or_no']==(row['top1_token_id'] in (row['yes_token_id'],row['no_token_id'])),'top1 differs')
            records+=1
    quota_cells=0
    for policy,view in rep['policies'].items():
        fixed=[]
        for row in base:
            if row.get('gold_is_abstention'):continue
            label=row['outcome'] if policy=='legacy' else apply_full_personal_refusal_policy(row['grade'],questions[row['id']],row['output'])['label']
            fixed.append({**row,'outcome':label})
        controls={'simple':[{**r,'score':r['simple_score']} for r in fixed],
                  'random':[{**r,'score':(int(tie(r['id'],plan['seed'])[:14],16)>>3)/float(2**53)} for r in fixed]}
        for name,rs in controls.items():
            for q,reported in view['controls'][name]['quotas'].items():count(rs,q,plan['seed'],reported);quota_cells+=1
        for name,summary in view['conditions'].items():
            joined=[{**r,'score':maps[name][r['id']]} for r in fixed];available=[r for r in joined if r['outcome']!='abstain']
            require(auc_rank([r['score'] for r in joined],[r['outcome']=='correct' for r in joined])==summary['correctness_auc_all'],'all-cohort AUC differs')
            require(auc_rank([r['score'] for r in available],[r['outcome']=='correct' for r in available])==summary['correctness_auc_among_answers'],'answered AUC differs')
            native=[r for r in available if r['score']>0]
            require(summary['native']['answered']==len(native) and summary['native']['correct']==sum(r['outcome']=='correct' for r in native),'native counts differ')
            for q,reported in summary['quotas'].items():count(joined,q,plan['seed'],reported);quota_cells+=1
            for control,rs in controls.items():count(rs,Fraction(len(native),len(joined)),plan['seed'],summary['native_same_K_controls'][control]);quota_cells+=1
    pilot=read(here/'results'/model/'pilot_pairs/report.json');pilotraw=read(here/'results'/model/'pilot_pairs/readouts.json')
    pp=read(here/'results'/model/'pilot_pairs/plan.json');expected=pp['pilot_expected_original_answer']
    require([r['id'] for r in pilotraw['baseline_original_answers']]==[r['id'] for r in pp['question_rows']] and len(expected)==6,'pilot baseline cohort differs')
    require(all(r['literal_answer_polarity']==expected[r['id']] and r['finished'] for r in pilotraw['baseline_original_answers']),'pilot original answer mismatch')
    return {'model':model,'all_match':True,'readout_records_checked':records,'quota_cells_checked':quota_cells,
            'pilot_six_original_answers_manually_checked':True,'pilot_ability_yes':{k:v['yes_count'] for k,v in pilot['probability_audit'].items()},
            'policies':rep['policies'],'correct_yes_no_cases':rep['correct_yes_no_cases'],
            'sha256':{n:digest(folder/n) for n in ('plan.json','readouts.json','report.json')}}

def main():
    out={'all_match':True,'runs':[run(m) for m in ('llama3.1-8b','qwen3-8b')]};p=Path(__file__).with_name('independent_audit.json');p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    for r in out['runs']:print(r['model'],r['readout_records_checked'],r['quota_cells_checked'],r['pilot_ability_yes'])
if __name__=='__main__':main()
