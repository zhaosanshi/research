"""Recount E37 selections directly from raw/grades/scores, no metric imports."""
import argparse
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tie(pid):return hashlib.sha256(json.dumps(['actual_coverage_v1',42,pid],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def recount(model):
    run=Path('/tmp/td_r21_source_inputs')/model/'full192'
    sidepath=HERE.parent/'source_order/sensitivity'/model/'report.json'
    raw,grade,scores,side=[read(p) for p in (run/'tune.json',run/'tune_scored_items.json',run/'tune_simple_scores.json',sidepath)]
    assert raw['completed'] and grade['grading_completed'] and raw['model']==grade['model']==side['model']==model
    assert grade['grading_source_generation_sha256']==scores['identity']['source_generation_sha256']==sha(run/'tune.json')
    assert side['source_files']['tune.json']==sha(run/'tune.json') and side['source_files']['tune_scored_items.json']==sha(run/'tune_scored_items.json')
    assert side['common_score_sha256']==sha(run/'tune_simple_scores.json')
    assert grade['outcome_policy']=='judge_primary_v2_full_personal_refusal_guard_v1'
    ids=[r['pair_id'] for r in raw['evaluations']]
    assert len(ids)==192 and len(set(ids))==192 and ids==[r['pair_id'] for r in grade['evaluations']]==[r['pair_id'] for r in scores['rows']]
    names=raw['conditions'];assert len(names)==10 and names==grade['conditions']
    points=[]
    for quality in ('R1','R2'):
        for scoring in ('common','native'):
            table=(side['common_unmodified_confidence_corrected' if scoring=='common' else 'native_policy_confidence_corrected'] if quality=='R1'
                   else side['quality_sensitivity']['common_unmodified_model' if scoring=='common' else 'native_policy'])
            for i,q in enumerate((.25,.5,.75)):
                k=int(q*192);counts={}
                for name in names:
                    rows=[]
                    for r,g,s in zip(raw['evaluations'],grade['evaluations'],scores['rows']):
                        assert r['outputs']==g['outputs'] and r['finished']==g['finished'] and not g['gold_is_abstention']
                        outcome=g['outcome'][name];assert outcome==g['judge'][name]['label'] and outcome in ('correct','wrong','abstain')
                        if quality=='R2' and outcome=='correct' and (not r['finished'][name] or '\ufffd' in r['outputs'][name]):outcome='wrong'
                        if scoring=='common':score=s['scores'][name]
                        else:
                            tokens=r['generation_token_ids'][name];tr=r['trace'][name];assert len(tokens)==len(tr)
                            for j,t in enumerate(tr):assert t['next_top1_id']==tokens[j] and t['accepted_prefix_ids']==tokens[:j] and 0<t['next_top1_probability']<=1
                            use=tr[:-1] if r['finished'][name] else tr
                            score=math.fsum(math.log(t['next_top1_probability']) for t in use)/len(use) if use and r['outputs'][name].strip() else -1e6
                        assert math.isfinite(score)
                        rows.append({'id':r['pair_id'],'score':score,'outcome':outcome})
                    eligible=[r for r in rows if r['outcome']!='abstain'];feasible=len(eligible)>=k
                    selected=sorted(eligible,key=lambda r:(-r['score'],tie(r['id']),r['id']))[:k] if feasible else None
                    count=sum(r['outcome']=='correct' for r in selected) if feasible else None
                    official=table['coverages'][i]['arms'][name]
                    assert official['feasible']==feasible and official['correct']==count
                    assert official['selected_ids']==(sorted(r['id'] for r in selected) if feasible else None)
                    counts[name]=count
                points.append({'view':quality+'_'+scoring,'coverage':q,'n':192,'k':k,'correct':counts,'all_selected_ids_match':True})
    result={'model':model,'point_count':len(points)*len(names),'all_points_and_ids_match':True,'points':points,
            'source_sha256':{p.name:sha(p) for p in (run/'tune.json',run/'tune_scored_items.json',run/'tune_simple_scores.json')},'sidecar_sha256':sha(sidepath)}
    (HERE/(model+'_full_points.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(model,result['point_count'],'arm/coverage/view points PASS')
    for p in points:
        if p['view']=='R2_common':print(p['coverage'],{k:v for k,v in p['correct'].items() if k=='E_source' or k.startswith('EC')})
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args();recount(a.model)
