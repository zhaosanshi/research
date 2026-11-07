"""Uniform seven-arm E34 sensitivity; preserves every original generation and grade."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from audit_and_sidecars import (read,digest,save,eval_arms,correct_arms,natural,audit_source)
from native_confidence import native_score,validate_generation
from core.selective_eval import evaluate_coverage
from predicate_blind_judge import parse,payload,digest as object_digest

LABELS={'A':'correct','B':'wrong','C':'abstain'}

def quality_view(arms,raw):
    value=deepcopy(arms);by_id={r['pair_id']:r for r in raw['evaluations']};changes=[]
    for name,rows in value.items():
        for row in rows:
            r=by_id[row['pair_id']]
            reasons=[]
            if not r['finished'][name]:reasons.append('no_natural_stop_within_generation_budget')
            if '\ufffd' in r['outputs'][name]:reasons.append('unicode_replacement_character')
            if row['outcome']=='correct' and reasons:
                changes.append({'pair_id':row['pair_id'],'condition':name,'original_outcome':'correct',
                                'sensitivity_outcome':'wrong','reasons':reasons,'output':row['output']})
                row['outcome']='wrong'
    return value,changes

def score_views(arms,raw,common,n_boot):
    by_id={r['pair_id']:r for r in raw['evaluations']};scores={r['pair_id']:r['scores'] for r in common['rows']}
    result={}
    for key in ('common_unmodified_model','native_policy'):
        prepared={}
        for name,rows in arms.items():
            prepared[name]=[{**r,'score':scores[r['pair_id']][name] if key=='common_unmodified_model'
                            else native_score(by_id[r['pair_id']],name)['score']}
                           for r in rows if not r.get('gold_is_abstention')]
        result[key]=evaluate_coverage(prepared,(.25,.5,.75),n_boot=n_boot,seed=42)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,default=Path('/tmp/td_r20_e34_inputs'))
    p.add_argument('--n-boot',type=int,default=2000);p.add_argument('--blind-labels',type=Path)
    a=p.parse_args();inventory=[];lines=['# E34 七组统一口径侧表','',
        '全部为已经看过的调参题探索；原生成、原判分与选择一律保留。',
        'R1 只修完整个人拒答；R2 在 R1 上将未自然结束或含替换乱码的正确回答计作失败。R2 是严格交付质量敏感性，不声称所有截断回答的已说内容都错误。',
        '每一组都用同一规则；拒答仍留在分母，A/B 都有交付资格。两种置信读出并列，不择优。','']
    blind={}
    if a.blind_labels:
        for r in read(a.blind_labels):
            if not r['completed']:raise ValueError('blind grading is incomplete')
            parsed=parse(r['raw'],r['payload']['prediction'])
            if parsed!=r['parsed'] or r['id']!=object_digest(r['payload']):raise ValueError('blind result identity mismatch')
            if r['id'] in blind:raise ValueError('duplicate blind answer')
            blind[r['id']]=r
    for model in ('llama3.1-8b','qwen3-8b'):
        run=a.input_root/model;raw,grade,raw_path,grade_path=validate_generation(run,'tune')
        common=read(run/'tune_simple_scores.json')
        if (not common['completed'] or common['identity']['source_generation_sha256']!=digest(raw_path)
                or common['identity']['conditions']!=raw['conditions']
                or [r['pair_id'] for r in common['rows']]!=[r['pair_id'] for r in raw['evaluations']]):
            raise ValueError('common score identity differs')
        arms,questions=eval_arms(grade);guarded,audit=correct_arms(arms,questions)
        result=audit_source(run,run,a.n_boot)
        inventory.append({'category':'source_E34','run':model,'model':model,'split':'tune',**audit})
        strict,quality_changes=quality_view(guarded,raw)
        result['quality_sensitivity']={'definition':'R2: full-personal-refusal guard, then any semantic correct answer without natural stop or with U+FFFD counts wrong; C unchanged',
            'natural':{n:natural(rs) for n,rs in strict.items()},'changes':quality_changes,
            **score_views(strict,raw,common,a.n_boot)}
        result['quality_counts']={n:{'unfinished':sum(not r['finished'][n] for r in raw['evaluations']),
            'unicode_replacement':sum('\ufffd' in r['outputs'][n] for r in raw['evaluations']),
            'guarded_correct_unfinished':sum(r['outcome']=='correct' and not raw['evaluations'][i]['finished'][n] for i,r in enumerate(guarded[n]))}
            for n in raw['conditions']}
        if blind:
            derived=deepcopy(arms);used=[];changes=[]
            for i,r in enumerate(grade['evaluations']):
                for name in raw['conditions']:
                    request={'question':r['question'],'references':r['judge_reference_answers'],'prediction':r['outputs'][name]}
                    key=object_digest(request)
                    if key not in blind:raise ValueError('blind answer missing')
                    b=blind[key];label=LABELS[b['parsed']['label']];used.append(key)
                    derived[name][i]['outcome']=label
                    if label!=r['outcome'][name]:changes.append({'pair_id':r['pair_id'],'condition':name,'original':r['outcome'][name],
                        'audited':label,'output':r['outputs'][name],'evidence':b['parsed']})
            derived,blind_guard=correct_arms(derived,questions)
            complete,blind_quality=quality_view(derived,raw)
            result['evidence_audit']={'scope':'same DeepSeek provider/model with evidence-bearing separate rubric; condition/model blind, not independent-model replication',
                'source_sha256':digest(a.blind_labels),'unique_answers_used':len(set(used)),'changes_from_original':changes,
                'guard_applied_after_audit':blind_guard,'natural':{n:natural(rs) for n,rs in derived.items()},
                **score_views(derived,raw,common,a.n_boot),
                'quality_sensitivity':{'natural':{n:natural(rs) for n,rs in complete.items()},'changes':blind_quality,
                    **score_views(complete,raw,common,a.n_boot)}}
        result['sidecar_script_sha256']=digest(Path(__file__))
        result['no_original_files_modified']=True
        save(HERE/'sidecars'/('source_E34_'+model)/'report.json',result)
        cases=[]
        for r in grade['evaluations']:
            n='pair_preference_source_swap'
            if r['outcome'][n]!=r['outcome']['baseline'] or not r['finished'][n]:
                cases.append({k:r[k] for k in ('pair_id','question','judge_reference_answers','outputs','outcome','finished')})
        save(HERE/'sidecars'/('source_E34_'+model)/'raw_changes.json',cases)
        views={'R1_guard':{'natural':result['natural_corrected'],'common_unmodified_model':result['common_unmodified_confidence_corrected'],
                         'native_policy':result['native_policy_confidence_corrected']},'R2_guard_complete':result['quality_sensitivity']}
        if blind:views.update(R3_evidence=result['evidence_audit'],R4_evidence_complete=result['evidence_audit']['quality_sensitivity'])
        lines += ['## '+model,'','| 视图 | 方法 | 自然答对/实答 | 共同分 25/50/75% 答对 | 各自置信 25/50/75% 答对 |','|---|---|---|---|---|']
        for view,value in views.items():
            for name in raw['conditions']:
                cells=[]
                for readout in ('common_unmodified_model','native_policy'):
                    cells.append('/'.join(str(p['arms'][name]['correct']) if p['arms'][name]['feasible'] else '不可行' for p in value[readout]['coverages']))
                n=value['natural'][name]
                lines.append('| %s | %s | %s/%s | %s | %s |'%(view,name,n['correct'],n['answered'],*cells))
        print(model,'guard changes',audit['changed_record_occurrences'],'quality',result['quality_counts']['pair_preference_source_swap'],flush=True)
    # Append the completed fixed-loss experiment to the existing canonical guard inventory.
    summary=read(HERE/'audit_summary.json')
    summary['artifact_audits']=[r for r in summary['artifact_audits'] if r['category']!='source_E34']+inventory
    summary['changed_occurrences_total']=sum(r['changed_record_occurrences'] for r in summary['artifact_audits'])
    summary['distinct_changed_model_question_text']=sorted({(r['model'],c['pair_id'],c['output']) for r in summary['artifact_audits'] for c in r['changes']})
    summary['E34_extension_script_sha256']=digest(Path(__file__))
    summary['scope']+='; completed E34 seven-arm fixed-loss experiment' if 'completed E34' not in summary['scope'] else ''
    save(HERE/'audit_summary.json',summary)
    (HERE/'E34_sidecars.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
