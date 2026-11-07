"""Bind every matching full answer to the same-budget reference's semantic grade.

This delivery has zero label drift. A nonzero drift fails before publication,
requiring a separate statistics recomputation instead of reusing old summaries.
No model generation or API call occurs. Originals are never overwritten.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fp(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def checked(raw,graded,raw_hash,cfg,rubric):
    if (not raw.get('completed') or not graded.get('completed') or not graded.get('grading_completed')
            or graded.get('grading_source_generation_sha256')!=raw_hash
            or graded.get('grading_judge_config_sha256')!=fp(cfg)
            or graded.get('grading_judge_name')!=cfg['name']
            or raw['identity']!=graded['identity'] or raw['conditions']!=graded['conditions']
            or len(raw['evaluations'])!=192 or len(graded['evaluations'])!=192):
        raise ValueError('incomplete or foreign raw/grade/config identity')
    ids=[r['pair_id'] for r in raw['evaluations']]
    if len(set(ids))!=192 or ids!=[r['pair_id'] for r in graded['evaluations']]:raise ValueError('cohort identity differs')
    for r,g in zip(raw['evaluations'],graded['evaluations']):
        for key in ('pair_id','question','gold','incorrect','best_answer','outputs','finished'):
            if r.get(key)!=g.get(key):raise ValueError('reference-lock raw and graded content differ')
        if set(g['outcome'])!=set(raw['conditions']) or set(g['judge'])!=set(raw['conditions']):raise ValueError('missing grades')
        for name in raw['conditions']:
            if g['judge'][name].get('rubric_id')!=rubric or g['outcome'][name]!=g['judge'][name]['label']:
                raise ValueError('semantic rubric/label inconsistency')


def perform(args):
    sys.path.insert(0,str(args.code_dir))
    from core.qa_scoring import apply_configured_outcome_policy
    from core.judge import qa_cache_key,QA_GRADER_VERSION
    cfg=read(args.code_dir/'configs/experiments/E4_qa.json')['judge']
    old_cfg=read(args.code_dir/'configs/experiments/E4_qa_legacy_r20.json')['judge']
    if not cfg.get('full_personal_refusal_guard'):raise ValueError('uniform corrected policy is required')
    for key in ('name','model','base_url','max_tokens','extra_body'):
        if cfg.get(key)!=old_cfg.get(key):raise ValueError('semantic judge request/model differs between reference and current')
    run,reference,dest=args.run_dir,args.reference_run,args.output_dir
    if dest.exists():raise ValueError('use a new derived directory; never overwrite a prior lock')
    raw,grade=read(run/'tune.json'),read(run/'tune_scored_items.json')
    parent_raw,parent=read(reference/'tune.json'),read(reference/'tune_scored_items.json')
    checked(raw,grade,sha(run/'tune.json'),cfg,QA_GRADER_VERSION)
    checked(parent_raw,parent,sha(reference/'tune.json'),old_cfg,QA_GRADER_VERSION)
    if raw['model']!=parent_raw['model'] or [r['pair_id'] for r in raw['evaluations']]!=[r['pair_id'] for r in parent_raw['evaluations']]:
        raise ValueError('reference does not cover the same model/cohort')
    for a,b in zip(grade['evaluations'],parent['evaluations']):
        if any(a.get(k)!=b.get(k) for k in ('question','gold','incorrect','best_answer','judge_reference_answers')):
            raise ValueError('reference question or semantic reference fields differ')
    if any(raw['identity'].get(k)!=parent_raw['identity'].get(k) for k in ('source','layer','config')):
        raise ValueError('same-budget E/base reference source differs')
    parent_budget=parent_raw['identity'].get('budget_override',{}).get('max_new_tokens',parent_raw['identity']['config']['max_new_tokens'])
    if raw['identity']['max_new_tokens']!=parent_budget:raise ValueError('reference budget differs')
    index={}
    def key(row,text):
        return qa_cache_key(cfg['model'],row['question'],row['judge_reference_answers'],text,row.get('incorrect') or None)
    for row in parent['evaluations']:
        for name,text in row['outputs'].items():
            k=key(row,text);old=row['judge'][name]
            entry={'raw_grade':old,'grade':apply_configured_outcome_policy(old,row['question'],text,cfg),
                   'sources':[{'pair_id':row['pair_id'],'condition':name}]}
            if k in index:
                if index[k]['raw_grade']['label']!=old['label'] or index[k]['grade']['label']!=entry['grade']['label']:
                    raise ValueError('same reference key has conflicting old labels; no favorable choice allowed')
                index[k]['sources']+=entry['sources']
            else:index[k]=entry
    before_summary=read(run/'tune_scored.json');records=[];changes=[]
    for row in grade['evaluations']:
        for name,text in row['outputs'].items():
            k=key(row,text);old=copy.deepcopy(row['judge'][name])
            if k in index:
                replacement=copy.deepcopy(index[k]['grade'])
                source_info={'decision':'same_question_references_and_complete_output_reference_lock',
                             'reference_key':k,'reference_conditions':index[k]['sources']}
                record={'pair_id':row['pair_id'],'condition':name,'key':k,'old_label':old['label'],
                        'reference_original_label':index[k]['raw_grade']['label'],'new_label':replacement['label'],
                        'old_grade':old,'reference_grade':index[k]['raw_grade'],**source_info}
                records.append(record)
                if old['label']!=replacement['label']:changes.append(record)
            else:
                expected=apply_configured_outcome_policy(old,row['question'],text,cfg)
                if expected['label']!=old['label']:raise ValueError('new-text current policy is inconsistent')
    if changes:raise ValueError('nonzero reference-label drift requires statistics regeneration; original artifacts preserved')
    filename='judge_cache_%s.jsonl'%re.sub(r'[^A-Za-z0-9_.-]','_',cfg['model'])
    def cache(path):
        out={}
        for line in path.read_text().splitlines():
            if line.strip():
                item=json.loads(line)
                if item.get('model')!=cfg['model']:raise ValueError('foreign judge model in private cache')
                out[item['key']]=item
        return out
    current_cache_path=run/'judge_cache'/filename;parent_cache_path=reference/'judge_cache'/filename
    current_cache,parent_cache=cache(current_cache_path),cache(parent_cache_path)
    # With zero final-label drift, preserve completed current cache responses
    # and all grade metadata verbatim. The sidecar locks the demonstrated
    # equivalence; it does not pretend the old API was actually consulted.
    merged=dict(current_cache);cache_overrides=[]
    for k,item in index.items():
        if k not in parent_cache or parent_cache[k]['answer']!=item['raw_grade']['raw']:
            raise ValueError('reference cache does not reproduce its frozen semantic response')
        if k in merged and merged[k]['answer']!=parent_cache[k]['answer']:cache_overrides.append(k)
    for row in grade['evaluations']:
        for name,text in row['outputs'].items():
            k=key(row,text);r=merged.get(k)
            if r is None:
                if not row['judge'][name].get('semantic_call_skipped_by_guard'):raise ValueError('missing non-guard semantic cache')
                continue
            if r['answer']!=row['judge'][name]['raw']:raise ValueError('merged cache differs from adopted grade')
    identity={'protocol':'same_text_reference_grade_lock_v1','model':raw['model'],
        'matching_fields':['question','semantic_reference_answers','incorrect_references','complete_generated_output'],
        'current_generation_sha256':sha(run/'tune.json'),'current_grade_sha256':sha(run/'tune_scored_items.json'),
        'reference_generation_sha256':sha(reference/'tune.json'),'reference_grade_sha256':sha(reference/'tune_scored_items.json'),
        'current_judge_config_sha256':fp(cfg),'reference_judge_config_sha256':fp(old_cfg),
        'judge_model':cfg['model'],'semantic_rubric':QA_GRADER_VERSION,'reference_conflicts':0,
        'matched_question_conditions':len(records),'unmatched_question_conditions':1920-len(records),
        'changed_final_labels':len(changes),'all_existing_v1_results_preserved':True,'no_new_model_or_API_call':True,
        'current_cache_sha256':sha(current_cache_path),'reference_cache_sha256':sha(parent_cache_path),
        'different_raw_cache_responses_with_same_verified_final_label':cache_overrides,
        'cache_policy':'completed current cache copied unchanged; all matching parent labels independently verified after the current guard',
        'derivation_script_sha256':sha(Path(__file__)),
        'statistics_reused_because_no_labels_changed':True}
    dest.mkdir(parents=True);(dest/'judge_cache').mkdir()
    shutil.copyfile(current_cache_path,dest/'judge_cache'/filename)
    identity['merged_cache_sha256']=sha(dest/'judge_cache'/filename)
    for name in ('tune.json','plan.json','tune_scored_items.json','tune_scored.json','tune_scored.md',
                 'tune_simple_scores.json','tune_matched_coverage.json','tune_matched_coverage.md'):
        shutil.copyfile(run/name,dest/name)
    shutil.copytree(run/'code_snapshot',dest/'code_snapshot')
    shutil.copyfile(Path(__file__),dest/'reference_lock_script.py')
    write(dest/'reference_lock_audit.json',{'identity':identity,'decisions':records,'changed_labels':changes})
    write(dest/'reference_lock_manifest.json',{'identity':identity,'files':{n:sha(dest/n) for n in
        ('tune.json','tune_scored_items.json','tune_scored.json','tune_simple_scores.json','tune_matched_coverage.json',
         'reference_lock_audit.json','reference_lock_script.py')}})
    print('REFERENCE_LOCK_READY',raw['model'],'matched',len(records),'changed',len(changes),str(dest),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('code-dir','run-dir','reference-run','output-dir'):p.add_argument('--'+key,type=Path,required=True)
    perform(p.parse_args())
