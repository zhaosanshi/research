"""Audit our own reference/cache assumptions against new judge request fields."""
import copy
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;REVIEW=HERE.parents[2];CODE=HERE/'source_cf8f752'
sys.path.insert(0,str(CODE))
from core.qa_reference_grades import reference_index,semantic_key
from core.qa_generation_adapter import fingerprint
from core.judge import QA_GRADER_VERSION


def main():
    cfg={'name':'judge','model':'gpt-offline','base_url':'https://example.invalid',
         'max_tokens':8,'max_completion_tokens':1024,'reasoning_effort':'low','send_temperature':False}
    row={'pair_id':'q','question':'Q?','gold':['gold'],'incorrect':['bad'],'best_answer':'gold',
         'outputs':{'baseline':'answer'},'finished':{'baseline':True}}
    raw={'completed':True,'model':'generator','identity':{},'conditions':['baseline'],
         'condition_specs':[{'name':'baseline'}],'evaluations':[row]}
    grade=copy.deepcopy(raw);grade.update(grading_completed=True,grading_source_generation_sha256='raw-hash',
        grading_judge_config_sha256=fingerprint(cfg),grading_judge_name=cfg['name'])
    grade['evaluations'][0].update(judge_reference_answers=['gold'],gold_is_abstention=False,
        judge={'baseline':{'label':'correct','raw':'A','rubric_id':QA_GRADER_VERSION}},outcome={'baseline':'correct'})
    changes={}
    for key,value in [('max_completion_tokens',2048),('reasoning_effort','high'),('send_temperature',True)]:
        try:
            reference_index(raw,grade,raw_sha256='raw-hash',reference_config=cfg,target_config={**cfg,key:value})
            accepted=True
        except ValueError:accepted=False
        changes[key]={'from':cfg[key],'to':value,'accepted_as_compatible':accepted}
    r=grade['evaluations'][0];a='x'*4000+' TRUE TAIL';b='x'*4000+' FALSE TAIL'
    collision=semantic_key(cfg,r,a)==semantic_key(cfg,r,b)
    old_config=Path('/tmp/td_r21_20261107/claude_gpt_review/from_claude/code/configs/experiments/E4_qa_legacy_r20.json')
    new_config=CODE/'configs/experiments/E4_qa_legacy_r20.json'
    compatibility={}
    if old_config.exists():
        previous=json.loads(old_config.read_text())['judge'];current=json.loads(new_config.read_text())['judge']
        compatibility={'old_model':previous['model'],'current_legacy_named_model':current['model'],
            'old_config_fingerprint':fingerprint(previous),'current_fingerprint':fingerprint(current)}
        oldraw=copy.deepcopy(raw);oldgrade=copy.deepcopy(grade)
        oldgrade['grading_judge_config_sha256']=fingerprint(previous);oldgrade['grading_judge_name']=previous['name']
        oldgrade['evaluations'][0]['judge']['baseline']['rubric_id']='qa_semantic_v3'
        try:
            reference_index(oldraw,oldgrade,raw_sha256='raw-hash',reference_config=previous,target_config=previous)
            compatibility['old_readonly_reference_rejected']=False
        except ValueError as exc:
            compatibility['old_readonly_reference_rejected']=True;compatibility['error']=str(exc)
    result={'snapshot':'cf8f752','network_calls':0,'request_fields_not_checked':changes,
            'full_answer_reference_key_collision':collision,'legacy_compatibility':compatibility,
            'interpretation':'Legacy refusal is fail-closed, not silent grade reuse. New live judging must remain explicitly Pro; old frozen provenance must not be relabeled as v4/v5.'}
    (HERE/'reference_compatibility.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    assert all(v['accepted_as_compatible'] for v in changes.values()) and collision


if __name__=='__main__':main()
