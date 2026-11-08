"""Reference-grade consistency by complete semantic request, not arm name."""
from .judge import QA_GRADER_VERSION
from .qa_generation_adapter import fingerprint
from .judge_two_axis_contract import canonical_sha,request_contract
from .qa_scoring import apply_configured_outcome_policy, reference_answers


def semantic_key(config, row, prediction):
    if not isinstance(prediction,str):raise ValueError('reference identity requires the complete answer string')
    return canonical_sha({'protocol':'qa_complete_reference_key_v2','rubric':QA_GRADER_VERSION,
        'request':request_contract(config),'question':row['question'],'gold_refs':row['judge_reference_answers'],
        'incorrect':row.get('incorrect') or None,'prediction':prediction})


def reference_index(raw, graded, *, raw_sha256, reference_config, target_config):
    """Validate the complete frozen parent; uniformly convert only its policy."""
    previous,current=request_contract(reference_config),request_contract(target_config)
    changed=[key for key in previous if previous[key]!=current[key]]
    if changed:raise ValueError('reference and current semantic judge requests differ: '+', '.join(changed))
    if (not raw.get('completed') or not graded.get('completed') or not graded.get('grading_completed')
            or raw['model']!=graded['model'] or raw['identity']!=graded['identity']
            or raw['conditions']!=graded['conditions'] or raw['condition_specs']!=graded['condition_specs']
            or graded.get('grading_source_generation_sha256')!=raw_sha256
            or graded.get('grading_judge_config_sha256')!=fingerprint(reference_config)
            or graded.get('grading_judge_name')!=reference_config['name']):
        raise ValueError('reference generation/grades/judge identity differs')
    ids=[r['pair_id'] for r in raw['evaluations']]
    if (not ids or len(ids)!=len(set(ids)) or ids!=[r['pair_id'] for r in graded['evaluations']]
            or not raw['conditions'] or len(raw['conditions'])!=len(set(raw['conditions']))):
        raise ValueError('reference grade cohort differs')
    index={};rows=[]
    for r,g in zip(raw['evaluations'],graded['evaluations']):
        if any(r.get(k)!=g.get(k) for k in ('pair_id','question','gold','incorrect','best_answer','outputs','finished')):
            raise ValueError('reference raw and graded answer differ')
        if set(r['outputs'])!=set(raw['conditions']) or set(r['finished'])!=set(raw['conditions']):
            raise ValueError('reference raw condition set is incomplete')
        refs,only=reference_answers(r['gold'])
        if g.get('judge_reference_answers')!=refs or g.get('gold_is_abstention') is not only:
            raise ValueError('reference semantic inputs or partition changed')
        if set(g['judge'])!=set(raw['conditions']) or set(g['outcome'])!=set(raw['conditions']):
            raise ValueError('reference grades omit a condition')
        converted={**g,'judge':{},'outcome':{}}
        for name,text in r['outputs'].items():
            old=g['judge'][name]
            if (old.get('rubric_id')!=QA_GRADER_VERSION or old.get('label') not in ('correct','wrong','abstain')
                    or g['outcome'][name]!=old['label']):
                raise ValueError('reference label/rubric is unresolved or inconsistent')
            new=apply_configured_outcome_policy(old,r['question'],text,target_config)
            key=semantic_key(target_config,g,text)
            if key in index:
                if index[key]['original_label']!=old['label'] or index[key]['grade']['label']!=new['label']:
                    raise ValueError('same reference semantic request has conflicting labels')
                index[key]['source_conditions'].append({'pair_id':r['pair_id'],'condition':name})
            else:
                index[key]={'original_label':old['label'],'grade':new,
                            'source_conditions':[{'pair_id':r['pair_id'],'condition':name}]}
            converted['judge'][name]=new;converted['outcome'][name]=new['label']
        rows.append(converted)
    return {'index':index,'policy_converted_rows':rows,
            'reference_config_sha256':fingerprint(reference_config),
            'target_config_sha256':fingerprint(target_config),'reference_conflicts':0}


def verify_matching_grades(raw, graded, reference, *, raw_sha256, target_config):
    """Every matching answer, including newly named C/EC arms, stays frozen."""
    if reference.get('read_only_historical'):
        raise ValueError('historical identity validation cannot seed a new judge or rubric')
    if (not raw.get('completed') or not graded.get('completed') or not graded.get('grading_completed')
            or graded.get('grading_source_generation_sha256')!=raw_sha256
            or graded.get('grading_judge_config_sha256')!=fingerprint(target_config)
            or graded.get('grading_judge_name')!=target_config['name'] or graded['model']!=raw['model']
            or graded['identity']!=raw['identity'] or graded['conditions']!=raw['conditions']
            or graded['condition_specs']!=raw['condition_specs']
            or [r['pair_id'] for r in raw['evaluations']]!=[r['pair_id'] for r in graded['evaluations']]):
        raise ValueError('current reference-grade verification identity differs')
    matches=[]
    for r,g in zip(raw['evaluations'],graded['evaluations']):
        if any(r.get(k)!=g.get(k) for k in ('question','gold','incorrect','best_answer','outputs','finished')):
            raise ValueError('current grading content differs from generated answers')
        if set(r['outputs'])!=set(raw['conditions']) or set(r['finished'])!=set(raw['conditions']):
            raise ValueError('current raw condition set is incomplete')
        refs,only=reference_answers(r['gold'])
        if g.get('judge_reference_answers')!=refs or g.get('gold_is_abstention') is not only:
            raise ValueError('current semantic reference inputs differ from the original generation')
        if set(g['judge'])!=set(raw['conditions']) or set(g['outcome'])!=set(raw['conditions']):
            raise ValueError('current grade condition set is incomplete')
        for name,text in r['outputs'].items():
            actual=g['judge'][name]
            if (actual.get('rubric_id')!=QA_GRADER_VERSION or actual.get('label') not in ('correct','wrong','abstain')
                    or g['outcome'][name]!=actual['label']):
                raise ValueError('current semantic outcome/rubric differs')
            key=semantic_key(target_config,g,text)
            if key not in reference['index']:
                if name in ('baseline','E_source'):
                    raise ValueError('baseline/E has no matching full-answer reference grade')
                continue
            wanted=reference['index'][key]['grade']['label']
            if (actual.get('rubric_id')!=QA_GRADER_VERSION or actual.get('label')!=wanted
                    or g['outcome'][name]!=wanted):
                raise ValueError('identical full answer changed its frozen reference label: '+r['pair_id']+'/'+name)
            matches.append({'pair_id':r['pair_id'],'condition':name,'reference_key':key,'label':wanted})
    return {'matched_question_conditions':len(matches),'all_reference_labels_identical_after_current_policy':True,
            'decisions':matches,'reference_config_sha256':reference['reference_config_sha256'],
            'current_config_sha256':fingerprint(target_config)}


def audit_historical_reference(raw,graded,*,raw_sha256,recorded_config,expected_rubric):
    """Read-only validation of explicitly named historical metadata, never live reuse.

    Supply the original archived config, not a mutable current file bearing a
    legacy name. Recorded reference annotations are retained; this function
    does not pretend a new rubric or refusal policy was applied in the past.
    """
    if not isinstance(expected_rubric,str) or not expected_rubric:
        raise ValueError('historical rubric must be named explicitly')
    if (not raw.get('completed') or not graded.get('completed') or not graded.get('grading_completed')
            or raw['model']!=graded['model'] or raw['identity']!=graded['identity']
            or raw['conditions']!=graded['conditions'] or raw['condition_specs']!=graded['condition_specs']
            or graded.get('grading_source_generation_sha256')!=raw_sha256
            or graded.get('grading_judge_config_sha256')!=fingerprint(recorded_config)
            or graded.get('grading_judge_name')!=recorded_config['name']):
        raise ValueError('historical raw/grade/config identity differs')
    ids=[r['pair_id'] for r in raw['evaluations']];names=raw['conditions']
    if not ids or len(ids)!=len(set(ids)) or ids!=[r['pair_id'] for r in graded['evaluations']] or len(names)!=len(set(names)):
        raise ValueError('historical cohort or condition identity differs')
    n=0
    for r,g in zip(raw['evaluations'],graded['evaluations']):
        if any(r.get(k)!=g.get(k) for k in ('pair_id','question','gold','incorrect','best_answer','outputs','finished')):
            raise ValueError('historical answer or question changed')
        if (any(set(r[key])!=set(names) for key in ('outputs','finished'))
                or set(g['judge'])!=set(names) or set(g['outcome'])!=set(names)
                or not isinstance(g.get('judge_reference_answers'),list) or type(g.get('gold_is_abstention')) is not bool):
            raise ValueError('historical judgments/reference annotations are incomplete')
        for name in names:
            grade=g['judge'][name]
            if (grade.get('rubric_id')!=expected_rubric or grade.get('label') not in ('correct','wrong','abstain')
                    or g['outcome'][name]!=grade['label']):
                raise ValueError('historical label/rubric differs from its explicit declared version')
            n+=1
    return {'read_only_historical':True,'can_seed_new_judge':False,'model':raw['model'],
        'recorded_judge_model':recorded_config['model'],'recorded_rubric':expected_rubric,
        'recorded_config_sha256':fingerprint(recorded_config),'source_generation_sha256':raw_sha256,
        'questions':len(ids),'question_condition_grades_checked':n,
        'scope':'Identity-only historical check. No new semantic/rubric/guard claim and no API calls.'}
