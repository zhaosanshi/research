"""Explicit grading-only implementation override; model snapshot stays immutable."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())

def verify_parent(raw_path,grade_path):
    from core.qa_scoring import reference_answers
    from core.judge import QA_GRADER_VERSION
    raw,grade=load(raw_path),load(grade_path)
    if (not raw.get('completed') or not grade.get('completed') or not grade.get('grading_completed')
            or grade['grading_source_generation_sha256']!=sha(raw_path)
            or raw['conditions']!=grade['conditions'] or raw['model']!=grade['model']
            or raw['identity']!=grade['identity']
            or [r['pair_id'] for r in raw['evaluations']]!=[r['pair_id'] for r in grade['evaluations']]):
        raise ValueError('completion parent raw/grade identity mismatch')
    for r,g in zip(raw['evaluations'],grade['evaluations']):
        for key in ('question','gold','outputs','finished','generation_token_ids','trace','prompt_token_ids'):
            if r[key]!=g[key]:raise ValueError('completion parent grade changed original '+key)
        for name in raw['conditions']:
            if (g['judge'][name].get('label') not in ('correct','wrong','abstain')
                    or g['judge'][name].get('rubric_id')!=QA_GRADER_VERSION or g['outcome'][name]!=g['judge'][name]['label']):
                raise ValueError('completion parent grade has unresolved/inconsistent outcomes')
        refs,only=reference_answers(r['gold'])
        if g['judge_reference_answers']!=refs or g['gold_is_abstention'] is not only:
            raise ValueError('completion parent reference partition changed')
    return {'generation_sha256':sha(raw_path),'scored_items_sha256':sha(grade_path),'question_count':len(raw['evaluations']),
            'condition_count':len(raw['conditions']),'all_original_fields_and_labels_consistent':True}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('code-root','judge-code-file','driver-file','proof-dir','masked-run','controls256-run'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('driver_args',nargs=argparse.REMAINDER);a=p.parse_args()
    args=a.driver_args[1:] if a.driver_args[:1]==['--'] else a.driver_args
    if not args or args[0]!='judge':raise ValueError('this override is restricted to grading, never train/generate')
    for option,path in (('--code-root',a.code_root),('--masked-run',a.masked_run),
                        ('--controls256-run',a.controls256_run),('--output-dir',a.proof_dir)):
        if args.count(option)!=1 or Path(args[args.index(option)+1]).resolve()!=path.resolve():
            raise ValueError('grading proof paths do not match the actual driver arguments')
    sys.path[:0]=[str(a.code_root/'scripts'),str(a.code_root),str(a.code_root/'scripts/review_checks')]
    import e25_qa_das as original
    spec=importlib.util.spec_from_file_location('qa_checkpointed_grader',a.judge_code_file)
    replacement=importlib.util.module_from_spec(spec);spec.loader.exec_module(replacement)
    proof={'scope':'grading only; only E25 checkpoint save scheduling differs from the model snapshot',
        'original_grader_file_sha256':sha(original.__file__),'replacement_grader_file_sha256':sha(a.judge_code_file),
        'driver_sha256':sha(a.driver_file),'wrapper_sha256':sha(__file__),
        'parents':{'masked48':verify_parent(a.masked_run/'tune.json',a.masked_run/'tune_scored_items.json'),
                   'controls256':verify_parent(a.controls256_run/'tune.json',a.controls256_run/'tune_scored_items.json')},
        'grader_rubric_unchanged':replacement.QA_GRADER_VERSION==original.QA_GRADER_VERSION,
        'model_code_files_not_modified':True,'completed':False}
    if not proof['grader_rubric_unchanged']:raise ValueError('grading rubric changed')
    a.proof_dir.mkdir(parents=True,exist_ok=True);proof_path=a.proof_dir/'checkpointed_grading_implementation.json'
    if proof_path.exists():
        old=load(proof_path)
        if {k:v for k,v in old.items() if k not in ('completed','source_files_unchanged_after')}!={k:v for k,v in proof.items() if k!='completed'}:
            raise ValueError('existing grading implementation proof differs')
    proof_path.write_text(json.dumps(proof,indent=2)+'\n')
    previous,previous_argv=original.judge,sys.argv
    try:
        # A documented callback substitution lets the model/source verifier
        # inspect its original immutable files, while grading uses this exact
        # separately hashed implementation. No generated field or cache changes.
        original.judge=replacement.judge
        spec=importlib.util.spec_from_file_location('completion_grading_driver',a.driver_file)
        driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
        sys.argv=[str(a.driver_file),*args];driver.main()
        proof['completed']=True
    finally:
        original.judge=previous;sys.argv=previous_argv
        proof['source_files_unchanged_after']=sha(original.__file__)==proof['original_grader_file_sha256'] and sha(a.judge_code_file)==proof['replacement_grader_file_sha256']
        proof_path.write_text(json.dumps(proof,indent=2)+'\n')
    if not proof['source_files_unchanged_after']:raise RuntimeError('grading source changed while executing')

if __name__=='__main__':main()
