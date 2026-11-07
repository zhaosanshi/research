"""Freeze complete norm-null runs and independently verify every archive member."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    for model in ('llama3.1-8b','qwen3-8b'):
        base=args.root/model
        for phase,stem in [('pilot3','check'),('full192','tune')]:
            raw=json.loads((base/phase/(stem+'.json')).read_text());scored=json.loads((base/phase/(stem+'_scored_items.json')).read_text())
            assert raw['completed'] and scored['completed'] and scored['grading_completed']
            assert scored['grading_source_generation_sha256']==sha(base/phase/(stem+'.json'))
            assert scored['frozen_control_grade_verification']['question_condition_grades_checked']==(21 if phase=='pilot3' else 1344)
        assert (base/'full192/tune_simple_scores.json').exists() and (base/'full192/tune_matched_coverage.json').exists()
        files=[f for f in base.rglob('*') if f.is_file() and '__pycache__' not in f.parts
               and 'judge_cache' not in f.parts and not f.name.endswith('_grading_input.json')
               and f.suffix in ('.json','.py','.log','.exit','.md')]
        files.append(args.root/'native_stop_config.json');files=sorted(set(files))
        manifest=[{'path':str(f.relative_to(args.root)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in files]
        name='norm_random_'+model;archive=args.output/(name+'.tar.xz')
        with tarfile.open(archive,'w:xz',preset=8) as tar:
            for f in files:tar.add(f,arcname=str(f.relative_to(args.root)),recursive=False)
        with tarfile.open(archive) as tar:
            actual={m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in tar.getmembers() if m.isfile()}
        assert actual=={f['path']:f['sha256'] for f in manifest}
        result={'archive':archive.name,'archive_sha256':sha(archive),'bytes':archive.stat().st_size,
                'files':manifest,'all_archived_files_verified':True,
                'scope':'fixed random axis, learned scalar same-state stored-norm target, three seeds, tuning only',
                'excluded':'mutable judge cache and duplicate derived grading-input JSON'}
        (args.output/(name+'_manifest.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False))
        print(name,len(files),archive.stat().st_size,'verified',flush=True)


if __name__=='__main__':main()
