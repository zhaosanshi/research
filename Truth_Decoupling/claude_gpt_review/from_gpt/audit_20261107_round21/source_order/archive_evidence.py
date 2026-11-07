"""Archive completed E37 generation, grading, calibration and matched scores."""
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
        base=args.root/model;cal=base/'calibration/c_calibration.json';calibration=json.loads(cal.read_text())
        assert calibration['completed'] and calibration['states_sha256']==sha(base/'calibration/unmodified_prompt_states.pt')
        for tag,stem,n in [('original_sigma_pilot3','check',3),('calibrated_pilot3','check',3),('full192','tune',192)]:
            rawpath=base/tag/(stem+'.json');raw=json.loads(rawpath.read_text());graded=json.loads((base/tag/(stem+'_scored_items.json')).read_text())
            assert raw['completed'] and len(raw['evaluations'])==n and len(raw['conditions'])==10
            assert raw['all_controls_freshly_generated'] and raw['baseline_E_fresh_exact_reference']
            assert graded['completed'] and graded['grading_completed'] and graded['grading_source_generation_sha256']==sha(rawpath)
            if tag!='original_sigma_pilot3':assert raw['identity']['c_calibration_sha256']==sha(cal)
        full=base/'full192';assert (full/'tune_simple_scores.json').exists() and (full/'tune_matched_coverage.json').exists()
        locked=base/'reference_locked';lock=json.loads((locked/'reference_lock_audit.json').read_text())
        assert lock['identity']['changed_final_labels']==0 and lock['identity']['reference_conflicts']==0
        for name in ('tune.json','tune_scored_items.json','tune_scored.json','tune_simple_scores.json','tune_matched_coverage.json'):
            assert sha(full/name)==sha(locked/name)
        files=sorted(f for f in base.rglob('*') if f.is_file() and not f.is_symlink()
            and 'judge_cache' not in f.parts and '__pycache__' not in f.parts
            and not f.name.endswith('_grading_input.json')
            and f.suffix in ('.json','.py','.pt','.md','.log','.exit','.sh'))
        entries=[{'path':str(f.relative_to(args.root)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in files]
        name='source_commitment_order_'+model;archive=args.output/(name+'.tar.xz')
        first_by_hash={}
        with tarfile.open(archive,'w:xz',preset=8) as tar:
            for f,entry in zip(files,entries):
                if entry['sha256'] in first_by_hash:
                    target=first_by_hash[entry['sha256']];info=tar.gettarinfo(str(f),arcname=entry['path'])
                    info.type=tarfile.LNKTYPE;info.linkname=target;info.size=0;tar.addfile(info)
                    entry['hardlink_to']=target
                else:
                    tar.add(f,arcname=entry['path'],recursive=False);first_by_hash[entry['sha256']]=entry['path']
        with tarfile.open(archive) as tar:
            actual={m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in tar.getmembers() if m.isfile() or m.islnk()}
        assert actual=={v['path']:v['sha256'] for v in entries}
        result={'archive':archive.name,'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':entries,
            'all_members_verified':True,'unique_content_members':len(first_by_hash),
            'hardlink_policy':'only identical SHA256 content links to earlier internal archive members; every logical member is rehashed',
            'scope':'original-sigma three-question diagnostic; 92-training-question native-dtype C calibration; calibrated three-question and full192 ten-arm generation/grades/common scores; byte-identical reference-locked primary with explicit audit; immutable runtime snapshots',
            'excluded':'mutable API cache and duplicate derived grading-input JSON'}
        (args.output/(name+'_manifest.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False))
        print(name,len(entries),archive.stat().st_size,'all verified',flush=True)


if __name__=='__main__':main()
