"""Independent synthetic reproductions; no model calls or real human verdicts."""
import copy,json,os,sys,tempfile,subprocess,tarfile
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve()
if 'R24_REPRO_REVIEW' not in os.environ:
    with tempfile.TemporaryDirectory() as frozen:
        frozen=Path(frozen)
        with tarfile.open(HERE.with_name('blind_n7_reviewed_source.tar.gz')) as source:
            for member in source.getmembers():
                target=frozen/member.name
                if member.name.startswith('/') or '..' in Path(member.name).parts or not (member.isfile() or member.isdir()):
                    raise ValueError('unsafe reviewed-source archive member')
                if member.isdir():target.mkdir(parents=True,exist_ok=True)
                else:
                    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.extractfile(member).read())
        env=dict(os.environ,R24_REPRO_REVIEW=str(frozen/'Truth_Decoupling/claude_gpt_review'))
        raise SystemExit(subprocess.run([sys.executable,str(HERE)],env=env).returncode)
REVIEW=Path(os.environ['R24_REPRO_REVIEW']);CODE=REVIEW/'from_claude/code'
sys.path[:0]=[str(CODE),str(CODE/'scripts/review_checks'),str(REVIEW/'from_gpt/audit_20261108_round22/pipeline')]
import judge2_blind_audit as B
import n7_curve_table as N
import e41_analysis as E
import test_two_axis_pipeline_r22 as F
from core.io_utils import write_jsonl

def one_raw(pred='Paris.'):
    raw=F.generation();raw['evaluations']=raw['evaluations'][:1]
    raw['evaluations'][0]['outputs']={k:pred for k in raw['conditions']}
    return raw

def main():
    out={'scope':'Synthetic offline reproduction on 77ad5b1; no actual historical/model/human result altered','new_api_calls':0}
    with tempfile.TemporaryDirectory() as td,patch.dict(os.environ,{},clear=False):
        for name in ('TD_JUDGE2_HUMAN','TD_JUDGE2_CACHE','TD_JUDGE2_CONFIG'):os.environ.pop(name,None)
        td=Path(td);run=F.setup(td/'run',one_raw());cfg=F.run.load_cfg();F.allenai(run,cfg)
        F.run.run_llm(str(run),'pro',cfg,judge=F.FakeJudge('pro',cfg))
        F.run.human_set(str(run),'q0|EC','A','synthetic-assisted-only',note='Previously adjudicated after viewing machine votes; not blind')
        B.sample([str(run)],str(td/'blind'),'trial',1,1,[])
        before=B.report(str(td/'blind'),'trial')
        out['assisted_human_counted_as_blind_without_apply']={'report':before,'final_file_exists':(run/'judge2/final.jsonl').exists()}
        key=json.loads((td/'blind/trial_key.jsonl').read_text().splitlines()[0])
        # An old exported sheet survives a legitimate fresh run at the same path.
        run.rename(td/'archived-original-run');run=F.setup(td/'run',one_raw('Lyon.'))
        accepted=B.apply(str(td/'blind'),'trial','1A','synthetic-sheet-reviewer')
        recorded=F.run._human_verdicts(str(run))[key['cell_id']]
        out['stale_sheet_applied_to_replacement_answer']={'accepted':accepted,'sheet_answer':key['prediction'],
             'recorded_reviewed_answer':recorded['reviewed_input']['prediction'],'sheet_input_sha':key['input_sha'],
             'recorded_input_sha':recorded['input_sha'],'recorded_label':recorded['label']}
        # Actual production verifier rejects this row, but blind report reads it anyway.
        old=td/'archived-original-run';B.sample([str(old)],str(td/'other'),'badrow',1,1,[])
        rows=F.run._rows(old/'judge2/pro.jsonl')
        for row in rows.values():row['input_sha']='wrong-source';row['truth']='F';row['truth_raw']='T'
        write_jsonl(list(rows.values()),str(old/'judge2/pro.jsonl'))
        valid,stats=F.run._verify_rows(F.run._rows(old/'judge2/pro.jsonl'),F.run._cells(str(old)),'pro',cfg)
        report=B.report(str(td/'other'),'badrow')
        out['unverified_machine_votes_in_blind_report']={'normal_verifier_stale':stats['stale'],
               'normal_verifier_valid_rows':len(valid),'blind_report_pro':report['pro']}
    # Six strength choices; old per-strength four-way Holm treats .01 as .04.
    rows=[];raw_ps=[]
    for alpha in [.25,.5,.75,1,1.5,2]:
        p=.01 if alpha==.25 else 1.;ps=[1.,p,1.,1.];raw_ps.extend(ps);adjusted=E.holm(ps)
        rows.append({'alpha':alpha,'abst':{'P2':1,'P2_holm':float(adjusted[1])},'ans':{'P2':0,'P2_holm':float(adjusted[3])}})
    current=N.verdict({'llama3.1-8b':rows,'qwen3-8b':copy.deepcopy(rows)})
    adjusted_all=E.holm(raw_ps)
    out['n7_strength_selection_outside_correction']={'current_verdict':current,'raw_best_p':.01,
        'current_best_per_strength_holm':rows[0]['abst']['P2_holm'],'corrected_best_across_24_contrasts':float(adjusted_all[1]),
        'all_strength_family_significant':bool(min(adjusted_all)<.05)}
    out['n7_no_result_has_negative_verdict']=N.verdict({'llama3.1-8b':[],'qwen3-8b':[],'gemma3-4b':[]})
    dest=HERE.with_name('blind_n7_reproductions.json');dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
