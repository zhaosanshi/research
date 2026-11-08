"""Offline reproductions against reviewed 77ad5b1; no network or actual human labels."""
import argparse, copy, hashlib, importlib.util, json, os, re, subprocess, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class Stub:
    def __init__(self, answer):
        self.answer=answer;self.calls=[];self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(model=kw['model'],choices=[SimpleNamespace(message=SimpleNamespace(content=self.answer))])


def main():
    p=argparse.ArgumentParser();p.add_argument('--code-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    review=Path(__file__).resolve().parents[2]
    sys.path[:0]=[str(a.code_root),str(a.code_root/'scripts/review_checks'),str(review/'from_gpt/audit_20261108_round22/pipeline')]
    from core import judge as J, judge_two_axis as T
    from core.judge_two_axis_contract import judge_contract,canonical_sha,cell_sha
    from core.io_utils import write_jsonl
    import judge_two_axis_run as run
    import judge2_cache_merge as merge
    import test_two_axis_pipeline_r22 as fixture
    out={'reviewed_commit':'77ad5b1','no_network_calls':True,'no_real_human_labels':True,'findings':{}}
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);path=root/'judge2_cache_m.jsonl';tag={'judge_contract_sha':'synthetic'}
        left=J.LLMJudge('m',api_key='offline-only',cache_path=str(path),sleep=0,key_extra=tag)
        right=J.LLMJudge('m',api_key='offline-only',cache_path=str(path),sleep=0,key_extra=tag)
        left._client=Stub('T');right._client=Stub('F')
        replies=[left.complete('system','user'),right.complete('system','user')]
        assert replies==['T','F'] and len(right._client.calls)==1
        out['findings']['shared_cache_stale_memory']={'replies':replies,'requests':len(left._client.calls)+len(right._client.calls),
            'disk_rows':len(path.read_text().splitlines()),'automatically_created_conflicts':(root/'conflicts.json').exists()}

        legacy=root/'legacy.jsonl'
        x=J.LLMJudge('m',base_url='https://one.example.invalid',api_key='offline-only',cache_path=str(legacy),max_tokens=8,sleep=0);x._client=Stub('A')
        x.qa_grade('q',['gold'],'answer')
        y=J.LLMJudge('m',base_url='https://two.example.invalid',api_key='offline-only',cache_path=str(legacy),max_tokens=1024,sleep=0);y._client=Stub('B')
        result=y.qa_grade('q',['gold'],'answer')
        assert result['label']=='correct' and not y._client.calls
        out['findings']['legacy_new_requests_ignore_parameters']={'second_label':result['label'],'second_network_stub_calls':len(y._client.calls),
            'different_endpoint_and_budget':True,'same_key':x.cache_key('s','u')==y.cache_key('s','u'),
            'long_tail_prompt_collision':J.qa_grade_user('q',['g'],'x'*4000+' TRUE')==J.qa_grade_user('q',['g'],'x'*4000+' FALSE')}

        cfg=run.load_cfg();before=canonical_sha(judge_contract('pro',cfg));parse_before=T.parse_truth('T: TRUTHFUL')
        with patch.object(T,'_SEP',re.compile(r'\s+')):
            after=canonical_sha(judge_contract('pro',cfg));parse_after=T.parse_truth('T: TRUTHFUL')
        assert parse_before=='T' and parse_after is None and before==after
        out['findings']['parser_globals_not_hashed']={'before':parse_before,'after':parse_after,'contract_unchanged':before==after}

        directory=fixture.setup(root/'run');fixture.allenai(directory,cfg)
        run.run_llm(str(directory),'pro',cfg,judge=fixture.FakeJudge('pro',cfg))
        run.compute_disputes(str(directory),cfg)
        run.run_llm(str(directory),'gpt',cfg,only='disputes',judge=fixture.FakeJudge('gpt',cfg))
        run.human_set(str(directory),'q2|EC','A','synthetic-test-only')
        shared=root/'shared_human.jsonl';shared.write_text('')
        with patch.dict(os.environ,{'TD_JUDGE2_HUMAN':str(shared)}):
            run.adjudicate(str(directory),cfg)
            original=run.load_verified_final(str(directory),directory/'tune.json')[2]['q0|EC']['label']
            c=next(c for c in run._cells(str(directory)) if c['cell_id']=='q0|EC')
            record=run.normalize_human_verdict({'cell_id':c['cell_id'],'label':'B','info_subtype':'S','by':'synthetic-test-only',
                'input_sha':cell_sha(c),'version':T.ADJUDICATION_VERSION,
                'reviewed_input':{k:c[k] for k in ('question','prediction','gold','gold_refs','incorrect','best_answer','gold_is_abstention')}})
            write_jsonl([record],str(shared))
            effective=run._human_verdicts(str(directory))['q0|EC']['label']
            accepted=run.load_verified_final(str(directory),directory/'tune.json')[2]['q0|EC']['label']
            assert original==accepted=='A' and effective=='B'
            out['findings']['external_human_not_bound_in_final']={'stored_final':accepted,'current_human':effective,'stale_final_accepted':True}
            shared.write_text('')
        cache=directory/'judge2/cache';cache.mkdir(exist_ok=True)
        keys=run.judge_request_keys(c,'pro',cfg)
        (cache/'conflicts.json').write_text(json.dumps({'keys':{'synthetic_file':[keys[0]]}}))
        pending=run._inconsistent_cells(run._cells(str(directory)),cfg,str(cache))
        accepted=run.load_verified_final(str(directory),directory/'tune.json')[2]['q0|EC']['label']
        assert 'pro' in pending['q0|EC'] and accepted=='A'
        out['findings']['external_conflicts_not_bound_in_final']={'new_conflict_requires_human':True,'stale_final_accepted':True}

        other=fixture.setup(root/'other')
        (other/'judge2/final.jsonl').write_text('')
        proc=subprocess.run([sys.executable,str(a.code_root/'scripts/review_checks/judge2_cross_run_check.py'),str(directory),str(other)],capture_output=True,text=True)
        assert proc.returncode==0
        report=json.loads(proc.stdout.splitlines()[0])
        out['findings']['cross_run_missing_finals_pass']={'exit_code':proc.returncode,'report':report,'empty_final_file':True}
    a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
