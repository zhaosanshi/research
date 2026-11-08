"""Offline provenance counterexamples against R22 cf8f752; no real API calls.

Every case executes the production readers/writers on isolated temporary files.
The JSON records observed unsafe behavior, not fabricated scientific scores.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

REVIEW=Path(__file__).resolve().parents[3]
CODE=Path(os.environ.get('R22_AUDIT_CODE',str(Path(__file__).with_name('source_cf8f752'))))
sys.path[:0]=[str(CODE),str(CODE/'scripts')]
from core import judge as J
from core import judge_two_axis as T
from core.io_utils import read_jsonl,write_jsonl
from scripts.review_checks import judge_two_axis_run as R
from scripts.review_checks import judge_human_review as H


class Stub:
    def __init__(self,replies):
        self.replies=list(replies);self.calls=[]
        self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
    def create(self,**kw):
        self.calls.append(kw)
        return SimpleNamespace(model=kw['model']+'-resolved',choices=[SimpleNamespace(message=SimpleNamespace(content=self.replies.pop(0)))])


def config(model='model-old',**extra):
    return {'version':T.TWO_AXIS_VERSION,'pro':{'model':model,'api_key':'offline-placeholder','sleep':0.,'max_tokens':8,**extra},
            'gpt':{'model':'gpt-offline'},'allenai':{'threshold':.5},'audit':{'n':0,'seed':0}}


def generation(root,prediction='Paris.',gold=None,conditions=('base',)):
    value={'completed':True,'model':'generator-toy','conditions':list(conditions),
           'evaluations':[{'pair_id':'q','question':'What is the capital of France?','gold':gold or ['Paris'],
                          'incorrect':['Lyon'],'outputs':{n:prediction for n in conditions}}]}
    (root/'tune.json').write_text(json.dumps(value));R.build_cells(str(root),audit_n=0)


def wrapped(root,cfg,replies):
    judge=T.TwoAxisLLMJudge(cfg['pro'],str(root/'judge2/cache'),name='pro');stub=Stub(replies);judge.llm._client=stub
    return judge,stub


def synthetic_votes(root,truth='F'):
    cells=R._cells(str(root));al=[];pro=[]
    for cell in cells:
        sha=T.qa_sha(cell['question'],cell['prediction'])
        al.append({'cell_id':cell['cell_id'],'qa_sha':sha,'truth_p_yes':.9 if truth=='T' else .1,'info_p_yes':.9})
        pro.append({'cell_id':cell['cell_id'],'qa_sha':sha,'rubric':T.TWO_AXIS_VERSION,'model':'model-old','truth':truth,'info':'S'})
    write_jsonl(al,str(root/'judge2/allenai.jsonl'));write_jsonl(pro,str(root/'judge2/pro.jsonl'))


def main():
    observations={}
    with tempfile.TemporaryDirectory() as td:
        base=Path(td)
        # A changed gold reference changes the actual truth prompt but is not in qa_sha.
        root=base/'references';root.mkdir();generation(root);cfg=config();j,s=wrapped(root,cfg,['T','S'])
        R.run_llm(str(root),'pro',cfg,judge=j);generation(root,gold=['The reference was corrected'])
        j,s=wrapped(root,cfg,['F','S']);summary=R.run_llm(str(root),'pro',cfg,judge=j)
        retained=list(read_jsonl(str(root/'judge2/pro.jsonl')))[0]
        observations['changed_references_reuses_old_vote']={'new_api_calls':len(s.calls),'retained_truth':retained['truth'],
            'stale_regraded':summary['stale_regraded'],'unsafe_reproduced':len(s.calls)==0 and retained['truth']=='T'}
        # A changed requested model still retains the old model's row, while summary names the new model.
        cfg2=config('model-new');j,s=wrapped(root,cfg2,['F','S']);summary=R.run_llm(str(root),'pro',cfg2,judge=j)
        retained=list(read_jsonl(str(root/'judge2/pro.jsonl')))[0]
        observations['changed_model_reuses_old_vote']={'new_api_calls':len(s.calls),'summary_model':summary['model'],
            'retained_row_model':retained['model'],'unsafe_reproduced':len(s.calls)==0 and retained['model']!=summary['model']}
        # A legacy unbound row is certified against a current answer without any independent source proof.
        cell=R._cells(str(root))[0]
        legacy={'cell_id':cell['cell_id'],'model':'old-model','rubric':T.TWO_AXIS_VERSION,'truth':'T','info':'S'}
        accepted,stats=R._verify_rows({cell['cell_id']:legacy},[dict(cell,prediction='An entirely different answer')])
        observations['unbound_legacy_row_backfilled']={'stats':stats,'accepted':list(accepted.values()),
            'unsafe_reproduced':stats['legacy_backfilled']==1 and not stats['stale']}
        # Human decisions are keyed by cell ID, not the reviewed input: old Paris verdict transfers to Lyon.
        root=base/'human_stale';root.mkdir();generation(root);R.human_set(str(root),'q|base','A','offline-reviewer')
        generation(root,prediction='Lyon.');synthetic_votes(root,'F');R.adjudicate(str(root),config())
        final=list(read_jsonl(str(root/'judge2/final.jsonl')))[0]
        observations['human_verdict_transfers_to_changed_answer']={'current_answer':'Lyon.','old_reviewed_answer':'Paris.',
            'final_label':final['label'],'source':final['source'],'machine_label':final['judges_label'],
            'unsafe_reproduced':final['label']=='A' and final['judges_label']=='B'}
        # Same question/reference/full answer in two conditions can receive different final human/AI labels.
        root=base/'human_duplicates';root.mkdir();generation(root,conditions=('base','EC'));synthetic_votes(root,'F')
        R.human_set(str(root),'q|EC','A','offline-reviewer');R.adjudicate(str(root),config())
        finals=list(read_jsonl(str(root/'judge2/final.jsonl')))
        observations['identical_answers_different_final_labels']={'labels':{r['condition']:r['label'] for r in finals},
            'unsafe_reproduced':len({r['label'] for r in finals})>1}
        # Two-axis cache tag misses effective token budget and provider-specific thinking parameters.
        root=base/'params';root.mkdir();cfg=config(extra_body={'thinking':{'type':'disabled'}})
        j,s=wrapped(root,cfg,['T','S']);j.grade('q',['a'],'a')
        changed=config(max_tokens=1024,extra_body={'thinking':{'type':'enabled'}})
        j2,s2=wrapped(root,changed,['F','R']);result=j2.grade('q',['a'],'a')
        observations['changed_request_params_share_cache']={'new_api_calls':len(s2.calls),
            'old_request':j.llm.request_kwargs(),'new_request':j2.llm.request_kwargs(),
            'retained_votes':[result['truth'],result['info']],
            'unsafe_reproduced':not s2.calls and j.llm.request_kwargs()!=j2.llm.request_kwargs()}
        # Full-output hashes differ, but both v5 prompts and cached requests discard everything after 4000 chars.
        a='x'*4000+' Paris is the capital.';b='x'*4000+' Lyon is the capital.'
        observations['different_long_tails_same_judging_request']={
            'full_qa_sha_different':T.qa_sha('q',a)!=T.qa_sha('q',b),
            'v5_truth_prompt_equal':T.truth_user('q',['Paris'],a)==T.truth_user('q',['Paris'],b),
            'legacy_key_equal':J.qa_cache_key('m','q',['Paris'],a)==J.qa_cache_key('m','q',['Paris'],b),
            'unsafe_reproduced':T.qa_sha('q',a)!=T.qa_sha('q',b) and T.truth_user('q',['Paris'],a)==T.truth_user('q',['Paris'],b)}
        # Queue contract: a retried malformed answer becomes A, but pending still lists it.
        root=base/'legacy_pending';root.mkdir();path=root/J.cache_filename('m')
        j=J.LLMJudge('m',api_key='offline-placeholder',cache_path=str(path),sleep=0)
        stub=Stub(['not a verdict','A']);j._client=stub
        first=j.qa_grade('q',['a'],'a');second=j.qa_grade('q',['a'],'a');pending=H.pending(str(root))
        observations['resolved_malformed_reply_stays_pending']={'first':first['label'],'second':second['label'],
            'pending_count':len(pending),'unsafe_reproduced':second['label']=='correct' and len(pending)==1}
    out=Path(__file__).with_name('reproduced_findings.json')
    result={'snapshot':'cf8f752','network_calls':0,'scope':'synthetic isolated filesystem and stub clients; no production data changed',
            'files':{str(p.relative_to(REVIEW)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [CODE/'core/judge.py',CODE/'core/judge_two_axis.py',CODE/'core/qa_reference_grades.py',
                 CODE/'scripts/review_checks/judge_two_axis_run.py',CODE/'scripts/review_checks/judge_human_review.py']},
            'observations':observations}
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(observations,indent=2,ensure_ascii=False))
    assert all(r['unsafe_reproduced'] for r in observations.values())


if __name__=='__main__':main()
