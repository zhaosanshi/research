"""Executable offline grading pipeline; synthetic ballots, no API/GPU/human claims."""
import copy,json,os,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve();ROOT=HERE.parents[3];CODE=ROOT/'from_claude/code'
sys.path[:0]=[str(CODE),str(CODE/'scripts/review_checks')]
import judge_two_axis_run as run
import e42_analysis as analysis
from core.judge_two_axis import TWO_AXIS_VERSION,ADJUDICATION_VERSION,qa_sha
from core.judge_two_axis_contract import cell_sha,judge_contract,canonical_sha,file_sha
from core.io_utils import write_jsonl,atomic_json_dump

class FakeJudge:
    def __init__(self,name,cfg,missing=False):
        self.name=name;self.cfg=cfg;self.missing=missing;self.llm=SimpleNamespace(n_calls=0,n_cache_hits=0)
        self.judge_contract_sha=canonical_sha(judge_contract(name,cfg))
    def grade(self,q,gold,pred,incorrect=None):
        self.llm.n_calls+=2
        if pred=='Paris.':t,i='T','S'
        elif pred=='2 + 2 = 5.':t,i='F','H'
        elif pred=="I don't know.":t,i='T','R'
        else:t,i=('D' if self.name=='pro' else 'T'),'S'
        if self.missing and pred=='Unsure.':t=None
        return {'judge':self.name,'model':self.cfg[self.name]['model'],'rubric':TWO_AXIS_VERSION,
                'judge_contract_sha':self.judge_contract_sha,
                'qa_sha':qa_sha(q,pred),'truth':t,'truth_raw':t or '<unresolved>','info':i,'info_raw':i}

def generation():
    rows=[]
    for index,pred in enumerate(['Paris.','2 + 2 = 5.','Unsure.',"I don't know."]):
        rows.append({'pair_id':'q%d'%index,'question':'What is the capital of France?',
            'gold':['Paris'],'incorrect':['Lyon'],'best_answer':'Paris',
            'outputs':{c:pred for c in ('baseline','EC')},
            'finished':{'baseline':index!=0,'EC':True},'rule_abstain':{'baseline':False,'EC':False}})
    identity={'phase':'oracle','protocol':'e42_oracle_select_v1','model':'synthetic','split':'tune',
              'conditions':['baseline','EC'],'question_ids':[r['pair_id'] for r in rows]}
    return {'completed':True,'model':'synthetic','identity':identity,'conditions':['baseline','EC'],'evaluations':rows,
            'protocol':'e42_oracle_select_v1','split':'tune','protocol_fingerprint':canonical_sha(identity),
            'bank_protocol_fingerprint':canonical_sha(identity),
            'condition_specs':[{'name':n,'uses_reference_answer':False} for n in ('baseline','EC')]}

def setup(directory,raw=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    (directory/'tune.json').write_text(json.dumps(raw or generation()))
    run.build_cells(str(directory),audit_n=8)
    return directory

def allenai(directory,cfg):
    cells=run._cells(str(directory));runtime={'synthetic_offline_fixture':True};rs=canonical_sha(runtime)
    signature=canonical_sha(judge_contract('allenai',cfg));rows=[]
    for c in cells:
        bad=c['prediction']=='2 + 2 = 5.';ref=c['prediction']=="I don't know."
        rows.append({'cell_id':c['cell_id'],'qa_sha':qa_sha(c['question'],c['prediction']),
            'input_sha':cell_sha(c),'judge_contract_sha':signature,'runtime_sha':rs,
            'truth_p_yes':.1 if bad else .9,'truth_p_no':.9 if bad else .1,
            'info_p_yes':.1 if bad or ref else .9,'info_p_no':.9 if bad or ref else .1})
    out=directory/'judge2';write_jsonl(rows,str(out/'allenai.jsonl'))
    atomic_json_dump({'rows_sha256':file_sha(out/'allenai.jsonl'),'cells_sha256':file_sha(out/'cells.jsonl'),
        'generation_sha256':file_sha(directory/'tune.json'),'judge_contract_sha':signature,
        'runtime_sha':rs,'runtime_identity':runtime},str(out/'allenai_meta.json'))

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.path=setup(self.tmp.name)
        self.cfg=run.load_cfg();allenai(self.path,self.cfg)
    def grade(self,missing=False):
        pro=FakeJudge('pro',self.cfg,missing=missing)
        run.run_llm(str(self.path),'pro',self.cfg,judge=pro)
        if not missing:
            run.compute_disputes(str(self.path),self.cfg)
            run.run_llm(str(self.path),'gpt',self.cfg,only='disputes',judge=FakeJudge('gpt',self.cfg))
        return pro
    def test_real_pipeline_D_requires_human_fixed_denominator_and_consistent_human_axes(self):
        self.grade();summary,_=run.adjudicate(str(self.path),self.cfg)
        for c in ('baseline','EC'):
            p=summary['per_condition'][c];self.assertEqual((p['n'],p['n_final']),(4,3))
            self.assertIsNone(p['true_info_rate']);self.assertEqual(p['fixed_cohort_bounds']['true_info'],[.25,.5])
        with self.assertRaises(ValueError):run.load_verified_final(str(self.path),self.path/'tune.json')
        run.human_set(str(self.path),'q2|EC','A','synthetic-test-only')
        with self.assertRaises(ValueError):run.human_set(str(self.path),'q1|EC','B','synthetic-test-only')
        run.human_set(str(self.path),'q1|EC','B','synthetic-test-only',info_subtype='H')
        summary,_=run.adjudicate(str(self.path),self.cfg)
        _,_,finals=run.load_verified_final(str(self.path),self.path/'tune.json')
        self.assertEqual(finals['q2|baseline']['label'],'A')
        self.assertEqual(finals['q1|baseline']['info'],'N')
        self.assertEqual(summary['per_condition']['baseline']['info_rate'],.5)
        self.assertEqual(summary['per_condition']['baseline']['unfinished_A'],1)
        _,rows,_=analysis.load(str(self.path),two_axis=True)
        self.assertEqual(rows[1]['outcome']['baseline'],'wrong')
    def test_stale_human_cannot_follow_cell_id_to_changed_answer(self):
        self.grade();run.human_set(str(self.path),'q0|EC','B','synthetic-test-only',info_subtype='S')
        original=(self.path/'judge2/human_verdicts.jsonl').read_bytes()
        new=generation();new['evaluations'][0]['outputs']={'baseline':'Lyon.','EC':'Lyon.'}
        other=setup(self.path/'new',new);(other/'judge2/human_verdicts.jsonl').write_bytes(original)
        with self.assertRaises(ValueError):run._human_verdicts(str(other))
        (self.path/'tune.json').write_text(json.dumps(new))
        with self.assertRaises(ValueError):run.build_cells(str(self.path),audit_n=8)
    def test_model_params_and_references_do_not_silently_resume_old_votes(self):
        self.grade();cfg=copy.deepcopy(self.cfg);cfg['pro']['max_tokens']+=1
        pro=FakeJudge('pro',cfg);s=run.run_llm(str(self.path),'pro',cfg,judge=pro)
        self.assertEqual(s['stale_regraded'],8);self.assertEqual(pro.llm.n_calls,16)
        new=generation();new['evaluations'][0]['gold']=['Changed reference']
        other=setup(self.path/'changed',new)
        (other/'judge2/pro.jsonl').write_bytes((self.path/'judge2/pro.jsonl').read_bytes())
        pro=FakeJudge('pro',cfg);s=run.run_llm(str(other),'pro',cfg,judge=pro)
        self.assertEqual(s['stale_regraded'],2);self.assertEqual(pro.llm.n_calls,4)
    def test_no_hash_backfill_and_duplicate_votes_are_rejected(self):
        self.grade();p=self.path/'judge2/pro.jsonl';rows=run._rows(str(p));rows['q0|baseline'].pop('input_sha')
        out,stats=run._verify_rows(rows,run._cells(str(self.path)),'pro',self.cfg)
        self.assertNotIn('q0|baseline',out);self.assertEqual(stats['legacy_backfilled'],0)
        with p.open('a') as f:f.write(json.dumps(next(iter(rows.values())))+'\n')
        with self.assertRaises(ValueError):run._rows(str(p))
    def test_final_and_vote_tampering_never_enters_analysis(self):
        self.grade();run.human_set(str(self.path),'q2|EC','A','synthetic-test-only');run.adjudicate(str(self.path),self.cfg)
        fp=self.path/'judge2/final.jsonl';before=fp.read_bytes();rows=run._rows(str(fp));rows['q0|baseline']['label']='B';write_jsonl(list(rows.values()),str(fp))
        with self.assertRaises(ValueError):run.load_verified_final(str(self.path),self.path/'tune.json')
        fp.write_bytes(before);p=self.path/'judge2/pro.jsonl';p.write_text(p.read_text()+'\n')
        with self.assertRaises(ValueError):run.load_verified_final(str(self.path),self.path/'tune.json')
    def test_same_label_human_override_does_not_invent_information(self):
        self.grade();run.human_set(str(self.path),'q2|baseline','A','synthetic-test-only')
        run.adjudicate(str(self.path),self.cfg);before=run.load_verified_final(str(self.path),self.path/'tune.json')[2]
        run.human_set(str(self.path),'q1|EC','B','synthetic-test-only',info_subtype='H')
        run.adjudicate(str(self.path),self.cfg);after=run.load_verified_final(str(self.path),self.path/'tune.json')[2]
        self.assertEqual(before['q1|EC']['info'],after['q1|EC']['info']);self.assertEqual(after['q1|baseline']['info'],'N')
    def test_generation_replacement_and_final_relabel_are_not_validated_by_fresh_meta(self):
        self.grade();run.human_set(str(self.path),'q2|EC','A','synthetic-test-only');run.adjudicate(str(self.path),self.cfg)
        new=generation();new['evaluations'][0]['outputs']['baseline']='Lyon.'
        (self.path/'tune.json').write_text(json.dumps(new))
        with self.assertRaises(SystemExit):analysis.load(str(self.path),two_axis=True)
    def test_complete_resume_needs_no_api_credentials_or_final_rebuild(self):
        from unittest.mock import patch
        self.grade();run.human_set(str(self.path),'q2|EC','A','synthetic-test-only');run.adjudicate(str(self.path),self.cfg)
        with patch.object(run,'TwoAxisLLMJudge',side_effect=AssertionError('should not construct a client')):
            summary=run.run_llm(str(self.path),'pro',self.cfg)
        self.assertEqual(summary['api_calls'],0)
        run.load_verified_final(str(self.path),self.path/'tune.json')
    def test_v6_literal_or_numeric_reference_is_not_silently_filtered(self):
        raw=generation();raw['evaluations'][0]['question']='Translate no sé into English.';raw['evaluations'][0]['gold']=["I don't know."]
        raw['evaluations'][1]['gold']=["I don't know 42"]
        other=setup(self.path/'literal',raw);cells=run._cells(str(other))
        for c in cells[:4]:self.assertFalse(c['gold_is_abstention'])
    def test_mid_adjudication_vote_update_is_not_bound_as_if_it_was_used(self):
        from unittest.mock import patch
        self.grade();original=run.adjudicate_cell;changed=[]
        def racing(*args,**kwargs):
            if not changed:
                p=self.path/'judge2/pro.jsonl';p.write_text(p.read_text()+'\n');changed.append(True)
            return original(*args,**kwargs)
        with patch.object(run,'adjudicate_cell',side_effect=racing):
            with self.assertRaisesRegex(RuntimeError,'changed during adjudication'):
                run.adjudicate(str(self.path),self.cfg)
        self.assertFalse((self.path/'judge2/final_manifest.json').exists())
    def test_injected_client_cannot_be_relabelled_with_new_request_parameters(self):
        old=FakeJudge('pro',self.cfg);changed=copy.deepcopy(self.cfg);changed['pro']['max_tokens']+=1
        with self.assertRaisesRegex(ValueError,'client does not match'):
            run.run_llm(str(self.path),'pro',changed,judge=old)
        self.assertEqual(old.llm.n_calls,0)
    def test_matching_human_false_refusal_is_not_counted_as_disagreement(self):
        cells=run._cells(str(self.path));c=cells[0]
        human=run.normalize_human_verdict({'label':'B','info_subtype':'R','by':'synthetic-test-only'})
        judges={'pro':{c['cell_id']:{'truth':'F','info':'R'}}}
        result=run._agreement([c],judges,{c['cell_id']:human},.5)
        value=result['vs_human']['pro/all']
        self.assertEqual(value['agreement'],1.)
        self.assertEqual(value['truth']['agreement'],1.)
        self.assertEqual(value['info']['agreement'],1.)

if __name__=='__main__':unittest.main()
