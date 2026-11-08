"""Request cache, full-input, and explicit historical-read regression tests."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent;CODE=HERE.parent/'from_claude/code'
sys.path[:0]=[str(CODE),str(CODE/'scripts')]
from core import judge as J
from core import judge_two_axis as T
from core.judge_two_axis_contract import canonical_sha,cell_sha,request_contract,judge_contract,resolve_base_url,DEFAULT_OPENAI_BASE_URL
from core.qa_generation_adapter import fingerprint
from core.qa_reference_grades import reference_index,semantic_key,audit_historical_reference,verify_matching_grades


class Stub:
    def __init__(self,answer):
        self.answer=answer;self.calls=[];self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
    def create(self,**kw):
        self.calls.append(kw)
        return SimpleNamespace(model=kw['model'],choices=[SimpleNamespace(message=SimpleNamespace(content=self.answer))])


class ContractTests(unittest.TestCase):
    def test_cell_identity_is_full_input_and_condition_independent(self):
        cell={'question':'q','prediction':'a','gold':['g'],'gold_refs':['g'],'incorrect':['b'],
              'best_answer':'g','gold_is_abstention':False,'condition':'base','cell_id':'q|base'}
        initial=cell_sha(cell)
        self.assertEqual(initial,cell_sha({**cell,'condition':'EC','cell_id':'q|EC','audit':True}))
        for key,value in [('question','q2'),('prediction','a2'),('gold',['g2']),('gold_refs',['g2']),
                          ('incorrect',['b2']),('best_answer','g2'),('gold_is_abstention',True)]:
            self.assertNotEqual(initial,cell_sha({**cell,key:value}),key)

    def test_effective_request_not_secrets_or_inactive_options(self):
        cfg={'model':'m','max_completion_tokens':1024,'send_temperature':False}
        a=request_contract(cfg)
        self.assertEqual(a,request_contract({**cfg,'api_key':'not-a-real-secret','temperature':1.5,'max_tokens':999,'sleep':99}))
        for key,value in [('base_url','https://other.invalid'),('max_completion_tokens',2048),('reasoning_effort','high'),
                          ('send_temperature',True),('extra_body',{'thinking':{'type':'enabled'}})]:
            self.assertNotEqual(a,request_contract({**cfg,key:value}),key)
        self.assertNotIn('api_key',json.dumps(a))
        for change in ({'max_tokens':0},{'max_completion_tokens':False},{'send_temperature':'false'},{'temperature':float('nan')}):
            with self.assertRaises(ValueError):request_contract({'model':'m',**change})

    def test_environment_endpoint_changes_key_and_initialization_freezes_client(self):
        first='https://endpoint-a.example.invalid/v1';second='https://endpoint-b.example.invalid/v1'
        # request-bound (v2) keys are the two-axis judges' (key_extra set); a legacy judge keeps the historical key
        tag={'judge_contract_sha':'test'}
        with patch.dict(os.environ,{'OPENAI_BASE_URL':first}):
            a=J.LLMJudge('m',api_key='offline-only',key_extra=tag)
            key=a.cache_key('system','user')
            self.assertEqual(request_contract({'model':'m'})['base_url'],first)
        with patch.dict(os.environ,{'OPENAI_BASE_URL':second}):
            b=J.LLMJudge('m',api_key='offline-only',key_extra=tag)
            self.assertNotEqual(key,b.cache_key('system','user'))
            self.assertEqual(key,a.cache_key('system','user'))
            with patch('openai.OpenAI',return_value=object()) as factory:
                a._get_client()
            self.assertEqual(factory.call_args.kwargs['base_url'],first)
            self.assertEqual(a.effective_request()['base_url'],first)

    def test_explicit_endpoint_beats_environment_and_default_is_explicit(self):
        explicit='https://explicit.example.invalid/v1'
        with patch.dict(os.environ,{'OPENAI_BASE_URL':'https://environment.example.invalid/v1'}):
            self.assertEqual(resolve_base_url(explicit),explicit)
            self.assertEqual(request_contract({'model':'m','base_url':explicit})['base_url'],explicit)
        with patch.dict(os.environ):
            os.environ.pop('OPENAI_BASE_URL',None)
            self.assertEqual(resolve_base_url(),DEFAULT_OPENAI_BASE_URL)
            judge=J.LLMJudge('m',api_key='offline-only')
            with patch('openai.OpenAI',return_value=object()) as factory:judge._get_client()
            self.assertEqual(factory.call_args.kwargs['base_url'],DEFAULT_OPENAI_BASE_URL)

    def test_new_cache_separates_effective_budget_and_thinking(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'cache.jsonl';tag={'judge_contract_sha':'test'}   # request-bound keys: judges with key_extra (two-axis)
            a=J.LLMJudge('m',api_key='offline-only',cache_path=str(path),sleep=0,max_tokens=8,
                         extra_body={'thinking':{'type':'disabled'}},key_extra=tag)
            a._client=Stub('A');self.assertEqual(a.qa_grade('q',['g'],'a')['label'],'correct')
            b=J.LLMJudge('m',api_key='offline-only',cache_path=str(path),sleep=0,max_tokens=1024,
                         extra_body={'thinking':{'type':'enabled'}},key_extra=tag)
            b._client=Stub('B');self.assertEqual(b.qa_grade('q',['g'],'a')['label'],'wrong')
            self.assertEqual(len(b._client.calls),1)
            records=[json.loads(s) for s in path.read_text().splitlines()]
            self.assertNotEqual(records[0]['key'],records[1]['key'])
            self.assertEqual(records[1]['params']['max_tokens'],1024)
            self.assertEqual(records[1]['params']['extra_body'],{'thinking':{'type':'enabled'}})
            self.assertEqual(records[1]['cache_key_version'],J.CACHE_KEY_VERSION)
            c=J.LLMJudge('m',api_key='offline-only',cache_path=str(path),sleep=0,max_tokens=1024,
                         extra_body={'thinking':{'type':'enabled'}},key_extra=tag)
            c._client=Stub('A');self.assertEqual(c.qa_grade('q',['g'],'a')['label'],'wrong');self.assertEqual(c._client.calls,[])

    def test_legacy_cache_rows_still_hit_for_legacy_judges(self):
        # Claude, after the red team of GPT round 22 (blocker 1): the legacy single-letter judge keeps the historical
        # key sha1(model, system, user), so the E4/E24/E25/E26/E27/E28 caches and seed_control_cache() stay valid;
        # only a judge with key_extra (the two-axis judges) uses the request-bound v2 key.
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'cache.jsonl';key=J.legacy_qa_cache_key('m','q',['g'],'a')
            old={'key':key,'model':'m','answer':'A'};path.write_text(json.dumps(old)+'\n')
            j=J.LLMJudge('m',api_key='offline-only',cache_path=str(path),sleep=0);j._client=Stub('B')
            self.assertEqual(j.qa_grade('q',['g'],'a')['label'],'correct');self.assertEqual(j._client.calls,[])
            self.assertEqual(j.cache_key(J.QA_GRADER_SYSTEM,J.qa_grade_user('q',['g'],'a')),key)
            self.assertEqual(J.qa_cache_key('m','q',['g'],'a',config={'model':'m'}),key)
            tagged=J.LLMJudge('m',api_key='offline-only',cache_path=str(path),sleep=0,key_extra={'judge_contract_sha':'x'})
            self.assertNotEqual(tagged.cache_key(J.QA_GRADER_SYSTEM,J.qa_grade_user('q',['g'],'a')),key)
            self.assertEqual(tagged.cache_key_version(),J.CACHE_KEY_VERSION);self.assertNotEqual(j.cache_key_version(),J.CACHE_KEY_VERSION)

    def test_long_tails_change_prompt_api_key_and_reference_key(self):
        a='x'*4000+' correct tail';b='x'*4000+' false tail'
        # the legacy rubric prompt is frozen byte-for-byte, including its historical 4000-character clip
        self.assertEqual(J.qa_grade_user('q',['g'],a),J.qa_grade_user('q',['g'],b))
        self.assertEqual(J.qa_cache_key('m','q',['g'],a),J.qa_cache_key('m','q',['g'],b))
        self.assertEqual(J.legacy_qa_cache_key('m','q',['g'],a),J.legacy_qa_cache_key('m','q',['g'],b))
        row={'question':'q','judge_reference_answers':['g'],'incorrect':['b']}
        self.assertNotEqual(semantic_key({'model':'m'},row,a),semantic_key({'model':'m'},row,b))
        judge=J.LLMJudge('m',api_key='offline-only',sleep=0);judge._client=Stub('No')
        judge.abstains('x'*2100+' FINAL_SENTINEL')
        self.assertIn('FINAL_SENTINEL',judge._client.calls[0]['messages'][1]['content'])

    def test_judge_contract_binds_prompt_version_and_effective_fields(self):
        cfg={'model':'m','max_tokens':8};a=judge_contract('pro',cfg)
        self.assertEqual(a,judge_contract('pro',{'version':T.TWO_AXIS_VERSION,'pro':cfg}))
        self.assertEqual(a['prompts']['truth_system'],T.TRUTH_SYSTEM)
        self.assertNotEqual(canonical_sha(a),canonical_sha(judge_contract('pro',{**cfg,'max_tokens':16})))
        with self.assertRaises(ValueError):judge_contract('pro',{'version':'wrong-old-version','pro':cfg})
        gpu={'truth_model':'truth','info_model':'info','max_chars':None,'runtime_identity':{'tokenizer':'a'}}
        self.assertNotEqual(judge_contract('allenai',gpu),judge_contract('allenai',{**gpu,'runtime_identity':{'tokenizer':'b'}}))

    def test_our_reference_guard_rejects_all_new_semantic_request_fields(self):
        cfg={'name':'judge','model':'m','max_tokens':8,'max_completion_tokens':1024,'send_temperature':False,'reasoning_effort':'low'}
        raw,graded=self.fixture(cfg,J.QA_GRADER_VERSION)
        for key,value in [('max_completion_tokens',2048),('reasoning_effort','high'),('send_temperature',True)]:
            with self.assertRaisesRegex(ValueError,'requests differ'):
                reference_index(raw,graded,raw_sha256='raw',reference_config=cfg,target_config={**cfg,key:value})

    def test_v3_is_explicit_read_only_and_cannot_become_current_grade(self):
        cfg={'name':'old','model':'deepseek-chat','max_tokens':8};raw,graded=self.fixture(cfg,'qa_semantic_v3')
        audit=audit_historical_reference(raw,graded,raw_sha256='raw',recorded_config=cfg,expected_rubric='qa_semantic_v3')
        self.assertTrue(audit['read_only_historical']);self.assertFalse(audit['can_seed_new_judge'])
        self.assertEqual(audit['question_condition_grades_checked'],1)
        with self.assertRaisesRegex(ValueError,'rubric'):
            reference_index(raw,graded,raw_sha256='raw',reference_config=cfg,target_config=cfg)
        with self.assertRaisesRegex(ValueError,'historical'):
            verify_matching_grades(raw,graded,audit,raw_sha256='raw',target_config=cfg)

    @staticmethod
    def fixture(cfg,rubric):
        row={'pair_id':'q','question':'q','gold':['g'],'incorrect':['b'],'best_answer':'g','outputs':{'baseline':'a'},'finished':{'baseline':True}}
        raw={'completed':True,'model':'generator','identity':{},'conditions':['baseline'],'condition_specs':[{'name':'baseline'}],'evaluations':[row]}
        graded=copy.deepcopy(raw);graded.update(grading_completed=True,grading_source_generation_sha256='raw',
            grading_judge_config_sha256=fingerprint(cfg),grading_judge_name=cfg['name'])
        graded['evaluations'][0].update(judge_reference_answers=['g'],gold_is_abstention=False,
            judge={'baseline':{'label':'correct','raw':'A','rubric_id':rubric}},outcome={'baseline':'correct'})
        return raw,graded


if __name__=='__main__':unittest.main()
