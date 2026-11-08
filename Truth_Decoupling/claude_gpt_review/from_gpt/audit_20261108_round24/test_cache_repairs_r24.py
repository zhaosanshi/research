"""Shared-cache races and frozen-history boundary tests; all clients are offline stubs."""
import json
import multiprocessing
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

CODE=Path(__file__).resolve().parents[2]/'from_claude/code'
sys.path[:0]=[str(CODE),str(CODE/'scripts/review_checks')]
from core import judge as J, judge_two_axis as T
from core.judge_two_axis_contract import judge_contract, canonical_sha, request_identity
from core.judge_cache_store import cache_lock
from core.io_utils import append_jsonl
import judge2_cache_merge as merge


class Stub:
    def __init__(self, answer):
        self.answer=answer;self.calls=[];self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(model=kw['model'],choices=[SimpleNamespace(message=SimpleNamespace(content=self.answer))])


def worker(path, answer, ready, start, result, count):
    judge=J.LLMJudge('m',api_key='offline-only',cache_path=path,sleep=0,key_extra={'request':'same'})
    stub=Stub(answer)
    original=stub.create
    def counted(**kw):
        with count.get_lock():count.value+=1
        time.sleep(.03)
        return original(**kw)
    stub.chat.completions.create=counted;judge._client=stub
    ready.put(True);start.wait(5)
    result.put(judge.complete('system','user'))


class SharedCacheTests(unittest.TestCase):
    def judge(self, path, answer='T', **kwargs):
        obj=J.LLMJudge('m',api_key='offline-only',cache_path=str(path),sleep=0,**kwargs)
        obj._client=Stub(answer)
        return obj

    def test_two_initialized_instances_reload_before_cache_use(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'judge2_cache_m.jsonl'
            a=self.judge(path,'T',key_extra={'x':1});b=self.judge(path,'F',key_extra={'x':1})
            self.assertEqual(a.complete('s','u'),'T');self.assertEqual(b.complete('s','u'),'T')
            self.assertEqual(b._client.calls,[]);self.assertEqual(len(path.read_text().splitlines()),1)

    def test_identical_request_is_one_call_across_processes(self):
        ctx=multiprocessing.get_context('fork')
        with tempfile.TemporaryDirectory() as td:
            path=str(Path(td)/'judge2_cache_m.jsonl');ready=ctx.Queue();result=ctx.Queue();start=ctx.Event();count=ctx.Value('i',0)
            jobs=[ctx.Process(target=worker,args=(path,answer,ready,start,result,count)) for answer in ('T','F')]
            for job in jobs:job.start()
            try:
                for _ in jobs:ready.get(timeout=5)
                start.set();values=[result.get(timeout=5) for _ in jobs]
                for job in jobs:job.join(5);self.assertEqual(job.exitcode,0)
                self.assertEqual(len(set(values)),1);self.assertEqual(count.value,1)
                self.assertEqual(len(Path(path).read_text().splitlines()),1)
            finally:
                for job in jobs:
                    if job.is_alive():job.terminate();job.join()

    def test_bounded_retry_survives_new_instance_and_records_conflict(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'judge2_cache_m.jsonl';tag={'x':1}
            a=self.judge(path,'bad-one',key_extra=tag);self.assertEqual(a.complete('s','u'),'bad-one')
            b=self.judge(path,'bad-two',key_extra=tag)
            self.assertEqual(b.complete('s','u',retry_cached_if=lambda x:True),'bad-two')
            c=self.judge(path,'T',key_extra=tag)
            self.assertEqual(c.complete('s','u',retry_cached_if=lambda x:True),'bad-two');self.assertEqual(c._client.calls,[])
            self.assertEqual(c.cached_replies('s','u'),['bad-one','bad-two'])
            report=json.loads((Path(td)/'conflicts.json').read_text())
            self.assertIn(c.cache_key('s','u'),report['keys'][path.name])

    def test_merge_retains_append_during_source_read_and_distinct_observations(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);shared=root/'shared';source=root/'source';shared.mkdir();source.mkdir();name='judge2_cache_m.jsonl'
            out=shared/name;old={'key':'same','answer':'bad','observation_id':'one'};new={**old,'observation_id':'two'}
            out.write_text(json.dumps(old)+'\n');(source/name).write_text(json.dumps({'key':'new','answer':'T'})+'\n')
            original=merge.read_cache_rows;seen=[]
            def racing(path):
                if not seen:
                    with cache_lock(out):append_jsonl(new,str(out))
                    seen.append(True)
                return original(path)
            with patch.object(merge,'read_cache_rows',side_effect=racing):merge.merge(str(shared),[str(source)])
            rows=[json.loads(x) for x in out.read_text().splitlines()]
            self.assertIn(old,rows);self.assertIn(new,rows);self.assertEqual(len(rows),3)
            before=out.read_bytes();merge.merge(str(shared),[str(source)]);self.assertEqual(out.read_bytes(),before)

    def test_legacy_unknown_history_hits_but_known_parameter_change_fails(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'old.jsonl';key=J.legacy_qa_cache_key('m','q',['g'],'answer')
            old={'key':key,'model':'m','answer':'A'};path.write_text(json.dumps(old)+'\n')
            old_judge=self.judge(path,'B');self.assertEqual(old_judge.qa_grade('q',['g'],'answer')['label'],'correct')
            self.assertEqual(old_judge._client.calls,[])
            other=Path(td)/'known.jsonl';a=self.judge(other,'A',base_url='https://one.invalid',max_tokens=8)
            a.qa_grade('q',['g'],'answer')
            # Even a later human row without params must not erase the known API request evidence.
            append_jsonl({'key':key,'model':'m','answer':'A','source':'human'},str(other))
            b=self.judge(other,'B',base_url='https://two.invalid',max_tokens=1024)
            with self.assertRaisesRegex(ValueError,'parameter mismatch'):b.qa_grade('q',['g'],'answer')
            self.assertEqual(b._client.calls,[])

    def test_legacy_prefix_helpers_stay_frozen_but_full_answer_grade_is_rejected(self):
        x='x'*4000+' true tail';y='x'*4000+' false tail'
        self.assertEqual(J.qa_grade_user('q',['g'],x),J.qa_grade_user('q',['g'],y))
        with tempfile.TemporaryDirectory() as td:
            j=self.judge(Path(td)/'old.jsonl','A')
            with self.assertRaisesRegex(ValueError,'4000-character prefix'):j.qa_grade('q',['g'],x)
            self.assertEqual(j._client.calls,[])

    def test_parser_constants_change_row_identity_but_not_request_identity(self):
        cfg={'model':'m'};first=judge_contract('pro',cfg)
        with patch.object(T,'_SEP',re.compile(r'\s+')):second=judge_contract('pro',cfg)
        self.assertNotEqual(canonical_sha(first),canonical_sha(second))
        self.assertEqual(request_identity(first),request_identity(second))
        with patch.object(T,'_WORDS_INFO',(('SUBSTANTIVE','H'),)):third=judge_contract('pro',cfg)
        self.assertNotEqual(canonical_sha(first),canonical_sha(third))
        self.assertEqual(request_identity(first),request_identity(third))


if __name__=='__main__':unittest.main()
