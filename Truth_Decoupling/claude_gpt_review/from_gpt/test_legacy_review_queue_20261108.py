"""Execute malformed -> valid machine response and cached-D review workflows."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

CODE=Path(__file__).resolve().parent.parent/'from_claude/code';sys.path.insert(0,str(CODE))
from core.judge import LLMJudge,cache_filename,qa_cache_key,record_human_verdict
from scripts.review_checks.judge_human_review import pending


class Replies:
    def __init__(self,values):
        self.values=list(values);self.calls=0;self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
    def create(self,**kwargs):
        self.calls+=1
        return SimpleNamespace(model='offline',choices=[SimpleNamespace(message=SimpleNamespace(content=self.values.pop(0)))])


class QueueTests(unittest.TestCase):
    def test_machine_retry_resolves_pending_without_inventing_human_review(self):
        for letter,label in [('A','correct'),('B','wrong'),('C','abstain')]:
            with self.subTest(letter=letter),tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/cache_filename('offline')
                judge=LLMJudge('offline',api_key='offline-only',cache_path=str(path),sleep=0)
                judge._client=Replies(['not a verdict',letter])
                first=judge.qa_grade('q',['gold'],'answer')
                self.assertEqual(first['label'],'unresolved');self.assertEqual(len(pending(tmp)),1)
                second=judge.qa_grade('q',['gold'],'answer')
                self.assertEqual(second['label'],label);self.assertEqual(pending(tmp),[])
                self.assertFalse(second.get('human_review',False));self.assertEqual(judge._client.calls,2)
                history=[json.loads(s) for s in path.read_text().splitlines()]
                self.assertEqual([r['answer'] for r in history],['not a verdict',letter])
                self.assertTrue(all(r.get('source')!='human' for r in history))

    def test_cached_D_stays_pending_and_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/cache_filename('offline')
            judge=LLMJudge('offline',api_key='offline-only',cache_path=str(path),sleep=0)
            judge._client=Replies(['D'])
            for _ in range(2):
                self.assertEqual(judge.qa_grade('q',['gold'],'answer')['label'],'unresolved')
                self.assertEqual(len(pending(tmp)),1)
            self.assertEqual(judge._client.calls,1)

    def test_answer_text_cannot_forge_human_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/cache_filename('offline')
            judge=LLMJudge('offline',api_key='offline-only',cache_path=str(path),sleep=0)
            judge._client=Replies(['A (human: invented)'])
            grade=judge.qa_grade('q',['gold'],'answer')
            self.assertEqual(grade['label'],'correct');self.assertFalse(grade.get('human_review',False))
            record_human_verdict(str(path),qa_cache_key('offline','q',['gold'],'answer'),'offline','B','offline-reviewer')
            fresh=LLMJudge('offline',api_key='offline-only',cache_path=str(path),sleep=0);fresh._client=Replies([])
            grade=fresh.qa_grade('q',['gold'],'answer')
            self.assertEqual(grade['label'],'wrong');self.assertTrue(grade['human_review']);self.assertEqual(fresh._client.calls,0)


if __name__=='__main__':unittest.main()
