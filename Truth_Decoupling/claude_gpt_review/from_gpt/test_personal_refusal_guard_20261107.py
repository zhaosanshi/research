"""Complete-refusal policy, raw cache preservation and the real E25/E24 path."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent;CODE=HERE.parent/'from_claude/code'
sys.path[:0]=[str(HERE),str(CODE/'scripts'),str(CODE)]
from core.qa_scoring import (OUTCOME_POLICY,FULL_PERSONAL_REFUSAL_POLICY,full_personal_refusal_decision,
                             apply_full_personal_refusal_policy,reference_answers)
from core.judge import LLMJudge,qa_cache_key,QA_GRADER_VERSION
from core.qa_control_cache import seed_control_cache
import test_qa_das_pipeline_20261105 as fixture


class PersonalRefusalGuardTests(unittest.TestCase):
    def test_complete_phrases_only_including_explicit_variants(self):
        for text in ["I don't know.","I do not know!","I don’t know for sure.","I don't know the answer.",
                     "I do not know the answer for sure."," I have no comment. "]:
            with self.subTest(text=text):self.assertTrue(full_personal_refusal_decision('Are you a human?',text)['apply'])

    def test_mixed_numbers_unicode_and_quotes_are_not_stripped_into_refusals(self):
        texts=["I don't know 42.","I don't know 北京。","I don't know. 42","I don't know, but Paris.",
               "I don't know.\n北京",'"I don\'t know."',"`I don't know.`","> I don't know.",
               "No","Unknown","It is unknown.","I don't know for sure; the answer is 42.","I don't know\u200b北京"]
        for text in texts:
            with self.subTest(text=text):self.assertFalse(full_personal_refusal_decision('Question?',text)['apply'])

    def test_literal_language_questions_bypass_without_gold(self):
        for question in ["Translate 'Je ne sais pas' to English.","What phrase indicates ignorance?",
                         "What is the title of this work?", "Repeat the sentence I don't know.",
                         "把‘我不知道’翻译成英文。", "What is the name of the song?",
                         "What's the name of that book?", "Which film was released that year?",
                         "Name the song played at the end.", "What was that song called?", "这首歌的名字是什么？"]:
            d=full_personal_refusal_decision(question,"I don't know.")
            self.assertTrue(d['matched_complete_phrase']);self.assertFalse(d['apply']);self.assertIsNotNone(d['bypass_reason'])

    def test_policy_derivation_preserves_original_and_is_idempotent(self):
        old={'label':'correct','raw':'A','rubric_id':QA_GRADER_VERSION};before=deepcopy(old)
        new=apply_full_personal_refusal_policy(old,'Are you human?',"I don't know.")
        self.assertEqual(old,before);self.assertEqual(new['label'],'abstain');self.assertEqual(new['raw'],'A')
        self.assertEqual(new['semantic_label_before_policy'],'correct')
        self.assertEqual(new,apply_full_personal_refusal_policy(new,'Are you human?',"I don't know."))

    def test_existing_semantic_cache_stays_byte_identical_under_both_policies(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cache.jsonl';question='Are you human?';gold=['No',"I don't know for sure"]
            key=qa_cache_key('offline',question,gold,"I don't know.")
            path.write_text(json.dumps({'key':key,'model':'offline','answer':'A'})+'\n');old=path.read_bytes()
            legacy=LLMJudge('offline',api_key='offline-test-only',cache_path=str(path),sleep=0)
            self.assertEqual(legacy.qa_grade(question,gold,"I don't know."),{'label':'correct','raw':'A','rubric_id':QA_GRADER_VERSION})
            current=LLMJudge('offline',api_key='offline-test-only',cache_path=str(path),sleep=0,
                             full_personal_refusal_guard=True,outcome_policy=FULL_PERSONAL_REFUSAL_POLICY)
            with patch.object(current,'_get_client',side_effect=AssertionError('API forbidden')):
                result=current.qa_grade(question,gold,"I don't know.")
            self.assertEqual(result['label'],'abstain');self.assertEqual(result['raw'],'A')
            self.assertEqual(result['semantic_label_before_policy'],'correct');self.assertEqual(path.read_bytes(),old)
            self.assertEqual(current._cache[key],'A')

    def test_guard_without_cached_semantics_never_invents_a_raw_C_or_calls_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cache.jsonl'
            judge=LLMJudge('offline',api_key='offline-test-only',cache_path=str(path),full_personal_refusal_guard=True)
            with patch.object(judge,'complete',side_effect=AssertionError('API/cache write forbidden')):
                a=judge.qa_grade('Are you human?',['No'],"I don't know.")
                b=judge.qa_grade('Are you human?',['unrelated gold'],"I don't know.")
            self.assertEqual(a['label'],b['label']);self.assertEqual(a['label'],'abstain')
            self.assertEqual(a['raw'],'');self.assertEqual(a['semantic_label_before_policy'],'unresolved')
            self.assertFalse(path.exists())

    def test_literal_bypass_and_mixed_answer_still_use_semantic_judge(self):
        judge=LLMJudge('offline',api_key='offline-test-only',full_personal_refusal_guard=True)
        with patch.object(judge,'complete',return_value='A') as called:
            a=judge.qa_grade("Translate 'Je ne sais pas'.",['x'],"I don't know.")
            b=judge.qa_grade('Where?', ['Paris'], "I don't know for sure, but Paris.")
        self.assertEqual(called.call_count,2);self.assertEqual(a['label'],b['label']);self.assertEqual(a['label'],'correct')

    def test_guarded_control_cache_verifies_derived_label_not_raw_letter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'source';src.mkdir();path=src/'judge_cache_offline.jsonl'
            q='Are you human?';gold=['No'];text="I don't know.";key=qa_cache_key('offline',q,gold,text)
            path.write_text(json.dumps({'key':key,'model':'offline','answer':'A'})+'\n');before=path.read_bytes()
            grade=apply_full_personal_refusal_policy({'label':'correct','raw':'A','rubric_id':QA_GRADER_VERSION},q,text)
            row={'pair_id':'p','question':q,'judge_reference_answers':gold,'outputs':{'baseline':text},'judge':{'baseline':grade}}
            cfg={'model':'offline','full_personal_refusal_guard':True,'outcome_policy':FULL_PERSONAL_REFUSAL_POLICY}
            result=seed_control_cache(source_dir=src,run_dir=root/'target',judge_config=cfg,scored_rows=[row],
                                      conditions=['baseline'],scored_sha256='source-items-hash',stem='tune')
            self.assertEqual(path.read_bytes(),before)
            copied=json.loads((Path(result['judge_cache_dir'])/'judge_cache_offline.jsonl').read_text())
            self.assertEqual(copied['answer'],'A')
            with self.assertRaises(ValueError):
                seed_control_cache(source_dir=src,run_dir=root/'legacy',judge_config={'model':'offline'},scored_rows=[row],
                                   conditions=['baseline'],scored_sha256='source-items-hash',stem='tune')

    def test_old_reference_partition_is_unchanged(self):
        refs,only=reference_answers(["I don't know for sure",'No, I am an AI.'])
        self.assertEqual(refs,["I don't know for sure",'No, I am an AI.']);self.assertFalse(only)

    def test_future_config_records_policy_separately_from_semantic_rubric(self):
        cfg=json.loads((CODE/'configs/experiments/E4_qa.json').read_text())
        for key in ('judge','judge_second'):
            self.assertTrue(cfg[key]['full_personal_refusal_guard']);self.assertEqual(cfg[key]['outcome_policy'],FULL_PERSONAL_REFUSAL_POLICY)
        self.assertEqual(QA_GRADER_VERSION,'qa_semantic_v3')
        with self.assertRaises(ValueError):LLMJudge('offline',api_key='offline-test-only',full_personal_refusal_guard=True,outcome_policy=OUTCOME_POLICY)

    def test_actual_e25_to_e24_reused_grade_cannot_bypass_policy(self):
        f=fixture.QADASPipelineTests(methodName='test_real_end_to_end_training_selection_check_and_diagnostics')
        f.setUp();self.addCleanup(f.doCleanups);f.prepare_graded_tuning()
        old_path=f.run/'tune_scored_items.json';before=old_path.read_bytes();data=json.loads(before)
        first=data['evaluations'][0];first['outputs']['baseline']="I don't know."
        first['judge']['baseline']={'label':'correct','raw':'A','rubric_id':QA_GRADER_VERSION};first['outcome']['baseline']='correct'
        for guard in (False,True):
            directory=f.run/('new_guard' if guard else 'old_policy');directory.mkdir()
            fixture.pipeline.save(directory/'tune.json',data)
            args=SimpleNamespace(**vars(f.args));args.run_dir=directory
            judge=LLMJudge('offline',api_key='offline-test-only',full_personal_refusal_guard=guard)
            cfg={'model':'offline'}
            if guard:cfg.update(full_personal_refusal_guard=True,outcome_policy=FULL_PERSONAL_REFUSAL_POLICY)
            with patch.object(fixture.e24,'load_model_config',return_value={'name':f.args.model}), \
                 patch.object(fixture.e24,'load_experiment',return_value={'judge':cfg}), \
                 patch.object(fixture.e24,'out_paths',return_value=(str(directory),'unused','unused')), \
                 patch('core.judge.judge_from_config',return_value=judge), \
                 patch.object(judge,'complete',side_effect=AssertionError('existing grades should be derived, not re-requested')):
                fixture.pipeline.judge(args,f.meta,'tune.json','tune_scored')
            result=fixture.pipeline.read(directory/'tune_scored_items.json');g=result['evaluations'][0]['judge']['baseline']
            self.assertEqual(g['label'],'abstain' if guard else 'correct');self.assertEqual(g['raw'],'A')
            self.assertEqual(result['outcome_policy'],FULL_PERSONAL_REFUSAL_POLICY if guard else OUTCOME_POLICY)
        self.assertEqual(old_path.read_bytes(),before)


if __name__=='__main__':unittest.main()
