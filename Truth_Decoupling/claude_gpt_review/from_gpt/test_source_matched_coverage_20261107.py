"""Three-question scoring through an actual toy network; no network/API calls."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
CODE=HERE.parent/'from_claude/code'
sys.path[:0]=[str(HERE),str(CODE/'scripts'),str(CODE)]
import e32_source_matched_coverage as matched
import test_qa_das_pipeline_20261105 as fixture
from core.judge import QA_GRADER_VERSION


class MatchedSourceTests(unittest.TestCase):
    def setUp(self):
        self.f=fixture.QADASPipelineTests(methodName='test_real_end_to_end_training_selection_check_and_diagnostics')
        self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.args=SimpleNamespace(source_run=self.f.run,run_dir=self.f.run/'matched',stem='check',n_boot=8)
        self.args.run_dir.mkdir()
        self.names=matched.CONTROL_NAMES
        identity={'protocol':matched.SOURCE_PROTOCOLS[0],
                  'source':matched.source_loader.source_identity(self.f.args,self.f.meta,[0]),
                  'layer':0,'gold_used_at_inference':False}
        self.raw={'protocol':matched.SOURCE_PROTOCOLS[0],'identity':identity,'model':self.f.args.model,
                  'split':'tune','completed':True,'pilot':True,'wiring_pilot':True,
                  'conditions':self.names,'condition_specs':[{'name':n,'uses_reference_answer':False} for n in self.names],
                  'protocol_fingerprint':matched.qa.fingerprint(identity),'bank_protocol_fingerprint':matched.qa.fingerprint(identity),
                  'evaluations':[]}
        self.expected_calls=0
        for i,q in enumerate(self.f.prepared['splits']['tune']):
            outputs={n:['Paris','I do not know.','Rome'][i] for n in self.names}
            if i==0:outputs['source_only']='London'
            if i==1:outputs['source_only']=''
            self.expected_calls+=len({v for v in outputs.values() if v.strip()})
            prompt=self.f.bundle.build_chat(self.f.cfg['qa_system'],self.f.cfg['qa_user'].format(question=q['question']))
            tokens=self.f.bundle.encode(prompt)['input_ids'][0].tolist()
            self.raw['evaluations'].append({**deepcopy(q),'idx':i,'outputs':outputs,'finished':{n:True for n in self.names},
                'prompt_token_ids':{'base':tokens}})
        self.raw_path=self.args.run_dir/'check.json';matched.qa.save(self.raw_path,self.raw)
        self.graded=deepcopy(self.raw)
        self.graded.update(grading_completed=True,grading_source_generation_sha256=matched.qa.digest(self.raw_path))
        for i,row in enumerate(self.graded['evaluations']):
            refs,only=matched.reference_answers(row['gold']);row['judge_reference_answers']=refs;row['gold_is_abstention']=only
            labels={n:('abstain' if i==1 else 'correct' if i==0 and row['outputs'][n]=='Paris' else 'wrong') for n in self.names}
            row['outcome']=labels;row['judge']={n:{'label':v,'rubric_id':QA_GRADER_VERSION,'raw':'synthetic'} for n,v in labels.items()}
        self.grade_path=self.args.run_dir/'check_scored_items.json';matched.qa.save(self.grade_path,self.graded)

    def score(self):
        return matched.score(self.args,bundle=self.f.bundle)

    def test_actual_three_question_score_report_deduplicates_and_never_reads_gold_into_score(self):
        before={p.name:matched.qa.digest(p) for p in (self.raw_path,self.grade_path)}
        old_calls=self.f.model.forward_calls
        encode=matched.encode_answer
        with patch.object(matched,'encode_answer',wraps=encode) as encoding:
            scored=self.score()
        self.assertEqual(self.f.model.forward_calls-old_calls,self.expected_calls)
        expected={(r['question'],v) for r in self.raw['evaluations'] for v in r['outputs'].values() if v.strip()}
        self.assertEqual({(call.args[2],call.args[3]) for call in encoding.call_args_list},expected)
        self.assertTrue(all(len(call.args)==5 and not call.kwargs for call in encoding.call_args_list))
        self.assertTrue(scored['completed'])
        self.assertEqual(scored,self.score())
        report=matched.report(self.args)
        self.assertEqual(report['natural']['baseline'],{'n':3,'correct':1,'answered':2})
        seventyfive=next(r for r in report['matched_coverage']['coverages'] if r['target_coverage']==.75)
        self.assertEqual(seventyfive['arms']['baseline']['answered'],2)
        self.assertEqual(seventyfive['arms']['source_only']['correct'],0)
        self.assertEqual(before,{p.name:matched.qa.digest(p) for p in (self.raw_path,self.grade_path)})

    def test_missing_raw_question_and_reference_informed_condition_are_rejected(self):
        bad=deepcopy(self.raw);bad['evaluations'].pop();matched.qa.save(self.raw_path,bad)
        with self.assertRaises(ValueError):matched.inputs(self.args)
        bad=deepcopy(self.raw);bad['condition_specs'][0]['uses_reference_answer']=True;matched.qa.save(self.raw_path,bad)
        with self.assertRaisesRegex(ValueError,'reference-informed'):matched.inputs(self.args)

    def test_different_score_model_or_source_cannot_enter_report(self):
        scored=self.score();path=self.args.run_dir/'check_simple_scores.json'
        for key,value in [('model','another-model'),('source',{'foreign':True}),('uses_gold_in_score',True)]:
            bad=deepcopy(scored);bad['identity'][key]=value;matched.qa.save(path,bad)
            with self.assertRaisesRegex(ValueError,'foreign'):matched.report(self.args)
        matched.qa.save(path,scored)

    def test_different_rubric_or_reference_cannot_enter_report(self):
        self.score()
        for kind in ('rubric','references'):
            bad=deepcopy(self.graded)
            if kind=='rubric':bad['evaluations'][0]['judge']['baseline']['rubric_id']='obsolete-rubric'
            else:bad['evaluations'][0]['judge_reference_answers']=['different reference']
            matched.qa.save(self.grade_path,bad)
            with self.assertRaises(ValueError):matched.report(self.args)

    def test_identical_answer_with_conflicting_score_is_rejected_on_resume(self):
        scored=self.score();path=self.args.run_dir/'check_simple_scores.json'
        scored['rows'][0]['scores']['source_random_swap']+=1
        matched.qa.save(path,scored)
        with self.assertRaisesRegex(ValueError,'different scores'):self.score()

    def test_identical_answer_with_conflicting_grade_is_rejected(self):
        self.score();bad=deepcopy(self.graded)
        bad['evaluations'][0]['outcome']['source_random_swap']='wrong'
        bad['evaluations'][0]['judge']['source_random_swap']['label']='wrong'
        matched.qa.save(self.grade_path,bad)
        with self.assertRaisesRegex(ValueError,'different score or grade'):matched.report(self.args)

    def test_prompt_mismatch_stops_actual_model_scoring(self):
        bad=deepcopy(self.raw);bad['evaluations'][0]['prompt_token_ids']['base'].append(8)
        matched.qa.save(self.raw_path,bad)
        with self.assertRaisesRegex(ValueError,'scoring prompt differs'):self.score()

    def test_seven_arm_preference_executes_and_rejects_foreign_parent_or_extra_arm(self):
        ce = 'trained_source_swap_das_init'
        preference = 'pair_preference_source_swap'
        parent = {**deepcopy(self.raw['identity']), 'protocol': matched.SOURCE_PROTOCOLS[1],
                  'model': self.raw['model'], 'condition': ce,
                  'config': {k:self.f.cfg[k] for k in ('qa_system','qa_user','max_new_tokens','batch_size')},
                  'question_ids': [r['pair_id'] for r in self.raw['evaluations']],
                  'alpha': 1., 'same_prefix_tokens': True}
        raw = deepcopy(self.raw)
        raw['control_identity'] = parent
        raw['protocol'] = matched.SOURCE_PROTOCOLS[2]
        raw['identity'] = {**deepcopy(parent), 'protocol': raw['protocol'], 'condition': preference,
                           'control_generation_sha256': 'a'*64}
        raw['conditions'] = list(self.names) + [ce, preference]
        raw['condition_specs'] += [{'name':name,'uses_reference_answer':False} for name in (ce,preference)]
        raw['protocol_fingerprint'] = raw['bank_protocol_fingerprint'] = matched.qa.fingerprint(raw['identity'])
        for row in raw['evaluations']:
            for name in (ce,preference):
                row['outputs'][name] = row['outputs']['baseline']; row['finished'][name] = row['finished']['baseline']
        matched.qa.save(self.raw_path,raw)
        graded = deepcopy(raw)
        graded.update(grading_completed=True, grading_source_generation_sha256=matched.qa.digest(self.raw_path))
        for row, original in zip(graded['evaluations'],self.graded['evaluations']):
            for k in ('outcome','judge','judge_reference_answers','gold_is_abstention'):
                row[k] = deepcopy(original[k])
            for name in (ce,preference):
                row['outcome'][name] = row['outcome']['baseline']; row['judge'][name] = deepcopy(row['judge']['baseline'])
        matched.qa.save(self.grade_path,graded)
        self.score(); result = matched.report(self.args)
        self.assertEqual(result['natural'][preference],result['natural']['baseline'])
        self.assertEqual(len(result['matched_coverage']['coverages'][0]['arms']),7)
        bad = deepcopy(raw);bad['control_identity']['layer']=1;matched.qa.save(self.raw_path,bad)
        with self.assertRaisesRegex(ValueError,'matching frozen six-arm parent'):matched.inputs(self.args)
        bad = deepcopy(raw);bad['conditions'][5]='reference_oracle';matched.qa.save(self.raw_path,bad)
        with self.assertRaises(ValueError):matched.inputs(self.args)

    def test_pilot_parent_may_be_full_but_never_an_arbitrary_subset(self):
        full = ['q0', 'q1', 'q2', 'q3', 'q4']
        self.assertTrue(matched.parent_cohort_matches(full, full[:3], full, True))
        self.assertTrue(matched.parent_cohort_matches(full[:3], full[:3], full, True))
        self.assertFalse(matched.parent_cohort_matches(full, full[1:4], full, True))
        self.assertFalse(matched.parent_cohort_matches(full[:4], full[:3], full, True))
        self.assertFalse(matched.parent_cohort_matches(full[:3], full, full, False))
        self.assertTrue(matched.parent_cohort_matches(full, full, full, False))

    def test_extended_controls_require_frozen_parent_and_method_specific_provenance(self):
        ids = [r['pair_id'] for r in self.raw['evaluations']]
        parent = {**deepcopy(self.raw['identity']), 'protocol':matched.SOURCE_PROTOCOLS[2],
                  'model':self.raw['model'], 'question_ids':ids, 'alpha':1., 'same_prefix_tokens':True,
                  'config':{k:self.f.cfg[k] for k in ('qa_system','qa_user','max_new_tokens','batch_size')},
                  'source_system':'truthful source', 'adapter_sha256':'a'*64}
        for protocol in matched.SOURCE_PROTOCOLS[3:]:
            p = deepcopy(parent)
            if protocol == matched.SOURCE_PROTOCOLS[6]:p['protocol']=matched.SOURCE_PROTOCOLS[3]
            identity = {**deepcopy(parent), 'protocol':protocol, 'control_generation_sha256':'b'*64,
                'parent_generation_sha256':'b'*64, 'parent_scored_sha256':'c'*64,
                'masked_parent_generation_sha256':'d'*64,'masked_parent_grade_sha256':'e'*64,
                'control_grade_sha256':'f'*64, 'condition':matched.MASKED_NAME,
                'source_mode':matched.MASK_MODE, 'source_mask_manifest_sha256':'e'*64,
                'seeds':[42,43,44], 'training_called':False, 'target_adapter_sha256':'a'*64,
                'learned_information_borrowed':'scalar write norm only',
                'budget_override':{'parent_max_new_tokens':self.f.cfg['max_new_tokens'],'max_new_tokens':256,
                    'only_python_generation_loop_limit_changes':True,'only_generation_loop_limit_changes':True},
                'no_forced_eos':True,'no_text_trimming':True,'all_questions_retained':True}
            raw = {'protocol':protocol, 'identity':identity,'control_identity':p}
            if protocol==matched.SOURCE_PROTOCOLS[6]:
                raw['masked_parent_identity']={**deepcopy(identity),'protocol':matched.SOURCE_PROTOCOLS[4]}
            names = matched.extended_source_names(raw,self.f.meta,ids,ids,True)
            self.assertEqual(names[:7],matched.PREFERENCE_NAMES)
            self.assertEqual(len(names),10 if protocol==matched.SOURCE_PROTOCOLS[5] else
                             8 if protocol in (matched.SOURCE_PROTOCOLS[4],matched.SOURCE_PROTOCOLS[6]) else 7)
            bad = deepcopy(raw);bad['control_identity']['source']={'foreign':True}
            with self.assertRaisesRegex(ValueError,'matching frozen parent'):
                matched.extended_source_names(bad,self.f.meta,ids,ids,True)
            bad = deepcopy(raw)
            if protocol==matched.SOURCE_PROTOCOLS[5]:bad['identity']['training_called']=True
            elif protocol in (matched.SOURCE_PROTOCOLS[4],matched.SOURCE_PROTOCOLS[6]):bad['identity']['source_mask_manifest_sha256']='missing'
            else:bad['identity']['no_forced_eos']=False
            with self.assertRaises(ValueError):matched.extended_source_names(bad,self.f.meta,ids,ids,True)
            if protocol==matched.SOURCE_PROTOCOLS[6]:
                bad=deepcopy(raw);bad['masked_parent_identity']['adapter_sha256']='f'*64
                with self.assertRaisesRegex(ValueError,'48-token parent'):
                    matched.extended_source_names(bad,self.f.meta,ids,ids,True)


if __name__=='__main__':unittest.main()
