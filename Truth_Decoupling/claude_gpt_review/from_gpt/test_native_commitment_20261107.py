"""Fixed-E native-gate diagnostics: cohort safety and an executable toy-model run."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / 'from_claude/code'
sys.path[:0] = [str(HERE), str(CODE / 'scripts'), str(CODE)]
import e28_native_commitment as native
import test_e26_pipeline_20261106 as fixture


def row(i, score, outcome='correct', simple=0.):
    return {'id': str(i), 'pair_id': str(i), 'score': score, 'simple_score': simple,
            'outcome': outcome, 'output': 'answer', 'finished': True,
            'gold_is_abstention': False}


class NativePolicyTests(unittest.TestCase):
    def test_positive_gate_and_actual_refusal_both_required(self):
        rs = [row(1, 3, 'abstain'), row(2, 2), row(3, 0), row(4, -1, 'wrong')]
        out = native.native_delivery(rs)
        self.assertEqual((out['n'], out['answered'], out['correct']), (4, 1, 1))
        self.assertEqual(out['selected_ids'], ['2'])

    def test_empty_and_all_rejected_have_no_accuracy(self):
        for rs in ([], [row(1, -1)], [row(1, 1, 'abstain')]):
            out = native.native_delivery(rs)
            self.assertEqual(out['answered'], 0)
            self.assertIsNone(out['accuracy_answered'])

    def test_gate_outcome_not_used_to_replace_answer_outcome(self):
        answer = [row(1, -2, 'wrong')]
        gate = [row(1, 4, 'correct')]
        mixed = native.gate_only_rows(answer, gate)
        self.assertEqual((mixed[0]['score'], mixed[0]['outcome']), (4, 'wrong'))
        self.assertEqual(answer[0]['score'], -2)

    def test_gate_order_mismatch_is_rejected(self):
        rs = [row(1, 1), row(2, 2)]
        with self.assertRaises(ValueError):
            native.gate_only_rows(rs, rs[::-1])

    def test_gate_length_mismatch_is_rejected(self):
        rs = [row(1, 1), row(2, 2)]
        with self.assertRaises(ValueError):
            native.gate_only_rows(rs, rs[:1])

    def test_gate_group_mismatch_is_rejected(self):
        a, b = [row(1, 1)], [row(1, 1)]
        b[0]['pair_id'] = 'different'
        with self.assertRaises(ValueError):
            native.gate_only_rows(a, b)

    def test_dose_names_do_not_round_distinct_operators_together(self):
        a = native.condition('qa', 'das', {'layer': 2, 'alpha': 3}, 3, .1234567, 'EC')
        b = native.condition('qa', 'das', {'layer': 2, 'alpha': 3}, 3, .1234568, 'EC')
        self.assertNotEqual(a['name'], b['name'])
        self.assertFalse(native.same_operator(a, b))

    def test_fixed_k_controls_do_not_silently_serve_fewer_answers(self):
        baseline = [row(1, 1), row(2, 2, 'abstain'), row(3, -1, 'abstain')]
        ec = [row(1, 1), row(2, 1, 'wrong'), row(3, -1)]
        p = {'completed': True, 'plan': {}, 'tasks': {'qa': {'baseline': baseline, 'das_EC': ec}}}
        out = native.summarize(p)['tables']['qa']['das_EC']
        self.assertEqual(out['native']['answered'], 2)
        self.assertFalse(out['same_actual_count_controls']['baseline_simple_threshold']['feasible'])
        self.assertIsNone(out['same_actual_count_controls']['baseline_simple_threshold']['answered'])

    def test_refusal_only_reference_exclusion_is_shared(self):
        rs = [row(1, 1), row(2, 1)]
        rs[1]['gold_is_abstention'] = True
        p = {'completed': True, 'plan': {}, 'tasks': {'qa': {'baseline': rs, 'das_E': deepcopy(rs)}}}
        out = native.summarize(p)['tables']['qa']
        self.assertTrue(all(v['native']['n'] == 1 for v in out.values()))

    def test_same_k_fraction_does_not_lose_one_to_roundoff(self):
        rs = [row(i, 1 if i < 58 else -1, simple=-i) for i in range(100)]
        p = {'completed': True, 'plan': {}, 'tasks': {'qa': {'baseline': rs}}}
        out = native.summarize(p)['tables']['qa']['baseline']
        self.assertEqual(out['native']['answered'], 58)
        self.assertEqual(out['same_actual_count_controls']['baseline_simple_threshold']['answered'], 58)

    def test_unknown_or_reference_informed_outcomes_fail(self):
        with self.assertRaises(ValueError):
            native.native_delivery([row(1, 1, 'unresolved')])
        r = row(1, 1); r['uses_reference_answer'] = True
        with self.assertRaises(ValueError):
            native.native_delivery([r])

    def test_flipping_correct_wrong_keeps_eligibility_and_selected_ids(self):
        original = [row('a',4,'correct'),row('b',3,'wrong'),row('c',9,'abstain'),
                    row('d',2,'correct'),row('e',1,'wrong'),row('f',0,'abstain')]
        changed = [{**r,'outcome':{'correct':'wrong','wrong':'correct','abstain':'abstain'}[r['outcome']]} for r in original]
        a=native.quota_select(original,.5); b=native.quota_select(changed,.5)
        self.assertEqual(a['selected_ids'],b['selected_ids'])
        self.assertEqual(a['selected_ids'],['a','b','d'])
        self.assertEqual(a['answered'],b['answered'])
        self.assertNotEqual(a['correct'],b['correct'])


class NativeExecutionTests(unittest.TestCase):
    def test_collect_grade_report_real_toy_network_and_reject_bad_resume(self):
        f = fixture.E26IntegrationTests(methodName='test_complete_real_inference_judging_freeze_report_and_actual_deliveries')
        f.setUp()
        self.addCleanup(f.doCleanups)
        meta, bank, data = f.prepare()
        f.tune(meta, bank, data)
        args = SimpleNamespace(source_run=f.args.run_dir, run_dir=f.args.run_dir / 'native',
            coverage=.5, methods=['das', 'md'], alphas=[0., .25, .5], limit=2, cache_dir=None)
        raw = native.collect(args, meta, bank, data, bundle=f.bundle)
        self.assertTrue(raw['completed'])
        self.assertEqual(len(raw['tasks']['tf']['baseline']), 4)
        self.assertEqual(len(raw['tasks']['qa']['baseline']), 2)
        self.assertEqual(raw, native.collect(args, meta, bank, data, bundle=f.bundle))
        with patch.object(native, 'load_experiment', return_value={'judge': {}}):
            first = fixture.LocalJudge()
            grade = first.qa_grade
            def fail_after_ten(*a, **kw):
                if len(first.calls) == 10:
                    raise RuntimeError('simulated API interruption')
                return grade(*a, **kw)
            with patch.object(first, 'qa_grade', side_effect=fail_after_ten):
                with self.assertRaisesRegex(RuntimeError, 'API interruption'):
                    native.judge(args, data, judge=first)
            partial = native.joint.read(args.run_dir / 'native_scored.json')
            self.assertFalse(partial['completed'])
            self.assertEqual(sum(r['outcome'] == 'correct' for rs in partial['tasks']['qa'].values() for r in rs), 10)
            second = fixture.LocalJudge()
            with patch.object(native.joint, 'save', wraps=native.joint.save) as saving:
                scored = native.judge(args, data, judge=second)
            expected_grades = sum(len(rs) for rs in raw['tasks']['qa'].values())
            self.assertEqual(len(second.calls), expected_grades - 10)
            self.assertLess(saving.call_count, len(second.calls))
            self.assertTrue(scored['completed'])
            native.report(args, data)
            original = deepcopy(scored)
            bad = deepcopy(original)
            bad['tasks']['qa']['baseline'].pop()
            native.joint.save(args.run_dir / 'native_scored.json', bad)
            with self.assertRaisesRegex(ValueError, 'cohort'):
                native.report(args, data)
            bad = deepcopy(original)
            bad['tasks']['qa']['baseline'][0]['outcome'] = 'wrong'
            native.joint.save(args.run_dir / 'native_scored.json', bad)
            with self.assertRaisesRegex(ValueError, 'semantic grade'):
                native.report(args, data)
            bad = deepcopy(original)
            del bad['tasks']['tf']['baseline']
            native.joint.save(args.run_dir / 'native_scored.json', bad)
            with self.assertRaisesRegex(ValueError, 'arm set'):
                native.report(args, data)
            native.joint.save(args.run_dir / 'native_scored.json', original)
            native.report(args, data)


if __name__ == '__main__':
    unittest.main()
