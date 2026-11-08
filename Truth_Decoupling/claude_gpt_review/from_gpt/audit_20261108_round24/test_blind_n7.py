"""Meaningful offline blind-review and multiple-strength regression tests."""
import copy,json,os,sys,tempfile,unittest,subprocess
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve();REVIEW=HERE.parents[2];CODE=REVIEW/'from_claude/code'
sys.path[:0]=[str(CODE),str(CODE/'scripts/review_checks'),str(REVIEW/'from_gpt/audit_20261108_round22/pipeline')]
import judge2_blind_audit as B
import n7_curve_table as N
import test_two_axis_pipeline_r22 as F
from core.io_utils import write_jsonl

def raw(preds=('Paris.',)):
    r=F.generation();r['evaluations']=r['evaluations'][:len(preds)]
    for row,pred in zip(r['evaluations'],preds):row['outputs']={c:pred for c in r['conditions']}
    return r

class BlindTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        clean={k:v for k,v in os.environ.items() if k not in ('TD_JUDGE2_CACHE','TD_JUDGE2_HUMAN','TD_JUDGE2_CONFIG')}
        self.env=patch.dict(os.environ,clean,clear=True);self.env.start();self.addCleanup(self.env.stop)
        self.run=self.prepare(self.root/'run');self.out=self.root/'blind';self.cfg=F.run.load_cfg()
    def prepare(self,path,preds=('Paris.',)):
        run=F.setup(path,raw(preds));cfg=F.run.load_cfg();F.allenai(run,cfg)
        F.run.run_llm(str(run),'pro',cfg,judge=F.FakeJudge('pro',cfg));F.run.adjudicate(str(run),cfg)
        return run
    def sample(self,n=25):return B.sample([str(self.run)],str(self.out),'x',n,1,[])
    def key(self):return list(B.read_jsonl(str(self.out/'x_key.jsonl')))
    def test_actual_blind_workflow_preserves_axes_and_frozen_machine_baseline(self):
        self.assertEqual(self.sample()['n'],1)
        sheet=(self.out/'x_sheet.md').read_text();self.assertNotIn('baseline',sheet);self.assertNotIn('truth_p_yes',sheet)
        self.assertEqual(B.report(str(self.out),'x')['n_with_blind_verdict'],0)
        B.apply(str(self.out),'x','1B(H)','test-reviewer')
        F.run.adjudicate(str(self.run),self.cfg)
        report=B.report(str(self.out),'x')
        self.assertEqual(report['n_with_blind_verdict'],1);self.assertEqual(report['pipeline_before_human']['agreement'],0)
        h=F.run._human_verdicts(str(self.run))[self.key()[0]['cell_id']]
        self.assertEqual((h['label'],h['info'],h['info_subtype']),('B','N','H'))
        B.apply(str(self.out),'x','1B(H)','test-reviewer')  # safe resume
    def test_known_assisted_decision_is_excluded_not_blind(self):
        F.run.human_set(str(self.run),'q0|EC','A','assisted-test-only',note='seen votes')
        F.run.adjudicate(str(self.run),self.cfg)
        self.assertEqual(self.sample()['n'],0)
        self.assertEqual(B.report(str(self.out),'x')['n_with_blind_verdict'],0)
    def test_arbitration_added_after_export_is_not_counted_or_relabelled_blind(self):
        self.sample();F.run.human_set(str(self.run),'q0|EC','A','assisted-test-only')
        self.assertEqual(B.report(str(self.out),'x')['n_with_blind_verdict'],0)
        with self.assertRaisesRegex(ValueError,'prior human'):B.apply(str(self.out),'x','1A','blind-test-only')
        self.assertFalse((self.out/'x_verdicts.json').exists())
    def test_replaced_run_cannot_receive_old_sheet_verdict(self):
        self.sample();self.run.rename(self.root/'old');self.prepare(self.run,('Lyon.',))
        with self.assertRaises(ValueError):B.apply(str(self.out),'x','1A','test-reviewer')
        self.assertFalse((self.run/'judge2/human_verdicts.jsonl').exists())
    def test_last_moment_input_swap_is_rejected_at_human_write(self):
        self.sample();original=B.R.human_set
        def race(*args,**kwargs):
            self.run.rename(self.root/'old-race');self.prepare(self.run,('Lyon.',))
            return original(*args,**kwargs)
        with patch.object(B.R,'human_set',side_effect=race):
            with self.assertRaisesRegex(ValueError,'input|content|answer'):
                B.apply(str(self.out),'x','1A','test-reviewer')
        self.assertFalse((self.run/'judge2/human_verdicts.jsonl').exists())
        ledger=json.loads((self.out/'x_verdicts.json').read_text())
        self.assertEqual(ledger[0]['reviewed_input']['prediction'],'Paris.')
    def test_unknown_later_number_does_not_partially_apply_first(self):
        self.sample()
        with self.assertRaises(ValueError):B.apply(str(self.out),'x','1A 999C','test-reviewer')
        self.assertFalse((self.out/'x_verdicts.json').exists());self.assertFalse((self.run/'judge2/human_verdicts.jsonl').exists())
    def test_modified_sheet_or_hidden_key_is_rejected(self):
        self.sample();p=self.out/'x_key.jsonl';p.write_text(p.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'sheet/key'):B.apply(str(self.out),'x','1A','test-reviewer')
    def test_stale_machine_votes_cannot_enter_report(self):
        self.sample();p=self.run/'judge2/pro.jsonl';rows=F.run._rows(p)
        for row in rows.values():row['truth']='F';row['input_sha']='wrong'
        write_jsonl(list(rows.values()),str(p))
        with self.assertRaisesRegex(ValueError,'ballots changed'):B.report(str(self.out),'x')
    def test_old_unbound_packet_needs_explicit_new_export(self):
        self.out.mkdir();(self.out/'x_key.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'legacy/unbound'):B.apply(str(self.out),'x','1A','test-reviewer')
    def test_sample_does_not_overwrite_export_and_validates_exclusions(self):
        self.sample()
        with self.assertRaisesRegex(ValueError,'already exists'):self.sample()
        exclusion=self.root/'badexclude.json';exclusion.write_text('["q0"]')
        with self.assertRaises(ValueError):B.sample([str(self.run)],str(self.root/'other'),'x',1,1,[str(exclusion)])
        self.assertFalse((self.root/'other').exists())
    def test_different_reviewer_or_changed_label_cannot_overwrite_blind_record(self):
        self.sample();B.apply(str(self.out),'x','1A','reviewer-one')
        before=(self.out/'x_verdicts.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'already recorded'):B.apply(str(self.out),'x','1C','reviewer-two')
        self.assertEqual(before,(self.out/'x_verdicts.json').read_bytes())

class N7Tests(unittest.TestCase):
    def rows(self,p=.01):
        return [{'alpha':a,'E':'E_L11_a3','n_test':100,'baseline':{'K':100,'correct':70},'E_only':{'K':100,'correct':80},
                 'abst':{'P1':0,'P1_raw_p':1.,'P2':1,'P2_raw_p':p if a==.25 else 1.},
                 'ans':{'P1':0,'P1_raw_p':1.,'P2':0,'P2_raw_p':1.}} for a in N.DEFAULT_ALPHAS]
    def test_strength_winner_is_corrected_over_all_24_contrasts(self):
        a=self.rows();b=self.rows();v=N.verdict({'Llama':a,'Qwen':b})
        self.assertAlmostEqual(a[0]['abst']['P2_holm'],.24);self.assertFalse(v['criterion_met']);self.assertEqual(v['holm_tests_per_model'],24)
    def test_missing_strengths_are_incomplete_even_when_two_models_have_small_p(self):
        v=N.verdict({'Llama':self.rows(.00001)[:1],'Qwen':self.rows(.00001)[:1]})
        self.assertEqual(v['status'],'incomplete');self.assertIsNone(v['criterion_met'])
    def test_positive_E_comparison_never_certifies_any_single_direction(self):
        v=N.verdict({'Llama':self.rows(.00001),'Qwen':self.rows(.00001)})
        self.assertTrue(v['criterion_met']);self.assertFalse(v['full_any_single_direction_claim_established'])
    def test_cohort_E_or_undeclared_strength_changes_fail(self):
        rows=self.rows();rows[1]['E']='E_L12_a3'
        with self.assertRaises(ValueError):N.verdict({'Llama':rows})
        rows=self.rows();rows[1]['alpha']=.6
        with self.assertRaises(ValueError):N.verdict({'Llama':rows})
    def test_complete_analysis_files_run_through_real_table_cli(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for model in ('Llama','Qwen'):
                (root/model).mkdir()
                for alpha in N.DEFAULT_ALPHAS:
                    c0,c1='Cp_L25_abst_a%g'%alpha,'Cp_L26_ans_a%g'%alpha;e='E_L11_a3'
                    summaries={name:{'K':50,'correct_natural':40,'J':.5} for name in ('baseline',e,c0,c1,e+'+'+c0,e+'+'+c1)}
                    primary={}
                    for c in (c0,c1):
                        for typ,score in [('P1','gate'),('P2','conf')]:
                            p=.01 if alpha==.25 and c==c0 and typ=='P2' else 1.
                            primary[typ+' '+e+'+'+c+' vs E@'+score]={'diff':1 if p<1 else 0,'ci95':[-1,2],'boot_p':p,'holm_p':min(1,p*4)}
                    payload={'model':model,'fixed_C':{c0:{'sign':-1,'alpha':alpha},c1:{'sign':1,'alpha':alpha}},'chosen_E':e,'n_test':100,
                             'secondary':{'summaries':summaries},'primary':primary}
                    (root/model/('e41_n7_a%g_analysis.json'%alpha)).write_text(json.dumps(payload))
            proc=subprocess.run([sys.executable,str(CODE/'scripts/review_checks/n7_curve_table.py'),str(root),'--models','Llama','Qwen','--out',str(root/'table')],capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            output=json.loads((root/'table.json').read_text());self.assertFalse(output['verdict']['criterion_met'])
            self.assertAlmostEqual(output['rows']['Llama'][0]['abst']['P2_holm'],.24)
            self.assertIn('尚未在此检验同K的单C',proc.stdout)
            file=root/'Llama/e41_n7_a0.25_analysis.json';bad=json.loads(file.read_text());bad['fixed_C'][next(iter(bad['fixed_C']))]['alpha']=2.;file.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'filename strength'):N.load_model(str(root),'Llama')
    def test_no_results_are_not_a_negative_scientific_result(self):
        v=N.verdict({'Llama':[],'Qwen':[]});self.assertIsNone(v['criterion_met']);self.assertEqual(v['status'],'incomplete')

if __name__=='__main__':unittest.main(verbosity=2)
