import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CHECKS = ROOT / 'from_claude/code/scripts/review_checks'
RESULTS = ROOT / 'from_claude/results/03_E26同作答率_E27生成适配'


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, file); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


recount = module('r20_recount', CHECKS/'e26_independent_recount.py')
probe = module('r20_probe', CHECKS/'e26_c_dose_probe.py')


class RecountTests(unittest.TestCase):
    def test_exact_respects_question_grouping(self):
        a={str(i):'correct' for i in range(4)};b={str(i):'wrong' for i in range(4)}
        self.assertEqual(recount.exact(a,b,{str(i):str(i//2) for i in range(4)})['p_value'],.5)
        self.assertEqual(recount.exact(a,b,{str(i):str(i) for i in range(4)})['p_value'],.125)

    def test_quota_does_not_count_refusals_or_peek_at_correctness(self):
        rows=[{'id':str(i),'pair_id':str(i),'outcome':o,'score':3-i,'simple_score':3-i}
              for i,o in enumerate(['abstain','wrong','correct'])]
        q=recount.quota(rows,'0.67','score',42)
        self.assertEqual(q,{'0':'abstain','1':'wrong','2':'correct'})
        self.assertIsNone(recount.quota(rows,'1','score',42))

    def test_fraction_not_binary_rounding(self):
        rows=[{'id':str(i),'pair_id':str(i),'outcome':'correct','score':i,'simple_score':i} for i in range(100)]
        self.assertEqual(sum(v!='abstain' for v in recount.quota(rows,'.58','score',42).values()),58)

    @unittest.skipUnless((RESULTS/'E26_main/llama3.1-8b/report.json').exists(),'archived full E26 data not present')
    def test_actual_report_missing_comparison_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'run';shutil.copytree(RESULTS/'E26_main/llama3.1-8b',target)
            p=target/'report.json';value=json.loads(p.read_text());value['tables']['tf']['0.25']['quota']['coverages'][0]['comparisons'].pop()
            p.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'missing/duplicate/extra comparison'):recount.audit_run(target)

    @unittest.skipUnless((RESULTS/'E26_main/llama3.1-8b/report.json').exists(),'archived full E26 data not present')
    def test_actual_delivery_text_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'run';shutil.copytree(RESULTS/'E26_main/llama3.1-8b',target)
            p=target/'delivered_answers.json';value=json.loads(p.read_text());value['tf']['0.25']['baseline_simple'][0]['output']='changed'
            p.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'delivery output changed'):recount.audit_run(target)

    def test_primary_refuses_incomplete_family(self):
        p=subprocess.run(['python3',str(CHECKS/'e26_primary_analysis.py'),'/tmp/unused_r20_primary.json'],capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0);self.assertIn('both pre-declared models required',p.stderr)


class ProbeTests(unittest.TestCase):
    def test_spearman_averages_ties(self):
        self.assertAlmostEqual(probe.spearman([1,1,2],[1,2,3]),math.sqrt(.75))
        self.assertEqual(probe.average_ranks([3,1,1,2]).tolist(),[3.,.5,.5,2.])

    def test_degenerate_rank_is_null(self):
        self.assertIsNone(probe.spearman([1,1],[0,2]));self.assertIsNone(probe.spearman([1],[2]))

    def test_probe_rejects_invalid_parameters(self):
        for n,alphas in [(0,[1]),(-1,[1]),(3,[]),(3,[float('nan')]),(3,[1,1])]:
            with self.assertRaises(ValueError):probe.validate_probe(n,alphas)

    def test_probe_uses_production_batch_size_and_fresh_specs(self):
        seen=[];specs=[]
        def step(bundle,texts,s):
            seen.append(list(texts));return np.array([[int(t),0.] for t in texts])
        def interventions(*args):
            s=object();specs.append(s);return s
        fake=SimpleNamespace(gate_text=lambda b,m,t,r:str(r['i']),gate_map=lambda *a:{'yes':0,'no':1},
                             one_step=step,intervention_specs=interventions)
        got=probe.batched_margins(fake,None,{'config':{'batch_size':8}},{},'qa',[{'i':i} for i in range(17)],{})
        self.assertEqual([len(x) for x in seen],[8,8,1]);self.assertEqual(got.tolist(),list(range(17)))
        self.assertEqual(len({id(x) for x in specs}),3)


class LauncherTests(unittest.TestCase):
    def run_chain(self,fail=None,required=True):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);binpath=root/'bin';binpath.mkdir();records=root/'calls.jsonl'
            stub=binpath/'docker'
            stub.write_text('#!/usr/bin/env python3\nimport os,sys,json\nargs=sys.argv[1:]\nphase=args[args.index("scripts/e26_matched_coverage.py")+1]\nwith open(os.environ["RECORD"],"a") as f:f.write(json.dumps({"phase":phase,"args":args})+"\\n")\nraise SystemExit(7 if phase==os.environ.get("FAIL_PHASE") else 0)\n')
            stub.chmod(0o755)
            env=dict(os.environ,PATH=str(binpath)+os.pathsep+os.environ['PATH'],RECORD=str(records),
                     TD_PROJECT_ROOT='/container/project with space',TD_HOST_PROJECT_ROOT=str(root/'host project'))
            env.pop('CODE_PATH',None);env.pop('RESULTS_PATH',None)
            if required:env.update(CODE_PATH='/container/reviewed code',RESULTS_PATH='/container/project with space/new results')
            if fail:env['FAIL_PHASE']=fail
            p=subprocess.run(['bash',str(CHECKS/'e26_full_chain.sh'),'llama3.1-8b','3','test'],env=env,capture_output=True,text=True)
            calls=[json.loads(x) for x in records.read_text().splitlines()] if records.exists() else []
            return p,calls

    def test_chain_has_explicit_paths_and_correct_stage_order(self):
        p,calls=self.run_chain();self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual([c['phase'] for c in calls],['prepare','collect-tune','judge-tune','freeze','collect-test','judge-test','report'])
        args=calls[0]['args'];self.assertIn('/container/reviewed code',args)
        self.assertIn('HF_HOME=/container/project with space/hf_cache',args)
        self.assertFalse(any(a.startswith('HOME=') for a in args))
        self.assertIn('TD_DATA_DIR=/container/project with space/code/data/processed',args)

    def test_chain_stops_on_generation_failure(self):
        p,calls=self.run_chain('collect-tune');self.assertNotEqual(p.returncode,0)
        self.assertEqual([c['phase'] for c in calls],['prepare','collect-tune'])

    def test_chain_refuses_implicit_stale_code_path(self):
        p,calls=self.run_chain(required=False);self.assertNotEqual(p.returncode,0)
        self.assertEqual(calls,[]);self.assertIn('CODE_PATH',p.stderr)


extension = module('r20_extension', Path(__file__).with_name('joint_simple_extension.py'))


class ExtensionTests(unittest.TestCase):
    def fixture(self, root):
        manifest=root/'manifest.json';manifest.write_text('{}')
        cohort=[{'pair_id':str(i),'question':'q'+str(i),'gold':['answer']} for i in range(2)]
        spec={'name':'joint','kind':'EC'}
        mr={'model':'fake','bank_fingerprint':'bank','tasks':{'qa':{'missing_condition_specs':[spec]}}}
        rows=[{'id':str(i),'pair_id':str(i),'output':'answer','score':float(i),'simple_score':float(i),
               'outcome':'unresolved','finished':True,'edits':{'E':1,'C':1}} for i in range(2)]
        raw={'protocol':'E26_equal_simple_selector_extension_v1','exploratory':True,'model':'fake','bank_fingerprint':'bank',
             'extension_manifest_sha256':recount.digest(manifest),'completed':True,'pilot_limit':None,
             'cohort_sha256':recount.fingerprint(cohort),'condition_specs':[spec],'rows':{'joint':rows}}
        (root/'qa_test_extension.json').write_text(json.dumps(raw))
        graded=json.loads(json.dumps(raw));graded['generation_sha256']=recount.digest(root/'qa_test_extension.json')
        for r in graded['rows']['joint']:
            r.update(outcome='correct',gold_is_abstention=False,grade={'label':'correct','rubric_id':'qa_semantic_v3'})
        (root/'qa_test_extension_scored.json').write_text(json.dumps(graded))
        base=[dict(r) for r in graded['rows']['joint']]
        return manifest,mr,{'qa':{'test':cohort}},base

    def test_complete_extension_runs_and_keeps_rows(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);args=self.fixture(p);got=extension.load_extension(p,'qa',*args)
            self.assertEqual([r['id'] for r in got['joint']],['0','1'])

    def test_extension_rejects_pilot_and_missing_test_rows(self):
        for mutation in ('pilot','missing'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as td:
                p=Path(td);args=self.fixture(p);f=p/'qa_test_extension_scored.json';d=json.loads(f.read_text())
                if mutation=='pilot':d['pilot_limit']=1
                else:d['rows']['joint'].pop()
                f.write_text(json.dumps(d))
                with self.assertRaises(ValueError):extension.load_extension(p,'qa',*args)

    def test_identical_answer_cannot_receive_new_grade(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);args=self.fixture(p);f=p/'qa_test_extension_scored.json';d=json.loads(f.read_text())
            d['rows']['joint'][0]['outcome']='wrong';d['rows']['joint'][0]['grade']['label']='wrong';f.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'identical question/answer'):extension.load_extension(p,'qa',*args)

    def test_same_answer_scores_dont_change_under_joint_constant_shift(self):
        rows=[{'id':str(i),'pair_id':str(i),'outcome':'correct' if i<2 else 'wrong','score':float(i),'simple_score':float(i)} for i in range(4)]
        shifted=[{**r,'score':r['score']+100,'simple_score':r['simple_score']+100} for r in rows]
        self.assertEqual(recount.quota(rows,.5,'simple',42),recount.quota(shifted,.5,'simple',42))
        bs=extension.bootstrap_pair(rows,shifted,.5,42,n_boot=20)
        self.assertEqual(bs['ci95_accuracy'],[0.,0.]);self.assertEqual(bs['infeasible_resamples'],0)

if __name__=='__main__':unittest.main()
