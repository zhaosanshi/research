"""Real CLI return codes, harmless hostile pickle, and actual child-process waits."""
import copy,json,os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3]
CHECKS=ROOT/'from_claude/code/scripts/review_checks'
sys.path.insert(0,str(CHECKS))
import r19_compare_outputs as outputs
import r19_validation as vectors


def invoke(script,*args,env=None):
    return subprocess.run([sys.executable,str(CHECKS/script),*map(str,args)],capture_output=True,text=True,env=env)


def document(n):
    return {'completed':True,'model':'fixture','conditions':['baseline'],
            'evaluations':[{'pair_id':'q'+str(i),'idx':i,'question':'Question '+str(i)+'?', 'gold':['yes'],'incorrect':['no'],
                'outputs':{'baseline':'same text'},'finished':{'baseline':True},
                'prompt_token_ids':{'baseline':[1,i+2]},'generation_token_ids':{'baseline':[5,6]}} for i in range(n)]}


class OutputComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);p=Path(self.tmp.name)
        self.a=p/'a';self.b=p/'b';self.out=p/'report.json'
        for root in (self.a,self.b):
            root.mkdir()
            for name in outputs.TRACES:(root/name).write_text(json.dumps({'completed':True,'x':1}))
            for name in outputs.RUNS:
                run=root/name;run.mkdir(parents=True);(run/'tune.json').write_text(json.dumps(document(3 if name.endswith('pilot3') else 192)))
    def call(self,*flags):return invoke('r19_compare_outputs.py',self.a,self.b,self.out,*flags)
    def mutate(self,fn):
        p=self.a/outputs.RUNS[0]/'tune.json';d=json.loads(p.read_text());fn(d);p.write_text(json.dumps(d))
    def test_identical_complete_records_succeed(self):
        r=self.call();self.assertEqual(r.returncode,0,r.stderr);self.assertTrue(json.loads(self.out.read_text())['all_identical'])
    def test_duplicate_id_is_not_silently_replaced_by_last_dictionary_row(self):
        self.mutate(lambda d:d['evaluations'].append(copy.deepcopy(d['evaluations'][-1])))
        r=self.call();self.assertEqual(r.returncode,2);self.assertIn('duplicate pair_id',self.out.read_text())
    def test_identical_output_with_changed_question_or_tokens_fails(self):
        for key,value in [('question','DIFFERENT QUESTION'),('generation_token_ids',{'baseline':[9,6]}),('prompt_token_ids',{'baseline':[1,900]})]:
            with self.subTest(key=key):
                p=self.a/outputs.RUNS[0]/'tune.json';p.write_text(json.dumps(document(3)))
                self.mutate(lambda d:d['evaluations'][0].__setitem__(key,value))
                self.assertEqual(self.call().returncode,1)
    def test_missing_arm_incomplete_run_and_nonfinite_json_fail(self):
        for mutation in (lambda d:d['evaluations'][0]['finished'].clear(),lambda d:d.__setitem__('completed',False),
                         lambda d:d['evaluations'][0].__setitem__('score',float('nan'))):
            p=self.a/outputs.RUNS[0]/'tune.json';p.write_text(json.dumps(document(3)));self.mutate(mutation)
            self.assertEqual(self.call().returncode,2)
    def test_both_sides_same_wrong_size_still_fail(self):
        for root in (self.a,self.b):(root/outputs.RUNS[1]/'tune.json').write_text(json.dumps(document(3)))
        self.assertEqual(self.call().returncode,2)
    def test_missing_tokens_never_claim_all_identical(self):
        for root in (self.a,self.b):
            p=root/outputs.RUNS[0]/'tune.json';d=json.loads(p.read_text())
            for r in d['evaluations']:r.pop('generation_token_ids')
            p.write_text(json.dumps(d))
        self.assertEqual(self.call().returncode,2)
        r=self.call('--allow-missing-token-evidence');self.assertEqual(r.returncode,0)
        d=json.loads(self.out.read_text());self.assertFalse(d['all_identical']);self.assertTrue(d['all_recorded_fields_identical'])
    def test_context_hash_is_checked_not_only_compared_as_a_string(self):
        for root in (self.a,self.b):
            p=root/outputs.RUNS[-1]/'tune.json';d=json.loads(p.read_text());context=p.with_name('context.json');context.write_text(json.dumps(document(192)))
            d.update(generation_context_file='context.json',generation_context_sha256=outputs.sha(context));p.write_text(json.dumps(d))
        p=self.a/outputs.RUNS[-1]/'context.json';p.write_text(p.read_text()+' ')
        self.assertEqual(self.call().returncode,2)
    def test_row_order_and_duplicate_json_keys_are_checked(self):
        self.mutate(lambda d:d['evaluations'].reverse());self.assertEqual(self.call().returncode,1)
        (self.a/outputs.TRACES[0]).write_text('{"completed":true,"completed":true}')
        self.assertEqual(self.call().returncode,2)


class EvilMetadata:
    def __init__(self,marker):self.marker=marker
    def __reduce__(self):return eval,("open(%r,'w').write('UNSAFE LOAD EXECUTED')"%str(self.marker),)


class TensorCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)/'early';self.release=Path(self.tmp.name)/'release'
        for model in ('llama3.1-8b','qwen3-8b'):
            for run,alpha in (('ce_alpha1_ep3',1.),('ce_alpha3_ep3',3.)):
                for init in ('das','random','md'):
                    path=self.root/model/run/init/'adapter.pt';path.parent.mkdir(parents=True)
                    v=F.normalize(torch.randn(4,generator=torch.Generator().manual_seed(44)),dim=0) if init=='random' else F.normalize(torch.arange(1.,5.),dim=0)
                    a={'model':model,'hidden_size':4,'alpha':alpha,'initial_W':{'2':v},'W':{'2':F.normalize(v+.1,dim=0)}};torch.save(a,path)
                    if alpha==3:
                        dest=self.release/model/run/(init+'_s42')/'adapter.pt';dest.parent.mkdir(parents=True);torch.save(a,dest)
    def test_random_mismatch_changes_process_exit_status(self):
        r=invoke('r19_random_init_check.py',self.root);self.assertEqual(r.returncode,0,r.stderr)
        p=self.root/'llama3.1-8b/ce_alpha1_ep3/random/adapter.pt';a=torch.load(p,weights_only=True);a['initial_W']['2'][0]+=.01;torch.save(a,p)
        r=invoke('r19_random_init_check.py',self.root);self.assertEqual(r.returncode,1);self.assertIn('False',r.stdout)
    def test_zero_nan_and_infinite_directions_do_not_print_fake_angles(self):
        p=self.root/'llama3.1-8b/ce_alpha3_ep3/das/adapter.pt';saved=torch.load(p,weights_only=True)
        for v in (torch.zeros(4),torch.tensor([float('nan'),1,0,0]),torch.tensor([float('inf'),1,0,0])):
            a=copy.deepcopy(saved);a['W']['2']=v;torch.save(a,p)
            r=invoke('r19_angle_check.py',self.root);self.assertEqual(r.returncode,2);self.assertIn('nonfinite or zero',r.stderr)
    def test_restricted_loading_does_not_execute_harmless_pickle_payload(self):
        marker=Path(self.tmp.name)/'payload_was_executed';p=self.root/'llama3.1-8b/ce_alpha1_ep3/random/adapter.pt'
        a=torch.load(p,weights_only=True);a['extra']=EvilMetadata(marker);torch.save(a,p)
        r=invoke('r19_random_init_check.py',self.root);self.assertEqual(r.returncode,2);self.assertFalse(marker.exists());self.assertIn('restricted checkpoint',r.stderr)
    def test_adapter_difference_returns_nonzero_unless_report_only_requested(self):
        self.assertEqual(invoke('r19_compare_adapters.py',self.root,self.release).returncode,0)
        p=self.release/'llama3.1-8b/ce_alpha3_ep3/md_s42/adapter.pt';a=torch.load(p,weights_only=True);a['W']['2'][0]+=.01;torch.save(a,p)
        self.assertEqual(invoke('r19_compare_adapters.py',self.root,self.release).returncode,1)
        r=invoke('r19_compare_adapters.py',self.root,self.release,'--report-only');self.assertEqual(r.returncode,0);self.assertIn('NONIDENTICAL',r.stdout)
    def test_stable_angle_for_extreme_finite_scales(self):
        x=torch.tensor([1e300,0.],dtype=torch.float64);y=torch.tensor([0.,1e-300],dtype=torch.float64)
        self.assertEqual(vectors.cosine(x,y),0.)


class DriverProcessTests(unittest.TestCase):
    DRIVERS=(('r19_reproduce.sh','ALL',2),('r19_reproduce_full.sh','FULL',3),('r19_reproduce_llama.sh','LLAMA',2))
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        b=self.root/'bin';b.mkdir();stub=b/'docker'
        stub.write_text('#!/usr/bin/env python3\nimport json,os,sys,time\na=sys.argv[1:]\ng=next(x.split("=",1)[1] for x in a if x.startswith("CUDA_VISIBLE_DEVICES="))\ntime.sleep(.02 if g==os.environ.get("FAIL_GPU") else .16)\nwith open(os.environ["EVENTS"],"a") as f:f.write(json.dumps({"gpu":g,"args":a,"finished":True})+"\\n")\nraise SystemExit(7 if g==os.environ.get("FAIL_GPU") else 0)\n');stub.chmod(0o755)
        self.env=dict(os.environ,PATH=str(b)+os.pathsep+os.environ['PATH'],TD_PROJECT_ROOT='/container/project with spaces',
                      TD_HOST_PROJECT_ROOT=str(self.root/'host'),CODE_PATH='/container/review code',EVENTS=str(self.root/'events.jsonl'))
    def launch(self,script,tag,fail=None):
        env=dict(self.env,RESULTS_PATH='/container/project with spaces/'+tag)
        if fail is not None:env['FAIL_GPU']=str(fail)
        return subprocess.run(['/bin/bash',str(CHECKS/script)],capture_output=True,text=True,env=env),self.root/'host'/tag/'logs'
    def test_each_driver_waits_every_child_and_never_writes_done_on_any_failure(self):
        for script,group,n in self.DRIVERS:
            for failed in range(n):
                with self.subTest(script=script,failed=failed):
                    event=Path(self.env['EVENTS']);event.unlink(missing_ok=True)
                    r,logs=self.launch(script,group+str(failed),failed)
                    self.assertEqual(r.returncode,1,r.stderr);self.assertFalse((logs/(group+'.DONE')).exists());self.assertTrue((logs/(group+'.FAILED')).exists())
                    events=[json.loads(x) for x in event.read_text().splitlines()];self.assertEqual(len(events),n)
                    codes=[int(x.read_text()) for x in logs.glob('*.exit')];self.assertEqual(len(codes),n);self.assertEqual(codes.count(7),1)
                    self.assertFalse(list(logs.glob('*.RUNNING')))
    def test_success_done_and_existing_evidence_is_never_overwritten(self):
        for script,group,n in self.DRIVERS:
            with self.subTest(script=script):
                r,logs=self.launch(script,group+'_ok');self.assertEqual(r.returncode,0,r.stderr);self.assertTrue((logs/(group+'.DONE')).exists())
                before={p.name:p.read_bytes() for p in logs.iterdir() if p.is_file()}
                again,_=self.launch(script,group+'_ok');self.assertEqual(again.returncode,2)
                self.assertEqual(before,{p.name:p.read_bytes() for p in logs.iterdir() if p.is_file()})
                self.assertEqual([int(p.read_text()) for p in logs.glob('*.exit')],[0]*n)
        events=[json.loads(x) for x in Path(self.env['EVENTS']).read_text().splitlines()]
        self.assertTrue(all(any(v.startswith('HF_HOME=') for v in e['args']) and not any(v.startswith('HOME=') for v in e['args']) for e in events))
    def test_no_implicit_historical_output_directory(self):
        env=dict(self.env);env.pop('RESULTS_PATH',None);r=subprocess.run(['/bin/bash',str(CHECKS/'r19_reproduce.sh')],capture_output=True,text=True,env=env)
        self.assertNotEqual(r.returncode,0);self.assertIn('RESULTS_PATH',r.stderr);self.assertFalse(Path(self.env['EVENTS']).exists())

if __name__=='__main__':unittest.main()
