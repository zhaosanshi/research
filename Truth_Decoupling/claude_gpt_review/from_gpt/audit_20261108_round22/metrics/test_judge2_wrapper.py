import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[3]
SCRIPT=ROOT/'from_claude/code/scripts/review_checks/judge2_run.sh'

class WrapperTests(unittest.TestCase):
 def run_wrapper(self,args,code=0):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);out=d/'args.json';exe=d/'docker'
   exe.write_text('#!'+sys.executable+'\nimport json,os,sys\njson.dump(sys.argv[1:],open(os.environ["CAPTURE_ARGS"],"w"))\nsys.exit(int(os.environ["MOCK_EXIT"]))\n');exe.chmod(0o755)
   env=dict(os.environ,PATH=str(d)+os.pathsep+os.environ.get('PATH',''),CODE_PATH='/code snapshot',CAPTURE_ARGS=str(out),MOCK_EXIT=str(code),GPU='')
   proc=subprocess.run(['/bin/bash',str(SCRIPT),*args],env=env,capture_output=True,text=True)
   return proc,json.loads(out.read_text()) if out.exists() else None
 def test_config_precedes_subcommand_and_spaces_preserved(self):
  proc,argv=self.run_wrapper(['/results with space','pro','--config','/config with space.json','--only','disputes'])
  self.assertEqual(proc.returncode,0,proc.stderr)
  p=argv.index('scripts/review_checks/judge_two_axis_run.py')
  self.assertEqual(argv[p+1:p+6],['--config','/config with space.json','llm','/results with space','--judge'])
  self.assertIn('HF_HOME=/workspace/Truth_Decoupling/hf_cache',argv)
  self.assertFalse(any(x.startswith('HOME=') for x in argv));self.assertIn('CUDA_VISIBLE_DEVICES=',argv)
 def test_empty_extra_args_work_under_mac_bash_and_child_failure_propagates(self):
  p,argv=self.run_wrapper(['/r','adjudicate'],19);self.assertEqual(p.returncode,19,p.stderr);self.assertIsNotNone(argv)
 def test_e42_wrapper_uses_requested_snapshot_and_propagates_failure(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);out=d/'args.json';exe=d/'docker'
   exe.write_text('#!'+sys.executable+'\nimport json,os,sys\njson.dump(sys.argv[1:],open(os.environ["CAPTURE_ARGS"],"w"))\nsys.exit(23)\n');exe.chmod(0o755)
   env=dict(os.environ,PATH=str(d)+os.pathsep+os.environ.get('PATH',''),CODE_PATH='/new immutable code',CAPTURE_ARGS=str(out),TD_HOST_RESULTS_ROOT=str(d/'results'),TD_RESULTS_DIR='/new results')
   script=SCRIPT.with_name('e42_run.sh')
   p=subprocess.run(['/bin/bash',str(script),'llama3.1-8b','3','select-check','--n-check','3'],env=env,capture_output=True,text=True)
   self.assertEqual(p.returncode,23,p.stderr);args=json.loads(out.read_text())
   self.assertEqual(args[args.index('-w')+1],'/new immutable code')
   self.assertFalse(any(v.startswith('HOME=') for v in args))
   self.assertEqual(args[-2:],['--n-check','3'])
   self.assertTrue((d/'results/llama3.1-8b/select-check').is_dir())
 def test_invalid_step_never_launches_docker(self):
  p,argv=self.run_wrapper(['/r','unknown']);self.assertEqual(p.returncode,2);self.assertIsNone(argv)

if __name__=='__main__':unittest.main()
