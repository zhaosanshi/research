"""Run the audited judging/selection regressions with frozen source fingerprints."""
import argparse,hashlib,importlib.util,json,platform,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;TEST=HERE.parent;REVIEW=TEST.parent;CODE=REVIEW/'from_claude/code'
sys.path[:0]=[str(TEST),str(CODE),str(CODE/'scripts'),str(CODE/'scripts/review_checks')]
PREVIOUS=TEST/'audit_20261108_round22'
ROOT_TESTS=('test_judge_contract_20261108.py','test_legacy_review_queue_20261108.py','test_two_axis_semantics_20261108.py',
            'test_qa_judge_20261103.py','test_personal_refusal_guard_20261107.py','test_source_reference_grades_20261107.py')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def flatten(s):
 for x in s:
  if isinstance(x,unittest.TestSuite):yield from flatten(x)
  else:yield x

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 files=[TEST/n for n in ROOT_TESTS]+[f for f in HERE.rglob('test*.py') if '__pycache__' not in f.parts and 'snapshot' not in str(f)]
 files += list((PREVIOUS/'metrics').glob('test*.py')) + list((PREVIOUS/'pipeline').glob('test*.py'))
 files=sorted(set(files))
 def source():
  ps={p for p in CODE.rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.json') and '__pycache__' not in p.parts}
  ps.update(files);ps.update(p for p in HERE.rglob('*.py') if '__pycache__' not in p.parts and 'snapshot' not in str(p))
  return {str(p.relative_to(REVIEW)):digest(p) for p in sorted(ps)}
 before=source();cases={};names={}
 for f in files:
  name=f.stem
  if name in names and names[name]!=f:raise RuntimeError('duplicate test module '+name)
  names[name]=f;sys.path.insert(0,str(f.parent))
  try:
   mod=sys.modules.get(name)
   if mod is None:
    spec=importlib.util.spec_from_file_location(name,f);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
   if Path(mod.__file__).resolve()!=f.resolve():raise RuntimeError('test imported from wrong checkout')
   for case in flatten(unittest.defaultTestLoader.loadTestsFromModule(mod)):
    if case.__class__.__module__==mod.__name__:cases[case.id()]=case
  finally:sys.path.remove(str(f.parent))
 result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(cases[k] for k in sorted(cases)));after=source()
 report={'scope':'R24 review18 regression: prior 85 grading checks plus parser, shared cache, external provenance, cross-run, blind review and N7 counterexamples',
         'python':platform.python_version(),'platform':platform.platform(),'tests_run':result.testsRun,
         'failures':[t.id() for t,_ in result.failures],'errors':[t.id() for t,_ in result.errors],
         'skipped':[(t.id(),v) for t,v in result.skipped],'source_unchanged_during_test':before==after,
         'success':result.wasSuccessful() and before==after,'test_ids':sorted(cases),'source_sha256':after}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 raise SystemExit(0 if report['success'] else 1)
if __name__=='__main__':main()
