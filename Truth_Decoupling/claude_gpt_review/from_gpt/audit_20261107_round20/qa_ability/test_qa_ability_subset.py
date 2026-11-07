from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'from_gpt'),str(ROOT/'from_claude/code/scripts'),str(ROOT/'from_claude/code')]
import test_e26_pipeline_20261106 as fixture
import e26_matched_coverage as joint
from e35_qa_gate_prompt_diagnostic import ABILITY_SYSTEM,ABILITY_USER
class QAOnlyIntegration(unittest.TestCase):
    def test_real_qa_only_pipeline_uses_explicit_prompt_and_excludes_tf_results(self):
        c=fixture.E26IntegrationTests(methodName='test_complete_real_inference_judging_freeze_report_and_actual_deliveries')
        c.setUp();self.addCleanup(c.doCleanups);c.cfg['evaluation_tasks']=['qa']
        meta,bank,data=c.prepare()
        self.assertEqual(meta['config']['qa_gate_system'],ABILITY_SYSTEM)
        self.assertEqual(meta['config']['qa_gate_user'],ABILITY_USER)
        joint.collect(c.args,meta,bank,data,'tune',bundle=c.bundle)
        joint.judge_qa(c.args,meta,data,'tune',judge=fixture.LocalJudge())
        sel=joint.freeze(c.args,meta);self.assertEqual(set(sel['tasks']),{'qa'})
        before=(c.args.run_dir/'selection.json').read_bytes()
        joint.collect(c.args,meta,bank,data,'test',bundle=c.bundle)
        joint.judge_qa(c.args,meta,data,'test',judge=fixture.LocalJudge())
        report=joint.report(c.args,meta)
        self.assertEqual(set(report['tables']),{'qa'});self.assertEqual(report['evaluation_tasks'],['qa'])
        self.assertFalse((c.args.run_dir/'tf_tune.json').exists());self.assertFalse((c.args.run_dir/'tf_test.json').exists())
        self.assertEqual(before,(c.args.run_dir/'selection.json').read_bytes())
        with self.assertRaisesRegex(ValueError,'not enabled'):joint.scored_rows(c.args,meta,'tf','test')
if __name__=='__main__':unittest.main()
