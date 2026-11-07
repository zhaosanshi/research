from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'from_gpt'),str(ROOT/'from_claude/code/scripts'),str(ROOT/'from_claude/code')]
import test_e26_pipeline_20261106 as fixture
import e26_matched_coverage as joint
import e33_gate_prompt_diagnostic as frozen_prompt


class TFOnlyIntegration(unittest.TestCase):
    def setUp(self):
        self.ctx=fixture.E26IntegrationTests(methodName='test_complete_real_inference_judging_freeze_report_and_actual_deliveries')
        self.ctx.setUp();self.addCleanup(self.ctx.doCleanups)

    def test_tf_only_real_toy_network_complete_chain_never_generates_or_grades_qa(self):
        c=self.ctx;c.cfg['evaluation_tasks']=['tf'];meta,bank,data=c.prepare()
        count=len(c.model.generated_prompts);judge=fixture.LocalJudge()
        joint.collect(c.args,meta,bank,data,'tune',bundle=c.bundle)
        self.assertTrue(joint.judge_qa(c.args,meta,data,'tune',judge=judge)['skipped'])
        selection=joint.freeze(c.args,meta);before=(c.args.run_dir/'selection.json').read_bytes()
        self.assertEqual(set(selection['tasks']),{'tf'});self.assertEqual(set(selection['sources']),{'tf_tune.json'})
        joint.collect(c.args,meta,bank,data,'test',bundle=c.bundle)
        self.assertTrue(joint.judge_qa(c.args,meta,data,'test',judge=judge)['skipped'])
        report=joint.report(c.args,meta)
        self.assertEqual(set(report['tables']),{'tf'});self.assertEqual(report['evaluation_tasks'],['tf'])
        self.assertIn('not evaluated',report['qa_evaluation_scope'])
        self.assertEqual(count,len(c.model.generated_prompts));self.assertEqual(judge.calls,[])
        self.assertEqual(list(c.args.run_dir.glob('qa_*')),[])
        self.assertEqual(before,(c.args.run_dir/'selection.json').read_bytes())
        with self.assertRaisesRegex(ValueError,'not enabled'):joint.scored_rows(c.args,meta,'qa','test')

    def test_old_bank_prompt_fallback_and_new_prompt_are_explicit(self):
        c=self.ctx
        meta={'config':{'qa_gate_user':'{question}','tf_gate_user':'{statement}'},
              'qa_config':{'qa_system':'qa system'},'prompts':{'statement':{'system':'legacy fact checker'}}}
        b=SimpleNamespace(build_chat=lambda system,user:(system,user))
        self.assertEqual(joint.gate_text(b,meta,'tf',{'text':'statement'}),('legacy fact checker','statement'))
        meta['config']['tf_gate_system']=frozen_prompt.NEUTRAL_SYSTEM
        meta['config']['tf_gate_user']=frozen_prompt.EXPLICIT_USER
        self.assertEqual(joint.gate_text(b,meta,'tf',{'text':'statement'}),
                         (frozen_prompt.NEUTRAL_SYSTEM,frozen_prompt.EXPLICIT_USER.format(statement='statement')))
        self.assertEqual(joint.gate_text(b,meta,'qa',{'question':'q'}),('qa system','q'))
        self.assertEqual(joint.evaluation_tasks({'config':{}}),('tf','qa'))

    def test_task_subset_configuration_rejects_empty_duplicates_and_unknown(self):
        for tasks in ([],['tf','tf'],['other'],'tf'):
            cfg=deepcopy(self.ctx.cfg);cfg['evaluation_tasks']=tasks
            with self.assertRaises(ValueError):joint.validate_config(cfg)


class LegacyFrozenRunTests(unittest.TestCase):
    def test_both_original_models_refreeze_to_identical_bytes(self):
        root=ROOT/'from_claude/results/03_E26同作答率_E27生成适配/E26_main'
        if not (root/'llama3.1-8b/selection.json').exists():self.skipTest('archived full runs absent')
        for model in ('llama3.1-8b','qwen3-8b'):
            args=SimpleNamespace(run_dir=root/model,model=model)
            meta,_,_=joint.load_run(args)
            regenerated=joint.freeze(args,meta,persist=False)
            exact=json.dumps(regenerated,ensure_ascii=False,indent=2,allow_nan=False).encode()
            self.assertEqual(exact,(root/model/'selection.json').read_bytes())

if __name__=='__main__':unittest.main()
