import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REVIEW = Path(__file__).resolve().parents[3]
CODE = REVIEW / 'from_claude/code'
sys.path[:0] = [str(REVIEW / 'from_gpt'), str(CODE / 'scripts'), str(CODE)]
import test_qa_das_pipeline_20261105 as fixtures
import e29_qa_source_interchange as script
from core.qa_source_layer_plan import resolve_layer_plan, freeze_plan
from core.qa_control_cache import seed_control_cache
from core.judge import qa_cache_key, QA_GRADER_VERSION


class LayerPlanTests(unittest.TestCase):
    def test_only_fixed_depth_layers_and_frozen_training_variables_are_allowed(self):
        path = Path(__file__).with_name('source_layer_plan.json')
        meta = {'model': 'llama3.1-8b', 'num_layers': 32, 'das_candidate_layers': list(range(31)),
                'config': {'max_new_tokens': 48}}
        args = SimpleNamespace(layer=15, layer_plan=path, run_dir=None, phase='train', epochs=3,
                               lr=.001, clip=1., seed=42, init='das', pilot=None)
        layer, record = resolve_layer_plan(args, meta, 29, script.SOURCE_SYSTEM)
        self.assertEqual(layer, 15)
        with tempfile.TemporaryDirectory() as directory:
            freeze_plan(directory, record)
            self.assertEqual((Path(directory) / 'layer_plan.json').read_bytes(), path.read_bytes())
            freeze_plan(directory, record)
        args.layer = 7
        with self.assertRaisesRegex(ValueError, 'finite frozen'): resolve_layer_plan(args, meta, 29, script.SOURCE_SYSTEM)
        args.layer = 15; args.seed = 43
        with self.assertRaisesRegex(ValueError, 'outside'): resolve_layer_plan(args, meta, 29, script.SOURCE_SYSTEM)
        args.seed = 42; args.pilot = 3; args.epochs = 1
        resolve_layer_plan(args, meta, 29, script.SOURCE_SYSTEM)
        args.layer_plan = None
        with self.assertRaisesRegex(ValueError, 'requires'): resolve_layer_plan(args, meta, 29, script.SOURCE_SYSTEM)

    def test_cache_copy_checks_original_labels_and_is_independent(self):
        row = {'pair_id': 'q', 'question': 'Q?', 'judge_reference_answers': ['A'], 'incorrect': ['B'],
               'outputs': {'baseline': 'A'}, 'judge': {'baseline': {'label': 'correct', 'raw': 'A', 'rubric_id': QA_GRADER_VERSION}}}
        cfg = {'model': 'toy'}
        key = qa_cache_key('toy', row['question'], ['A'], 'A', ['B'])
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / 'source', Path(directory) / 'target'
            source.mkdir(); cache = source / 'judge_cache_toy.jsonl'
            cache.write_text(json.dumps({'key': key, 'model': 'toy', 'answer': 'A'}) + '\n')
            original = cache.read_bytes()
            seeded = seed_control_cache(source_dir=source, run_dir=target, judge_config=cfg,
                                         scored_rows=[row], conditions=['baseline'], scored_sha256='a' * 64, stem='tune')
            self.assertEqual(seeded['frozen_control_grades']['q']['baseline']['label'], 'correct')
            self.assertEqual(cache.read_bytes(), original)
            self.assertNotEqual(Path(seeded['judge_cache_dir']).resolve(), source.resolve())
            wrong = copy.deepcopy(row); wrong['judge']['baseline']['label'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'differs'):
                seed_control_cache(source_dir=source, run_dir=target, judge_config=cfg,
                                    scored_rows=[wrong], conditions=['baseline'], scored_sha256='a' * 64, stem='tune')


class ControlReuseTests(unittest.TestCase):
    def test_three_controls_are_reused_exactly_and_prompt_token_mismatch_fails(self):
        fixture = fixtures.QADASPipelineTests(methodName='runTest'); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        fixture.run.joinpath('selection.json').write_text('{"synthetic_frozen_layer":0}')
        args = SimpleNamespace(source_run=fixture.run, run_dir=fixture.run / 'parent', phase='check',
                               pair_ids=None, seed=42, layer=None, layer_plan=None, parent_controls=None)
        selected = {'selected': {'qa_das': {'layer': 0}}}
        with patch.object(script.qa, 'selection', return_value=selected):
            parent = script.generate(args, script.inputs(args), bundle=fixture.bundle)
            child_args = copy.copy(args); child_args.run_dir = fixture.run / 'child'
            child_args.parent_controls = args.run_dir / 'check.json'
            child = script.generate(child_args, script.inputs(child_args), bundle=fixture.bundle)
            self.assertEqual(child['identity']['reused_conditions'], ['baseline', 'source_only', 'equal_logit_mix'])
            for a, b in zip(parent['evaluations'], child['evaluations']):
                for condition in child['identity']['reused_conditions']:
                    for key in ('outputs', 'finished', 'generation_token_ids', 'trace'):
                        self.assertEqual(a[key][condition], b[key][condition])
                    self.assertFalse(b['reused_control_provenance'][condition]['generated_again'])
                self.assertEqual(a['outputs'], b['outputs'])
            damaged = copy.deepcopy(parent); damaged['evaluations'][0]['prompt_token_ids']['base'][0] += 1
            child_args.parent_controls.write_text(json.dumps(damaged))
            child_args.run_dir = fixture.run / 'damaged_child'
            with self.assertRaisesRegex(ValueError, 'prompt token IDs'):
                script.generate(child_args, script.inputs(child_args), bundle=fixture.bundle)


if __name__ == '__main__': unittest.main()
