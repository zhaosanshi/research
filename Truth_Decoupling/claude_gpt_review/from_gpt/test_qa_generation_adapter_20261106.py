"""Real CPU forward/backward and source-run integration for generation adaptation."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / 'from_claude/code'
sys.path[:0] = [str(HERE), str(CODE / 'scripts'), str(CODE)]
import test_qa_das_pipeline_20261105 as fixtures
from core import qa_generation_adapter as ga
from core.intervene import steer
from core.model_io import ModelBundle, forward_logits, position_ids_from_mask
from core.qa_context import pack_answers
import e27_qa_generation as script


class AdapterTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = fixtures.ToyTokenizer()
        self.model = fixtures.PipelineModel(self.tokenizer)
        self.bundle = ModelBundle(self.model, self.tokenizer, {'name': 'adapter-cpu', 'layers_attr': 'layers', 'chat': {}})
        self.cfg = {'qa_system': 'Answer briefly.', 'qa_user': 'Question: {question}\nAnswer:',
                    'max_new_tokens': 2, 'batch_size': 2}
        self.records = [{'pair_id': 'p1', 'question': 'Where?', 'positive_answer': 'Paris', 'negative_answer': 'Rome'},
                        {'pair_id': 'p2', 'question': 'Who?', 'positive_answer': 'A', 'negative_answer': 'Someone else'}]
        self.initial = {0: torch.nn.functional.normalize(torch.arange(1., 9.), dim=0)}
        self.centers = {0: {'prompt': torch.zeros(8), 'decode': torch.arange(8.) * .03}}

    def train(self, **kw):
        return ga.train_generation_adapter(self.bundle, self.records, self.cfg, self.initial, self.centers,
                                           epochs=2, log_fn=None, **kw)

    def test_real_gradient_updates_only_direction_and_restores_model_state(self):
        self.model.train(); self.model.layers[0].eval()
        next(self.model.parameters()).requires_grad_(True)
        flags = [p.requires_grad for p in self.model.parameters()]
        modes = [m.training for m in self.model.modules()]
        originals = [p.detach().clone() for p in self.model.parameters()]
        result = self.train(alpha=1., ranking_weight=.3)
        self.assertEqual(result['method'], 'das_generation_adapter')
        self.assertGreater(result['gradient_audit']['0']['maximum'], 0)
        self.assertGreater(float((result['W']['0'] - self.initial[0]).norm()), 1e-6)
        self.assertAlmostEqual(float(result['W']['0'].norm()), 1., places=6)
        self.assertTrue(result['base_parameter_versions_unchanged'])
        for p, old in zip(self.model.parameters(), originals):
            self.assertTrue(torch.equal(p, old))
            self.assertIsNone(p.grad)
        self.assertEqual(flags, [p.requires_grad for p in self.model.parameters()])
        self.assertEqual(modes, [m.training for m in self.model.modules()])
        self.assertTrue(all(not b._forward_hooks for b in self.model.layers))
        for tr in result['position_trace']:
            positions = {position for write in tr['writes'] for row, position in write['positions']}
            self.assertEqual(positions, set(range(tr['prompt_length'] - 1, tr['final_answer_position'])))
            self.assertNotIn(tr['final_answer_position'], positions)
        self.assertGreater(result['losses'][0]['pair_ranking'], 0)

    def test_default_ce_has_no_negative_answer_forward_and_controls_are_explicit(self):
        before = self.model.forward_calls
        result = self.train()
        self.assertEqual(self.model.forward_calls - before, 2 * len(self.records))
        self.assertTrue(all(x['pair_ranking'] == 0 for x in result['losses']))
        for init in ['md', 'random']:
            other = self.train(init=init)
            self.assertEqual(other['method'], init + '_generation_adapter')
            self.assertEqual(other['gradient_audit']['0']['updates'], result['gradient_audit']['0']['updates'])
            if init == 'random':
                self.assertFalse(torch.equal(other['initial_W']['0'], result['initial_W']['0']))
                repeated = self.train(init='random')
                self.assertTrue(torch.equal(other['initial_W']['0'], repeated['initial_W']['0']))

    def test_bfloat16_backbone_keeps_live_direction_gradients_and_original_weights(self):
        self.model.to(torch.bfloat16)
        before = [p.detach().clone() for p in self.model.parameters()]
        result = self.train(alpha=1.5)
        self.assertGreater(result['gradient_audit']['0']['maximum'], 0)
        self.assertTrue(all(t['residual_dtype'] == 'torch.bfloat16' for t in result['position_trace']))
        self.assertGreater(float((result['W']['0'] - self.initial[0]).norm()), 1e-6)
        self.assertTrue(all(torch.equal(p, old) for p, old in zip(self.model.parameters(), before)))
        self.assertEqual(self.model.head.weight.dtype, torch.bfloat16)
        self.assertNotIn('forward', self.model.head.__dict__)

    def test_training_mask_operator_matches_deployment_specs_with_padding_and_two_layers(self):
        result = self.train()
        result['layers'] = [0, 1]
        result['W']['1'] = torch.nn.functional.normalize(torch.arange(8., 0., -1), dim=0)
        result['centers']['1'] = {'prompt': torch.ones(8) * .2, 'decode': torch.ones(8) * -.1}
        encoded = ga.prepare_generation_pairs(self.bundle, self.cfg, self.records)
        packed = pack_answers([r['positive'] for r in encoded], 0, 'cpu')
        raw = {l: torch.nn.Parameter(result['W'][str(l)].clone()) for l in result['layers']}
        centers = {l: result['centers'][str(l)] for l in result['layers']}
        traces = []
        with ga.differentiable_amplification(self.bundle, raw, centers, packed, result['alpha'], traces):
            trained = forward_logits(self.bundle, packed['inputs'], grad=True)
        with steer(self.bundle, ga.adapter_specs(result, packed)):
            deployed = forward_logits(self.bundle, packed['inputs'])
        torch.testing.assert_close(trained, deployed, atol=1e-6, rtol=1e-6)
        for tr in traces:
            written = {tuple(p) for write in tr['writes'] for p in write['positions']}
            self.assertEqual(written, {tuple(p) for p in packed['predictor_mask'].nonzero().tolist()})

    def test_full_teacher_forcing_matches_continuous_cached_decode_at_every_prediction(self):
        for include_eos in (False, True):
            with self.subTest(include_eos=include_eos):
                result = self.train(include_eos=include_eos)
                item = ga.prepare_generation_pairs(self.bundle, self.cfg, self.records[:1], include_eos=include_eos)[0]['positive']
                packed = pack_answers([item], 0, 'cpu')
                with steer(self.bundle, ga.adapter_specs(result, packed)):
                    full = forward_logits(self.bundle, packed['inputs'])[0][packed['predictor_mask'][0]]
                ids = torch.tensor([item['prefix_ids']]); mask = torch.ones_like(ids); cache = None; rows = []
                with steer(self.bundle, ga.adapter_specs(result)), torch.no_grad():
                    for target in item['answer_ids']:
                        out = self.model(input_ids=ids, attention_mask=mask,
                                         position_ids=position_ids_from_mask(mask)[:, -ids.shape[1]:],
                                         use_cache=True, past_key_values=cache)
                        cache = out.past_key_values; rows.append(out.logits[0, -1])
                        ids = torch.tensor([[target]])
                        mask = torch.cat([mask, torch.ones((1, 1), dtype=mask.dtype)], dim=1)
                torch.testing.assert_close(full, torch.stack(rows), atol=2e-6, rtol=1e-5)

    def test_eos_is_a_real_ce_target_with_live_gradient_and_correct_predictor_mask(self):
        plain = ga.prepare_generation_pairs(self.bundle, self.cfg, self.records[:1])[0]['positive']
        item = ga.prepare_generation_pairs(self.bundle, self.cfg, self.records[:1], include_eos=True)[0]['positive']
        self.assertEqual(item['full_ids'], plain['full_ids'] + [2])
        self.assertEqual(item['answer_ids'], plain['answer_ids'] + [2])
        self.assertEqual(item['answer_length'], plain['answer_length'] + 1)
        self.assertEqual(item['text_audit']['stop_token_id'], 2)
        packed = pack_answers([item], 0, 'cpu')
        self.assertEqual(int(packed['target_mask'].sum()), plain['answer_length'] + 1)
        self.assertTrue(packed['predictor_mask'][0, -2])
        self.assertFalse(packed['predictor_mask'][0, -1])
        raw = {0: torch.nn.Parameter(self.initial[0].clone())}
        with ga.frozen_model(self.model), ga.fp32_output_projection(self.bundle):
            with ga.differentiable_amplification(self.bundle, raw, self.centers, packed, 1.):
                logits = forward_logits(self.bundle, packed['inputs'], grad=True)
                mean_ce = -ga.answer_logprobs(logits, packed)[0] / item['answer_length']
                selected = logits[0][packed['predictor_mask'][0]]
                per_token_ce = torch.nn.functional.cross_entropy(selected, torch.tensor(item['answer_ids']), reduction='none')
                torch.testing.assert_close(mean_ce, per_token_ce.mean())
                # Isolate the terminal loss to prove it contributes a live direction gradient.
                per_token_ce[-1].backward()
                self.assertGreater(float(raw[0].grad.norm()), 0)
        trained = self.train(include_eos=True)
        self.assertEqual(ga.stop_training_config(trained), (True, 2))
        self.assertGreater(trained['gradient_audit']['0']['maximum'], 0)
        for trace in trained['position_trace']:
            positions = {p for write in trace['writes'] for _, p in write['positions']}
            self.assertIn(trace['final_content_position'], positions)
            self.assertNotIn(trace['stop_token_position'], positions)

    def test_stop_id_is_declared_and_never_guessed_from_generation_list(self):
        self.model.generation_config.eos_token_id = [2, 99]
        self.assertEqual(ga.resolve_stop_token_id(self.bundle), 2)
        self.model.generation_config.eos_token_id = [99, 2]
        self.assertEqual(ga.resolve_stop_token_id(self.bundle), 2)
        for token in [None, -1, True, [2], 3]:
            with self.subTest(tokenizer_eos=token):
                self.tokenizer.eos_token_id = token
                with self.assertRaises(ValueError): ga.resolve_stop_token_id(self.bundle)
        self.tokenizer.eos_token_id = 3
        self.assertEqual(ga.resolve_stop_token_id(self.bundle, 2), 2)
        for invalid in [None, [], [2, None], [-1], True]:
            self.model.generation_config.eos_token_id = invalid
            with self.subTest(generation_eos=invalid), self.assertRaises(ValueError):
                ga.resolve_stop_token_id(self.bundle, 2)
        self.model.generation_config.eos_token_id = [256]
        with self.assertRaisesRegex(ValueError, 'vocabulary'): ga.resolve_stop_token_id(self.bundle, 256)
        with self.assertRaises(ValueError): ga.prepare_generation_pairs(self.bundle, self.cfg, self.records, stop_token_id=2)

    def test_default_content_only_encoding_weights_losses_and_signature_do_not_drift(self):
        default_pairs = ga.prepare_generation_pairs(self.bundle, self.cfg, self.records)
        self.assertEqual(default_pairs, ga.prepare_generation_pairs(self.bundle, self.cfg, self.records, include_eos=False))
        for record, pair in zip(self.records, default_pairs):
            for side, key in [('positive', 'positive_answer'), ('negative', 'negative_answer')]:
                old_encoding = ga.encode_answer(self.bundle, self.cfg['qa_system'], record['question'], record[key], self.cfg['qa_user'])
                self.assertEqual(pair[side], old_encoding)
        implicit, explicit = self.train(), self.train(include_eos=False)
        self.assertTrue(torch.equal(implicit['W']['0'], explicit['W']['0']))
        self.assertEqual(implicit['losses'], explicit['losses'])
        self.assertEqual(implicit['training_tokens_sha256'], explicit['training_tokens_sha256'])
        self.assertEqual(ga.stop_training_config(implicit), (False, None))
        with_eos = self.train(include_eos=True)
        self.assertNotEqual(implicit['training_tokens_sha256'], with_eos['training_tokens_sha256'])

    def test_native_llama_and_qwen_generation_stop_on_the_supervised_id(self):
        # Isolated optional-import workaround matches the existing random-family verifier;
        # actual HF generation is exercised, with a forced stop solely as a wiring control.
        code = textwrap.dedent('''
            import sys
            sys.path[:0] = [CODE, HERE]
            import torch
            torch.set_num_threads(1)
            import transformers.utils.import_utils as iu
            iu._sklearn_available = False
            iu._librosa_available = False
            iu._torchvision_available = False
            from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM, LogitsProcessor
            from test_e25_pipeline_20261103 import ToyTokenizer
            from core.model_io import ModelBundle
            from core.qa_generation_adapter import train_generation_adapter, adapter_specs, resolve_stop_token_id
            from core.intervene import steer
            class Stop(LogitsProcessor):
                def __call__(self, input_ids, scores):
                    scores[:] = -float('inf'); scores[:, 2] = 0; return scores
            cfg = dict(vocab_size=256, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                       num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                       max_position_embeddings=256, pad_token_id=0, bos_token_id=1, eos_token_id=[2, 3])
            for cls, config in [(LlamaForCausalLM, LlamaConfig(**cfg)), (Qwen3ForCausalLM, Qwen3Config(**cfg))]:
                torch.manual_seed(12)
                model = cls(config).eval()
                bundle = ModelBundle(model, ToyTokenizer(), {'name': cls.__name__, 'layers_attr': 'model.layers'})
                w = torch.nn.functional.normalize(torch.arange(1., 17.), dim=0)
                adapter = train_generation_adapter(bundle,
                    [{'pair_id':'p', 'question':'Q?', 'positive_answer':'A', 'negative_answer':'B'}],
                    {'qa_system':'', 'qa_user':'{question}', 'max_new_tokens':6, 'batch_size':1},
                    {0:w}, {0:{'prompt':torch.zeros(16), 'decode':torch.zeros(16)}},
                    epochs=1, include_eos=True, log_fn=None)
                assert resolve_stop_token_id(bundle) == 2
                assert adapter['train_config']['stop_token_id'] == 2
                assert adapter['gradient_audit']['0']['maximum'] > 0
                inputs = bundle.encode([bundle.build_chat('', 'Q?')])
                specs = adapter_specs(adapter)
                with steer(bundle, specs):
                    out = model.generate(**inputs, max_new_tokens=6, do_sample=False, pad_token_id=0, logits_processor=[Stop()])
                assert out.shape[1] == inputs['input_ids'].shape[1] + 1
                assert int(out[0, -1]) == 2 and specs[0].n_applied == 1
                print(cls.__name__, 'EOS_CE_AND_STOP_OK')
        ''')
        prefix = 'CODE = %r\nHERE = %r\n' % (str(CODE), str(HERE))
        result = subprocess.run([sys.executable, '-c', prefix + code], text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count('EOS_CE_AND_STOP_OK'), 2)

    def test_failure_removes_hooks_and_restores_parameter_flags_eval_modes_and_head(self):
        self.model.train(); next(self.model.parameters()).requires_grad_(True)
        flags = [p.requires_grad for p in self.model.parameters()]
        modes = [m.training for m in self.model.modules()]
        with patch.object(ga, 'forward_logits', side_effect=RuntimeError('injected forward failure')):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                self.train()
        self.assertEqual(flags, [p.requires_grad for p in self.model.parameters()])
        self.assertEqual(modes, [m.training for m in self.model.modules()])
        self.assertTrue(all(not b._forward_hooks for b in self.model.layers))
        self.assertNotIn('forward', self.model.head.__dict__)

    def test_export_round_trip_hash_alpha_and_model_guards(self):
        result = self.train()
        with tempfile.TemporaryDirectory() as d:
            ga.save_adapter(result, d)
            loaded = ga.load_adapter(d, expected_model='adapter-cpu', expected_alpha=1.)
            self.assertTrue(torch.equal(loaded['W']['0'], result['W']['0']))
            with self.assertRaises(ValueError): ga.load_adapter(d, expected_model='other')
            with self.assertRaises(ValueError): ga.adapter_specs(loaded, expected_alpha=2.)
            with self.assertRaises(FileExistsError): ga.save_adapter(result, d)
            p = Path(d) / 'adapter.pt'; p.write_bytes(p.read_bytes() + b'corruption')
            with self.assertRaisesRegex(ValueError, 'manifest'): ga.load_adapter(d)

    def test_invalid_training_input_fails_without_modifying_model(self):
        for kw in [{'alpha': 0.}, {'alpha': float('nan')}, {'ranking_weight': -1.}, {'init': 'pure_das'}]:
            with self.subTest(kw=kw), self.assertRaises(ValueError): self.train(**kw)
        with self.assertRaises(ValueError):
            ga.prepare_generation_pairs(self.bundle, self.cfg, self.records + self.records)


class AdapterSourceRunTests(unittest.TestCase):
    def test_eos_identity_is_checked_and_legacy_content_only_artifacts_resume_unchanged(self):
        fixture = fixtures.QADASPipelineTests(methodName='runTest')
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        args = SimpleNamespace(run_dir=fixture.run / 'content', layers=[0], pilot=2, init='das', alpha=1.,
                               epochs=1, lr=.001, clip=1., ranking_weight=0., seed=42, limit=None,
                               model=fixture.args.model, judge_exp=script.DEFAULT_JUDGE_EXPERIMENT)
        ns, meta, bank, prepared = script.source_run(fixture.run)
        content = script.train(args, ns, meta, bank, prepared, bundle=fixture.bundle)
        old = copy.deepcopy(content)
        old['train_config'].pop('include_eos'); old['train_config'].pop('stop_token_id')
        legacy_dir = fixture.run / 'legacy'
        ga.save_adapter(old, legacy_dir)
        legacy_args = copy.copy(args); legacy_args.run_dir = legacy_dir
        before = {p.name: p.read_bytes() for p in legacy_dir.iterdir() if p.is_file()}
        with patch.object(script.qa, 'open_bundle', side_effect=AssertionError('legacy resume must not load model')):
            resumed = script.train(legacy_args, ns, meta, bank, prepared)
        self.assertNotIn('include_eos', resumed['train_config'])
        self.assertEqual(ga.stop_training_config(script.verified_adapter(legacy_args, ns, meta)), (False, None))
        self.assertEqual(before, {p.name: p.read_bytes() for p in legacy_dir.iterdir() if p.is_file()})
        legacy_args.include_eos = True
        with self.assertRaises(ValueError): script.train(legacy_args, ns, meta, bank, prepared, bundle=fixture.bundle)
        self.assertEqual(before, {p.name: p.read_bytes() for p in legacy_dir.iterdir() if p.is_file()})
        eos_args = copy.copy(args); eos_args.run_dir = fixture.run / 'eos'; eos_args.include_eos = True
        eos = script.train(eos_args, ns, meta, bank, prepared, bundle=fixture.bundle)
        self.assertEqual(ga.stop_training_config(eos), (True, 2))
        self.assertNotEqual(eos['training_tokens_sha256'], content['training_tokens_sha256'])
        script.verified_adapter(eos_args, ns, meta)
        eos_args.include_eos = False
        with self.assertRaisesRegex(ValueError, 'include_eos'): script.verified_adapter(eos_args, ns, meta)
        eos_args.include_eos = True; eos_args.stop_token_id = 3
        with self.assertRaisesRegex(ValueError, 'stop_token_id'): script.verified_adapter(eos_args, ns, meta)
        eos_args.stop_token_id = 2
        script.verified_adapter(eos_args, ns, meta)

    def test_actual_default_judge_config_resolves_and_non_judge_config_is_rejected(self):
        self.assertEqual(script.DEFAULT_JUDGE_EXPERIMENT, 'E4_qa')
        cfg = script.validate_judge_experiment(script.DEFAULT_JUDGE_EXPERIMENT)
        self.assertEqual(cfg['model'], 'deepseek-chat')
        self.assertEqual(cfg['api_key_env'], 'DEEPSEEK_API_KEY')
        with self.assertRaisesRegex(ValueError, 'no usable judge'):
            script.validate_judge_experiment('E2b_tqa')

    def test_default_cli_judge_phase_executes_real_reporting_with_mocked_judge_api(self):
        fixture = fixtures.QADASPipelineTests(methodName='runTest')
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        args = SimpleNamespace(run_dir=fixture.run / 'adapter', layers=[0], pilot=2, init='das', alpha=1.,
                               epochs=1, lr=.001, clip=1., ranking_weight=0., seed=42, limit=None,
                               model=fixture.args.model, judge_exp=script.DEFAULT_JUDGE_EXPERIMENT)
        ns, meta, bank, prepared = script.source_run(fixture.run)
        adapter = script.train(args, ns, meta, bank, prepared, bundle=fixture.bundle)
        for phase, split in [('check', 'tune'), ('tune', 'tune'), ('test', 'test')]:
            with self.subTest(phase=phase):
                raw = script.generate(args, ns, meta, bank, prepared, adapter, split,
                                      check=(phase == 'check'), bundle=fixture.bundle)
                self.assertFalse(raw['diagnostic_only'])
                self.assertEqual(raw['wiring_pilot'], phase == 'check')
                self.assertTrue(all(not c['uses_reference_answer'] for c in raw['condition_specs']))
                grader = fixtures.LocalOutcomeJudge({q['question']: q['outputs']['baseline'] for q in raw['evaluations']})
                requested = []

                def mock_api_factory(cfg, cache_dir):
                    # Keep real config loading and the real E25/E24 reporting path.
                    # Only construction of the external judge client is replaced.
                    requested.append(copy.deepcopy(cfg))
                    self.assertEqual(cfg['model'], 'deepseek-chat')
                    return grader

                argv = ['e27_qa_generation.py', 'judge-' + phase, '--source-run', str(fixture.run), '--run-dir', str(args.run_dir)]
                with patch.object(sys, 'argv', argv), \
                        patch.object(fixtures.e24, 'load_model_config', return_value=fixture.model_config), \
                        patch.object(fixtures.e24, 'out_paths', return_value=(str(args.run_dir), 'unused', 'unused')), \
                        patch('core.judge.judge_from_config', side_effect=mock_api_factory):
                    script.main()
                self.assertTrue(requested)
                self.assertEqual(grader.n_calls, 3 * len(raw['evaluations']))
                summary = json.loads((args.run_dir / (phase + '_scored.json')).read_text())
                items = json.loads((args.run_dir / (phase + '_scored_items.json')).read_text())
                self.assertTrue(summary['completed'])
                self.assertTrue(items['grading_completed'])
                self.assertEqual(summary['source_generation_sha256'], ga.digest(args.run_dir / (phase + '.json')))
                self.assertEqual(set(summary['rates']), set(raw['conditions']))
                self.assertTrue(all(r['n_unresolved'] == 0 for r in summary['rates'].values()))
                report = (args.run_dir / (phase + '_scored.md')).read_text()
                self.assertIn('小样本线路检查', report)
                self.assertNotIn('含金标来源的条件不是可部署性能', report)
                self.assertTrue(all(set(r['outcome']) == set(raw['conditions']) for r in items['evaluations']))

    def test_verified_source_to_train_export_check_and_resume_without_network(self):
        fixture = fixtures.QADASPipelineTests(methodName='runTest')
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        args = SimpleNamespace(run_dir=fixture.run / 'adapter', layers=[0], pilot=2, init='das', alpha=1.,
                               epochs=2, lr=.001, clip=1., ranking_weight=0., seed=42, limit=None,
                               model=fixture.args.model, judge_exp='offline')
        ns, meta, bank, prepared = script.source_run(fixture.run)
        adapted = script.train(args, ns, meta, bank, prepared, bundle=fixture.bundle)
        self.assertEqual(adapted['training_pair_ids'], [r['pair_id'] for r in prepared['e_train'][:2]])
        loaded = script.verified_adapter(args, ns, meta)
        raw = script.generate(args, ns, meta, bank, prepared, loaded, 'tune', check=True, bundle=fixture.bundle)
        self.assertEqual(raw['conditions'], ['baseline', 'original_das', 'das_generation_adapter'])
        self.assertTrue(raw['completed'])
        self.assertEqual(len(raw['evaluations']), 3)
        self.assertTrue(all(set(r['outputs']) == set(raw['conditions']) for r in raw['evaluations']))
        fixture.assert_question_only_generation()
        source_hash = ga.digest(args.run_dir / 'adapter.pt')
        again = script.train(args, ns, meta, bank, prepared, bundle=fixture.bundle)
        self.assertEqual(ga.digest(args.run_dir / 'adapter.pt'), source_hash)
        self.assertTrue(torch.equal(again['W']['0'], loaded['W']['0']))
        self.assertEqual(script.generate(args, ns, meta, bank, prepared, loaded, 'tune', check=True,
                                        bundle=fixture.bundle), raw)
        pilot_path = args.run_dir / 'check.json'
        saved = pilot_path.read_bytes()
        legacy_label = copy.deepcopy(raw)
        legacy_label.pop('wiring_pilot')
        legacy_label['diagnostic_only'] = True
        pilot_path.write_text(json.dumps(legacy_label))
        legacy_bytes = pilot_path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'legacy pilot/reference label'):
            script.generate(args, ns, meta, bank, prepared, loaded, 'tune', check=True, bundle=fixture.bundle)
        self.assertEqual(pilot_path.read_bytes(), legacy_bytes)
        pilot_path.write_bytes(saved)
        args.alpha = 2.
        with self.assertRaises(ValueError): script.train(args, ns, meta, bank, prepared, bundle=fixture.bundle)
        with self.assertRaisesRegex(ValueError, 'alpha'): script.verified_adapter(args, ns, meta)
        args.alpha = 1.
        original = fixture.run / 'bank.json'; value = json.loads(original.read_text()); value['rows']['changed'] = {}
        original.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'changed'): script.verified_adapter(args, ns, meta)


if __name__ == '__main__':
    unittest.main()
