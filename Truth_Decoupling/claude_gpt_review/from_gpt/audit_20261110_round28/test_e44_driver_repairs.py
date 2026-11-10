"""Concrete E44 driver/MC/export regressions; no pretrained model or service."""
import contextlib
import copy
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parent.parent
CODE = REVIEW / 'from_claude/code'
sys.path[:0] = [str(CODE), str(CODE / 'scripts'), str(CODE / 'tests')]
import e44_factorial_das as D
import e44_export_judge as X
import e43_qa_prompt_das as E43
import test_qa_factorial_das as upstream


class PureDriverRepairs(unittest.TestCase):
    def test_C_per_question_uses_the_actual_factor_edge(self):
        spec = D.family_spec('C', 'E1')
        h = {'conf': torch.tensor([[0., 0.]]), 'caut': torch.tensor([[2., 0.]]),
             'conf_E1': torch.tensor([[10., 0.]]), 'caut_E1': torch.tensor([[8., 0.]])}
        rendered = {n: [{'prefix_text': n}] for n in h}
        with patch.object(D.pd, 'capture_prompt_end', side_effect=lambda b, texts, l, bs: h[texts[0]]):
            plain = D.question_differences(None, spec, rendered, 0, 1, 'conf')
            truth = D.question_differences(None, spec, rendered, 0, 1, 'conf_E1')
        self.assertEqual(plain.tolist(), [[2., 0.]])
        self.assertEqual(truth.tolist(), [[-2., 0.]])
        with self.assertRaisesRegex(ValueError, 'neutral'):
            D.question_pair(spec, 'x')

    def test_isolate_diagnostic_changes_other_factor_but_targets_base(self):
        for family, base, source in [('E', 'conf', 'caut'), ('E', 'caut', 'conf'),
                                      ('C', 'conf', 'conf_E1'), ('C', 'conf_E1', 'conf')]:
            spec = D.family_spec(family, 'E1')
            self.assertEqual(D.read_pair(spec, base, 'iso'), (source, base))
            self.assertEqual(D.arm_target_prompt(family, 'E1', base, 'iso', 1), base)
        m = D.family_spec('C', 'E1', 'M')
        self.assertEqual(D.question_pair(m, 'conf_M'), ('caut_M', 'conf_M'))
        self.assertEqual(D.read_pair(m, 'conf_M', 'iso'), ('conf_E1', 'conf_M'))
        self.assertEqual(D.read_pair(D.family_spec('E', 'E1', 'M'), 'conf_M', 'iso'), ('caut_M', 'conf_M'))
        self.assertEqual(D.arm_target_prompt('E', 'E1', 'conf_M', 'iso', 1., 'M'), 'conf_M')
        with self.assertRaises(ValueError):
            D.question_pair(m, 'conf')
        self.assertEqual(D.parse_arm('conf.iso.p1', {'conf': ''}), ('conf', 'iso', 1.))
        with self.assertRaises(SystemExit):
            D.parse_arm('conf.iso.m1', {'conf': ''})

    def test_unknown_M_training_fallback_is_not_silently_a_factor(self):
        with self.assertRaisesRegex(ValueError, 'E1/E2'):
            D.family_spec('E', 'M')

    def test_holdout_is_not_silently_removed_when_sample_is_small(self):
        with self.assertRaises(ValueError):
            D.split_train_val([{'pair_id': 'a'}, {'pair_id': 'b'}], 2, 42)
        train, val = D.split_train_val([{'pair_id': str(i)} for i in range(3)], 1, 42)
        self.assertEqual((len(train), len(val)), (2, 1))
        self.assertFalse({r['pair_id'] for r in train} & {r['pair_id'] for r in val})
        self.assertEqual(D.artifact_name('frozen_train.json', True), 't.frozen_train.json')

    def test_train_only_pilot_needs_no_test_or_e_train_records(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td)
            (source / 'bank.json').write_text(json.dumps({'config': {'qa_system': 's', 'qa_user': '{question}'}}))
            data = {'splits': {'train': [{'pair_id': 'a', 'question': 'Q?'}], 'tune': []}}
            (source / 'questions.json').write_text(json.dumps(data))
            args = SimpleNamespace(source_run=source, pair_ids=['a'], n_check=0, phase='freeze', splits=['train'])
            _, actual = D.load_source(args)
            self.assertNotIn('e_train', actual)
            self.assertNotIn('test', actual['splits'])
            args.splits = ['tune']
            with self.assertRaisesRegex(ValueError, 'empty'):
                D.load_source(args)
            args.pair_ids = None
            args.splits = ['train']
            with self.assertRaisesRegex(ValueError, 'empty'):
                D.load_source(args)

    def test_missing_reference_does_not_give_zero_accuracy_or_shrink_candidates(self):
        rows = [{'pair_id': 'q', 'question': 'Q?', 'scored': False, 'abstain_logp': None,
                 'true': [{'truncated': True, 'skipped': 'truncated_reference'}], 'false': []}]
        summary = D.mc_summary(rows)
        self.assertIsNone(summary['mc1'])
        self.assertIsNone(summary['mc2'])
        self.assertIsNone(summary['abstain_logp_mean'])
        self.assertEqual(summary['truncated_candidates'], 1)
        self.assertEqual(summary['n_abstain_unscored'], 1)
        with self.assertRaises(ValueError):
            D.mc_summary(rows, [dict(rows[0], pair_id='other')])

    def test_score_split_requires_all_candidates_and_preserves_best_answer_definition(self):
        q = {'pair_id': 'q', 'question': 'Q?', 'gold': ['Best', 'Other'], 'best_answer': 'Best', 'incorrect': ['Wrong']}
        args = SimpleNamespace(max_ref_tokens=48, mc_batch_size=4)
        def record(value, truncated=False):
            return {'sum_logp': value, 'n_tokens': 1, 'truncated': truncated, 'skipped': 'truncated_reference' if truncated else None}
        with patch.object(D.fd, 'score_answers', return_value=[record(-5), record(-4), record(-1), record(-2)]):
            r = D.score_split(args, None, [{}], [q], [1])[0]
        self.assertEqual((r['any_true_top1'], r['standard_mc1']), (1, 0))
        with patch.object(D.fd, 'score_answers', return_value=[record(-5), record(-4), record(None, True), record(-2)]):
            r = D.score_split(args, None, [{}], [q], [1])[0]
        self.assertFalse(r['scored'])
        self.assertEqual(r['unscorable_reason'], 'no_complete_reference_set')
        reference = torch.tensor([1., 0.])
        write = {'direction': torch.tensor([0., 1.]), 'reference_direction': reference, 'amount': .3}
        with patch.object(D.fd, 'score_answers', return_value=[record(-5), record(-4), record(-1), record(-2)]) as scorer:
            D.score_split(args, None, [{}], [q], [1], layer=1, write=write)
        self.assertIs(scorer.call_args.kwargs['reference_direction'], reference)


class PipelineDriverRepairs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path(upstream.TOKENIZER).exists():
            raise RuntimeError('TD_TEST_TOKENIZER must point to the supplied real tokenizer')
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.master = Path(cls.tmp.name)
        cls.source, cls.run_dir = cls.master / 'source', cls.master / 'run'
        cls.source.mkdir()
        cls.bundle = upstream.bundle.__wrapped__()
        cls.bundle._e43_model_files = {}
        topics = ['the moon', 'green tea', 'old maps', 'river stones', 'night trains', 'paper kites', 'copper wire', 'winter wheat',
                  'glass beads', 'sea salt', 'mountain goats', 'city bells']
        qs = [{'pair_id': 't:%d' % i, 'question': 'What is true about %s?' % text, 'gold': ['It is ordinary', 'Nothing special'],
               'incorrect': ['It is magic', 'It grants wishes'], 'best_answer': 'It is ordinary'} for i, text in enumerate(topics)]
        (cls.source / 'bank.json').write_text(json.dumps({'model': 'tiny-llama',
            'model_config': {'name': 'tiny-llama', 'path': str(cls.source)}, 'hidden_size': cls.bundle.hidden_size,
            'num_layers': cls.bundle.num_layers, 'config': {'qa_system': 'You are a knowledgeable assistant.', 'qa_user': upstream.USER}}))
        (cls.source / 'questions.json').write_text(json.dumps({'splits': {'train': qs[:8], 'tune': qs[8:]}, 'e_train': qs[:8]}))
        def call(*argv):
            args = ['e44_factorial_das.py', argv[0], '--source-run', str(cls.source), '--run-dir', str(cls.run_dir),
                    '--max-new-tokens', '5', '--batch-size', '4', *argv[1:]]
            with patch.object(sys, 'argv', args), patch.object(E43, 'open_model', return_value=cls.bundle), contextlib.redirect_stdout(io.StringIO()):
                D.main()
        call('freeze')
        call('train', '--family', 'E', '--layers', '1', '--epochs', '1', '--val-n', '2')
        call('generate', '--family', 'E', '--layers', '1', '--arms', 'conf.w.p1', 'conf.iso.p1', 'caut.wq.p1')
        call('mc', '--family', 'E', '--layers', '1', '--arms', 'conf.w.p1', 'conf.iso.p1')
        cls.call_master = staticmethod(call)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.source, self.run_dir = self.path / 'source', self.path / 'run'
        shutil.copytree(type(self).source, self.source)
        shutil.copytree(type(self).run_dir, self.run_dir)
        self.gen_path = self.run_dir / 'gen/E_L1_conf-w-p1_tune.json'

    def export(self, **kw):
        return X.export(self.source, self.run_dir, **kw)

    def test_actual_flow_exports_and_single_output_cannot_be_overwritten(self):
        out = self.export()
        self.assertEqual(out['n_questions'], 4)
        self.assertEqual(len(out['conditions']), 14)
        self.assertEqual(self.export(), out)
        Path(out['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'existing judge input differs'):
            self.export()

    def test_duplicate_or_extra_generation_rows_are_rejected(self):
        original = D.read(self.gen_path)
        for extra in [copy.deepcopy(original['rows'][0]), dict(original['rows'][0], pair_id='foreign')]:
            d = copy.deepcopy(original); d['rows'].append(extra); D.save(self.gen_path, D.seal(d))
            with self.assertRaises(ValueError):
                self.export()

    def test_resealed_foreign_question_is_rejected(self):
        d = D.read(self.gen_path); d['rows'][0]['question'] = 'Changed question'
        D.save(self.gen_path, D.seal(d))
        with self.assertRaisesRegex(ValueError, 'different question'):
            self.export()

    def test_changed_direction_or_training_freeze_invalidates_export(self):
        d = D.read(self.gen_path)
        direction = self.run_dir / 'dirs' / d['direction_file']
        before = direction.read_bytes(); direction.write_bytes(before + b'X')
        with self.assertRaisesRegex(ValueError, 'source file changed'):
            self.export()
        direction.write_bytes(before)
        p = self.run_dir / 'frozen_train.json';p.write_text(p.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'training freeze changed'):
            self.export()

    def test_E1_checkpoint_cannot_answer_an_E2_request(self):
        d = D.read(self.gen_path)
        args = SimpleNamespace(run_dir=self.run_dir, pair_ids=None, n_check=0, e_on='E2')
        with self.assertRaisesRegex(ValueError, 'E factor'):
            D.load_direction(args, 'E', 1, d['identity_sha256'])
        args.e_on = 'E1'; args.e_off = 'M'
        with self.assertRaisesRegex(ValueError, 'E factor'):
            D.load_direction(args, 'E', 1, d['identity_sha256'])

    def test_optimization_seed_does_not_change_holdout_and_split_seed_is_bound(self):
        frozen = D.read(self.run_dir / 'frozen_train.json')
        prepared = D.read(self.source / 'questions.json')
        meta = D.read(self.source / 'bank.json')
        cfg, systems = D.load_prompts(meta, CODE / 'configs/prompts/default_v1.json', CODE / 'configs/prompts/e44_factorial_v1.json')
        args = SimpleNamespace(run_dir=self.run_dir, source_run=self.source, family='E', e_on='E1',
            pair_ids=None, n_check=0, val_n=2, seed=7, split_seed=42, epochs=1, lr=1e-3, clip=1., loss='jsd',
            positions='content', init='random', batch_size=4, layers=[1], overwrite=False, max_target_tokens=24)
        original = D.split_train_val
        observed = []
        def record_split(questions, count, seed):
            train, val = original(questions, count, seed)
            observed.append((seed, [q['pair_id'] for q in val]))
            return train, val
        with patch.object(D, 'split_train_val', side_effect=record_split), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'different seed'):
                D.run_train(args, meta, prepared, type(self).bundle, cfg, systems, frozen['identity'], frozen['identity_sha256'])
        previous = torch.load(self.run_dir / 'dirs/E_L1.pt', map_location='cpu', weights_only=True)
        self.assertEqual(observed, [(42, previous['training_contract']['val_ids'])])
        args.seed = 42; args.split_seed = 7
        with patch.object(D, 'split_train_val', side_effect=record_split), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'different seed'):
                D.run_train(args, meta, prepared, type(self).bundle, cfg, systems, frozen['identity'], frozen['identity_sha256'])
        self.assertEqual(observed[-1][0], 7)

    def test_mc_parent_change_invalidates_existing_arm(self):
        path = self.run_dir / 'mc/E_L1_conf-w-p1_tune.json'; d = D.read(path)
        questions = D.read(self.source / 'questions.json')['splits']['tune']
        D.validate_mc(d, questions, self.run_dir)
        base = self.run_dir / 'mc' / d['baseline_file'];base.write_text(base.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'baseline source changed'):
            D.validate_mc(d, questions, self.run_dir)

    def test_summary_does_not_trust_old_self_seal_after_parent_change(self):
        d = D.read(self.gen_path);p = self.run_dir / 'dirs' / d['direction_file'];p.write_bytes(p.read_bytes() + b'X')
        args = SimpleNamespace(run_dir=self.run_dir, source_run=self.source, include_pilot=False)
        with self.assertRaisesRegex(ValueError, 'source changed'):
            D.run_summarize(args)

    def test_isolate_artifact_explicitly_records_target_baseline(self):
        d = D.read(self.run_dir / 'gen/E_L1_conf-iso-p1_tune.json')
        self.assertEqual((d['base_prompt'], d['target_prompt'], d['read_pair']), ('conf', 'conf', ['caut', 'conf']))
        self.assertEqual(d['summary']['baseline_matches_target_prompt'], 4)

    def test_filter_keeps_preselected_holdout_and_zero_mean_does_not_abort_random_init(self):
        new_run = self.path / 'new_run'; new_run.mkdir()
        shutil.copy2(self.run_dir / 'frozen_train.json', new_run / 'frozen_train.json')
        frozen = D.read(new_run / 'frozen_train.json')
        meta = D.read(self.source / 'bank.json')
        prepared = D.read(self.source / 'questions.json')
        cfg, systems = D.load_prompts(meta, CODE / 'configs/prompts/default_v1.json', CODE / 'configs/prompts/e44_factorial_v1.json')
        planned_train, planned_val = D.split_train_val(prepared['splits']['train'], 2, 42)
        omitted = planned_val[0]['pair_id']
        signed = {q['pair_id']: (1. if i % 2 == 0 else -1.) for i, q in enumerate(planned_train)}
        args = SimpleNamespace(run_dir=new_run, source_run=self.source, family='E', e_on='E1',
            pair_ids=None, n_check=0, val_n=2, seed=42, epochs=1, lr=1e-3, clip=1., loss='jsd',
            positions='content', init='random', batch_size=4, layers=[1], overwrite=False, max_target_tokens=24)
        original_build = D.build_questions
        def filtered(*a, **kw):
            built, skipped = original_build(*a, **kw)
            return [q for q in built if q['pair_id'] != omitted], skipped + [{'pair_id': omitted, 'reason': 'synthetic_unusable_target'}]
        def synthetic_states(bundle, built, spec, layer, batch_size, ref):
            h = {n: torch.zeros(len(built), bundle.hidden_size) for n in spec['prompts']}
            for i, q in enumerate(built):
                for n in ('conf_E1', 'caut_E1'):
                    h[n][i, 0] = signed.get(q['pair_id'], 0.)
                for row in q['rows']:
                    row['difference'] = h[row['source']][i] - h[row['base']][i]
                    row['ref_logp'] = ref[(q['pair_id'], row['target'])]
                    row['ref_target_ids'] = list(q['refs'][row['target']]['answer_ids'])
            return h
        with patch.object(D, 'build_questions', side_effect=filtered), patch.object(D, 'attach_layer', side_effect=synthetic_states), contextlib.redirect_stdout(io.StringIO()):
            D.run_train(args, meta, prepared, type(self).bundle, cfg, systems, frozen['identity'], frozen['identity_sha256'])
        record = torch.load(new_run / 'dirs/E_L1.pt', map_location='cpu', weights_only=True)
        self.assertEqual(record['training_contract']['train_ids'], [q['pair_id'] for q in planned_train])
        self.assertEqual(record['training_contract']['val_ids'], [planned_val[1]['pair_id']])
        self.assertEqual(record['planned_val_ids'], [q['pair_id'] for q in planned_val])
        self.assertFalse(record['mean_difference_defined'])
        self.assertEqual(record['md_norm'], 0.)
        self.assertIsNone(record['cosine_w_md'])
        self.assertEqual(record['losses']['train']['mean_difference_direction']['reference_status'], 'zero_mean_no_write_not_a_direction')


if __name__ == '__main__':
    unittest.main()
