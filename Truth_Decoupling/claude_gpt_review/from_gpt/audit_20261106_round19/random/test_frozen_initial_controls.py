"""Actual CPU generation with both source adapters; optimizer forbidden during controls."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
CODE = ROOT / 'from_claude/code'
sys.path[:0] = [str(ROOT / 'from_gpt'), str(CODE / 'scripts'), str(CODE)]
import test_qa_das_pipeline_20261105 as fixture
import e27_qa_generation as e27

spec = importlib.util.spec_from_file_location('frozen_initial_controls', CODE / 'scripts/review_checks/e27_frozen_initial_controls.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


class FrozenInitialTests(unittest.TestCase):
    def setUp(self):
        self.case = fixture.QADASPipelineTests('test_real_end_to_end_training_selection_check_and_diagnostics')
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.ns = self.case.args
        self.meta, self.bank, self.prepared, self.bundle = self.case.meta, self.case.bank, self.case.prepared, self.case.bundle
        for init in ('das', 'random'):
            args = SimpleNamespace(source_run=self.case.run, run_dir=self.case.run / ('adapter_' + init),
                alpha=3., init=init, layers=[0], pilot=None, epochs=1, lr=.001, clip=1., seed=42,
                ranking_weight=0., limit=None)
            a = e27.train(args, self.ns, self.meta, self.bank, self.prepared, bundle=self.bundle)
            e27.generate(args, self.ns, self.meta, self.bank, self.prepared, a, 'tune', bundle=self.bundle)
        self.args = SimpleNamespace(source_run=self.case.run, das_run=self.case.run/'adapter_das',
            random_run=self.case.run/'adapter_random', run_dir=self.case.run/'control', split='tune', limit=1)
        self.loaded = control.inputs(self.args)

    def test_saved_initial_is_used_and_no_training_occurs(self):
        ns, meta, bank, prepared, vectors, identity = self.loaded
        random = control.load_adapter(self.args.random_run)
        self.assertTrue(torch.equal(vectors['frozen_random_initial']['W']['0'], random['initial_W']['0']))
        self.assertFalse(torch.equal(vectors['frozen_random_initial']['W']['0'], random['W']['0']))
        hashes = {str(p): control.qa.digest(p) for p in [self.args.random_run/'adapter.pt', self.args.das_run/'adapter.pt']}
        original = [p.detach().clone() for p in self.bundle.model.parameters()]
        with patch('torch.optim.Adam', side_effect=AssertionError('optimizer forbidden')), \
             patch('core.qa_generation_adapter.train_generation_adapter', side_effect=AssertionError('training forbidden')):
            output = control.generate(self.args, ns, meta, prepared, vectors, identity, bundle=self.bundle)
        self.assertTrue(output['completed'])
        self.assertEqual(len(output['evaluations']), 1)
        self.assertEqual(set(output['evaluations'][0]['outputs']), set(control.CONDITIONS))
        self.assertTrue(all(len(row['source_batch_ids']) == 2 for row in output['audit']))
        self.assertTrue(all(len(row['stored_ids']) == 1 for row in output['audit']))
        self.assertTrue(all(x['exact_text_equal'] and x['finished_equal'] for x in output['cached_reproduction']))
        for p, h in hashes.items(): self.assertEqual(control.qa.digest(p), h)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(original,self.bundle.model.parameters())))
        for text in self.bundle.model.generated_prompts:
            self.assertNotIn('GOLD_',text); self.assertNotIn('WRONG_',text)
        self.assertTrue(output['evaluations'][0]['generation_token_ids']['frozen_random_initial'])
        # A good resume retains data rather than generating again.
        with patch.object(self.bundle.model, 'generate', side_effect=AssertionError('resume must not generate')):
            again = control.generate(self.args, ns, meta, prepared, vectors, identity, bundle=self.bundle)
        self.assertTrue(again['completed'])

    def test_reference_mismatch_cannot_be_resumed_as_success(self):
        ns, meta, bank, prepared, vectors, identity = self.loaded
        p = self.args.random_run / 'tune.json'
        data = control.qa.read(p)
        data['evaluations'][0]['outputs']['baseline'] = 'foreign output'
        control.qa.save(p,data)
        ns, meta, bank, prepared, vectors, identity = control.inputs(self.args)
        with self.assertRaisesRegex(RuntimeError,'did not reproduce'):
            control.generate(self.args,ns,meta,prepared,vectors,identity,bundle=self.bundle)
        result = control.qa.read(self.args.run_dir/'tune.json')
        self.assertFalse(result['completed']); self.assertTrue(result['reproduction_mismatches'])
        with self.assertRaisesRegex(ValueError,'failed reproduction'):
            control.generate(self.args,ns,meta,prepared,vectors,identity,bundle=self.bundle)


if __name__ == '__main__': unittest.main(verbosity=2)
