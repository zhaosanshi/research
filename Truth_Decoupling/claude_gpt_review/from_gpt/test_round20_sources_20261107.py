"""Reject mismatched E26 sources using actual on-disk checkpoints."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'from_claude/code/scripts'), str(HERE.parent / 'from_claude/code')]
import e26_matched_coverage as joint


class SourceBindingTests(unittest.TestCase):
    @staticmethod
    def checkpoint(path, seed=42, layer=0, offset=0.):
        w = torch.tensor([1., offset, 0., 0.]); w = w / w.norm()
        torch.save({'W': w[:, None], 'mean_gap': [1.], 'losses': [1., .5], 'seed': seed,
                    'n_pairs': 96, 'layer': layer, 'rank': 1, 'loss_kind': 'ce',
                    'bidirectional': True, 'positions': 'last', 'anchor': 'prompt',
                    'T': 1, 'epochs': 2, 'lr': .001, 'weighted': False, 'trim': 0.}, path)

    def test_source_relocation_requires_identical_bytes_not_matching_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, relocated = root / 'original', root / 'relocated'
            original.mkdir(); relocated.mkdir()
            name = 'das_E_L0_r1_s42_tf-ce-bidir.pt'
            self.checkpoint(original / name)
            (relocated / name).write_bytes((original / name).read_bytes())
            sweep = root / 'sweep.json'
            joint.save(sweep, {'das': {'dir': str(original), 'variant': 'tf-ce-bidir',
                'per_layer': {'0': {'files': [name]}}}})
            self.assertIn(name, joint.verify_tf_sweep_source(sweep, relocated)['checkpoint_sha256'])
            self.checkpoint(relocated / name, offset=.1)
            with self.assertRaisesRegex(ValueError, 'checkpoints differ'):
                joint.verify_tf_sweep_source(sweep, relocated)
            (original / name).unlink()
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                joint.verify_tf_sweep_source(sweep, relocated)
            # New sweeps carry immutable content hashes and remain portable
            # after the source mount is removed.
            metadata = joint.read(sweep)
            metadata['das']['checkpoint_sha256'] = {name: joint.digest(relocated / name)}
            joint.save(sweep, metadata)
            self.assertIsNone(joint.verify_tf_sweep_source(sweep, relocated)['legacy_limit'])
            (relocated / name).write_bytes(b'changed-after-sweep')
            with self.assertRaisesRegex(ValueError, 'checkpoints differ'):
                joint.verify_tf_sweep_source(sweep, relocated)

    def test_extra_seed_consumed_by_real_loader_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = 'das_E_L0_r1_s42_tf-ce-bidir.pt'
            self.checkpoint(root / name)
            sweep = root / 'sweep.json'
            joint.save(sweep, {'das': {'dir': str(root), 'variant': 'tf-ce-bidir',
                'per_layer': {'0': {'files': [name]}}}})
            before, _ = joint.load_das_direction(root, 'E', 0, variant='tf-ce-bidir')
            self.checkpoint(root / 'das_E_L0_r1_s43_tf-ce-bidir.pt', seed=43, offset=.1)
            after, used = joint.load_das_direction(root, 'E', 0, variant='tf-ce-bidir')
            self.assertEqual(len(used['files']), 2)
            self.assertFalse((before == after).all())
            with self.assertRaisesRegex(ValueError, 'loader checkpoint set'):
                joint.verify_tf_sweep_source(sweep, root)

    def test_layer_manifest_cannot_misassign_or_omit_deployed_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); name = 'das_E_L0_r1_s42_tf-ce-bidir.pt'
            self.checkpoint(root / name)
            sweep = root / 'sweep.json'
            joint.save(sweep, {'das': {'dir': str(root), 'variant': 'tf-ce-bidir',
                'per_layer': {'1': {'files': [name]}}}})
            with self.assertRaisesRegex(ValueError, 'loader checkpoint set'):
                joint.verify_tf_sweep_source(sweep, root)
            metadata = joint.read(sweep); metadata['das']['per_layer'] = {'0': {'files': [name]}}
            joint.save(sweep, metadata)
            with self.assertRaisesRegex(ValueError, 'layer coverage'):
                joint.verify_tf_sweep_source(sweep, root, expected_layers=[0, 1])

    def test_other_variants_and_ranks_not_consumed_are_not_false_positives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); name = 'das_E_L0_r1_s42_tf-ce-bidir.pt'
            self.checkpoint(root / name)
            for unused in ('das_E_L0_r1_s43_tf-kl.pt', 'das_E_L0_r2_s43_tf-ce-bidir.pt'):
                (root / unused).write_bytes(b'not consumed by this loader')
            sweep = root / 'sweep.json'
            joint.save(sweep, {'das': {'dir': str(root), 'variant': 'tf-ce-bidir',
                'per_layer': {'0': {'files': [name]}}}})
            result = joint.verify_tf_sweep_source(sweep, root, expected_layers=[0])
            self.assertEqual(result['loader_files'], {'0': [name]})

    def test_old_wrong_variant_and_absent_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'sweep.json'
            for source in ({}, {'dir': tmp, 'variant': 'tf-kl'},
                           {'dir': tmp, 'variant': 'tf-ce-bidir', 'per_layer': {}}):
                joint.save(path, {'das': source})
                with self.assertRaises(ValueError):
                    joint.verify_tf_sweep_source(path, tmp)


if __name__ == '__main__':
    unittest.main()
