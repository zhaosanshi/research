"""Exercise production review-script helpers without loading models or calling a judge."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


CODE = Path(__file__).resolve().parents[1] / 'from_claude/code'


def functions(path, names):
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    ns = {'np': np, 'Path': Path, 'json': json, 'hashlib': hashlib}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), ns)
    return ns


class DirectionCacheTests(unittest.TestCase):
    def prepare(self, configs, loader=None):
        p = CODE / 'scripts/review_checks/verify_own_answers.py'
        tree = ast.parse(p.read_text())
        ns = functions(p, ['dir_key', 'specs_for', 'keep_by_margin'])
        calls = []

        def load(directory, tag, layer, variant, align_to):
            calls.append((str(Path(directory).resolve()), variant, layer))
            return np.array([float(layer), 1. if variant == 'tf' else -1.]), {}

        layers = {l for v in configs.values() if v for l in v['layers']}
        ns.update(configs=configs, dirs={l: {'md': np.array([0., 1.]), 'cB': np.array([3., 4.])} for l in layers},
                  load_das_direction=loader or load,
                  SteerSpec=lambda w, l, **kw: {'W': w, 'layer': l, **kw})
        loop = next(n for n in tree.body if isinstance(n, ast.For)
                    and 'dir_key(v) not in dirs[l]' in ast.get_source_segment(p.read_text(), n))
        exec(compile(ast.Module(body=[loop], type_ignores=[]), str(p), 'exec'), ns)
        return ns, calls

    def spec(self, variant='tf', directory='/old', layers=None):
        return {'estimator': 'das', 'layers': layers or [11], 'alpha': 3.,
                'das_dir': directory, 'das_variant': variant}

    def test_new_and_old_same_layer_load_distinct_variants(self):
        old, new = self.spec(), self.spec('tf-ce-bidir')
        ns, calls = self.prepare({'old': old, 'new': new})
        self.assertEqual(len(calls), 2)
        np.testing.assert_array_equal(ns['specs_for'](old)[0]['W'], [11., 1.])
        np.testing.assert_array_equal(ns['specs_for'](new)[0]['W'], [11., -1.])

    def test_distinct_directory_is_not_aliased_and_relative_alias_is_reused(self):
        a, b = self.spec(directory='/a'), self.spec(directory='/b')
        alias = self.spec(directory='/a/../a')
        ns, calls = self.prepare({'a': a, 'b': b, 'alias': alias})
        self.assertEqual(len(calls), 2)
        self.assertEqual(len([k for k in ns['dirs'][11] if isinstance(k, tuple) and k[0] == 'das']), 2)

    def test_same_source_single_and_multilayer_window_share_only_matching_layers(self):
        single, window = self.spec(), self.spec(layers=[10, 11, 12])
        md = {'estimator': 'md', 'layers': [11], 'alpha': 2.}
        ns, calls = self.prepare({'single': single, 'window': window, 'md': md, 'absent': None})
        self.assertEqual(len(calls), 3)
        sp = ns['specs_for'](window)
        self.assertEqual([x['layer'] for x in sp], [10, 11, 12])
        for x in sp:
            np.testing.assert_array_equal(x['W'], [float(x['layer']), 1.])
            np.testing.assert_array_equal(x['center'], [3., 4.])
            self.assertEqual((x['mode'], x['alpha']), ('amplify', 3.))
        np.testing.assert_array_equal(ns['specs_for'](md)[0]['W'], [0., 1.])
        self.assertIsNone(ns['specs_for'](None))

    def test_missing_requested_direction_fails(self):
        with self.assertRaises(SystemExit):
            self.prepare({'old': self.spec()}, lambda *a, **kw: (None, None))

    def test_own_verdict_tie_matches_tf_and_tuned_gate_keeps_its_rule(self):
        ns, _ = self.prepare({})
        m = np.array([-1., 0., 1., np.nan]); o = np.array([1, -1, 1, 0])
        np.testing.assert_array_equal(ns['keep_by_margin'](m, o, 0., True), [False, False, True, True])
        np.testing.assert_array_equal(ns['keep_by_margin'](m, o, 0.), [False, True, True, True])


class DetectorPayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = functions(CODE / 'scripts/review_checks/own_answer_tfdetect.py',
                           ['auroc', 'file_sha256', 'json_sha256', 'vector_sha256',
                            'checkpoint_identity', 'detector_scores', 'audit_payload', 'legacy_payload_fields'])

    def fixture(self):
        h = np.array([[2., 0.], [1., 0.], [-1., 0.], [-2., 0.]])
        data = {'tune': {'y': np.array([1, 1, 0, 0]), 'ids': ['a', 'b', 'c', 'd'],
                         'H': {1: h, 5: -h}, 'verdict': np.array([2., 1., -1., -2.])},
                'test': {'y': np.array([0, 1, 0, 1]), 'ids': ['e', 'f', 'g', 'h'],
                         'H': {1: h[::-1], 5: -h[::-1]}, 'verdict': np.array([-2., 2., -1., 1.])}}
        dirs = {'md': {5: np.array([1., 0.]), 1: np.array([1., 0.])},
                'dasce': {5: np.array([1., 0.])}, 'daskl': {1: np.array([1., 0.])}}
        return data, dirs

    def test_orientation_tie_break_and_selection_do_not_read_test_labels(self):
        data, dirs = self.fixture()
        scores, picks, table = self.ns['detector_scores'](data, dirs)
        self.assertEqual(picks['md'], 'md_L1')
        self.assertEqual(table['md_L5']['sign'], -1.)
        self.assertEqual(table['md_L1']['sign'], 1.)
        altered = copy.deepcopy(data)
        altered['test']['y'] = 1 - data['test']['y']
        altered['test']['H'][1] = data['test']['H'][1] * -50
        _, picks2, table2 = self.ns['detector_scores'](altered, dirs)
        self.assertEqual(picks, picks2)
        self.assertEqual(table, table2)

    def test_every_score_id_label_sign_and_selection_can_be_replayed(self):
        data, dirs = self.fixture()
        scores, picks, table = self.ns['detector_scores'](data, dirs)
        payload = self.ns['audit_payload'](data, scores, picks, table, {'md': {}}, {'tune': {'sha256': 'abc'}}, {'records': []})
        # Actual JSON serialization is part of this test, not just dictionary construction.
        loaded = json.loads(json.dumps(payload, allow_nan=False))
        self.assertEqual(loaded['selection']['split'], 'tune')
        self.assertEqual(loaded['data']['tune']['ids'], ['a', 'b', 'c', 'd'])
        self.assertEqual(loaded['data']['test']['labels'], [0, 1, 0, 1])
        self.assertEqual(set(loaded['data']['test']['scores']), set(scores))
        for split, index in [('tune', 0), ('test', 1)]:
            for fam, pick in picks.items():
                self.assertEqual(loaded['selected_scores'][split][fam], scores[pick][index].tolist())
        replayed = max([n for n, v in loaded['selection']['candidates'].items() if v['family'] == 'md'],
                       key=lambda n: (loaded['selection']['candidates'][n]['tune_auroc'],
                                      -loaded['selection']['candidates'][n]['layer']))
        self.assertEqual(replayed, loaded['selection']['picks']['md'])

    def test_bad_identity_nonfinite_or_misaligned_payload_is_rejected(self):
        for mode in ['duplicate', 'overlap', 'nan', 'length', 'label']:
            with self.subTest(mode=mode):
                data, dirs = self.fixture()
                scores, picks, table = self.ns['detector_scores'](data, dirs)
                if mode == 'duplicate': data['test']['ids'][1] = data['test']['ids'][0]
                if mode == 'overlap': data['test']['ids'][0] = data['tune']['ids'][0]
                if mode == 'nan': scores['verdict'][1][0] = np.nan
                if mode == 'length': data['test']['ids'].append('extra')
                if mode == 'label': data['test']['y'][0] = 2
                with self.assertRaises(ValueError):
                    self.ns['audit_payload'](data, scores, picks, table, {}, {}, {})

    def test_claude_npz_schema_and_direction_sources_remain_compatible(self):
        data, dirs = self.fixture()
        scores, picks, candidates = self.ns['detector_scores'](data, dirs)
        dirs['empty'] = {}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'detector.json'
            fields = self.ns['legacy_payload_fields'](out, data, scores, dirs, candidates, '/root', 'model', 192)
            path = out.with_suffix('.items.npz')
            self.assertEqual(fields['items_payload'], 'detector.items.npz')
            self.assertEqual(fields['items_payload_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(fields['direction_sources']['dasce'],
                             {'dir': '/root/results_das_tfce/E1/model', 'variant': 'tf-ce-bidir'})
            self.assertEqual(fields['direction_sources']['md']['dir'], None)
            self.assertEqual(fields['signs_from_tuning']['md'], {'1': 1., '5': -1.})
            self.assertEqual(fields['per_layer_tune_auroc']['md'], {'1': 1., '5': 1.})
            with np.load(path, allow_pickle=False) as z:
                self.assertEqual(z['ids_tune'].tolist(), data['tune']['ids'])
                self.assertEqual(z['ids_test'].tolist(), data['test']['ids'])
                np.testing.assert_array_equal(z['y_tune'], data['tune']['y'])
                np.testing.assert_array_equal(z['y_test'], data['test']['y'])
                np.testing.assert_array_equal(z['verdict_test'], data['test']['verdict'])
                self.assertEqual(z['layers_md'].tolist(), [1, 5])
                np.testing.assert_array_equal(z['scores_tune_md'], np.stack([scores['md_L1'][0], scores['md_L5'][0]]))
                np.testing.assert_array_equal(z['scores_test_md'], np.stack([scores['md_L1'][1], scores['md_L5'][1]]))
                self.assertEqual(z['scores_tune_empty'].shape, (0, 4))
                self.assertEqual(z['scores_test_empty'].shape, (0, 4))

    def test_checkpoint_identity_binds_real_file_variant_and_actual_vector(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'checkpoint.pt'; p.write_bytes(b'old checkpoint bytes')
            fn = self.ns['checkpoint_identity']
            a = fn(d, 'tf', {'files': [p.name]}, [1., 0.])
            self.assertEqual(a['checkpoint_files'][p.name], hashlib.sha256(p.read_bytes()).hexdigest())
            p.write_bytes(b'new checkpoint bytes')
            b = fn(d, 'tf-ce-bidir', {'files': [p.name]}, [0., 1.])
            self.assertNotEqual(a['checkpoint_files'], b['checkpoint_files'])
            self.assertNotEqual(a['variant'], b['variant'])
            self.assertNotEqual(a['loaded_aligned_vector_sha256'], b['loaded_aligned_vector_sha256'])
            with self.assertRaises(ValueError): fn(d, 'tf', {}, [1., 0.])
            with self.assertRaises(FileNotFoundError): fn(d, 'tf', {'files': ['missing']}, [1., 0.])


if __name__ == '__main__':
    unittest.main()
