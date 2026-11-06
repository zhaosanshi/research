import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parents[2]
CODE = REVIEW / 'from_claude/code'
sys.path[:0] = [str(REVIEW / 'from_gpt'), str(CODE / 'scripts'), str(CODE)]
spec = importlib.util.spec_from_file_location('trace_corruption', CODE / 'scripts/review_checks/e27_trace_corruption.py')
trace = importlib.util.module_from_spec(spec); spec.loader.exec_module(trace)
from core.intervene import SteerSpec, steer
from core.model_io import ModelBundle, forward_logits
from core.qa_generation_adapter import adapter_specs, train_generation_adapter
import test_qa_das_pipeline_20261105 as fixtures


class ByteTokenizer:
    all_special_ids = [0]
    def __init__(self):
        self.pieces = {0: b'', 1: b'A', 2: b'\xf0\x9f', 3: b'\x98\x80',
                       4: b'.', 5: b'\xef\xbf\xbd', 6: '你好'.encode(), 7: b'\x80'}
        enc = {v: k for k, v in trace.byte_decoder().items()}
        self.tokens = {i: ''.join(enc[b] for b in bs) for i, bs in self.pieces.items()}
        self.tokens[0] = '<eos>'; self.tokens[6] = '你好'
    def convert_ids_to_tokens(self, ids):
        return [self.tokens[i] for i in ids] if isinstance(ids, list) else self.tokens[ids]
    def decode(self, ids, skip_special_tokens=True, **kwargs):
        return b''.join(self.pieces[i] for i in ids if i not in self.all_special_ids).decode('utf-8', errors='replace')


class BytesTests(unittest.TestCase):
    def test_split_valid_emoji_is_not_final_corruption(self):
        result = trace.final_byte_analysis(ByteTokenizer(), [2, 3, 0], [0])
        self.assertEqual(result['default_decoded'], '😀')
        self.assertEqual(result['decoded_prefixes'][0], '\ufffd')
        self.assertEqual(result['invalid_utf8_events'], [])
        self.assertEqual(result['literal_replacement_events'], [])
        self.assertEqual(result['temporary_prefix_replacement_steps'], [0])

    def test_incompatible_following_token_is_distinct_from_lead_byte(self):
        result = trace.final_byte_analysis(ByteTokenizer(), [2, 4, 0], [0])
        self.assertEqual(result['default_decoded'], '\ufffd.')
        event = result['invalid_utf8_events'][0]
        self.assertEqual(event['bytes_hex'], 'f09f')
        self.assertEqual(event['origin_token_index'], 0)
        self.assertEqual(event['deciding_token_index'], 1)

    def test_eos_can_close_an_incomplete_byte_sequence(self):
        result = trace.final_byte_analysis(ByteTokenizer(), [1, 2, 0], [0])
        event = result['invalid_utf8_events'][0]
        self.assertEqual(event['origin_token_index'], 1)
        self.assertEqual(event['deciding_token_index'], 2)
        self.assertEqual(event['reason'], 'unexpected end of data')

    def test_literal_replacement_is_not_invalid_utf8(self):
        result = trace.final_byte_analysis(ByteTokenizer(), [5, 0], [0])
        self.assertTrue(result['byte_mapping_verified'])
        self.assertEqual(result['invalid_utf8_events'], [])
        self.assertEqual(result['literal_replacement_events'][0]['bytes_hex'], 'efbfbd')

    def test_isolated_continuation_is_invalid_immediately(self):
        result = trace.final_byte_analysis(ByteTokenizer(), [1, 7, 4, 0], [0])
        event = result['invalid_utf8_events'][0]
        self.assertEqual(event['reason'], 'invalid start byte')
        self.assertEqual(event['origin_token_index'], 1)
        self.assertEqual(event['deciding_token_index'], 1)

    def test_unsupported_decoder_does_not_fabricate_attribution(self):
        result = trace.final_byte_analysis(ByteTokenizer(), [6, 0], [0])
        self.assertFalse(result['byte_mapping_verified'])
        self.assertEqual(result['default_decoded'], '你好')
        self.assertEqual(result['invalid_utf8_events'], [])
        self.assertIn('attribution_limit', result)


class ExecutionTests(unittest.TestCase):
    def test_replay_masks_preserve_padding_and_predictor_positions(self):
        inputs = {'input_ids': torch.tensor([[0, 1, 2], [3, 4, 5]]),
                  'attention_mask': torch.tensor([[0, 1, 1], [1, 1, 1]])}
        packed = trace.replay_packed(inputs, torch.tensor([[7, 8], [9, 10]]))
        self.assertEqual(packed['inputs']['input_ids'].tolist(), [[0, 1, 2, 7, 8], [3, 4, 5, 9, 10]])
        self.assertEqual(packed['prompt_end_mask'].nonzero().tolist(), [[0, 2], [1, 2]])
        self.assertEqual(packed['predictor_mask'].nonzero().tolist(), [[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4]])

    def test_observers_leave_production_output_unchanged_and_track_center_switch(self):
        block = torch.nn.Identity()
        bundle = SimpleNamespace(layer=lambda l: block, device=torch.device('cpu'))
        def spec():
            return SteerSpec([1., 0., 0., 0.], 0, mode='amplify', alpha=2., center=torch.zeros(4), decode_center=torch.ones(4) * 3)
        first = torch.tensor([[[1., 2., 3., 4.]], [[2., 3., 4., 5.]]])
        second = torch.tensor([[[7., 2., 3., 4.]], [[8., 3., 4., 5.]]])
        with steer(bundle, [spec()]):
            expected1, expected2 = block(first), block(second)
        records = []
        with trace.observed_steer(bundle, [spec()], [0, 1], records):
            actual1, actual2 = block(first), block(second)
        self.assertTrue(torch.equal(actual1, expected1)); self.assertTrue(torch.equal(actual2, expected2))
        self.assertEqual([x['center_kind'] for x in records], ['prompt', 'prompt', 'decode', 'decode'])
        self.assertEqual([x['step'] for x in records], [0, 0, 1, 1])
        self.assertEqual(records[2]['kept_delta_norm'], 8.)
        self.assertAlmostEqual(records[2]['kept_relative_norm'], 8 / float(second[0, -1].norm()))
        self.assertFalse(block._forward_hooks)
        with self.assertRaises(RuntimeError):
            with trace.observed_steer(bundle, [spec()], [0], []):
                block(first); raise RuntimeError('test failure')
        self.assertFalse(block._forward_hooks)

    def test_common_prefix_replays_use_real_production_operator(self):
        tok = fixtures.ToyTokenizer()
        tok.convert_ids_to_tokens = lambda value: [str(i) for i in value] if isinstance(value, list) else str(value)
        model = fixtures.PipelineModel(tok)
        bundle = ModelBundle(model, tok, {'name': 'trace-toy', 'layers_attr': 'layers', 'chat': {}})
        cfg = {'qa_system': 'Answer.', 'qa_user': 'Q: {question}\nA:', 'batch_size': 2, 'max_new_tokens': 2}
        records = [{'pair_id': 'p1', 'question': 'Where?', 'positive_answer': 'Paris', 'negative_answer': 'Rome'},
                   {'pair_id': 'p2', 'question': 'Who?', 'positive_answer': 'A', 'negative_answer': 'B'}]
        initial = {0: torch.nn.functional.normalize(torch.arange(1., 9.), dim=0)}
        centers = {0: {'prompt': torch.zeros(8), 'decode': torch.ones(8) * .02}}
        adapter = train_generation_adapter(bundle, records, cfg, initial, centers, epochs=1, log_fn=None)
        original = dict(adapter, W={'0': initial[0]})
        inputs = bundle.encode(['One prompt', 'Longer second prompt'])
        generated = torch.tensor([[80, 81], [82, 83]])
        for step in [0, 1]:
            packed = trace.replay_packed(inputs, generated[:, :step])
            with steer(bundle, adapter_specs(adapter, packed)):
                expected = forward_logits(bundle, packed['inputs'])[:, -1]
            scores = [expected] * 2
            result = trace.common_prefix_replays(bundle, inputs, generated, step, [0, 1], adapter, original, scores, 3)
            self.assertEqual(set(result['variants']), {'baseline', 'original_das', 'adapted', 'adapted_history_only'})
            for row in ['0', '1']:
                self.assertEqual(result['variants']['adapted'][row]['cached_processed_score_max_abs_difference'], 0.)
                self.assertTrue(result['variants']['adapted'][row]['cached_top1_match'])
                if step == 0:
                    self.assertEqual(result['variants']['baseline'][row], result['variants']['adapted_history_only'][row])
        self.assertTrue(all(not layer._forward_hooks for layer in model.layers))


if __name__ == '__main__':
    unittest.main()
