"""CPU tokenizer audit for the exact proposed QA-DAS self-check suffix.

Loads local tokenizers and processed data, never model weights or an API.
Requires the experiment's TD_DATA_DIR and TD_MODEL_ROOT environment variables.
"""
import argparse
import json
from pathlib import Path
import sys
import types

SUFFIX = ('\n\nIs the answer above factually correct?\n'
          'Reply with exactly one word: True or False.\nVerdict:')
LABELS = [' True', ' False']

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--code-root', type=Path, required=True)
parser.add_argument('--model-config-dir', type=Path,
                    help='fallback for model configurations absent from the code snapshot')
args = parser.parse_args()
sys.path[:0] = [str(args.code_root / 'scripts'), str(args.code_root)]
from core.config import load_json_config, resolve_config_path
from core.model_io import ModelBundle
from core.qa_context import encode_answer
import e25_qa_context as e25
from transformers import AutoTokenizer

cfg = load_json_config(resolve_config_path('experiments', 'E25_qa_context'))
cfg['qa_system'] = load_json_config(resolve_config_path('prompts', 'default_v1'))['qa']['system_idk_ok']
prepared = e25.split_questions(cfg)
report = {'suffix': SUFFIX, 'labels': LABELS, 'system': cfg['qa_system'], 'models': {}}
for name in ['llama3.1-8b', 'qwen3-8b', 'gemma3-4b']:
    path = args.code_root / 'configs/models' / (name + '.json')
    if not path.exists() and args.model_config_dir:
        path = args.model_config_dir / (name + '.json')
    model_cfg = load_json_config(str(path))
    tok = AutoTokenizer.from_pretrained(model_cfg['path'], local_files_only=True, trust_remote_code=True)
    bundle = types.SimpleNamespace(cfg=model_cfg, tokenizer=tok)
    bundle.build_chat = types.MethodType(ModelBundle.build_chat, bundle)
    result = {'n_answers': 0, 'natural_concat_retokenizes_answer': 0,
              'segmented_decode_mismatch': 0, 'label_prefix_mismatch': 0,
              'label_ids': {}, 'examples': [], 'model_config': model_cfg}
    suffix_ids = tok(SUFFIX, add_special_tokens=False)['input_ids']
    for split in ['e_train', 'e_tune']:
        for row in prepared[split]:
            for key in ['positive_answer', 'negative_answer']:
                encoded = encode_answer(bundle, cfg['qa_system'], row['question'], row[key], cfg['qa_user'])
                answer_ids = encoded['full_ids']
                text = encoded['full_text']
                natural = tok(text + SUFFIX, add_special_tokens=False)['input_ids']
                result['n_answers'] += 1
                if natural[:len(answer_ids)] != answer_ids:
                    result['natural_concat_retokenizes_answer'] += 1
                    if len(result['examples']) < 2:
                        result['examples'].append({
                            'pair_id': row['pair_id'], 'answer': row[key],
                            'answer_tail_ids': answer_ids[-5:],
                            'natural_prefix_tail_ids': natural[max(0, len(answer_ids)-5):len(answer_ids)+2],
                        })
                segmented = answer_ids + suffix_ids
                result['segmented_decode_mismatch'] += tok.decode(segmented, skip_special_tokens=False) != text + SUFFIX
                for label in LABELS:
                    full = tok(SUFFIX + label, add_special_tokens=False)['input_ids']
                    result['label_prefix_mismatch'] += full[:len(suffix_ids)] != suffix_ids
                    ids = full[len(suffix_ids):]
                    result['label_ids'].setdefault(label, [])
                    if ids not in result['label_ids'][label]:
                        result['label_ids'][label].append(ids)
                    assert ids and not (set(ids) & set(tok.all_special_ids))
                    assert tok.decode(segmented + ids, skip_special_tokens=False) == text + SUFFIX + label
    assert result['segmented_decode_mismatch'] == result['label_prefix_mismatch'] == 0
    assert result['label_ids'][LABELS[0]] != result['label_ids'][LABELS[1]]
    report['models'][name] = result
print(json.dumps(report, ensure_ascii=False, indent=2))
