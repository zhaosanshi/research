"""Three TRAIN questions only: real-model execution acceptance, never a quality result.

Uses a separate copy of bank/questions and a distinct run directory. Existing
services, model files, scientific runs and judge caches are not modified.
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--code', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    sys.path[:0] = [str(args.code), str(args.code / 'scripts')]
    import e43_qa_prompt_das as run
    from core.qa_precision import fp32_output_projection

    torch.set_num_threads(4)
    torch.manual_seed(42)
    args.out.mkdir(parents=True, exist_ok=False)
    source = args.out / 'source'
    source.mkdir()
    originals = {n: digest(args.source / n) for n in ('bank.json', 'questions.json')}
    meta = run.read(args.source / 'bank.json')
    # Reserve substantial headroom on each GPU. The head lives on GPU1, which
    # has fewer blocks because its temporary fp32 projection needs extra memory.
    mapping = {'model.embed_tokens': 0, 'model.norm': 3, 'lm_head': 1}
    for layer in range(meta['num_layers']):
        mapping['model.layers.%d' % layer] = 0 if layer < 8 else (1 if layer < 14 else (2 if layer < 22 else 3))
    meta['model_config'] = {**meta['model_config'], 'device_map': mapping, 'attn_implementation': 'eager'}
    run.save(source / 'bank.json', meta)
    (source / 'questions.json').write_bytes((args.source / 'questions.json').read_bytes())
    prepared = run.read(source / 'questions.json')
    selected = [q['pair_id'] for q in prepared['e_train'] if run.gold_target_text(q)[0] is not None][:3]
    if len(selected) != 3:
        raise RuntimeError('three eligible training questions are required')
    settings = SimpleNamespace(source_run=source, run_dir=args.out / 'pilot',
        prompts=args.code / 'configs/prompts/default_v1.json', phase='freeze', family=None,
        layers=[16], layer=16, arms=['none', 'main'], split='train', splits=['train'],
        epochs=1, lr=1e-3, clip=1.0, seed=42, batch_size=1,
        max_new_tokens=24, max_target_tokens=8, output_projection='float32',
        n_check=0, pair_ids=selected, overwrite=False, include_pilot=True)
    settings.run_dir.mkdir()
    start = time.time()
    report = {'scope': 'Three training questions; one epoch per E/C at L16. No scientific effect estimate.',
              'source_original_sha256': originals, 'selected_training_ids': selected,
              'device_map': mapping, 'max_new_tokens': 24, 'max_target_body_tokens': 8,
              'paid_judge_calls': 0, 'stages': []}

    def mark(stage):
        report['stages'].append({'stage': stage, 'elapsed_seconds': time.time() - start})
        run.save(args.out / 'smoke_progress.json', report)
        print('R27_SMOKE', stage, round(time.time() - start, 2), flush=True)

    mark('start')
    # Directly regress the actual cross-device fp32-head failure before loading
    # the real model. Both forward values and backward gradients must agree.
    from torch import nn
    from torch.nn import functional as F
    class TinyHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(4, 7, bias=True, dtype=torch.bfloat16, device='cuda:1')
            self.requires_grad_(False)
        def get_output_embeddings(self):
            return self.lm_head
    tiny = TinyHead()
    incoming = torch.randn(2, 4, device='cuda:3', requires_grad=True)
    reference = incoming.detach().clone().requires_grad_(True)
    with fp32_output_projection(SimpleNamespace(model=tiny)):
        actual = tiny.lm_head(incoming)
        expected = F.linear(reference.to('cuda:1'), tiny.lm_head.weight.float(), tiny.lm_head.bias.float())
        assert torch.equal(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        assert torch.equal(incoming.grad, reference.grad)
    report['cross_device_projection'] = {'input_device': 'cuda:3', 'head_device': 'cuda:1',
        'output_equal': True, 'input_gradient_equal': True, 'model_weights_unchanged': all(p.grad is None for p in tiny.parameters())}
    del tiny, incoming, reference, actual, expected
    torch.cuda.empty_cache()
    mark('cross_device_projection_regression')
    meta, prepared = run.load_source(settings)
    systems = run.prompt_systems(meta, settings.prompts)
    bundle = run.open_model(meta)
    mark('model_loaded')
    eot = run.pd.turn_terminator_ids(bundle)
    ident, ident_sha = run.run_identity(meta, systems, settings, eot, bundle)
    report['identity_sha256'] = ident_sha
    report['model_files'] = ident.get('model_files')
    with fp32_output_projection(bundle):
        run.run_freeze(settings, meta, prepared, bundle, systems, ident, ident_sha)
        mark('freeze_three_train_questions')
        for family in ('E', 'C'):
            settings.family = family
            run.run_check(settings, meta, prepared, bundle, systems, ident, ident_sha)
            mark(family + '_check_L16_and_L31')
            run.run_train(settings, meta, prepared, bundle, systems, ident, ident_sha)
            mark(family + '_three_pairs_one_epoch')
            settings.arms = ['none', 'main', 'md_main', 'random_main']
            if family == 'E':
                settings.arms += ['recover', 'perq']
            else:
                settings.arms += ['to_conf', 'conf_to_caut', 'caut_to_conf']
            run.run_generate(settings, meta, prepared, bundle, systems, ident, ident_sha)
            mark(family + '_generation_controls')
        run.run_summarize(settings)
    report['source_originals_unchanged'] = all(digest(args.source / n) == h for n, h in originals.items())
    assert report['source_originals_unchanged']
    report['peak_allocated_bytes'] = {str(i): torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())}
    report['elapsed_seconds'] = time.time() - start
    report['execution_passed'] = True
    report['interpretation'] = 'Execution/gradient/real-write checks only; the outputs are not judged and do not establish E/C improvement.'
    run.save(args.out / 'smoke_result.json', report)
    print('R27_SMOKE_COMPLETE', json.dumps({k: report[k] for k in ('execution_passed', 'elapsed_seconds', 'source_originals_unchanged')}), flush=True)


if __name__ == '__main__':
    main()
