"""Three train questions only, physical GPUs 1-3 only, no judges or protected-tree edits."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--code', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True, help='isolated train-only three-question input')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible or any(v not in ('1', '2', '3') for v in visible.split(',')):
        raise RuntimeError('physical CUDA_VISIBLE_DEVICES must be restricted to 1-3')
    sys.path[:0] = [str(a.code), str(a.code / 'scripts')]
    import e44_factorial_das as R
    from core.qa_precision import fp32_output_projection
    torch.set_num_threads(4)
    torch.manual_seed(42)
    a.out.mkdir(parents=True, exist_ok=False)
    args = SimpleNamespace(source_run=a.source, run_dir=a.out, phase='freeze',
        prompts=a.code / 'configs/prompts/default_v1.json',
        prompts_e44=a.code / 'configs/prompts/e44_factorial_v1.json',
        family=None, e_on='E1', e_off='none', layers=[16], arms=[], bases=[], split='train', splits=['train'],
        loss='jsd', positions='content', init='random', val_n=1, epochs=1,
        lr=1e-3, clip=1., seed=42, split_seed=42, batch_size=1, mc_batch_size=4,
        max_new_tokens=24, max_target_tokens=8, max_ref_tokens=96,
        output_projection='float32', n_check=3, pair_ids=None, overwrite=False, include_pilot=True)
    before = {n: R.file_sha(a.source / n) for n in ('bank.json', 'questions.json')}
    meta, prepared = R.load_source(args)
    if len(prepared['splits']['train']) != 3 or prepared['splits'].get('test'):
        raise RuntimeError('only the three copied training questions may enter this acceptance')
    cfg, systems = R.load_prompts(meta, args.prompts, args.prompts_e44)
    started = time.time()
    result = {'scope': 'three-train-question execution acceptance; not a science result',
              'physical_cuda_visible_devices': visible, 'question_ids': [q['pair_id'] for q in prepared['splits']['train']],
              'source_sha256': before, 'test_questions_accessed': 0, 'paid_calls': 0, 'stages': []}
    def mark(stage):
        result['stages'].append({'stage': stage, 'seconds': time.time() - started})
        R.save(a.out / 't.progress.json', result)
        print('R28', stage, round(time.time() - started, 2), flush=True)
    mark('start')
    bundle = R.e43.open_model(meta)
    eot = R.pd.turn_terminator_ids(bundle)
    ident, ident_sha = R.run_identity(meta, systems, cfg, args, eot, bundle)
    result['identity_sha256'] = ident_sha
    mark('model_loaded')
    with fp32_output_projection(bundle):
        R.run_freeze(args, meta, prepared, bundle, cfg, systems, ident, ident_sha)
        mark(str(len(systems)) + '_prompts_three_questions_frozen')
        args.bases = list(systems)
        R.run_mc(args, meta, prepared, bundle, cfg, systems, ident, ident_sha)
        mark('prompt_reference_scores_no_EOT')
        args.bases = []
        for family in ('E', 'C'):
            args.family = family
            R.run_train(args, meta, prepared, bundle, cfg, systems, ident, ident_sha)
            mark(family + '_L16_two_train_one_validation_one_epoch')
            args.arms = (['conf.wq.p1', 'conf_E1.wq.m1', 'caut.wq.p1', 'conf.iso.p1',
                          'x.w.p1', 'x.md.p1', 'x.rand.p1', 'conf.dq.p1'] if family == 'E' else
                         ['conf.wq.p1', 'caut.wq.m1', 'conf_E1.wq.p1', 'conf.iso.p1',
                          'x.w.p1', 'x.w.m1', 'x.md.p1', 'x.rand.p1', 'conf.dq.p1'])
            R.run_generate(args, meta, prepared, bundle, cfg, systems, ident, ident_sha)
            mark(family + '_generation_controls_and_isolation')
            args.arms = ['conf.wq.p1', 'conf.iso.p1', 'x.w.p1', 'x.rand.p1']
            R.run_mc(args, meta, prepared, bundle, cfg, systems, ident, ident_sha)
            mark(family + '_reference_scores_same_write')
        R.run_summarize(args)
    assert all(R.file_sha(a.source / n) == h for n, h in before.items())
    result.update(execution_passed=True, source_unchanged=True,
                  elapsed_seconds=time.time() - started,
                  peak_allocated_bytes={str(i): torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())})
    R.save(a.out / 't.smoke_result.json', result)
    print('R28_SMOKE_PASSED', result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
