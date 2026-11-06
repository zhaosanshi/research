#!/usr/bin/env python3
"""E26: actual-coverage method comparison, TF statements and free-form QA.

Separate commands: prepare, collect-tune, judge-tune, freeze, collect-test,
judge-test, report. Nothing silently starts a later/model/API phase.
The primary table is label-free ranking at a fixed batch quota; the separate
deployment table uses a threshold fixed on tuning data and reports its actual
test coverage. Reference answers are only supplied to the offline QA grader.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import torch

import common
from common import load_split, tf_prompt
from core.config import load_experiment, load_json_config, resolve_config_path
from core.directions import load_das_direction, mean_diff_direction
from core.intervene import steer
from core.joint_steering import (candidate_conditions, choose_conditions,
                                 commitment_direction, fingerprint, intervention_specs)
from core.model_io import capture_last_token, forward_logits
from core.qa_context import answer_logprobs, encode_answer, pack_answers
from core.qa_precision import fp32_output_projection
from core.qa_scoring import reference_answers
from core.readout import build_label_token_map, label_logprobs_from_logits
from core.selective_eval import (apply_threshold, calibrate_threshold,
                                 evaluate_coverage, quota_select, random_score)
import e25_qa_das as qa

PROTOCOL = 'joint_actual_coverage_v1'
TASKS = ('tf', 'qa')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value, immutable=False):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if immutable and path.exists():
        if read(path) != value:
            raise ValueError('refusing to replace a frozen artifact: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def validate_config(cfg):
    if cfg.get('protocol') != PROTOCOL:
        raise ValueError('wrong E26 protocol')
    for k in ('batch_size', 'max_new_tokens', 'e_top_k', 'c_top_k', 'n_boot'):
        if type(cfg.get(k)) is not int or cfg[k] < 1:
            raise ValueError('positive integer required: ' + k)
    if not cfg.get('coverages') or any(not math.isfinite(float(q)) or not 0 < float(q) <= 1 for q in cfg['coverages']):
        raise ValueError('coverages must be finite values in (0,1]')
    if len(set(cfg['coverages'])) != len(cfg['coverages']):
        raise ValueError('duplicate target coverages')
    if not cfg.get('c_alphas') or any(not math.isfinite(float(a)) or float(a) == 0 for a in cfg['c_alphas']):
        raise ValueError('C alphas must be finite nonzero')
    if set(cfg['gate_labels']) != {'yes', 'no'} or cfg['gate_labels']['yes'] == cfg['gate_labels']['no']:
        raise ValueError('need distinct Yes/No labels')


def source_args(args):
    return SimpleNamespace(run_dir=Path(args.qa_run), model=args.model)


def source_paths(args):
    return {'qa_run': str(Path(args.qa_run).resolve()),
            'tf_das_dir': str(Path(args.tf_das_dir).resolve()),
            'tf_das_sweep': str(Path(args.tf_das_sweep).resolve()),
            'tf_md_sweep': str(Path(args.tf_md_sweep).resolve()),
            'qa_adapters': [str(Path(p).resolve()) for p in (getattr(args, 'qa_adapters', None) or [])]}


def select_tf_candidates(path, model, method, expected, k):
    """Only saved tuning columns determine the candidate pool; verify raw split."""
    j = read(path)
    if j.get('model') != model or j.get('e_estimator') != ('das' if method == 'das' else 'meandiff'):
        raise ValueError('TF sweep model/estimator mismatch')
    zpath = Path(path).with_suffix('.npz')
    with np.load(zpath, allow_pickle=True) as z:
        ids, labels, tune = [str(x) for x in z['ids']], z['labels'].astype(bool), z['half_a'].astype(bool)
    expected_by_id = {r['id']: (r['label'], split == 'tune') for split in ('tune', 'test') for r in expected[split]}
    if len(ids) != len(set(ids)) or set(ids) != set(expected_by_id):
        raise ValueError('TF sweep and E26 question cohort differ')
    if any((bool(y), bool(t)) != expected_by_id[i] for i, y, t in zip(ids, labels, tune)):
        raise ValueError('TF labels/tuning split differ')
    scale = int(labels[tune].sum()) * int((~labels[tune]).sum())
    baseline = j['conds']['baseline']['A']
    per_layer = {}
    for name, row in j['conds'].items():
        match = re.fullmatch(r'E_L(\d+)_a([\d.]+)', name)
        if not match or abs(row['A']['ans'] - baseline['ans']) > .03 + 1e-12:
            continue
        layer, alpha = int(match[1]), float(match[2])
        key = (-round((row['A']['J'] - baseline['J']) * scale), alpha, layer)
        if layer not in per_layer or key < per_layer[layer][0]:
            per_layer[layer] = (key, {'layer': layer, 'alpha': alpha, 'mode': 'prefill'})
    choices = [x[1] for x in sorted(per_layer.values(), key=lambda x: x[0])[:k]]
    if not choices:
        raise ValueError('no eligible TF E candidate')
    return choices


def gate_text(bundle, meta, task, row):
    cfg = meta['config']
    system = meta['qa_config']['qa_system'] if task == 'qa' else meta['prompts']['statement']['system']
    template = cfg['qa_gate_user'] if task == 'qa' else cfg['tf_gate_user']
    return bundle.build_chat(system, template.format(question=row.get('question', ''), statement=row.get('text', '')))


def gate_map(bundle, text, cfg):
    m = build_label_token_map(bundle.tokenizer, text, {k: [v] for k, v in cfg['gate_labels'].items()})
    if any(len(set(m[k])) != 1 for k in ('yes', 'no')) or set(m['yes']) & set(m['no']):
        raise ValueError('gate needs distinct single first-token label IDs')
    return {k: m[k][0] for k in ('yes', 'no')}


def prepare(args, cfg, bundle=None):
    validate_config(cfg)
    if args.model not in cfg['models']:
        raise ValueError('model outside this experiment')
    if args.pilot is not None and args.pilot < 2:
        raise ValueError('C estimation pilot needs at least two questions')
    if (args.run_dir / 'bank.json').exists():
        old, _, _ = load_run(args)
        if old['config'] != cfg or old['sources']['paths'] != source_paths(args) or old['pilot'] != args.pilot:
            raise ValueError('prepare resume settings differ; use a new run directory')
        return old
    qargs = source_args(args)
    qmeta, qbank, prepared = qa.load_bank(qargs)
    qmeta, qbank = qa.runtime_bank(qargs, qmeta, qbank)
    if qmeta['pilot'] or qmeta['config']['seed'] != cfg['seed']:
        raise ValueError('main source bank must be a complete same-seed run')
    short, _ = qa.load_shortlist(qargs, qmeta, prepared)
    exp = load_experiment(cfg['source_experiment'])
    raw = []
    for ds in exp['eval_datasets']:
        raw += load_split(ds, 'calib') + load_split(ds, 'dev') + load_split(ds, exp.get('eval_split', 'test'))
    id_to_split = {str(r['pair_id']): split for split, rows in prepared['splits'].items() for r in rows}
    datasets = {'qa': {s: list(rs) for s, rs in prepared['splits'].items()},
                'tf': {s: [] for s in prepared['splits']}}
    for r in raw:
        pid = str(r.get('pair_id') or r['id'])
        if pid not in id_to_split:
            raise ValueError('TF question not in source QA partition')
        datasets['tf'][id_to_split[pid]].append({'id': str(r['id']), 'pair_id': pid,
            'text': r['text'], 'label': bool(r['label'])})
    for split in ('train', 'tune', 'test'):
        groups = defaultdict(list)
        for r in datasets['tf'][split]:
            groups[r['pair_id']].append(r['label'])
        if set(groups) != {str(q['pair_id']) for q in prepared['splits'][split]} or any(sorted(v) != [False, True] for v in groups.values()):
            raise ValueError('TF split needs one true/false statement per question')
    paths = source_paths(args)
    tf_summary = read(Path(paths['tf_das_dir']) / 'summary.json')
    tf_protocol = tf_summary.get('das') or {}
    expected_tf = {'variant': 'tf-ce-bidir', 'loss': 'ce', 'bidirectional': True,
                   'positions': 'last', 'anchor': 'prompt'}
    if (tf_summary.get('model') != args.model
            or tf_summary.get('num_layers') != qmeta['num_layers']
            or any(tf_protocol.get(k) != v for k, v in expected_tf.items())
            or (tf_summary.get('est_extra') or {}).get('pair_ids')
            or tf_protocol.get('n_pairs', {}).get('E') != len(prepared['splits']['train'])):
        raise ValueError('TF DAS source model/training protocol/estimation cohort differs')
    e_choices = {'tf': {}, 'qa': {}}
    for method in ('das', 'md'):
        e_choices['tf'][method] = select_tf_candidates(paths['tf_' + method + '_sweep'], args.model, method,
                                                      datasets['tf'], cfg['e_top_k'])
        e_choices['qa'][method] = [{'layer': c['layer'], 'alpha': c['alpha'], 'mode': c['mode']}
            for c in short['candidates']['qa_' + method][:cfg['e_top_k']]]
        if not e_choices['qa'][method]:
            raise ValueError('need a complete QA E candidate pool for both estimators')
    sources = {'paths': paths, 'qa_bank_fingerprint': qmeta['fingerprint'],
               'qa_training_sha256': qmeta['training_sha256'], 'files': {}}
    for p in [qargs.run_dir / 'bank.json', qargs.run_dir / 'bank.npz', qargs.run_dir / 'training.json',
              qargs.run_dir / 'questions.json', qargs.run_dir / 'shortlist.json',
              Path(paths['tf_das_dir']) / 'summary.json']:
        sources['files'][str(p.resolve())] = digest(p)
    for p in [Path(paths['tf_das_sweep']), Path(paths['tf_md_sweep'])]:
        for f in (p, p.with_suffix('.npz')):
            sources['files'][str(f.resolve())] = digest(f)
    for p in sorted((Path(paths['tf_das_dir'])).glob('das_E_*tf-ce-bidir.pt')):
        sources['files'][str(p.resolve())] = digest(p)
    adapters = {}
    for path in paths['qa_adapters']:
        from core.qa_generation_adapter import load_adapter
        adapter = load_adapter(path, expected_model=args.model)
        if (adapter['source']['bank_fingerprint'] != qmeta['fingerprint']
                or adapter['model_config'] != qmeta['model_config']
                or adapter['source']['questions_sha256'] != digest(qargs.run_dir / 'questions.json')
                or adapter['config'] != {k: qmeta['config'][k] for k in adapter['config']}):
            raise ValueError('generation adapter was trained from another QA source')
        if adapter.get('pilot') and not args.pilot:
            raise ValueError('a pilot adapter cannot enter a full main experiment')
        method = adapter['method']
        if method in e_choices['qa']:
            raise ValueError('one adapter per initialization per E26 run; compare other settings in separate runs')
        adapters[method] = adapter
        e_choices['qa'][method] = [{'layers': adapter['layers'], 'alpha': adapter['alpha'], 'mode': 'continuous'}]
        directory = Path(path) if Path(path).is_dir() else Path(path).parent
        for filename in ('adapter.pt', 'adapter.json'):
            sources['files'][str((directory / filename).resolve())] = digest(directory / filename)
    if args.pilot:
        for split in ('train', 'tune', 'test'):
            keep = {str(r['pair_id']) for r in datasets['qa'][split][:args.pilot]}
            datasets['qa'][split] = datasets['qa'][split][:args.pilot]
            datasets['tf'][split] = [r for r in datasets['tf'][split] if r['pair_id'] in keep]
    meta = {'protocol': PROTOCOL, 'model': args.model, 'model_config': qmeta['model_config'],
            'hidden_size': qmeta['hidden_size'], 'num_layers': qmeta['num_layers'], 'pilot': args.pilot,
            'config': cfg, 'qa_config': qmeta['config'], 'prompts': exp['prompts'], 'sources': sources,
            'e_choices': e_choices, 'c_estimation': {}, 'c_layers': {},
            'generation_adapters': {method: {'init': a['init'], 'layers': a['layers'], 'alpha': a['alpha'],
                'objective': a['objective'], 'source': a['source']} for method, a in adapters.items()},
            'score_provenance': {'uses_gold_labels': False, 'uses_reference_answer': False,
                'commitment': 'first-token Yes-No logit difference on question/statement prompt',
                'simple_tf': 'absolute predicted True-False log probability margin',
                'simple_qa': 'unsteered mean log probability of generated answer; no reference answers'}}
    if bundle is None:
        bundle = qa.open_bundle(qmeta, prepared)
    else:
        qa.verify_bundle(bundle, qmeta, prepared)
    layers = list(range(bundle.num_layers))
    arrays = {}
    tf_est = datasets['tf']['train']
    with fp32_output_projection(bundle):
        htf, _ = capture_last_token(bundle, [tf_prompt(bundle, exp['prompts'], r['text'], allow_idk=False)
                                            for r in tf_est], layers, batch_size=cfg['batch_size'])
        labels = np.array([r['label'] for r in tf_est])
        for layer in layers:
            h = htf[('resid', layer)].numpy().astype(float)
            wmd, _ = mean_diff_direction(h[labels], h[~labels])
            wd, dm = load_das_direction(paths['tf_das_dir'], 'E', layer, variant='tf-ce-bidir', align_to=wmd)
            if wd is None or any(n != len(prepared['splits']['train']) for n in dm['n_pairs']):
                raise ValueError('missing or foreign TF gold-label DAS checkpoint')
            arrays['tf_md_E_L%d' % layer], arrays['tf_das_E_L%d' % layer] = wmd, wd
            arrays['tf_prompt_center_L%d' % layer] = .5 * (h[labels].mean(0) + h[~labels].mean(0))
            for method in ('das', 'md'):
                key = 'E_qa_%s_L%d' % (method, layer)
                if key in qbank:
                    arrays['qa_%s_E_L%d' % (method, layer)] = qbank[key]
            arrays['qa_prompt_center_L%d' % layer] = qbank['prompt_center_L%d' % layer]
            arrays['qa_decode_center_L%d' % layer] = qbank['decode_center_L%d' % layer]
        for method, adapter in adapters.items():
            for layer in adapter['layers']:
                for label, key in [('prompt', 'prompt_center_L%d' % layer), ('decode', 'decode_center_L%d' % layer)]:
                    if not np.array_equal(adapter['centers'][str(layer)][label].numpy(), np.asarray(qbank[key], np.float32)):
                        raise ValueError('adapter training and main deployment centres differ')
                arrays['qa_%s_E_L%d' % (method, layer)] = adapter['W'][str(layer)].numpy()
        for task in TASKS:
            texts = [gate_text(bundle, meta, task, r) for r in datasets[task]['train']]
            acts, logits = capture_last_token(bundle, texts, layers, batch_size=cfg['batch_size'], return_logits=True)
            label_map = gate_map(bundle, texts[0], cfg)
            margins = logits[:, label_map['yes']].float().numpy() - logits[:, label_map['no']].float().numpy()
            meta['c_estimation'][task] = {'labels': label_map, 'methods': {}}
            meta['c_layers'][task] = {}
            for method in e_choices[task]:
                stats = {}
                for layer in layers:
                    h = acts[('resid', layer)].numpy().astype(float)
                    arrays['%s_gate_center_L%d' % (task, layer)] = h.mean(0)
                    ekey = '%s_%s_E_L%d' % (task, method, layer)
                    if ekey not in arrays:
                        continue
                    w, stat = commitment_direction(h, margins, arrays[ekey])
                    stats[str(layer)] = stat
                    if w is not None:
                        prefix = '%s_%s_C_L%d' % (task, method, layer)
                        arrays[prefix], arrays[prefix + '_scale'] = w, np.array(stat['scale'])
                valid = [int(l) for l, stat in stats.items() if stat['available']]
                meta['c_layers'][task][method] = sorted(valid, key=lambda l: (-stats[str(l)]['score_correlation'], l))[:cfg['c_top_k']]
                meta['c_estimation'][task]['methods'][method] = stats
    meta['conditions'] = {task: candidate_conditions(task, e_choices[task], meta['c_layers'][task], cfg['c_alphas']) for task in TASKS}
    audit_texts = []
    for task in TASKS:
        for row in datasets[task]['train'][:3]:
            audit_texts.append(gate_text(bundle, meta, task, row))
            audit_texts.append(tf_prompt(bundle, meta['prompts'], row['text'], allow_idk=False) if task == 'tf' else
                bundle.build_chat(meta['qa_config']['qa_system'], meta['qa_config']['qa_user'].format(question=row['question'])))
    meta['token_audit'] = [{'text': text, 'input_ids': bundle.encode([text])['input_ids'][0].detach().cpu().tolist()}
                           for text in audit_texts]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    save(args.run_dir / 'data.json', datasets)
    np.savez(args.run_dir / 'bank.npz', **arrays)
    meta['data_sha256'], meta['bank_sha256'] = digest(args.run_dir / 'data.json'), digest(args.run_dir / 'bank.npz')
    meta['fingerprint'] = fingerprint(meta)
    save(args.run_dir / 'bank.json', meta, immutable=True)
    print('PREPARED', {t: len(meta['conditions'][t]) for t in TASKS}, 'pilot=', args.pilot)
    return meta


def load_run(args):
    meta = read(args.run_dir / 'bank.json')
    if meta.get('model') != args.model or meta.get('protocol') != PROTOCOL:
        raise ValueError('wrong model or protocol')
    if fingerprint({k: v for k, v in meta.items() if k != 'fingerprint'}) != meta['fingerprint']:
        raise ValueError('bank metadata changed')
    validate_config(meta['config'])
    for f, key in [('bank.npz', 'bank_sha256'), ('data.json', 'data_sha256')]:
        if digest(args.run_dir / f) != meta[key]:
            raise ValueError('bank/data changed')
    with np.load(args.run_dir / 'bank.npz', allow_pickle=False) as z:
        bank = {k: z[k] for k in z.files}
    if any(not np.isfinite(x).all() for x in bank.values()):
        raise ValueError('nonfinite bank')
    return meta, bank, read(args.run_dir / 'data.json')


def open_bundle(meta):
    from core.model_io import load_model
    bundle = load_model(meta['model_config'])
    if (bundle.hidden_size, bundle.num_layers) != (meta['hidden_size'], meta['num_layers']):
        raise ValueError('model architecture changed')
    verify_tokens(bundle, meta)
    return bundle


def verify_tokens(bundle, meta):
    for row in meta['token_audit']:
        if bundle.encode([row['text']])['input_ids'][0].detach().cpu().tolist() != row['input_ids']:
            raise ValueError('tokenization changed since bank preparation')


def one_step(bundle, texts, specs):
    inputs = bundle.encode(texts)
    if not inputs['attention_mask'][:, -1].all():
        raise ValueError('left padding required')
    with steer(bundle, specs) as hooks:
        logits = forward_logits(bundle, inputs, logits_to_keep=1)[:, -1].detach().float().cpu().numpy()
    if any(s.n_applied != 1 for s in specs) or any(h.call_count != 1 for h in hooks.values()):
        raise RuntimeError('one-step intervention did not write exactly once')
    return logits


def output_records(bundle, meta, bank, task, rows, condition):
    """Inference sees inputs, generated outputs, and bank only; never gold QA answers."""
    cfg = meta['config']
    gate_texts = [gate_text(bundle, meta, task, row) for row in rows]
    label_map = gate_map(bundle, gate_texts[0], cfg)
    gl = one_step(bundle, gate_texts, intervention_specs(bank, condition, 'gate'))
    scores = gl[:, label_map['yes']] - gl[:, label_map['no']]
    if task == 'tf':
        texts = [tf_prompt(bundle, meta['prompts'], row['text'], allow_idk=False) for row in rows]
        labels = meta['prompts']['statement']['label_variants']
        mapping = build_label_token_map(bundle.tokenizer, texts[0], {k: labels[k] for k in ('True', 'False')})
        logits = one_step(bundle, texts, intervention_specs(bank, condition))
        result = []
        for row, logit, score in zip(rows, logits, scores):
            lp = label_logprobs_from_logits(logit, mapping, use_abstain=False)
            margin = float(lp['True'] - lp['False']); pred = margin > 0
            result.append({'id': row['id'], 'pair_id': row['pair_id'], 'output': 'True' if pred else 'False',
                'score': float(score), 'simple_score': abs(margin), 'predicted_true': pred,
                'outcome': 'correct' if pred == row['label'] else 'wrong', 'finished': True})
        return result
    texts = [bundle.build_chat(meta['qa_config']['qa_system'], meta['qa_config']['qa_user'].format(question=r['question'])) for r in rows]
    inputs = bundle.encode(texts); plen = inputs['input_ids'].shape[1]
    specs = intervention_specs(bank, condition)
    with torch.no_grad(), steer(bundle, specs) as hooks:
        output = bundle.model.generate(**inputs, max_new_tokens=cfg['max_new_tokens'], do_sample=False,
            num_beams=1, use_cache=True, pad_token_id=bundle.tokenizer.pad_token_id)
    steps = output.shape[1] - plen
    if any(s.n_applied != (1 if s.max_calls == 1 else steps) for s in specs):
        raise RuntimeError('joint generation E/C edit counts differ from declared positions')
    if any(h.first_shape[1] != plen or h.call_count != steps for h in hooks.values()):
        raise RuntimeError('joint generation call trace failed')
    eos = getattr(bundle.model.generation_config, 'eos_token_id', bundle.tokenizer.eos_token_id)
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    result = []
    for row, tokens, score in zip(rows, output[:, plen:], scores):
        answer = bundle.tokenizer.decode(tokens, skip_special_tokens=True)
        # A simple, unsteered score shared by all candidate generation methods.
        simple = -1e6
        if answer.strip():
            item = encode_answer(bundle, meta['qa_config']['qa_system'], row['question'], answer, meta['qa_config']['qa_user'])
            packed = pack_answers([item], bundle.tokenizer.pad_token_id, bundle.device)
            simple = float(answer_logprobs(forward_logits(bundle, packed['inputs']), packed)[0]) / item['answer_length']
        result.append({'id': str(row['pair_id']), 'pair_id': str(row['pair_id']), 'output': answer,
            'score': float(score), 'simple_score': simple, 'outcome': 'unresolved',
            'finished': any(int(t) in eos for t in tokens),
            'edits': {s.name: s.n_applied for s in specs}})
    return result


def collect(args, meta, bank, data, split, bundle=None):
    selected = frozen_selection(args, meta) if split == 'test' else None
    bundle = open_bundle(meta) if bundle is None else bundle
    verify_tokens(bundle, meta)
    with fp32_output_projection(bundle):
        for task in TASKS:
            specs = meta['conditions'][task]
            if selected:
                names = {'baseline'} | {n for point in selected['tasks'][task]['choices'].values() for n in point.values() if n is not None}
                specs = [s for s in specs if s['name'] in names]
            rows = data[task][split]
            path = args.run_dir / ('%s_%s.json' % (task, split))
            identity = fingerprint([meta['fingerprint'], task, split, specs, rows])
            artifact = read(path) if path.exists() else {'protocol': PROTOCOL, 'model': meta['model'],
                'task': task, 'split': split, 'fingerprint': identity, 'bank_fingerprint': meta['fingerprint'],
                'score_provenance': meta['score_provenance'], 'condition_specs': specs,
                'rows': {c['name']: [] for c in specs}, 'completed': False}
            if artifact['fingerprint'] != identity or artifact['condition_specs'] != specs:
                raise ValueError('collection resume data/config changed')
            ids = [str(r['id'] if task == 'tf' else r['pair_id']) for r in rows]
            for c in specs:
                existing = artifact['rows'][c['name']]
                if [r['id'] for r in existing] != ids[:len(existing)]:
                    raise ValueError('collection row order/identity changed')
                for start in range(len(existing), len(rows), meta['config']['batch_size']):
                    chunk = rows[start:start + meta['config']['batch_size']]
                    new = output_records(bundle, meta, bank, task, chunk, c)
                    if any(not math.isfinite(x['score']) or not math.isfinite(x['simple_score']) for x in new):
                        raise ValueError('nonfinite inference score')
                    existing.extend(new)
                    save(path, artifact)
                    if meta['pilot']:
                        for row in new:
                            print('PILOT', task, c['name'], row['id'], repr(row['output']), flush=True)
                print('COLLECTED', task, split, c['name'], len(existing), flush=True)
            artifact['completed'] = True
            save(path, artifact)


def judge_qa(args, meta, data, split, judge=None):
    raw_path = args.run_dir / ('qa_%s.json' % split)
    raw = read(raw_path)
    if not raw.get('completed') or raw['bank_fingerprint'] != meta['fingerprint']:
        raise ValueError('complete generation first')
    path = args.run_dir / ('qa_%s_scored.json' % split)
    if split == 'tune' and (args.run_dir / 'selection.json').exists():
        raise ValueError('tuning has been frozen; do not rejudge it in place')
    exp = load_experiment(args.judge_exp)
    if judge is None:
        from core.judge import judge_from_config
        judge = judge_from_config(exp['judge'], cache_dir=str(args.cache_dir or args.run_dir / 'judge_cache'))
    signature = fingerprint([digest(raw_path), exp['judge'], getattr(judge, 'name', None), getattr(judge, 'qa_rubric_id', None)])
    result = read(path) if path.exists() else {**raw, 'generation_sha256': digest(raw_path),
        'grading_fingerprint': signature, 'completed': False, 'judge': getattr(judge, 'name', None)}
    if result.get('grading_fingerprint') != signature:
        raise ValueError('judge/source changed; use a separate run')
    by_id = {str(q['pair_id']): q for q in data['qa'][split]}
    expected = list(by_id)
    for name, rows in result['rows'].items():
        if [r['id'] for r in rows] != expected:
            raise ValueError('QA grading cohort differs')
        raw_rows = raw['rows'][name]
        for row, original in zip(rows, raw_rows):
            if any(row.get(k) != original.get(k) for k in ('id', 'pair_id', 'output', 'score', 'simple_score', 'finished')):
                raise ValueError('graded record changed its raw generation or score')
            q = by_id[row['id']]
            gold, only_refusal = reference_answers(q['gold'])
            row['gold_is_abstention'] = bool(only_refusal)
            if row.get('outcome') not in ('correct', 'wrong', 'abstain'):
                row['grade'] = judge.qa_grade(q['question'], gold, row['output'], incorrect=q.get('incorrect'))
                row['outcome'] = row['grade']['label']
                save(path, result)
    unresolved = sum(r['outcome'] == 'unresolved' for rows in result['rows'].values() for r in rows)
    result['n_unresolved'], result['completed'] = unresolved, unresolved == 0
    save(path, result)
    if unresolved:
        raise RuntimeError('unresolved QA grades saved; rerun judge phase to retry, no rows removed')
    return result


def scored_rows(args, meta, task, split):
    filename = '%s_%s%s.json' % (task, split, '_scored' if task == 'qa' else '')
    result = read(args.run_dir / filename)
    if not result.get('completed') or result.get('bank_fingerprint') != meta['fingerprint']:
        raise ValueError('incomplete/foreign scored data')
    data = read(args.run_dir / 'data.json')
    if digest(args.run_dir / 'data.json') != meta['data_sha256']:
        raise ValueError('fixed cohort changed')
    cohort = data[task][split]
    expected_ids = [str(r['id'] if task == 'tf' else r['pair_id']) for r in cohort]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError('duplicate fixed-cohort IDs')
    specs = meta['conditions'][task]
    if split == 'test':
        selection = read(args.run_dir / 'selection.json')
        names = {'baseline'} | {n for point in selection['tasks'][task]['choices'].values()
                                for n in point.values() if n is not None}
        specs = [c for c in specs if c['name'] in names]
    expected_names = {c['name'] for c in specs}
    raw = read(args.run_dir / ('qa_%s.json' % split)) if task == 'qa' else result
    if (set(result['rows']) != expected_names or set(raw['rows']) != expected_names
            or result['condition_specs'] != specs or raw['condition_specs'] != specs
            or raw.get('fingerprint') != fingerprint([meta['fingerprint'], task, split, specs, cohort])):
        raise ValueError('conditions or raw cohort identity differ')
    if task == 'qa' and result['generation_sha256'] != digest(args.run_dir / ('qa_%s.json' % split)):
        raise ValueError('graded source changed')
    for name, rows in result['rows'].items():
        if [r['id'] for r in rows] != expected_ids or [r['id'] for r in raw['rows'][name]] != expected_ids:
            raise ValueError('every arm must contain the complete fixed cohort')
        for row, original, question in zip(rows, raw['rows'][name], cohort):
            if row['pair_id'] != str(question['pair_id']):
                raise ValueError('question group changed')
            if not all(math.isfinite(row[k]) for k in ('score', 'simple_score')):
                raise ValueError('nonfinite inference score')
            if task == 'qa':
                _, only_refusal = reference_answers(question['gold'])
                if row.get('gold_is_abstention') is not bool(only_refusal):
                    raise ValueError('reference-only refusal cohort flag changed')
                if row['outcome'] != row.get('grade', {}).get('label'):
                    raise ValueError('derived QA outcome disagrees with its semantic grade')
                if any(row[k] != original[k] for k in ('output', 'score', 'simple_score', 'finished')):
                    raise ValueError('graded record differs from raw output/score')
            else:
                if type(row.get('predicted_true')) is not bool or row['output'] != ('True' if row['predicted_true'] else 'False'):
                    raise ValueError('TF text and predicted label disagree')
                expected_outcome = 'correct' if row['predicted_true'] == question['label'] else 'wrong'
                if row['outcome'] != expected_outcome:
                    raise ValueError('TF outcome disagrees with fixed label')
            if row['outcome'] not in ('correct', 'wrong', 'abstain'):
                raise ValueError('unresolved records cannot enter selection or results')
    return {name: [dict(r) for r in rows if task != 'qa' or not r['gold_is_abstention']]
            for name, rows in result['rows'].items()}, filename


def score_variant(rows, which, seed):
    return [{**r, 'score': random_score(r['id'], seed) if which == 'random' else r['simple_score']} for r in rows]


def comparison_arms(rows, picked, cfg):
    arms = {'baseline_commitment': rows['baseline'],
            'baseline_simple': score_variant(rows['baseline'], 'simple', cfg['seed']),
            'baseline_random': score_variant(rows['baseline'], 'random', cfg['seed'])}
    for family, name in picked.items():
        if name is not None:
            arms[family] = (score_variant(rows[name], 'simple', cfg['seed'])
                            if family.endswith('_simple') else rows[name])
    return arms


def freeze(args, meta, persist=True):
    result = {'protocol': PROTOCOL, 'bank_fingerprint': meta['fingerprint'], 'tasks': {}, 'sources': {}}
    cfg = meta['config']
    for task in TASKS:
        rows, filename = scored_rows(args, meta, task, 'tune')
        choices = choose_conditions(rows, meta['conditions'][task], cfg['coverages'], cfg['seed'])
        simple_rows = {name: score_variant(rs, 'simple', cfg['seed']) for name, rs in rows.items()}
        simple_choices = choose_conditions(simple_rows, meta['conditions'][task], cfg['coverages'], cfg['seed'])
        for coverage, picked in choices.items():
            for family, name in simple_choices[coverage].items():
                if family.endswith(('_E', '_C')):
                    picked[family + '_simple'] = name
        thresholds = {}
        for coverage, picked in choices.items():
            point = comparison_arms(rows, picked, cfg)
            thresholds[coverage] = {name: calibrate_threshold(rs, float(coverage), seed=cfg['seed']) for name, rs in point.items()}
        result['tasks'][task] = {'choices': choices, 'thresholds': thresholds,
                                'tune_ids': [r['id'] for r in rows['baseline']]}
        result['sources'][filename] = digest(args.run_dir / filename)
    result['fingerprint'] = fingerprint(result)
    if persist:
        save(args.run_dir / 'selection.json', result, immutable=True)
    return result


def frozen_selection(args, meta):
    saved = read(args.run_dir / 'selection.json')
    if saved != freeze(args, meta, persist=False):
        raise ValueError('frozen decision/source changed')
    return saved


def report(args, meta):
    selection = frozen_selection(args, meta)
    cfg, tables = meta['config'], {}
    delivered = {}
    def number(value):
        return '—' if value is None else '%.4f' % value
    def interval(value):
        return '—' if value is None else '[%.4f, %.4f]' % tuple(value)
    lines = ['# E26 同实际作答率方法比较', '',
        '精确配额表是批量筛题评估；排序不看正确性。冻结门槛表另报真实测试作答率。',
        'QA分母为固定实质参考题；不够配额标不可行，未决判分不出表。', '',
        'PILOT ONLY — 线路检查，不作效果结论。' if meta['pilot'] else '完整配置运行。', '']
    for task in TASKS:
        rows, _ = scored_rows(args, meta, task, 'test')
        tables[task], delivered[task] = {}, {}
        for coverage in cfg['coverages']:
            key = str(coverage); picked = selection['tasks'][task]['choices'][key]
            arms = comparison_arms(rows, picked, cfg)
            quota = evaluate_coverage(arms, [coverage], n_boot=cfg['n_boot'], seed=cfg['seed'])
            deployment = {}
            for name, rs in arms.items():
                policy = selection['tasks'][task]['thresholds'][key][name]
                deployment[name] = (apply_threshold(rs, policy) if policy['feasible'] else
                    {'feasible': False, 'reason': 'tuning_cannot_fill_requested_quota',
                     'policy': policy, 'actual_coverage': None, 'answered': None,
                     'correct': None, 'accuracy_answered': None})
            tables[task][key] = {'quota': quota, 'frozen_threshold': deployment,
                'unavailable_families': [name for name, spec in picked.items() if spec is None],
                'native': {name: {'n': len(rs), 'correct': sum(r['outcome'] == 'correct' for r in rs),
                    'answered': sum(r['outcome'] != 'abstain' for r in rs)} for name, rs in arms.items()}}
            delivered[task][key] = {}
            point = quota['coverages'][0]
            lines += ['## %s / 目标 %.0f%%' % (task, coverage * 100), '',
                      '| 方法 | 实际回答 | 答对 | 已回答准确率 | 95%区间 | 状态 |', '|---|---:|---:|---:|---|---|']
            for name, rs in arms.items():
                q = quota_select(rs, coverage, seed=cfg['seed'])
                if q['feasible']:
                    selected_ids = set(q['selected_ids'])
                    delivered[task][key][name] = [{'id': r['id'], 'pair_id': r['pair_id'],
                        'output': r['output'] if r['id'] in selected_ids else cfg['abstention_output'],
                        'delivered_outcome': q['delivered_outcomes'][r['id']]} for r in rs]
                    boot = point['arms'][name]['bootstrap']
                    ci = (boot.get('ci95') or {}).get('accuracy_answered')
                    lines.append('| %s | %d/%d | %d | %s | %s | %s |' % (
                        name, q['answered'], q['n'], q['correct'], number(q['accuracy_answered']), interval(ci), boot['status']))
                else:
                    delivered[task][key][name] = None
                    lines.append('| %s | — | — | — | — | 无法填满配额：%s |' % (name, q['reason']))
            for family in tables[task][key]['unavailable_families']:
                lines.append('| %s | — | — | — | — | 调参无可行配置 |' % family)
            lines.append('')
            lines += ['组合与单方向／简单筛题的直接配对差；下表统一为“组合减对照”。', '',
                      '| 组合 | 对照 | 已回答准确率差 | 95%区间 | 固定交付精确p | 状态 |',
                      '|---|---|---:|---|---:|---|']
            comparisons = {(x['arm_a'], x['arm_b']): x for x in point['comparisons']}
            for joint in sorted(n for n in arms if n.endswith('_EC')):
                method = joint[:-3]
                for control in ['baseline_simple'] + [method + '_' + kind for kind in ('E', 'C', 'E_simple', 'C_simple')]:
                    entry = comparisons.get((joint, control)) or comparisons.get((control, joint))
                    if entry is None:
                        continue
                    sign = 1 if entry['arm_a'] == joint else -1
                    delta = (entry.get('point_delta') or {}).get('accuracy_answered')
                    ci = (entry.get('ci95') or {}).get('accuracy_answered')
                    if delta is not None:
                        delta *= sign
                    if ci is not None and sign == -1:
                        ci = [-ci[1], -ci[0]]
                    exact = entry.get('exact_fixed_deliveries') or {}
                    lines.append('| %s | %s | %s | %s | %s | %s |' % (
                        joint, control, number(delta), interval(ci), number(exact.get('p_value')), entry['bootstrap_status']))
            lines += ['', '冻结调参门槛后的实际测试表现（不强制达到目标配额）：', '',
                      '| 方法 | 实际作答率 | 实际回答数 | 答对数 | 已回答准确率 |', '|---|---:|---:|---:|---:|']
            for name, row in deployment.items():
                lines.append('| %s | %s | %s | %s | %s |' % (name, number(row.get('actual_coverage')),
                    '—' if row.get('answered') is None else row['answered'],
                    '—' if row.get('correct') is None else row['correct'], number(row.get('accuracy_answered'))))
            lines.append('')
    result = {'protocol': PROTOCOL, 'model': meta['model'], 'pilot': meta['pilot'],
              'bank_fingerprint': meta['fingerprint'], 'selection_sha256': digest(args.run_dir / 'selection.json'),
              'tables': tables, 'completed': True}
    save(args.run_dir / 'report.json', result)
    save(args.run_dir / 'delivered_answers.json', delivered)
    (args.run_dir / 'report.md').write_text('\n'.join(lines))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['prepare', 'collect-tune', 'judge-tune', 'freeze', 'collect-test', 'judge-test', 'report'])
    p.add_argument('--model', required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--config', default='E26_matched_coverage')
    p.add_argument('--qa-run', type=Path)
    p.add_argument('--tf-das-dir', type=Path)
    p.add_argument('--tf-das-sweep', type=Path)
    p.add_argument('--tf-md-sweep', type=Path)
    p.add_argument('--qa-adapters', nargs='*', type=Path, default=[],
                   help='optional verified E27 adapters; new bank re-estimates C and retunes all families')
    p.add_argument('--pilot', type=int)
    p.add_argument('--judge-exp', default='E4_qa')
    p.add_argument('--cache-dir', type=Path)
    args = p.parse_args()
    if args.phase == 'prepare':
        if any(getattr(args, k) is None for k in ('qa_run', 'tf_das_dir', 'tf_das_sweep', 'tf_md_sweep')):
            p.error('prepare needs --qa-run, --tf-das-dir, --tf-das-sweep and --tf-md-sweep')
        cfg = load_json_config(resolve_config_path('experiments', args.config))
        prepare(args, cfg)
        return
    meta, bank, data = load_run(args)
    if args.phase.startswith('collect-'):
        collect(args, meta, bank, data, args.phase.split('-')[1])
    elif args.phase.startswith('judge-'):
        judge_qa(args, meta, data, args.phase.split('-')[1])
    elif args.phase == 'freeze':
        freeze(args, meta)
    else:
        report(args, meta)


if __name__ == '__main__':
    main()
