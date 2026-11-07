#!/usr/bin/env python3
"""E42: two decisive diagnostics for the free-QA write problem (Claude, 2026-11-08; 记事本 N0).

Background. Every single-layer rank-one write tried so far leaves free-form answers unchanged at matched
answer rate (fixed-centre amplification, source interchange with a truthful instruction, trained W; and
in v1 even a gold-coordinate exchange at the rule-selected early layers), while the same coordinate READ
at late layers tells right from wrong answers (AUROC 0.67-0.78). Before another write operator, settle
two questions on the 192 tuning questions:

A. Oracle layer scan.  Rank-one source interchange h' = h + w (w.(s - h)) at every decode step, where the
   SOURCE stream is the same frozen model told the gold answer in its prompt. If even a gold-informed
   coordinate at layer L leaves the base answer unchanged, a rank-one write at L has no causal grip on
   generation (then: several layers / rank > 1 / a different operator). If it does change answers, a
   deployable source can be tried at L. Ceilings: the gold-informed source answering by itself, and the
   truthful-instruction source answering by itself (GPT's source_only).

B. Readout-guided selection.  Sample k candidate answers, write each as the TF statement
   'The answer to the question "Q" is: A.' and read the TF E coordinate at the readout layers (exactly
   own_answer_tfdetect.py), then pick the candidate the coordinate calls most true. This turns "the
   coordinate sees the error" directly into "the output changes" without any write operator. Controls:
   greedy baseline, random pick, highest mean log-probability pick, the model's own TF verdict margin.

Phases: oracle-check | select-check  (3 tuning questions, no judge; prints the raw outputs)
        oracle-tune  | select-tune   (192 tuning questions, raw generation only)
        judge-tune                   (e25's judge on tune.json; DeepSeek Pro, D -> human review queue)
The test split is never touched here. Gold reaches only the SOURCE stream of the oracle arms (named
oracle_*), never the base stream and never any deployable arm.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

import common  # noqa: F401
from common import load_split, tf_prompt
import e25_qa_das as qa
import e27_qa_generation as adapter_script
from core.config import load_experiment
from core.directions import load_das_direction, mean_diff_direction
from core.model_io import capture_last_token
from core.qa_generation_adapter import implementation_identity
from core.qa_precision import fp32_output_projection
from core.qa_source_interchange import SourceInterchangeDecoder
from core.readout import build_label_token_map, label_logprobs_from_logits, rule_abstains

PROTOCOL = 'e42_oracle_select_v1'
SOURCE_SYSTEM = ("Answer factual questions truthfully. Correct false premises rather than accepting them. "
                 "Give a short, direct answer of a few words. If you do not know the answer, say 'I don't know'.")
GOLD_SYSTEM = SOURCE_SYSTEM + " The user message states the correct answer; answer in agreement with it."
GOLD_NOTE = "{question}\n(Note: the correct answer is: {gold}.)"


# ----------------------------------------------------------------------------- prompts & directions
def chat_ids(bundle, system, user):
    inputs = bundle.encode(bundle.build_chat(system, user))
    if inputs['input_ids'].shape[0] != 1 or not bool(inputs['attention_mask'].all()):
        raise ValueError('single-question prompts must be unpadded')
    return inputs['input_ids'][0].tolist()


def gold_text(q):
    gold = q.get('best_answer') or (q['gold'][0] if q.get('gold') else None)
    if not isinstance(gold, str) or not gold.strip():
        raise ValueError('oracle needs a gold answer for %s' % q.get('pair_id'))
    return ' '.join(gold.split()).rstrip('.')


def prompts_for(bundle, cfg, q):
    user = cfg['qa_user'].format(question=q['question'])
    gold_user = cfg['qa_user'].format(question=GOLD_NOTE.format(question=q['question'], gold=gold_text(q)))
    return {'base': chat_ids(bundle, cfg['qa_system'], user),
            'truthful': chat_ids(bundle, SOURCE_SYSTEM, user),
            'gold': chat_ids(bundle, GOLD_SYSTEM, gold_user)}


def tf_estimation(bundle, layers):
    """TF estimation statements -> per-layer mean-difference direction, used only to orient DAS signs (true = +)."""
    exp = load_experiment('E2b_tqa')
    est = []
    for ds in exp['eval_datasets']:
        est += load_split(ds, 'calib') + load_split(ds, 'dev')
    y = np.array([bool(r['label']) for r in est])
    acts, _ = capture_last_token(bundle, [tf_prompt(bundle, exp['prompts'], r['text'], allow_idk=False) for r in est],
                                 list(layers), batch_size=8)
    wmd = {}
    for l in layers:
        H = acts[('resid', l)].numpy().astype(np.float64)
        wmd[l], _ = mean_diff_direction(H[y], H[~y])
    return exp['prompts'], wmd, len(est)


def directions(args, bank, meta, layers, wmd=None):
    """{family: {layer: unit torch vector}}; dasce = TF gold-label CE DAS (results_das_tfce/E1), qadas = QA-context DAS bank."""
    out = {f: {} for f in args.families}
    for l in layers:
        if 'dasce' in out:
            w, _ = load_das_direction('%s/results_das_tfce/E1/%s' % (args.direction_root, meta['model']), 'E', l,
                                      variant='tf-ce-bidir', align_to=None if wmd is None else wmd.get(l))
            if w is not None:
                out['dasce'][l] = torch.as_tensor(np.asarray(w, dtype=np.float64)).float()
        if 'qadas' in out and ('E_qa_das_L%d' % l) in bank:
            out['qadas'][l] = torch.as_tensor(bank['E_qa_das_L%d' % l]).float().clone()
    for f, by in out.items():
        if not by:
            raise ValueError('no %s direction found for layers %s' % (f, layers))
    return out


# ----------------------------------------------------------------------------- decoding with per-step audit
def stop_ids(bundle):
    stops = getattr(bundle.model.generation_config, 'eos_token_id', None)
    stops = set(stops if isinstance(stops, (tuple, list)) else [stops])
    if not stops or any(type(t) is not int or t < 0 for t in stops):
        raise ValueError('model must declare valid generation stop IDs')
    return sorted(stops)


def decode(bundle, base_ids, source_ids, *, mode, layer=None, direction=None, max_new_tokens=48,
           sampler=None):
    """Greedy (sampler None) or sampled decoding; records EOS probability, top-1 probability, entropy and write size per step."""
    stops = stop_ids(bundle)
    dec = SourceInterchangeDecoder(bundle, base_ids, source_ids, mode=mode, layer=layer, direction=direction)
    steps, logprob_sum, finished = [], 0.0, False
    for _ in range(max_new_tokens):
        logits, trace = dec.next_logits()
        logp = torch.log_softmax(logits, -1)
        probs = logp.exp()
        if sampler is None:
            token = int(logits.argmax())
        else:
            token = sampler(logits)
        write = trace['writes'][0]['stored_delta_over_base'] if trace.get('writes') else 0.0
        steps.append({'eos_p': float(probs[stops].sum()), 'top1_p': float(probs.max()),
                      'entropy': float(-(probs * logp).sum()), 'write_over_base': float(write),
                      'token_logp': float(logp[token])})
        logprob_sum += float(logp[token])
        dec.accept(token)
        if token in stops:
            finished = True
            break
    ids = dec.history
    content = [i for i in ids if i not in stops]
    text = bundle.tokenizer.decode(ids, skip_special_tokens=True)
    summary = {'n_steps': len(steps), 'finished': finished,
               'mean_token_logp': logprob_sum / max(1, len(steps)),
               'write_over_base_mean': float(np.mean([s['write_over_base'] for s in steps])) if steps else 0.0,
               'write_over_base_max': float(max(s['write_over_base'] for s in steps)) if steps else 0.0,
               'eos_p_at_end': steps[-1]['eos_p'] if steps else None,
               'min_top1_p': float(min(s['top1_p'] for s in steps)) if steps else None,
               'non_ascii_ratio': (sum(ord(c) > 127 for c in text) / len(text)) if text else 0.0,
               'n_content_tokens': len(content)}
    return {'output': text, 'token_ids': ids, 'finished': finished, 'steps': steps, 'summary': summary}


def make_sampler(seed, temperature, top_p):
    gen = torch.Generator(device='cpu').manual_seed(seed)

    def sample(logits):
        probs = torch.softmax(logits.float().cpu() / temperature, -1)
        sorted_p, order = probs.sort(descending=True)
        keep = sorted_p.cumsum(0) - sorted_p < top_p
        keep[0] = True
        filtered = torch.zeros_like(probs)
        filtered[order[keep]] = sorted_p[keep]
        return int(torch.multinomial(filtered / filtered.sum(), 1, generator=gen))
    return sample


# ----------------------------------------------------------------------------- phase A: oracle layer scan
def oracle_conditions(args, dirs):
    names = ['baseline', 'truthful_source_only', 'gold_source_only']
    for fam in args.families:
        for l in sorted(dirs[fam]):
            names.append('oracle_%s_L%d' % (fam, l))
            if l in args.deploy_layers:
                names.append('truthful_swap_%s_L%d' % (fam, l))
    return names


def run_oracle(args, bundle, meta, prepared, bank, questions, out_path):
    layers = sorted(set(args.layers) | set(args.deploy_layers))
    dirs = directions(args, bank, meta, layers)
    names = oracle_conditions(args, dirs)
    identity = {'protocol': PROTOCOL, 'phase': 'oracle', 'model': meta['model'],
                'source': adapter_script.source_identity(ns_of(args), meta, []),
                'base_system': meta['config']['qa_system'], 'truthful_system': SOURCE_SYSTEM, 'gold_system': GOLD_SYSTEM,
                'gold_note_template': GOLD_NOTE, 'user_template': meta['config']['qa_user'],
                'max_new_tokens': args.max_new_tokens, 'layers': layers, 'families': list(args.families),
                'deploy_layers': list(args.deploy_layers), 'conditions': names,
                'direction_sha256': {f: {str(l): qa.fingerprint(w.tolist()) for l, w in by.items()} for f, by in dirs.items()},
                'question_ids': [q['pair_id'] for q in questions], 'split': 'tune',
                'gold_used_at_inference': 'only in the SOURCE stream of oracle_* and gold_source_only arms',
                'implementation': implementation_identity()}
    result = {'protocol': PROTOCOL, 'model': meta['model'], 'identity': identity,
              'bank_protocol_fingerprint': qa.fingerprint(identity), 'protocol_fingerprint': qa.fingerprint(identity),
              'split': 'tune', 'pilot': args.phase.endswith('check'),
              'conditions': names,
              'condition_specs': [{'name': n, 'kind': 'baseline' if n == 'baseline' else 'E',
                                   'uses_reference_answer': n.startswith('oracle_') or n == 'gold_source_only',
                                   'eligible_for_selection': False} for n in names],
              'evaluations': [], 'completed': False}
    with fp32_output_projection(bundle):
        for q in questions:
            p = prompts_for(bundle, meta['config'], q)
            row = {**copy.deepcopy(q), 'idx': len(result['evaluations']), 'outputs': {}, 'finished': {}, 'rule_abstain': {},
                   'generation_token_ids': {}, 'diag': {}, 'prompt_token_ids': p}
            for name in names:
                if name == 'baseline':
                    g = decode(bundle, p['base'], p['base'], mode='baseline', max_new_tokens=args.max_new_tokens)
                elif name == 'truthful_source_only':
                    g = decode(bundle, p['base'], p['truthful'], mode='source_only', max_new_tokens=args.max_new_tokens)
                elif name == 'gold_source_only':
                    g = decode(bundle, p['base'], p['gold'], mode='source_only', max_new_tokens=args.max_new_tokens)
                else:
                    # oracle_<fam>_L<l> | truthful_swap_<fam>_L<l>
                    kind = name.split('_', 1)[0]
                    head, lay = name.rsplit('_L', 1)
                    fam, lay = head.split('_')[-1], int(lay)
                    src = p['gold'] if kind == 'oracle' else p['truthful']
                    g = decode(bundle, p['base'], src, mode='source_interchange', layer=lay, direction=dirs[fam][lay],
                               max_new_tokens=args.max_new_tokens)
                row['outputs'][name] = g['output']; row['finished'][name] = g['finished']
                row['rule_abstain'][name] = bool(rule_abstains(g['output']))
                row['generation_token_ids'][name] = g['token_ids']; row['diag'][name] = g['summary']
                if args.phase.endswith('check'):
                    row.setdefault('steps', {})[name] = g['steps']
                print('E42_ORACLE', meta['model'], q['pair_id'], '%-26s' % name,
                      'write %.3f/%.3f' % (g['summary']['write_over_base_mean'], g['summary']['write_over_base_max']),
                      'eos_end %.3f' % (g['summary']['eos_p_at_end'] or 0), repr(g['output'])[:120], flush=True)
            result['evaluations'].append(row)
            qa.save(out_path, result)
    result['completed'] = True
    qa.save(out_path, result)
    return result


# ----------------------------------------------------------------------------- phase B: readout-guided selection
def statement(q, a):
    a = ' '.join(a.split())
    return 'The answer to the question "%s" is: %s' % (q.strip().rstrip('?').strip() + '?', a.rstrip('.') + '.')


def run_select(args, bundle, meta, prepared, bank, questions, out_path):
    read_layers = sorted(args.read_layers)
    prompts, wmd, n_est = tf_estimation(bundle, read_layers)
    dirs = directions(args, bank, meta, read_layers, wmd=wmd)
    k = args.k
    cand_names = ['greedy', 'truthful_source_only'] + ['sample_%d' % i for i in range(k)]
    identity = {'protocol': PROTOCOL, 'phase': 'select', 'model': meta['model'],
                'source': adapter_script.source_identity(ns_of(args), meta, []),
                'base_system': meta['config']['qa_system'], 'truthful_system': SOURCE_SYSTEM,
                'user_template': meta['config']['qa_user'], 'max_new_tokens': args.max_new_tokens,
                'k': k, 'temperature': args.temperature, 'top_p': args.top_p, 'seed': args.seed,
                'read_layers': read_layers, 'families': list(args.families),
                'orientation': 'DAS signs aligned to the TF mean-difference direction of %d estimation statements (true = +)' % n_est,
                'statement_template': 'The answer to the question "Q" is: A.', 'conditions': cand_names,
                'direction_sha256': {f: {str(l): qa.fingerprint(w.tolist()) for l, w in by.items()} for f, by in dirs.items()},
                'question_ids': [q['pair_id'] for q in questions], 'split': 'tune',
                'gold_used_at_inference': False, 'implementation': implementation_identity()}
    result = {'protocol': PROTOCOL, 'model': meta['model'], 'identity': identity,
              'bank_protocol_fingerprint': qa.fingerprint(identity), 'protocol_fingerprint': qa.fingerprint(identity),
              'split': 'tune', 'pilot': args.phase.endswith('check'), 'conditions': cand_names,
              'condition_specs': [{'name': n, 'kind': 'candidate', 'uses_reference_answer': False,
                                   'eligible_for_selection': False} for n in cand_names],
              'evaluations': [], 'completed': False}
    st = prompts['statement']
    label_variants = {'True': st['label_variants']['True'], 'False': st['label_variants']['False']}
    with fp32_output_projection(bundle):
        for qi, q in enumerate(questions):
            p = prompts_for(bundle, meta['config'], q)
            row = {**copy.deepcopy(q), 'idx': len(result['evaluations']), 'outputs': {}, 'finished': {}, 'rule_abstain': {},
                   'generation_token_ids': {}, 'diag': {}, 'prompt_token_ids': {'base': p['base'], 'truthful': p['truthful']}}
            for ci, name in enumerate(cand_names):
                if name == 'greedy':
                    g = decode(bundle, p['base'], p['base'], mode='baseline', max_new_tokens=args.max_new_tokens)
                elif name == 'truthful_source_only':
                    g = decode(bundle, p['base'], p['truthful'], mode='source_only', max_new_tokens=args.max_new_tokens)
                else:
                    sampler = make_sampler(args.seed * 1000003 + qi * 101 + ci, args.temperature, args.top_p)
                    g = decode(bundle, p['base'], p['base'], mode='baseline', max_new_tokens=args.max_new_tokens, sampler=sampler)
                row['outputs'][name] = g['output']; row['finished'][name] = g['finished']
                row['rule_abstain'][name] = bool(rule_abstains(g['output']))
                row['generation_token_ids'][name] = g['token_ids']; row['diag'][name] = g['summary']
            # read every candidate as a TF statement under the TF prompt (own_answer_tfdetect rendering)
            texts = [tf_prompt(bundle, prompts, statement(q['question'], row['outputs'][n]), allow_idk=False) for n in cand_names]
            acts, lg = capture_last_token(bundle, texts, read_layers, batch_size=8, return_logits=True)
            label_map = build_label_token_map(bundle.tokenizer, texts[0], label_variants)
            row['readout'] = {}
            for ci, name in enumerate(cand_names):
                lp = label_logprobs_from_logits(lg[ci].numpy(), label_map, use_abstain=False)
                r = {'verdict_margin': float(lp['True'] - lp['False']), 'mean_token_logp': row['diag'][name]['mean_token_logp']}
                for fam, by in dirs.items():
                    for l, w in by.items():
                        r['coord_%s_L%d' % (fam, l)] = float(acts[('resid', l)][ci].double().numpy() @ w.double().numpy())
                row['readout'][name] = r
            row['picks'] = picks(row, cand_names, dirs, read_layers, args.seed * 7919 + qi)
            print('E42_SELECT', meta['model'], q['pair_id'], json.dumps(row['picks'], ensure_ascii=False), flush=True)
            for name in cand_names:
                print('    %-22s abst=%d coord=%s verdict=%+.2f logp=%+.2f  %r' % (
                    name, row['rule_abstain'][name],
                    ' '.join('%+.1f' % row['readout'][name][k] for k in sorted(row['readout'][name]) if k.startswith('coord_')),
                    row['readout'][name]['verdict_margin'], row['readout'][name]['mean_token_logp'], row['outputs'][name][:90]), flush=True)
            result['evaluations'].append(row)
            qa.save(out_path, result)
    result['completed'] = True
    qa.save(out_path, result)
    return result


def picks(row, cand_names, dirs, read_layers, seed):
    """Which candidate each rule selects; abstaining candidates are never selected when a non-abstaining one exists."""
    pool = [n for n in cand_names if not row['rule_abstain'][n] and row['outputs'][n].strip()] or list(cand_names)
    out = {'greedy': 'greedy', 'random': pool[np.random.default_rng(seed).integers(len(pool))],
           'max_logp': max(pool, key=lambda n: row['readout'][n]['mean_token_logp']),
           'verdict': max(pool, key=lambda n: row['readout'][n]['verdict_margin'])}
    for fam, by in dirs.items():
        for l in by:
            key = 'coord_%s_L%d' % (fam, l)
            out[key] = max(pool, key=lambda n: row['readout'][n][key])
        # the mean over the readout layers of this family (z-scored within the question)
        vals = {n: np.mean([row['readout'][n]['coord_%s_L%d' % (fam, l)] for l in by]) for n in pool}
        out['coord_%s_mean' % fam] = max(pool, key=lambda n: vals[n])
    return out


# ----------------------------------------------------------------------------- plumbing
def ns_of(args):
    ns, _, _, _ = adapter_script.source_run(args.source_run)
    return ns


def load_inputs(args):
    ns, meta, bank, prepared = adapter_script.source_run(args.source_run)
    questions = prepared['splits']['tune']
    if args.pair_ids:
        wanted = set(args.pair_ids)
        questions = [q for q in questions if q['pair_id'] in wanted]
    elif args.phase.endswith('check'):
        questions = questions[:args.n_check]
    if not questions:
        raise ValueError('empty tuning cohort')
    return ns, meta, bank, prepared, questions


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('phase', choices=('oracle-check', 'oracle-tune', 'select-check', 'select-tune', 'judge-tune'))
    p.add_argument('--source-run', type=Path, required=True, help='E25 v1u full run (bank, questions, splits)')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--direction-root', default='/workspace/Truth_Decoupling')
    p.add_argument('--families', nargs='+', default=['dasce', 'qadas'], choices=['dasce', 'qadas'])
    p.add_argument('--layers', nargs='+', type=int, default=[12, 16, 20, 24, 28, 29, 30, 31], help='oracle scan layers')
    p.add_argument('--deploy-layers', nargs='+', type=int, default=[29], help='layers that also get the truthful-source swap arm')
    p.add_argument('--read-layers', nargs='+', type=int, default=[28, 29, 30, 31], help='selection readout layers')
    p.add_argument('--k', type=int, default=6)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--max-new-tokens', type=int, default=48)
    p.add_argument('--n-check', type=int, default=3)
    p.add_argument('--pair-ids', nargs='+')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--judge-exp', default='E4_qa')
    args = p.parse_args()
    ns, meta, bank, prepared, questions = load_inputs(args)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == 'judge-tune':
        args.model = meta['model']
        args.judge_cache_dir = str(args.run_dir / 'judge_cache')
        Path(args.judge_cache_dir).mkdir(parents=True, exist_ok=True)
        adapter_script.validate_judge_experiment(args.judge_exp)
        data = qa.read(args.run_dir / 'tune.json')
        qa.judge(args, {**meta, 'protocol_fingerprint': data['bank_protocol_fingerprint']}, 'tune.json', 'tune_scored')
        return
    stem = 'check.json' if args.phase.endswith('check') else 'tune.json'
    out_path = args.run_dir / stem
    if out_path.exists() and qa.read(out_path).get('completed'):
        raise SystemExit('%s exists and is complete; use a new --run-dir' % out_path)
    bundle = qa.open_bundle(meta, prepared)
    if args.phase.startswith('oracle'):
        run_oracle(args, bundle, meta, prepared, bank, questions, out_path)
    else:
        run_select(args, bundle, meta, prepared, bank, questions, out_path)


if __name__ == '__main__':
    main()
