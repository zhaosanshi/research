"""Claim group 3: early (round-18 archive) vs Claude release rerun, item by item."""
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


ADAPTER_COND = {'das': 'das_generation_adapter', 'md': 'md_generation_adapter', 'random': 'random_generation_adapter'}
REL_DIRS = ['das_s42', 'md_s42', 'random_s42', 'random_s43', 'random_s44']

for model in ['llama3.1-8b', 'qwen3-8b']:
    print('=' * 25, model)
    rel_meta = {}
    for rd in REL_DIRS:
        a = load(RELEASE / model / rd / 'adapter.json')
        rel_meta[rd] = a
        print(f"  release {rd}: init={a['init']} seed={a['train_config']['seed']} ckpt={a['checkpoint_sha256'][:16]} "
              f"adapter.pt sha={sha(RELEASE / model / rd / 'adapter.pt')[:16]}")
    for init in ['das', 'md', 'random']:
        ed = X_R18 / model / 'ce_alpha3_ep3' / init
        ea = load(ed / 'adapter.json')
        print(f"-- early {init}: seed={ea['train_config']['seed']} ckpt={ea['checkpoint_sha256'][:16]} "
              f"adapter.pt sha={sha(ed / 'adapter.pt')[:16]} train_config={ {k: ea['train_config'][k] for k in ea['train_config']} }")
        cands = [rd for rd in REL_DIRS if rel_meta[rd]['init'] == init]
        for rd in cands:
            ra = rel_meta[rd]
            loss_eq = ea['losses'] == ra['losses']
            grad_eq = ea.get('gradient_audit') == ra.get('gradient_audit')
            tc_diff = {k: (ea['train_config'].get(k), ra['train_config'].get(k)) for k in set(ea['train_config']) | set(ra['train_config'])
                       if ea['train_config'].get(k) != ra['train_config'].get(k)}
            print(f"   vs release {rd}: losses_equal={loss_eq} gradient_audit_equal={grad_eq} train_config_diffs={tc_diff}")
            if not loss_eq:
                for i, (x, y) in enumerate(zip(ea['losses'], ra['losses'])):
                    print(f"      epoch{i}: early={x['loss']!r} release={y['loss']!r} diff={y['loss'] - x['loss']:.3e}")
            for split in ['tune', 'test']:
                ef = ed / f'{split}_scored_items.json'
                if not ef.exists():
                    print(f"      {split}: no early archive file -> not comparable")
                    continue
                E = by_pid(load(ef))
                R = by_pid(load(RELEASE / model / rd / f'{split}_scored_items.json'))
                Rraw = by_pid(load(RELEASE / model / rd / f'{split}.json'))
                Eraw = by_pid(load(ed / f'{split}.json'))
                assert set(E) == set(R) == set(Rraw) == set(Eraw)
                for cond in ['baseline', 'original_das', ADAPTER_COND[init]]:
                    td = [p for p in E if E[p]['outputs'][cond] != R[p]['outputs'][cond]]
                    tdraw = [p for p in E if Eraw[p]['outputs'][cond] != Rraw[p]['outputs'][cond]]
                    fd = [p for p in E if E[p]['finished'][cond] != R[p]['finished'][cond]]
                    rad = [p for p in E if E[p]['rule_abstain'][cond] != R[p]['rule_abstain'][cond]]
                    same_text_label_diff = [p for p in E if E[p]['outputs'][cond] == R[p]['outputs'][cond]
                                            and E[p]['outcome'][cond] != R[p]['outcome'][cond]]
                    ce, cr = counts(E.values(), cond)['correct'], counts(R.values(), cond)['correct']
                    print(f"      {split:4s} {cond:26s} n={len(E)} text_diff={len(td)} (raw files {len(tdraw)}) finished_diff={len(fd)} "
                          f"rule_abstain_diff={len(rad)} | correct early={ce} release={cr} | same-text-diff-label={len(same_text_label_diff)}")
                    for p in (td if len(td) <= 10 else []):
                        print(f"         TEXT {p}: early={E[p]['outputs'][cond]!r} [{E[p]['outcome'][cond]}] | release={R[p]['outputs'][cond]!r} [{R[p]['outcome'][cond]}]")
                    for p in (same_text_label_diff if len(td) <= 10 else []):
                        print(f"         LABEL {p}: text={E[p]['outputs'][cond]!r} early={E[p]['outcome'][cond]}({E[p]['judge'][cond]['raw']}) "
                              f"release={R[p]['outcome'][cond]}({R[p]['judge'][cond]['raw']}) gold_is_abstention={E[p]['gold_is_abstention']}")
