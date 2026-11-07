"""Claim group 5: single-step current dose (qwen_current_dose.json)."""
import codecs
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

d = load(R19 / 'unicode/evidence/qwen_current_dose.json')
tr = load(R19 / 'unicode/evidence/qwen_trace.json')
print('\n=== current dose: protocol', d['protocol'], 'history_alpha', d['history_alpha'], 'alphas', d['alphas'],
      'original_trace_sha256 matches qwen_trace.json sha:', d['original_trace_sha256'][:16])
EOS = {151645, 151643}
for b in d['batches']:
    print(f"batch start={b['start']} whole_batch_text_reproduced={b['whole_batch_text_reproduced']} "
          f"selected_raw_ids_reproduced={b['selected_raw_ids_reproduced']}")
    for st in b['steps']:
        s = st['step']
        hist = {x['history_output_sha256'] for x in st['doses']}
        cur = {x['current_before_sha256'] for x in st['doses']}
        ctrl = {k: (v['reference'], v['max_abs_logits_difference'], v['bitwise_logits_equal'], v['argmax_equal_all_batch_rows'])
                for k, v in st['controls'].items()}
        print(f"  step{s}: n_doses={len(st['doses'])} distinct history hashes={len(hist)} distinct current-input hashes={len(cur)} "
              f"flags hist={st['history_identical_across_doses']} cur={st['current_input_identical_across_doses']} controls={ctrl}")
        for x in st['doses']:
            for pid, r in x['rows'].items():
                # independent recomputation of legal-continuation mass from listed top tokens is not possible
                # (only top-8 listed), so report the stored mass and check argmax legality myself
                am = r['argmax_id']
                pend = bytes.fromhex(r['pending_utf8_hex']) if r['pending_utf8_hex'] else b''
                tb = tok_bytes(r['argmax_token']) if am not in EOS else None
                legal = None
                if pend:
                    if tb is None:
                        legal = False
                    else:
                        try:
                            codecs.getincrementaldecoder('utf-8')('strict').decode(pend + tb, final=False)
                            legal = True
                        except UnicodeDecodeError:
                            legal = False
                topsum = sum(t['probability'] for t in r['top'])
                lb = None
                if pend:
                    lb = sum(t['probability'] for t in r['top'] if t['id'] not in EOS and tok_bytes(t['token']) is not None
                             and utf8_prefix_ok(pend + tok_bytes(t['token'])))
                    assert lb <= (r['valid_continuation_probability_mass'] or 0) + 1e-9, (pid, x['alpha'], lb)
                eos_top = sum(t['probability'] for t in r['top'] if t['id'] in EOS)
                vm = r['valid_continuation_probability_mass']
                print(f"    {pid} a={x['alpha']:<4} argmax={am} {r['argmax_token']!r} p={r['argmax_probability']:.6f} "
                      f"eos_total={r['eos_total_probability'] * 100:.4f}% (eos in top-8 list {eos_top * 100:.4f}%) "
                      f"valid_mass={'-' if vm is None else f'{vm * 100:.4f}%'} myLegal={legal} fileLegal={r['argmax_is_legal_continuation']} "
                      f"rel_edit={r['edit']['kept_relative_norm'] * 100:.2f}% top8sum={topsum:.4f} legal_mass_lower_bound_from_top8={'-' if lb is None else f'{lb * 100:.4f}%'}")
