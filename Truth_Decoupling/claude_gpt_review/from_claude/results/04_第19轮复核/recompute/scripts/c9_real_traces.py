"""Claim group 9: Llama vs Qwen real-trace comparison (models/evidence/{llama,qwen}_trace.json)."""
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

T = {'llama3.1-8b': load(R19 / 'models/evidence/llama_trace.json'), 'qwen3-8b': load(R19 / 'models/evidence/qwen_trace.json')}
print('same six code hashes:', T['llama3.1-8b']['code_sha256'] == T['qwen3-8b']['code_sha256'])
REL = {'llama3.1-8b': by_pid(load(RELEASE / 'llama3.1-8b/das_s42/tune.json')),
       'qwen3-8b': by_pid(load(RELEASE / 'qwen3-8b/das_s42/tune.json'))}
for m, d in T.items():
    print('=' * 10, m, 'adapter', d['adapter_sha256'][:16], 'reference_generation_sha256', d['reference_generation_sha256'][:16],
          'gen-config do_sample (saved default)=', d['generation_config'].get('do_sample'))
    for pid in sorted(d['cases'], key=lambda p: d['cases'][p]['batch_start']):
        c = d['cases'][pid]
        ids = c['effective_ids']
        n = len(ids)
        # my own byte reconstruction and invalid-UTF-8 detection
        raw = b''.join(tok_bytes(t) or b'' for t in c['bytes']['token_strings'])
        my_text = raw.decode('utf-8', errors='replace')
        try:
            raw.decode('utf-8')
            my_invalid = 0
        except UnicodeDecodeError:
            my_invalid = 1
        edits = sorted(c['edits'], key=lambda e: e['step'])
        steps = [e['step'] for e in edits]
        assert len(set(steps)) == len(steps)
        act = [e for e in edits if e['step'] < n]
        assert [e['step'] for e in act] == list(range(n))
        dec = [e['kept_relative_norm'] for e in act if e['step'] > 0]
        rel_txt = REL[m][pid]['outputs']['das_generation_adapter']
        print(f"  {pid} out={c['original_output']!r} my_decode_equal={my_text == c['original_output']} == release das_s42 tune text: "
              f"{rel_txt == c['original_output']} | n_active={n} padding_edits_dropped={len(edits) - n} | first={100 * act[0]['kept_relative_norm']:.2f}% "
              f"decode_median={100 * statistics.median(dec):.2f}% max={100 * max(e['kept_relative_norm'] for e in act):.2f}% "
              f"(max incl. padding={100 * max(e['kept_relative_norm'] for e in edits):.2f}%) | invalid_utf8 file={len(c['bytes']['invalid_utf8_events'])} "
              f"mine={my_invalid} literal_FFFD_events={len(c['bytes']['literal_replacement_events'])} FFFD_in_text={FFFD in c['original_output']}")
        pref = c['bytes']['decoded_prefixes']
        if "I don't know." in pref:
            k = pref.index("I don't know.")
            s = k + 1
            if s < n:
                st = c['steps'][s]['token']
                am = st['argmax_id']
                tk = next(t for t in st['tokens'] if t['id'] == am)
                tb = tok_bytes(tk['token_string'])
                print(f"     after complete \"I don't know.\" -> step {s}: argmax={am} {tk['token_string']!r} bytes={tb.hex() if tb else None} "
                      f"p={100 * math.exp(tk['log_probability']):.4f}% rel_edit={100 * act[s]['kept_relative_norm']:.2f}%")
