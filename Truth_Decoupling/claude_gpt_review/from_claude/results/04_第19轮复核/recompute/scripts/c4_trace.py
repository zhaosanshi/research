"""Claim group 4: Qwen corruption trace (tqa:138 / tqa:150 / tqa:412), recomputed from qwen_trace.json."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa


def bytes_to_unicode():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


INV = bytes_to_unicode()


def tok_bytes(s):
    if s is None or (s.startswith('<|') and s.endswith('|>')):
        return None
    return bytes(INV[ch] for ch in s)


def legal_utf8_continuation(prefix: bytes, tb: bytes) -> bool:
    """True if prefix (a pending incomplete UTF-8 sequence) + tb does not create an invalid sequence at the junction."""
    data = prefix + tb
    # decode incrementally: valid if decoding data with an incremental decoder raises no error in the bytes so far
    import codecs
    dec = codecs.getincrementaldecoder('utf-8')('strict')
    try:
        dec.decode(data, final=False)
        return True
    except UnicodeDecodeError:
        return False


ALPHA = 3.0
d = load(R19 / 'unicode/evidence/qwen_trace.json')
print('trace completed', d['completed'], 'reference_reproduced', d['reference_reproduced'], 'adapter', d['adapter_sha256'][:16],
      'reference_generation_sha256', d['reference_generation_sha256'][:16])
rel = load(RELEASE / 'qwen3-8b/das_s42/tune.json')
relp = by_pid(rel)
for pid in ['tqa:138', 'tqa:150', 'tqa:412']:
    c = d['cases'][pid]
    ids = c['effective_ids']
    toks = c['bytes']['token_strings']
    mybytes = [tok_bytes(t) for t in toks]
    filehex = c['bytes']['token_bytes_hex']
    assert [b.hex() if b is not None else None for b in mybytes] == filehex, (pid, filehex)
    raw = b''.join(b for b in mybytes if b is not None)
    text = raw.decode('utf-8', errors='replace')
    print(f"\n{pid} {c['question']!r} batch_start={c['batch_start']} row={c['batch_row']}")
    print(f"  ids={ids}")
    print(f"  my byte decode of tokens={[b.hex() if b else None for b in mybytes]}")
    print(f"  my utf-8 decode={text!r} original_output={c['original_output']!r} equal={text == c['original_output']} "
          f"release tune.json das adapter text={relp[pid]['outputs']['das_generation_adapter']!r} finished={relp[pid]['finished']['das_generation_adapter']}")
    print(f"  step5 id={ids[5]} bytes={mybytes[5].hex()} | step6 bytes={mybytes[6].hex()} | step7 bytes={mybytes[7].hex()} | step8 id={ids[8]}")
    pend = mybytes[5][1:] + mybytes[6]
    print(f"  pending UTF-8 after step6 = {pend.hex()} ; step7 byte legal continuation? {legal_utf8_continuation(pend, mybytes[7])}")
    # edits: one record per step for this row? collect
    edits = {}
    for e in c['edits']:
        assert e['batch_row'] == c['batch_row']
        assert e['step'] not in edits
        edits[e['step']] = e
    n = len(ids)
    for s in (5, 7):
        e = edits[s]
        cb, ca, cc = e['coordinate_before'], e['coordinate_after'], e['center_coordinate']
        wn = e['weight_norm']
        centre_term = -ALPHA * cc
        dyn = ALPHA * cb
        intended_len = abs(ALPHA * (cb - cc)) * wn
        print(f"  step{s}: center_kind={e['center_kind']} h_norm={e['h_norm']:.4f} kept_delta={e['kept_delta_norm']:.4f} "
              f"intended_delta={e['intended_delta_norm']:.4f} kept_rel={100 * e['kept_relative_norm']:.4f}% "
              f"(recomputed kept/h={100 * e['kept_delta_norm'] / e['h_norm']:.4f}%) | centre_proj={cc:.4f} cur_proj={cb:.4f} "
              f"coord_after={ca:.4f} (4*cb-3*cc={4 * cb - 3 * cc:.4f}) | centre_term={centre_term:.4f} dynamic={dyn:+.4f} "
              f"|alpha(cb-cc)|*|w|={intended_len:.4f} weight_norm={wn}")
        top = e['largest_changed_dimensions']
        print('     top changed dims:', [(t['dimension'], t['before'], t['after']) for t in top])
    # active steps (exclude post-stop padding)
    act = [edits[s]['kept_relative_norm'] for s in range(n)]
    print(f"  active steps 0..{n - 1} rel edits %: {[round(100 * x, 2) for x in act]}")
    # generation-time scores at steps 5 and 7
    for s in (5, 7):
        st = c['steps'][s]['token']
        am = st['argmax_id']
        lp = next(t['log_probability'] for t in st['tokens'] if t['id'] == am)
        print(f"  generation scores step{s}: argmax={am} p={math.exp(lp):.6%}")

# replays at the common prefixes
print('\n--- common-prefix replays (full-prefix recompute, original batch)')
for b in d['batches']:
    for rp in b['common_prefix_replays']:
        s = rp['step']
        rowpid = [p for p in ['tqa:138', 'tqa:150', 'tqa:412'] if p in b['pair_ids']]
        assert rp['same_raw_prefix_for_every_variant'] and rp['original_batch_preserved']
        out = []
        for var in ['baseline', 'original_das', 'adapted', 'adapted_history_only']:
            v = list(rp['variants'][var].values())
            assert len(v) == 1
            v = v[0]
            am = v['argmax_id']
            tk = next(t for t in v['tokens'] if t['id'] == am)
            tb = tok_bytes(tk['token_string'])
            extra = ''
            if var == 'adapted':
                extra = f" cached_maxdiff={v['cached_processed_score_max_abs_difference']:.4f} cached_top1_match={v['cached_top1_match']}"
            out.append(f"{var}: argmax={am} {tk['token_string']!r} bytes={tb.hex() if tb else None} p={math.exp(tk['log_probability']):.6g}{extra}")
        print(f"  {rowpid} step{s}: " + ' | '.join(out))
