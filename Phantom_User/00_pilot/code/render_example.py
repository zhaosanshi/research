#!/usr/bin/env python3
"""render_example.py -- print the exact string the model sees in a short version of the probe conversation
(2 filler pairs instead of ~60), clean and contaminated, so the measurement point can be shown literally.

  PROBE_LANG=en python render_example.py --model $MODEL_ROOT/Llama-3.1-8B-Instruct \
      --filler results/filler_en_general.jsonl [--wrapper 1] [--kind meta]
"""
import argparse

import turn_boundary_probe as tbp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--filler", required=True)
    ap.add_argument("--kind", default="meta")
    ap.add_argument("--wrapper", type=int, default=1)
    ap.add_argument("--n-turns", type=int, default=2)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    pairs, _ = tbp.load_filler(args.filler)
    eot_str, eot_id, hdr_str, hdr_id, eom_id = tbp.detect_special(tok)
    for k in (0, 1):
        msgs, mp = tbp.build_conv(kind=args.kind, n_turns=args.n_turns, wrapper=bool(args.wrapper), contam_k=k,
                                  thin_input=True, pre_tool_signoff=True, filler=pairs[:args.n_turns])
        text = tbp.render_chat(msgs, tok, mp, eot_str)
        # shorten the 40-line tool result for display
        lines = text.split("\n")
        out, skipping = [], False
        for ln in lines:
            if ln.lstrip().startswith(("4  # line 4 of", "   4  # line 4 of")):
                out.append(ln)
                out.append("   ... (lines 5-38 of the tool result) ...")
                skipping = True
                continue
            if skipping:
                if ln.lstrip().startswith("39  # line 39"):
                    skipping = False
                    out.append(ln)
                continue
            out.append(ln)
        print("=" * 20, f"contam_k={k}  messages={len(msgs)}  measure={mp}  chars={len(text)}", "=" * 20)
        print("\n".join(out))
        print()


if __name__ == "__main__":
    main()
