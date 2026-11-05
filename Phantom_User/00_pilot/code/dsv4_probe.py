#!/usr/bin/env python3
"""dsv4_probe.py -- slot_start gate probe for DeepSeek-V4 (vLLM backend; transformers in the NGC image has no deepseek_v4).

Same conversations as turn_boundary_probe.py (tbp.build_conv, paper setting: English, wrapper=0, plain contaminant),
rendered with DeepSeek's own reference encoder (encoding/encoding_dsv4.py, thinking_mode='chat' -> prompt ends with
"<｜Assistant｜></think>").  Tool results sit inside the user turn as <tool_result>...</tool_result>, exactly like
DeepSeek's gold example.  Measures the full first-token distribution (vLLM logprobs=-1) at the slot: P of the special
user token <｜User｜>, of the plain words user/User/USER/Human/用户, top-10; then greedy continuations after forcing
'user', 'User' and '<｜User｜>', and the unforced greedy answer.

Contaminant variants (env, as in tbp): CONTAM_ROLE_WORD=user (plain "\\n\\nuser\\n<complaint>" inside the model's last k
answers) or DSV4_NATIVE_CONTAM=1 (the special-token form "<｜User｜><complaint>" with no newlines = what a real V4 leak
would look like).

  PROBE_LANG=en CONTAM_STYLE=plain python dsv4_probe.py --model $MODEL_ROOT/deepseek-v4-flash --tp 4 \
      --filler-coding results/filler_en_coding.jsonl --filler-meta results/filler_en_general.jsonl \
      --cells m00 m03 c00 c03 --seeds 1 2 3 --out results/dsv4_probe.jsonl
  ... --dry-run   # only build + encode + report token ids (no GPU)
"""
import argparse
import json
import math
import os
import sys
import time

import turn_boundary_probe as tbp

DSV4_ENCODING_DIR = os.environ.get("DSV4_ENCODING_DIR", os.path.join(os.environ.get("MODEL_ROOT", "models"), "deepseek-v4-flash", "encoding"))
sys.path.insert(0, DSV4_ENCODING_DIR)
from encoding_dsv4 import encode_messages, USER_SP_TOKEN, ASSISTANT_SP_TOKEN, eos_token  # noqa: E402

WATCH_WORDS = ["user", "User", "USER", "Human", "用户", " user", "Assistant"]
FORCE_WORDS = ["user", "User", USER_SP_TOKEN]


def build_prompt(kind, contam_k, seed, pairs_by_kind, toks_by_kind, target_tokens, native_contam):
    import random
    pairs = list(pairs_by_kind[kind])
    toks = list(toks_by_kind[kind])
    if seed:
        idx = list(range(len(pairs)))
        random.Random(seed).shuffle(idx)
        pairs = [pairs[i] for i in idx]
        toks = [toks[i] for i in idx]
    n_turns = tbp.fit_n_turns(toks, target_tokens)
    msgs, mp = tbp.build_conv(kind=kind, n_turns=n_turns, wrapper=False, contam_k=contam_k, thin_input=True,
                              pre_tool_signoff=True, filler=pairs)
    if native_contam and contam_k:
        marker = tbp.FAKE_USER_TURN_TEXT
        native = USER_SP_TOKEN + marker[len("\n\nuser\n"):]
        for m in msgs:
            if m["role"] == "assistant" and marker in m["content"]:
                m["content"] = m["content"].replace(marker, native)
    prompt = encode_messages(msgs, thinking_mode="chat")
    return msgs, prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(os.environ.get("MODEL_ROOT", "models"), "deepseek-v4-flash"))
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--kv-cache-dtype", default="fp8", help="DeepseekV4 fp8_ds_mla layout in vLLM requires an fp8 kv-cache")
    ap.add_argument("--filler-coding", required=True)
    ap.add_argument("--filler-meta", required=True)
    ap.add_argument("--cells", nargs="+", default=["m00", "m03", "c00", "c03"], help="{m|c}{0}{contam_k}")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--target-tokens", type=int, default=20000)
    ap.add_argument("--native-contam", action="store_true", help="contaminant in the <｜User｜> special-token form")
    ap.add_argument("--contam-words", nargs="+", default=["user"],
                    help="role words to put in the contaminant (one run per word, same model load); each is also forced")
    ap.add_argument("--max-forced", type=int, default=48)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pairs_by_kind, toks_by_kind = {}, {}
    for kind, path in (("coding", args.filler_coding), ("meta", args.filler_meta)):
        pairs_by_kind[kind], toks_by_kind[kind] = tbp.load_filler(path)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    user_sp_id = tok.convert_tokens_to_ids(USER_SP_TOKEN)
    asst_sp_id = tok.convert_tokens_to_ids(ASSISTANT_SP_TOKEN)
    eos_id = tok.convert_tokens_to_ids(eos_token)
    word_ids = {}
    for w in WATCH_WORDS:
        ids = tok.encode(w, add_special_tokens=False)
        word_ids[w] = (ids[0], len(ids), tok.decode(ids[:1])) if ids else None
    print(f"special ids: user={user_sp_id} assistant={asst_sp_id} eos={eos_id}; words={word_ids}", flush=True)

    base_contam = tbp.FAKE_USER_TURN_TEXT
    assert base_contam.startswith("\n\nuser\n"), "run with CONTAM_ROLE_WORD unset; use --contam-words instead"
    contam_rest = base_contam[len("\n\nuser"):]
    jobs = []
    for cw in args.contam_words:
        for cell in args.cells:
            kind = {"m": "meta", "c": "coding"}[cell[0]]
            contam_k = int(cell[2])
            for s in args.seeds:
                jobs.append((cell, kind, contam_k, s, cw))

    if args.dry_run:
        for cell, kind, contam_k, s, cw in jobs[:2]:
            tbp.FAKE_USER_TURN_TEXT = "\n\n" + cw + contam_rest
            msgs, prompt = build_prompt(kind, contam_k, s, pairs_by_kind, toks_by_kind, args.target_tokens, args.native_contam)
            n = len(tok.encode(prompt, add_special_tokens=False))
            print(f"=== {cell} seed {s}: {len(msgs)} messages, {n} tokens; prompt tail:\n{prompt[-700:]!r}")
            if contam_k:
                i = prompt.find(tbp.FAKE_USER_TURN_TEXT[len("\n\n"):].split("\n")[1][:30])
                print("    contaminant context:", repr(prompt[max(0, i - 120):i + 160]))
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=args.max_model_len, max_logprobs=-1,
              trust_remote_code=True, gpu_memory_utilization=0.92, enable_prefix_caching=True,
              kv_cache_dtype=args.kv_cache_dtype, disable_custom_all_reduce=True)
    sp_first = SamplingParams(max_tokens=1, temperature=0, logprobs=-1)
    sp_cont = SamplingParams(max_tokens=args.max_forced, temperature=0)
    sp_free = SamplingParams(max_tokens=80, temperature=0)
    out_f = open(args.out, "a", encoding="utf-8") if args.out else None

    for cell, kind, contam_k, s, cw in jobs:
        t0 = time.time()
        tbp.FAKE_USER_TURN_TEXT = "\n\n" + cw + contam_rest
        force_words = list(FORCE_WORDS) + ([cw] if cw not in FORCE_WORDS else [])
        cw_ids = tok.encode(cw, add_special_tokens=False)
        msgs, prompt = build_prompt(kind, contam_k, s, pairs_by_kind, toks_by_kind, args.target_tokens, args.native_contam)
        n_ctx = len(tok.encode(prompt, add_special_tokens=False))
        res = llm.generate([prompt], sp_first, use_tqdm=False)[0].outputs[0]
        lp = res.logprobs[0]            # {token_id: Logprob}
        logz = None                     # raw_logprobs are already log-probabilities (log-softmax); check normalisation
        total = sum(math.exp(v.logprob) for v in lp.values())
        def p_of(tid):
            v = lp.get(tid)
            return float(math.exp(v.logprob)) if v is not None else 0.0
        rec = {"condition": f"X20k_{cell}", "cell": cell, "kind": kind, "contam_k": contam_k, "filler_seed": s,
               "contam_word": cw, "p_contam_word_first": p_of(cw_ids[0]) if cw_ids else None,
               "contam_word_tokens": [tok.decode([t]) for t in cw_ids],
               "native_contam": bool(args.native_contam), "model": args.model, "n_ctx_tokens": n_ctx,
               "n_messages": len(msgs), "logprob_mass_check": total,
               "p_user_sp": p_of(user_sp_id), "p_assistant_sp": p_of(asst_sp_id), "p_eos": p_of(eos_id),
               "p_words": {w: (p_of(word_ids[w][0]) if word_ids[w] else None) for w in WATCH_WORDS},
               "word_first_tokens": {w: (word_ids[w][2] if word_ids[w] else None) for w in WATCH_WORDS}}
        top = sorted(lp.items(), key=lambda kv: -kv[1].logprob)[:10]
        rec["top10"] = [(tok.decode([tid]), round(float(math.exp(v.logprob)), 5)) for tid, v in top]
        # rank of the special user token and of plain 'user'
        ranked = sorted(lp.items(), key=lambda kv: -kv[1].logprob)
        rank_of = {tid: i + 1 for i, (tid, _) in enumerate(ranked)}
        rec["rank_user_sp"] = rank_of.get(user_sp_id)
        rec["rank_user_word"] = rank_of.get(word_ids["user"][0]) if word_ids["user"] else None
        # continuations
        rec["unforced"] = llm.generate([prompt], sp_free, use_tqdm=False)[0].outputs[0].text
        rec["force_first"] = {}
        for w in force_words:
            rec["force_first"][w] = w + llm.generate([prompt + w], sp_cont, use_tqdm=False)[0].outputs[0].text
        rec["seconds"] = round(time.time() - t0, 1)
        print(f"{rec['condition']} s{s} contam={cw!r} ctx={n_ctx} mass={total:.3f} P(<User>)={rec['p_user_sp']:.2e} r={rec['rank_user_sp']} "
              f"P(user)={rec['p_words']['user']:.2e} r={rec['rank_user_word']} P(User)={rec['p_words']['User']:.2e} "
              f"P(cw first)={rec['p_contam_word_first']} top3={rec['top10'][:3]}", flush=True)
        for w in force_words:
            print(f"    forced {w!r}: {rec['force_first'][w][:160]!r}", flush=True)
        print(f"    unforced: {rec['unforced'][:120]!r}", flush=True)
        if out_f:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()


if __name__ == "__main__":
    main()
