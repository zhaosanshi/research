# Iteration_01 analysis (Claude Opus 5.5, 2026-11-22, queue I section 7.3), read-only, no model, no judge call.
# Capability with the honest sentence appended to the SYSTEM prompt (decision of the lead session 23:30), E at s_sel
# (newsets/h22/n1s_choice_<model>.json) at the main-table layer:
#   MMLU (option probability, capability/full_h22/<model>.json from launchers/.../cap_h22_model.sh), the 13,042 items not
#   in capability/sets/mmlu_1000_20261119.json: base = x (capability/full/<model>.json), P = x_E2, P + E, and the
#   no-sentence E at the main-table strength (the existing full-set file that has it).
#   GSM8K (1,319 items, cap 512, rule scoring cap_score.judge_row): base = neutral (capability/gen_main/gsm8k/<model>/b8),
#   P = neutral_E2, P + E at "prompt" (capability/gen_main/gsm8k_h22/<model>/b8), the no-sentence E at "prompt" where the
#   file exists.
# Readings: P - base, (P + E) - base, (P + E) - P, and (E) - base for reference, in accuracy points with a paired
# bootstrap over items (4000, seed 20261119); five models: the mean of the five changes, items resampled jointly.
# Modes: mmlucheck MODEL [DIR] | gsmcheck MODEL | mmluparts MODEL S_HALF (the checks of the launchers) | table [--half]
# --half (section 9, 2026-11-23 05:2x): adds P + E(s_half) (capability/full_h22s, gsm8k_h22) and the GSM8K truncation counts;
# Qwen3.5-9B's no-sentence GSM8K E at "prompt" is read from gsm8k_h22 when the usual directory has none.
# usage (scoring container, from /workspace/Truth_Decoupling): python3 -B Iteration_01/diag/sentence_capability.py MODE [MODEL]
import glob
import json
import os
import random
import sys

sys.path.insert(0, "Iteration_01/diag")
import cap_score as CS            # noqa: E402
import n1_on_sentence as N        # noqa: E402

CAP = "Iteration_01/capability"
SEEN = CAP + "/sets/mmlu_1000_20261119.json"
ORDER = N.ORDER
MAIN_MMLU = ("full", "full_mid", "full_high", "full_low", "full_l16", "full_l16hi", "full_l24", "full_l28")


def read(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sel(model):
    return read(N.CHOICE % model)["strength_dot"]


HALF = "--half" in sys.argv       # section 9: add P + E(s_half) (and, for GSM8K, the truncation counts)
TRUNC = {}


def half(model):
    h = float(sel(model)) / 2
    return ("%.4f" % h).rstrip("0").rstrip(".")


def trunc(p):
    """Rows that hit the token cap without an explicit final answer (cap_score reason "truncated")."""
    if not os.path.exists(p):
        alt = p.replace("/gsm8k/", "/gsm8k_h22/")
        if not os.path.exists(alt):
            return None
        p = alt
    gold = CS.gold_of("gsm8k")
    return sum(CS.judge_row("gsm8k", r, gold(r["id"]))[1] == "truncated" for r in read(p)["rows"])


def mmlucheck(model, d="full_h22"):
    c = read("%s/%s/%s_check.json" % (CAP, d, model))["arms"]["x"]["rows"]
    f = read("%s/full/%s.json" % (CAP, model))["arms"]["x"]["rows"][:len(c)]
    same = sum(a["id"] == b["id"] and a["pick"] == b["pick"] for a, b in zip(c, f))
    gap = max(abs(a["logp"][L] - b["logp"][L]) for a, b in zip(c, f) for L in "ABCD")
    print("mmlu check %s: x on %d items vs the existing full-set file: same pick %d, largest log-prob difference %.2e" % (model, len(c), same, gap))
    if same != len(c) or gap > 1e-3:
        raise SystemExit(1)


def mmluparts(model, sh):
    """Section 9: the E write of full_h22s against the main-table E arm of the existing full-set files (same layer and kind;
    amount = main amount x s_half / main strength)."""
    display, layer, s_main = N.A.KEY[model]
    main_arm = "x:E%d.w.p%s" % (layer, N.dot(s_main))
    ref = None
    for d in MAIN_MMLU:
        p = "%s/%s/%s.json" % (CAP, d, model)
        if os.path.exists(p):
            j = read(p)
            if main_arm in j["arms"]:
                ref = j["arms"][main_arm]["writes"][0]
                break
    w = read("%s/full_h22s/%s.json" % (CAP, model))["arms"]["x_E2:E%d.w.p%s" % (layer, sh)]["writes"][0]
    ratio = float(sh) / float(N.dot(s_main))
    ok = ref is not None and all(w[k] == ref[k] for k in ("family", "layer", "kind", "gap")) and abs(w["amount"] - ref["amount"] * ratio) <= 1e-6 * abs(ref["amount"])
    print("mmlu write record %s: E%d amount %.6g, main %.6g x %.4g -> %s" % (model, layer, w["amount"], ref["amount"] if ref else float("nan"), ratio, "identical" if ok else "DIFFERS"))
    if not ok:
        raise SystemExit(1)


def gsmcheck(model):
    d = "%s/gen_main/gsm8k_h22/%s/b8" % (CAP, model)
    sub = sorted(glob.glob(d + "/s*.neutral.json"))
    if len(sub) != 1:
        raise SystemExit("gsm check: one subset neutral file expected")
    a = read(sub[0])["rows"]
    b = read("%s/gen_main/gsm8k/%s/b8/neutral.json" % (CAP, model))["rows"][:len(a)]
    same = sum(x["id"] == y["id"] and all(x[k] == y[k] for k in ("text", "token_ids", "finished", "stop_id")) for x, y in zip(a, b))
    print("gsm8k check %s: neutral on %d items vs the existing no-write pass: identical %d" % (model, len(a), same))
    if same != len(a):
        raise SystemExit(1)


def boot(vals_per_model, n_boot=4000, seed=20261119):
    """vals_per_model: [[d_1 .. d_n]] (same items for every model); mean over models of the mean difference, in points."""
    n = len(vals_per_model[0])
    per_item = [sum(v[i] for v in vals_per_model) / len(vals_per_model) for i in range(n)]
    rng = random.Random(seed)
    bs = sorted(sum(per_item[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return "%+.2f [%+.2f, %+.2f]" % (100 * sum(per_item) / n, 100 * bs[int(0.025 * n_boot)], 100 * bs[int(0.975 * n_boot) - 1])


def mmlu_model(model, seen):
    display, layer, s_main = N.A.KEY[model]
    main_arm = "x:E%d.w.p%s" % (layer, N.dot(s_main))
    base, emain = None, None
    for d in MAIN_MMLU:
        p = "%s/%s/%s.json" % (CAP, d, model)
        if not os.path.exists(p):
            continue
        j = read(p)
        if base is None:
            base = {r["id"]: r["pick"] == r["gold"] for r in j["arms"]["x"]["rows"]}
        if main_arm in j["arms"] and emain is None:
            emain = {r["id"]: r["pick"] == r["gold"] for r in j["arms"][main_arm]["rows"]}
    h = read("%s/full_h22/%s.json" % (CAP, model))["arms"]
    p_arm, pe_arm = "x_E2", "x_E2:E%d.w.p%s" % (layer, sel(model))
    P = {r["id"]: r["pick"] == r["gold"] for r in h[p_arm]["rows"]}
    PE = {r["id"]: r["pick"] == r["gold"] for r in h[pe_arm]["rows"]}
    ids = [i for i in base if i not in seen]
    out = {"base": base, "P": P, "P+E": PE, "E (no sentence)": emain}
    if HALF:
        p = "%s/full_h22s/%s.json" % (CAP, model)
        hs = read(p)["arms"].get("x_E2:E%d.w.p%s" % (layer, half(model))) if os.path.exists(p) else None
        out["P+E half"] = {r["id"]: r["pick"] == r["gold"] for r in hs["rows"]} if hs else None
    return ids, out


def gsm_model(model, gold):
    display, layer, s_main = N.A.KEY[model]

    def score(p):
        if not os.path.exists(p):
            return None
        g = read(p)
        if g["contract"]["max_new_tokens"] != 512 or g["contract"]["batch_size"] != 8 or not g["contract"]["full_pass"]:
            raise SystemExit("%s: not a full 512-token batch-8 pass" % p)
        return {r["id"]: CS.judge_row("gsm8k", r, gold(r["id"]))[0] == "R" for r in g["rows"]}
    tok = lambda s: s.replace(".", "-")  # noqa: E731
    d = {"base": score("%s/gen_main/gsm8k/%s/b8/neutral.json" % (CAP, model)),
         "P": score("%s/gen_main/gsm8k_h22/%s/b8/neutral_E2.json" % (CAP, model)),
         "P+E": score("%s/gen_main/gsm8k_h22/%s/b8/neutral_E2--E%d-w-p%s-at-prompt.json" % (CAP, model, layer, tok(sel(model)))),
         "E (no sentence)": score("%s/gen_main/gsm8k/%s/b8/neutral--E%d-w-p%s-at-prompt.json" % (CAP, model, layer, s_main))}
    if HALF:
        d["P+E half"] = score("%s/gen_main/gsm8k_h22/%s/b8/neutral_E2--E%d-w-p%s-at-prompt.json" % (CAP, model, layer, tok(half(model))))
        if d["E (no sentence)"] is None:      # section 9: the reference made for Qwen3.5-9B
            d["E (no sentence)"] = score("%s/gen_main/gsm8k_h22/%s/b8/neutral--E%d-w-p%s-at-prompt.json" % (CAP, model, layer, s_main))
        TRUNC[model] = {k: trunc("%s/gen_main/%s/%s/b8/%s.json" % (CAP, dd, model, f)) for k, dd, f in (
            ("base", "gsm8k", "neutral"), ("P", "gsm8k_h22", "neutral_E2"),
            ("P+E", "gsm8k_h22", "neutral_E2--E%d-w-p%s-at-prompt" % (layer, tok(sel(model)))),
            ("P+E half", "gsm8k_h22", "neutral_E2--E%d-w-p%s-at-prompt" % (layer, tok(half(model)))),
            ("E (no sentence)", "gsm8k", "neutral--E%d-w-p%s-at-prompt" % (layer, s_main)))}
    return list(d["base"]), d


def table():
    seen = set(read(SEEN)["ids"])
    gold = CS.gold_of("gsm8k")
    for task, fn in (("MMLU, the 13,042 items not seen before", lambda m: mmlu_model(m, seen)), ("GSM8K, 1,319 items, cap 512", lambda m: gsm_model(m, gold))):
        print("\n## %s (accuracy %%; differences in points, paired bootstrap)" % task)
        allv = {}
        for model in ORDER:
            try:
                ids, d = fn(model)
            except FileNotFoundError as e:
                print("    %-12s not ready (%s)" % (model, e))
                continue
            if d["P"] is None or d["P+E"] is None:
                print("    %-12s not ready" % model)
                continue
            acc = " | ".join("%s %.2f" % (k, 100 * sum(v[i] for i in ids) / len(ids)) if v is not None else "%s none" % k for k, v in d.items())
            diffs = [("P - base", "P", "base"), ("(P+E) - base", "P+E", "base"), ("(P+E) - P", "P+E", "P"), ("E (no sentence) - base", "E (no sentence)", "base")]
            if HALF:
                diffs += [("(P+E half) - base", "P+E half", "base"), ("(P+E half) - (P+E)", "P+E half", "P+E")]
            parts = []
            for name, a, b in diffs:
                if d[a] is None or d[b] is None:
                    parts.append("%s none" % name)
                    continue
                v = [int(d[a][i]) - int(d[b][i]) for i in ids]
                allv.setdefault(name, {})[model] = (ids, v)
                parts.append("%s %s" % (name, boot([v])))
            print("    %-12s %d items | %s || %s" % (model, len(ids), acc, " | ".join(parts)))
            if model in TRUNC:
                print("        truncated at 512 tokens: " + " | ".join("%s %s" % (k, "none" if v is None else v) for k, v in TRUNC.pop(model).items()))
        for name, per in allv.items():
            if len(per) == len(ORDER) and len({tuple(x[0]) for x in per.values()}) == 1:
                print("    five models, mean of the changes: %s %s" % (name, boot([per[m][1] for m in ORDER])))
            else:
                print("    five models: %s on %d of 5 models only, no total" % (name, len(per)))


def main():
    argv = [a for a in sys.argv if a != "--half"]
    sys.argv[:] = argv
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "mmlucheck":
        mmlucheck(sys.argv[2], *sys.argv[3:4])
    elif mode == "mmluparts":
        mmluparts(sys.argv[2], sys.argv[3])
    elif mode == "gsmcheck":
        gsmcheck(sys.argv[2])
    elif mode == "table":
        table()
    else:
        raise SystemExit("usage: sentence_capability.py mmlucheck|gsmcheck MODEL | table")


if __name__ == "__main__":
    main()
