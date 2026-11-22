# Iteration_01 analysis (Claude Opus 5.5, 2026-11-22, queue I: discussion/work_queue/2026-11-22_queue_i.md), read-only,
# no model, no judge call. The main table's method on top of the honest sentence: P = the prompt x_E2 (plain prompt +
# honest sentence, configs/prompts/e44_eval_extra_h22_v1.json), E at the main-table layer, C at the main-table C layer,
# both written at "prompt", TruthfulQA, DeepSeek Pro.
#   S0 = P, no write (baseline_x_E2, eval_extra/judge_<split>_h22c4_<split>);
#   SE(s) = P + E(s); SN1(s) = N1 on P: the conditions of eval_extra/judge_<split>_n1s_<split> (single E: E_L<L>_x_E2-...,
#   E + C: X_x_E2--...-at-prompt) are renamed to the main table's names (E_L<L>_x-..., X_x--..., baseline_x) and put into
#   the same structure as diag/main_table_prompt.prompt_split (newsets_score.NewSplit); systems by the unchanged
#   S.system and ML.n1_pick.
# Step 0 (queue section 4 item 5, hard): the same builder on the prefix x (the main table's own work dirs) must reproduce
# the main table's test numbers of single E and N1 for the five models; a mismatch stops the program.
# Modes:  plan MODEL          print "EL S HALF CL" from the main table's records (A.KEY, MT.Data)
#         choose MODEL        tune split, both strengths: SE and SN1 (n1_pick's caps); the strength with the higher SN1
#                             net, ties to the smaller one; writes newsets/h22/n1s_choice_<model>.json (never overwritten)
#         tune                the tune table of the five models (both strengths)
#         test                the test readings of queue section 3 (items 1-5), per model and the five models together
# usage (scoring container, from /workspace/Truth_Decoupling): python3 -B Iteration_01/diag/n1_on_sentence.py MODE [MODEL]
import json
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import iti_table as A  # noqa: E402

MP, ML, S, ps = A.MP, A.ML, A.S, A.ps
fr = MP.fr
V = {"R": 1, "W": -1, "X": 0, "H": 0}
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
SUF = "-at-prompt"
REF = {"llama3.1-8b": ((40, 319), (62, 476)), "qwen3.5-9b": ((168, 408), (186, 444)), "qwen3.8-27b": ((142, 354), (175, 490)),
       "gemma3-12b": ((103, 323), (158, 489)), "gemma3-27b": ((105, 423), (144, 510))}
CHOICE = "Iteration_01/newsets/h22/n1s_choice_%s.json"


def dot(s):
    return s.replace("-", ".")


def dash(s):
    return s.replace(".", "-")


def half(s):
    h = float(dot(s)) / 2
    return dash(("%.4f" % h).rstrip("0").rstrip("."))


def plan(model):
    display, layer, s = A.KEY[model]
    D = MP.MT.Data(display, 1)
    return layer, dot(s), dot(half(s)), D.m["c"]


def restricted(model, i):
    """The main table's question list of split i (main_table_prompt / tq_same_n_decomp restriction)."""
    display, layer, s = A.KEY[model]
    T, err = MP.prompt_split(display, layer, i, s)
    if T is None:
        raise SystemExit(err)
    src = [x for x in ML.sources(display)[2] if x[0] == layer]
    want = set(src[0][3 if i else 1].qs)
    return [q for q in T[3] if q in want]


def build(model, i, prefix, s, qs):
    """(NewSplit, Texts) of split i on the prompt prefix ("x" = the main table's work dirs, "x_E2" = this queue's)."""
    display, layer, _ = A.KEY[model]
    D = MP.MT.Data(display, i)
    split = ("tune", "test")[i]
    run = D.m["ec"][i].split("/eval_extra/")[0]
    cl = D.m["c"]
    st, bl, tx = {}, {}, {}

    def put(k, v, b, t):
        if k in st and st[k] != v:
            raise SystemExit("%s: condition %s %s appears twice with different verdicts" % (model, k[0], k[1]))
        st[k], bl[k], tx[k] = v, b, t

    if prefix == "x":
        utx = fr.raw_texts(D.m["ec"][i])
        for q in D.sp.qs:
            put((q, "baseline_x"), D.sp.st[(q, "baseline_x")], D.sp.blank[(q, "baseline_x")], utx[(q, "baseline_x")])
        dirs = [MP.work(run, split, "posprompt%s" % ("l%d" % layer)), MP.work(run, split, "posprompt2l%d" % layer)]
        rename = lambda c: c[:-len(SUF)] if c.endswith(SUF) else c  # noqa: E731
        base_name = "baseline_x"
    else:
        dirs = [MP.work(run, split, "h22c4"), MP.work(run, split, "n1s")]
        base_name = "baseline_x_E2"

        def rename(c):
            if c == "baseline_x_E2":
                return "baseline_x"
            if c.startswith("X_x_E2--") and c.endswith(SUF):
                return "X_x--" + c[len("X_x_E2--"):-len(SUF)]
            if c.startswith("E_L") and "_x_E2-" in c:
                return c.replace("_x_E2-", "_x-", 1)
            return None
    for w in dirs:
        if not MP.has(w):
            raise FileNotFoundError("%s: no Pro verdicts in %s" % (model, w))
        st1, bl1 = ps.load(w, "pro")
        t1 = fr.raw_texts(w)
        for (q, c), v in st1.items():
            if prefix == "x" and c == "baseline_x":
                continue                      # the usual no-write cells are taken from the main table's split
            if prefix == "x_E2" and c != base_name and not (c.startswith("X_x_E2--") or "_x_E2-" in c):
                continue                      # e.g. the cell-4 single-C condition of the h22c4 work dir
            c2 = rename(c)
            if c2 is None:
                continue
            put((q, c2), v, bl1[(q, c)], t1[(q, c)])
    sh = S.shell(layer, cl)
    need = ["baseline_x", sh.cE(s)] + [sh.cEC(s, k) for k in ps.LAD]
    sp = S.NewSplit(layer, cl, st, bl, need, qs=qs)
    return sp, fr.Texts(tx, tx)


def run_sys(sp, tx, cfg):
    states, ans = S.system(sp, tx, cfg)[:2]
    return {q: (V[x], bool(a)) for q, x, a in zip(sp.qs, states, ans)}


def net(d, qs=None):
    qs = list(d) if qs is None else qs
    return sum(d[q][0] for q in qs), sum(d[q][1] for q in qs)


def step0():
    """The main table's test numbers of single E and N1, from the same builder on the prefix x."""
    ok = True
    for model in ORDER:
        display, layer, s = A.KEY[model]
        ct, cx = restricted(model, 0), restricted(model, 1)
        tsp, ttx = build(model, 0, "x", s, ct)
        xsp, xtx = build(model, 1, "x", s, cx)
        cfg = ML.n1_pick(tsp, ttx, s)
        got = (net(run_sys(xsp, xtx, ("E", s))), net(run_sys(xsp, xtx, cfg)))
        hit = got == REF[model]
        ok &= hit
        print("# step 0, prefix x, %s: single E %+d (%d), N1 %+d (%d) [%s]; reference %+d (%d) / %+d (%d) -> %s" % (
            (model,) + got[0] + got[1] + (ML.ro.name_of(cfg) if hasattr(ML, "ro") else str(cfg),) + REF[model][0] + REF[model][1]
            + ("reproduced" if hit else "NOT REPRODUCED",)))
    if not ok:
        raise SystemExit("step 0 failed: the builder does not reproduce the main table; stop")


def tune_model(model):
    display, layer, s = A.KEY[model]
    ct = restricted(model, 0)
    rows = []
    for st_ in (s, half(s)):
        sp, tx = build(model, 0, "x_E2", st_, ct)
        cfg = ML.n1_pick(sp, tx, st_)
        se, sn = run_sys(sp, tx, ("E", st_)), run_sys(sp, tx, cfg)
        s0 = run_sys(sp, tx, ("base",))
        rows.append(dict(s=st_, cfg=cfg, qs=len(sp.qs), dropped=len(ct) - len(sp.qs), S0=net(s0), SE=net(se), SN1=net(sn)))
    return rows


def show_tune(model, rows):
    for r in rows:
        print("  %-12s E %-5s %d tune q (%d left out) | S0 %+d (%d) | SE %+d (%d) | SN1 %+d (%d) with %s" % (
            (model, dot(r["s"]), r["qs"], r["dropped"]) + r["S0"] + r["SE"] + r["SN1"] + (str(r["cfg"]),)))


def choose(model):
    rows = tune_model(model)
    show_tune(model, rows)
    best = max(rows, key=lambda r: (r["SN1"][0], -float(dot(r["s"]))))
    out = CHOICE % model
    if os.path.exists(out):
        old = json.load(open(out))
        if old["strength"] != best["s"] or old["cfg"] != list(best["cfg"]):
            raise SystemExit("%s exists with another choice" % out)
        print("choice exists and agrees: %s" % out)
    else:
        with open(out, "x") as f:
            json.dump({"model": model, "strength": best["s"], "strength_dot": dot(best["s"]), "cfg": list(best["cfg"]),
                       "tune": {r["s"]: {"SE": r["SE"], "SN1": r["SN1"], "cfg": list(r["cfg"])} for r in rows}}, f, indent=1)
    print("chosen %s: E %s, %s (tune SN1 %+d)" % (model, dot(best["s"]), best["cfg"], best["SN1"][0]))
    print(dot(best["s"]))


def test():
    step0()
    per = {}
    for model in ORDER:
        display, layer, s = A.KEY[model]
        ch = json.load(open(CHOICE % model))
        sc, cfg = ch["strength"], tuple(ch["cfg"])
        ct, cx = restricted(model, 0), restricted(model, 1)
        tsp, ttx = build(model, 0, "x", s, ct)
        xsp, xtx = build(model, 1, "x", s, cx)
        main_e, main_n1 = run_sys(xsp, xtx, ("E", s)), run_sys(xsp, xtx, ML.n1_pick(tsp, ttx, s))
        psp, ptx = build(model, 1, "x_E2", sc, cx)
        s0, se, sn = run_sys(psp, ptx, ("base",)), run_sys(psp, ptx, ("E", sc)), run_sys(psp, ptx, cfg)
        D = MP.MT.Data(display, 1)
        st1, bl1 = ps.load(D.m["d"][1], "pro")
        qs = [q for q in psp.qs if q in main_e and st1.get((q, "baseline_conf_E2"), "?") != "?"]
        t2 = {q: (s0[q] if s0[q][1] else (V[st1[(q, "baseline_conf_E2")]], not bl1[(q, "baseline_conf_E2")])) for q in qs}
        per[model] = dict(qs=qs, S0=s0, SE=se, SN1=sn, mainE=main_e, mainN1=main_n1, T2=t2, s=sc, cfg=cfg,
                          dropped=len(cx) - len(qs))
    diffs = (("SN1", "S0"), ("SE", "S0"), ("SN1", "SE"), ("SN1", "mainN1"), ("SE", "mainE"), ("SN1", "T2"))
    names = ("S0", "SE", "SN1", "mainE", "mainN1", "T2")
    for model in ORDER:
        d = per[model]
        qs = d["qs"]
        print("\n## TruthfulQA test, %s: E %s, %s, %d questions (%d of the main table's list left out)" % (
            model, dot(d["s"]), d["cfg"], len(qs), d["dropped"]))
        print("    " + " | ".join("%s %+d (%d)" % ((n,) + net(d[n], qs)) for n in names))
        print("    " + " | ".join("%s - %s %s" % (a, b, ps.ci([d[a][q][0] - d[b][q][0] for q in qs])) for a, b in diffs))
        print("    S0 -> SN1: " + transitions([(d["S0"][q], d["SN1"][q]) for q in qs]))
    common = sorted(set.intersection(*(set(per[m]["qs"]) for m in ORDER)))
    print("\n## TruthfulQA test, the five models together, %d common questions" % len(common))
    print("    " + " | ".join("%s %+d (%d)" % (n, sum(per[m][n][q][0] for m in ORDER for q in common),
                                             sum(per[m][n][q][1] for m in ORDER for q in common)) for n in names))
    print("    " + " | ".join("%s - %s %s" % (a, b, ps.ci([sum(per[m][a][q][0] - per[m][b][q][0] for m in ORDER) for q in common]))
                              for a, b in diffs))
    print("    S0 -> SN1: " + transitions([(per[m]["S0"][q], per[m]["SN1"][q]) for m in ORDER for q in common]))


def transitions(pairs):
    """Value classes: right (+1), wrong (-1), none (0: no answer, or an answer without information)."""
    cls = lambda x: {1: "right", -1: "wrong", 0: "none"}[x[0]]  # noqa: E731
    c = {}
    for a, b in pairs:
        c[(cls(a), cls(b))] = c.get((cls(a), cls(b)), 0) + 1
    order = [("wrong", "right"), ("wrong", "none"), ("right", "wrong"), ("right", "none"), ("none", "right"), ("none", "wrong"),
             ("right", "right"), ("wrong", "wrong"), ("none", "none")]
    return " | ".join("%s->%s %d" % (a, b, c.get((a, b), 0)) for a, b in order) + " (none = blank or no information)"


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "plan":
        print("%d %s %s %d" % plan(sys.argv[2]))
    elif mode == "choose":
        choose(sys.argv[2])
    elif mode == "tune":
        step0()
        print("\n## TruthfulQA tune, P = x_E2, both E strengths (S0 = P, SE = P + E, SN1 = N1 on P with n1_pick's caps)")
        for model in ORDER:
            show_tune(model, tune_model(model))
    elif mode == "test":
        test()
    elif mode == "step0":
        step0()
    else:
        raise SystemExit("usage: n1_on_sentence.py plan|choose MODEL | tune | test | step0")


if __name__ == "__main__":
    main()
