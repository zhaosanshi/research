# Iteration_01 analysis (Claude Fable 5.2, 2026-11-17), read-only, no model, no judge call.
# 赵磊 2026-11-17: "gemma4-12B这么敏感，不太对，随机怎么可能那么敏感呢，如果真是这样，找到官方的依据。"
# (1) config.json of Gemma-4-12B against Gemma-4-31B: the keys that differ (text model and top level).
# (2) Dose-response of the random-direction writes on the plain prompt (tune, 192), from the saved generations of the E line,
#     the C line and the E+C grid: every arm is one fixed random unit vector per layer (seed 42 + 7919 * layer; the same
#     vectors as Gemma-3-12B, which has the same width) times a norm; listed by the norm relative to the hidden state.
#     "answers changed" = the normalized text differs from the no-write answer; the zero-write control of the pipeline
#     reproduces the no-write answers exactly, so a change is an effect of the write.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/gemma4_random_dose.py
import glob, json, os, re, statistics as S
M = "models"
def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, pre + k + "."))
        else:
            out[pre + k] = v
    return out
a = flat(json.load(open(M + "/gemma-4-12B-it/config.json")))
b = flat(json.load(open(M + "/gemma-4-31B-it/config.json")))
print("## config.json: keys that differ between 12B and 31B (text / top level)")
for k in sorted(set(a) | set(b)):
    if a.get(k, "<absent>") != b.get(k, "<absent>"):
        va, vb = json.dumps(a.get(k, "<absent>"))[:90], json.dumps(b.get(k, "<absent>"))[:90]
        if not any(s in k for s in ("id2label", "label2id")):
            print("   %-58s 12B: %-40s 31B: %s" % (k[:58], va, vb))
# (2) all gen files of the two Gemma-4-12B runs and of Gemma-3-12B / 31B: random arms, norm of the write, relative size, changed answers
RUNS = [("Gemma-4-12B", "Iteration_01/results/ff/gemma4/12b/e44/gemma4-12b"), ("Gemma-4-12B stronger caution", "Iteration_01/results/ff_xu1/gemma4/12b/e44/gemma4-12b"),
        ("Gemma-4-31B", "Iteration_01/results/ff/gemma4/31b/e44/gemma4-31b-dm"), ("Gemma-3-12B", "Iteration_01/results/ff/gemma3/12b/e44/gemma3-12b")]
for name, run in RUNS:
    rows = []
    for f in sorted(glob.glob(run + "/gen/*_x-rand-*_tune.json")) + sorted(glob.glob(run + "/eval_extra/gen/x--*rand*_tune.json")):
        base = os.path.basename(f)
        if "-and-" in base:
            continue
        d = json.load(open(f, encoding="utf-8"))
        rel = [r["write"]["prefill_rel"] for r in d["rows"] if r.get("write")]
        amt = [abs(r["write"]["amount"]) for r in d["rows"] if r.get("write")]
        sm = d["summary"]
        new = sum(1 for r in d["rows"] if not r["baseline_refusal_rule"] and r["refusal_rule"])
        rel0 = sum(1 for r in d["rows"] if r["baseline_refusal_rule"] and not r["refusal_rule"])
        rows.append((S.mean(rel), S.mean(amt), base.replace("_tune.json", ""), sm["changed_normalized"], new, rel0))
    rows.sort()
    print("\n## %s: random-direction writes on the plain prompt, sorted by size relative to |h| (tune, 192)" % name)
    for rel, amt, b_, ch, new, rel0 in rows:
        print("   write/|h| %.4f  norm %9.3f  %-28s answers changed %3d | new refusals %2d | refusals released %2d" % (rel, amt, b_, ch, new, rel0))
