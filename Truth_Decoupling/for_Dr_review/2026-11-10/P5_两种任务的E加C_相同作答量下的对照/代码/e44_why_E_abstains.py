# -*- coding: utf-8 -*-
"""Read-only: on the plain prompt x, which questions does the E write turn into "I don't know", and why.
Uses only files already on disk (frozen answers, generations, both judges' labels, confidence_tune_v1.json).
usage: e44_why_E_abstains.py RUN_DIR"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e44_round1_tables as t  # noqa: E402

run = Path(sys.argv[1])
frozen = json.loads((run / "frozen_tune.json").read_text(encoding="utf-8"))
ids = [q["pair_id"] for q in frozen["questions"]]
ans = {q["pair_id"]: q["answers"] for q in frozen["questions"]}
base_ab = {q: ans[q]["x"]["refusal_rule"] for q in ids}
sent_ab = {q: ans[q]["x_E1"]["refusal_rule"] for q in ids}


def gen_ab(path):
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    return {r["pair_id"]: r["refusal_rule"] for r in rows}


print("## 1. dose: rule abstentions on x (192 questions), E at layer 10 vs a random direction of the same size\n")
print("| write | rule abstentions | newly abstained vs no write | newly answered |")
print("|---|---|---|---|")
print("| no write | %d | — | — |" % sum(base_ab.values()))
print("| prompt: + truth sentence (x_E1), no write | %d | %d | %d |" % (sum(sent_ab.values()), sum(sent_ab[q] and not base_ab[q] for q in ids), sum(base_ab[q] and not sent_ab[q] for q in ids)))
for s in ("0-25", "0-5", "0-75", "1", "1-5", "2"):
    p = run / "gen" / ("E_L10_x-w-p%s_tune.json" % s)
    if p.exists():
        a = gen_ab(p)
        print("| E +%s | %d | %d | %d |" % (s.replace("-", "."), sum(a.values()), sum(a[q] and not base_ab[q] for q in ids), sum(base_ab[q] and not a[q] for q in ids)))
for s in ("p2", "m2"):
    p = run / "eval_extra" / "gen" / ("x--E10-rand-%s_tune.json" % s)
    a = gen_ab(p)
    print("| random direction %s | %d | %d | %d |" % (s, sum(a.values()), sum(a[q] and not base_ab[q] for q in ids), sum(base_ab[q] and not a[q] for q in ids)))

E2 = gen_ab(run / "gen" / "E_L10_x-w-p2_tune.json")
EC = gen_ab(run / "eval_extra" / "gen" / "x--E10-w-p2-and-C31-w-m0-25_tune.json")
new = [q for q in ids if E2[q] and not base_ab[q]]              # answered without the write, "I don't know" with E +2
keep = [q for q in ids if not E2[q] and not base_ab[q]]          # answered both times
conf = json.loads((run / "eval_extra" / "confidence_tune_v1.json").read_text(encoding="utf-8"))
cx = {r["pair_id"]: r["mean_logp"] for r in conf["targets"]["x"]["rows"]}
print("\n## 2. the %d questions E +2 turns into \"I don't know\" (answered without the write)\n" % len(new))
for judge in ("pro", "allenai"):
    gen, specs, wids, state, rule, words = t.load(run / "eval_extra" / "judge_tune_ec_tune", judge)
    b = {q: state[(q, "baseline_x")] for q in ids}
    e = {q: state[(q, "E_L10_x-w-p2")] for q in ids}
    ec = {q: state[(q, "X_x--E10-w-p2-and-C31-w-m0-25")] for q in ids}
    cnt = lambda qs, lab: {k: sum(1 for q in qs if lab[q] == k) for k in "RWHX"}  # noqa: E731
    nb, kb = cnt(new, b), cnt(keep, b)
    print("- %s: their answers WITHOUT the write were R / W / H = %d / %d / %d (right %.0f%%); questions answered both times were %d / %d / %d (right %.0f%%)" % (
        judge, nb["R"], nb["W"], nb["H"], 100.0 * nb["R"] / max(1, nb["R"] + nb["W"]), kb["R"], kb["W"], kb["H"], 100.0 * kb["R"] / max(1, kb["R"] + kb["W"])))
    back = [q for q in new if not EC[q]]
    nbk = cnt(back, ec)
    print("  with E + C -0.25, %d of them are answered again: R / W / H = %d / %d / %d" % (len(back), nbk["R"], nbk["W"], nbk["H"]))
mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
srt = sorted(cx[q] for q in ids if not base_ab[q])
rank = {v: i for i, v in enumerate(srt)}
print("- confidence of the no-write answer (mean log-prob per token): those E turns into \"I don't know\" %.3f, those still answered %.3f; "
      "their average percentile among all answered questions %.0f%% vs %.0f%%" % (
          mean([cx[q] for q in new]), mean([cx[q] for q in keep]),
          100 * mean([rank[cx[q]] / (len(srt) - 1) for q in new]), 100 * mean([rank[cx[q]] / (len(srt) - 1) for q in keep])))
both = sum(1 for q in new if sent_ab[q])
print("- overlap with the prompt: %d of the %d are also \"I don't know\" when the truth sentence is simply added to the prompt (the prompt alone newly abstains on %d)" % (
    both, len(new), sum(sent_ab[q] and not base_ab[q] for q in ids)))
