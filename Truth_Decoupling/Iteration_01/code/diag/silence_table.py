# Iteration_01 analysis (Claude Opus 5.5, 2026-11-17), read-only: no model, no GPU, no judge call.
# Task A of the 11-17 offline batch: "who gets silenced". For each model and split, and for each write group, the questions
# that were ANSWERED with no write (Pro state R / W / H in baseline_x) and are NOT answered with the write (Pro state X),
# counted by their no-write state (wrong / right / hedge). Reference: the wrong share W / (R + W) of the no-write answers.
# Groups: E toward truth at 1, 2; E-layer random directions (each, and pooled); C toward caution (C_L<c>_x-w-p..., read
# from the separate judged dirs that hold it, at the C layer of the E+C grid; Pro in the main table, AllenAI-only dirs in a
# separate auxiliary table); C-layer random directions; the stronger-caution prompt (Gemma-4-12B: baseline_x of the plain
# prompt -> baseline_x of the stronger-caution prompt, same questions); C toward answering (-w-m), for completeness.
# States: policy_select2.load (Pro: A R / B W / H H / C X; empty output X). Questions: policy_select2.Split.qs for the E+C
# grid; a separate dir uses its own baseline_x questions. A missing verdict ('?') in a used cell stops the script.
# Share of wrong among the silenced: W / (W + R); 95% interval by resampling the questions (4000, seed 20261117,
# A random-direction group with a cell that has no verdict is printed as NOT COMPUTED (not imputed, not replaced).
# percentiles 101st / 3900th of the sorted resamples, as policy_select2.ci). Under 10 silenced (W + R): counts only.
# The last column (answered R / W -> hedge H with the write) is printed for reference only; it is NOT part of "silenced".
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/silence_table.py [MODEL ...]
import glob
import os
import random
import re
import sys

sys.path.insert(0, "Iteration_01/diag")
import policy_select2 as ps
import c_first

SEED = 20261117
NB = 4000
ORDER = ["Llama-3.1-8B", "Qwen3.5-9B", "Qwen3.8-27B", "Gemma-3-12B", "Gemma-3-27B", "Gemma-4-12B",
         "Gemma-4-12B stronger caution", "Gemma-4-31B"]
MODELS = c_first.MODELS
# extra run roots that hold the C-toward-caution dirs of the model (the main run of the same model, not the E2 side run)
EXTRA_ROOTS = {"Llama-3.1-8B": ["results_e44_factorial_das/llama3.1-8b"], "Qwen3.5-9B": ["results_newmodels_ff_20261112/e44/qwen3.5-9b"]}


def run_root(m):
    w = m["ec"][0]
    if "/eval_extra/" not in w:
        raise SystemExit("cannot find the run root of %s" % w)
    return w.split("/eval_extra/")[0]


def boot_ratio(rows, seed=SEED):
    """rows: per question (w, r) counts; ratio sum(w) / sum(w + r) and its resampled 95% interval."""
    n = len(rows)
    tw, tr = sum(a for a, _ in rows), sum(b for _, b in rows)
    rng = random.Random(seed)
    bs = []
    for _ in range(NB):
        w = r = 0
        for _ in range(n):
            a, b = rows[rng.randrange(n)]
            w += a
            r += b
        bs.append(w / (w + r) if w + r else 0.0)
    bs.sort()
    return tw / (tw + tr), bs[100], bs[3899]


def pct(t):
    return "%.0f%% [%.0f, %.0f]" % (100 * t[0], 100 * t[1], 100 * t[2])


def silence(qs, base_st, arm_sts):
    """base_st: {q: state}; arm_sts: list of {q: state}. Per question (w, r, h) silenced counts pooled over the arms."""
    rows = []
    for q in qs:
        w = r = h = wh = rh = 0
        for a in arm_sts:
            if base_st[q] in "RWH" and a[q] == "X":
                w += base_st[q] == "W"
                r += base_st[q] == "R"
                h += base_st[q] == "H"
            if base_st[q] in "RW" and a[q] == "H":            # supplementary, NOT counted as silenced: answered -> hedge
                wh += base_st[q] == "W"
                rh += base_st[q] == "R"
        rows.append((w, r, h, wh, rh))
    return rows


def line(label, qs, rows, narms=1):
    w, r, h, wh, rh = (sum(x[i] for x in rows) for i in range(5))
    share = "counts only (%d < 10)" % (w + r) if w + r < 10 else pct(boot_ratio([(x[0], x[1]) for x in rows]))
    return "| %s | %d | %d | %d | %d | %d | %s | %d / %d |" % (label, len(qs) * narms, w, r, h, w + r + h, share, wh, rh)


HEAD = ("| group | questions x groups | silenced: wrong | right | hedge | all | wrong share W/(W+R) of the silenced [95%] | not silenced, for reference: wrong -> hedge / right -> hedge |",
        "|---|---|---|---|---|---|---|---|")


def get(st, q, c, where):
    s = st.get((q, c), "?")
    if s == "?":
        raise SystemExit("%s: no verdict for %s / %s" % (where, q, c))
    return s


def scan_cp(name, m, judge):
    """Judged dirs (single_<judge>.jsonl) holding C_L<c>_x-w-p... at the grid's C layer; {split: [(dir, [conds])]}."""
    roots = [run_root(m)] + EXTRA_ROOTS.get(name, [])
    pat = re.compile(r"^C_L%d_x-w-p[0-9-]+$" % m["c"])
    out = {"tune": [], "test": []}
    for root in roots:
        for f in sorted(glob.glob(root + "/judge_*/judge2/single_%s.jsonl" % judge) + glob.glob(root + "/eval_extra/judge_*/judge2/single_%s.jsonl" % judge)):
            work = f.split("/judge2/")[0]
            st, _ = ps.load(work, judge)
            conds = sorted({c for _, c in st if pat.match(c)})
            if not conds:
                continue
            base = os.path.basename(work)
            split = "tune" if "_tune" in base else ("test" if "_test" in base else None)
            if split is None:
                raise SystemExit("cannot tell the split of %s" % work)
            out[split].append((work, conds, st))
    return out


def report(name):
    m = MODELS[name]
    print("\n## %s (E layer %d, C layer %d; judge Pro on both splits)" % (name, m["e"], m["c"]))
    for i, split in ((1, "test"), (0, "tune")):
        sp = ps.Split(m, i)
        if m["judge"][i] != "pro":
            raise SystemExit("%s: split %s is not judged by Pro" % (name, split))
        qs = sp.qs
        where = m["ec"][i]
        b = {q: get(sp.st, q, "baseline_x", where) for q in qs}
        ref = [(1 if b[q] == "W" else 0, 1 if b[q] == "R" else 0) for q in qs]
        nr, nw, nh, nx = (sum(b[q] == c for q in qs) for c in "RWHX")
        print("\n### %s split, %s (%d questions; no write R %d W %d H %d X %d; wrong share of the no-write answers W/(R+W) = %s)" % (
            split, where, len(qs), nr, nw, nh, nx, pct(boot_ratio(ref))))
        print(HEAD[0])
        print(HEAD[1])
        have = {c for _, c in sp.st}
        arm = lambda c: {q: get(sp.st, q, c, where) for q in qs}

        def row(label, conds):
            # a write group with a missing verdict is NOT computed (no imputation, no substitute); the row says so loudly
            miss = [(q, c) for c in conds for q in qs if sp.st.get((q, c), "?") == "?"]
            if miss:
                print("| %s | NOT COMPUTED: %d cell(s) without a verdict, e.g. %s |" % (label, len(miss), miss[:3]))
                return
            print(line(label, qs, silence(qs, b, [arm(c) for c in conds]), len(conds)))
        for s in "12":
            print(line("E %s" % s, qs, silence(qs, b, [arm(sp.cE(s))])))
        rnd = sorted(c for c in have if re.fullmatch(r"E_L%d_x-rand-[mp][12]" % m["e"], c))
        if len(rnd) != 4:
            raise SystemExit("%s: E-layer random groups are %s, expected 4" % (where, rnd))
        for c in rnd:
            row("E random %s" % c.split("rand-")[1], [c])
        row("E random, 4 pooled", rnd)
        crnd = sorted(c for c in have if re.fullmatch(r"C_L%d_x-rand-[mp][0-9-]+" % m["c"], c))
        if not crnd:
            raise SystemExit("%s: no C-layer random group" % where)
        for c in crnd:
            row("C random %s" % c.split("rand-")[1], [c])
        row("C random, %d pooled" % len(crnd), crnd)
        for k in ps.LAD:
            c = sp.cC(k)
            if c in have:
                print(line("C toward answering %s" % ps.NAME[k], qs, silence(qs, b, [arm(c)])))
        if name == "Gemma-4-12B":
            sx = ps.Split(MODELS["Gemma-4-12B stronger caution"], i)
            if sorted(sx.qs) != sorted(qs):
                only = sorted(set(qs) ^ set(sx.qs))
                raise SystemExit("plain and stronger-caution prompts hold different questions on %s: %s" % (split, only[:5]))
            bx = {q: get(sx.st, q, "baseline_x", MODELS["Gemma-4-12B stronger caution"]["ec"][i]) for q in qs}
            print(line("stronger-caution prompt, no write (vs plain prompt, no write)", qs, silence(qs, b, [bx])))
    # C toward caution, separate dirs
    for judge, title in (("pro", "Pro"), ("allenai", "AllenAI, auxiliary (not comparable with the Pro rows)")):
        found = scan_cp(name, m, judge)
        if not found["tune"] and not found["test"]:
            print("\n### C toward caution at layer %d, %s: no judged dir" % (m["c"], title))
            continue
        print("\n### C toward caution at layer %d, %s (each dir against its own baseline_x)" % (m["c"], title))
        print(HEAD[0].replace("| group |", "| split | dir | group |").replace("questions x groups", "questions") +
              " no write W/(R+W) | baseline_x states agree with the E+C grid's (common questions) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for split in ("test", "tune"):
            i = 1 if split == "test" else 0
            gsp = ps.Split(m, i) if judge == m["judge"][i] else None
            for work, conds, st in found[split]:
                qs = sorted(p for p, c in st if c == "baseline_x")
                b = {q: get(st, q, "baseline_x", work) for q in qs}
                ref = pct(boot_ratio([(1 if b[q] == "W" else 0, 1 if b[q] == "R" else 0) for q in qs]))
                if gsp is not None:
                    com = [q for q in qs if q in set(gsp.qs)]
                    agree = "%d / %d" % (sum(gsp.st[(q, "baseline_x")] == b[q] for q in com), len(com))
                else:
                    agree = "grid not judged by %s" % judge
                for c in conds:
                    a = {q: get(st, q, c, work) for q in qs}
                    l = line("C toward caution %s" % c.split("w-p")[1].replace("-", "."), qs, silence(qs, b, [a]))
                    print("| %s | %s |%s %s | %s |" % (split,os.path.relpath(work, run_root(m)) if work.startswith(run_root(m)) else work, l[1:], ref, agree))


def main():
    names = sys.argv[1:] or ORDER
    for n in names:
        if n not in MODELS:
            raise SystemExit("unknown model %s" % n)
    print("# Task A: who gets silenced (Pro states; answered with no write = R / W / H, silenced = X with the write)")
    for n in names:
        report(n)


if __name__ == "__main__":
    main()
