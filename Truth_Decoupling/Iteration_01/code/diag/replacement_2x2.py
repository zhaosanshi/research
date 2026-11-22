# Iteration_01 analysis (Claude Opus 5.5, 2026-11-22, queue H), read-only, no model, no judge call. Exploration.
# The 2x2 of discussion/work_queue/2026-11-22_queue_h.md: honesty by the sentence (1, 4) or by E (2, 3) x answering
# by the must-answer instruction (1, 2) or by C (3, 4); every write at the core rule "prompt", C at 0.25.
#   cell 1 = must-answer + honest sentence, no write (conf_E2)        cell 2 = must-answer + E (conf:E<L>.w.p<S>)
#   cell 3 = plain prompt + E + C 0.25 (x:E..+C<L>.w.m0.25)            cell 4 = plain prompt + honest sentence + C 0.25
#                                                                       only (x_E2:C<L>.w.m0.25, queue H, new)
# Step 0 reproduces GPT's four numbers on (QA)^2 with cells 1-3 (discussion/validation_dataset/2026-11-22f_replay/
# replacement.py, same arms, same verdicts, same bootstrap: S.joint_ci on the per-question sums over the five models):
# pilot 2 - 1 = +57 [16, 98], 3 - 2 = +3 [-22, 30]; held-out +180 [104, 258], +60 [3, 115]; a mismatch stops.
# Then the readings of section 2 of the queue (rules fixed there), per set, per model and the five models together:
#   (a) the questions on which all four cells answer (blank = rule abstention or empty output): net of each cell;
#   (b) every question: net and answer count of each cell;
#   E's effect 2 - 1 and 3 - 4; C's effect 3 - 2 and 4 - 1; main statistic 2 - 4; interaction (3 - 4) - (2 - 1);
#   paired intervals: questions resampled, a question's models together (S.joint_ci; per model the same on one list).
# Sets: (QA)^2 100 pilot (tune) questions and the 470 held-out ones; TruthfulQA tune (cell 2 has no tune answers at
# "prompt": "none") and test (the main table's question list, questions with an unresolved verdict in a cell left out).
# usage (scoring container, from /workspace/Truth_Decoupling):
#   python3 -B Iteration_01/diag/replacement_2x2.py dev|held   (repro: step 0 only)
# --with-zero (added 2026-11-22 20:4x, queue H section 5): extra rows for cell 0 = plain prompt + honest sentence, no write
# (x_E2; (QA)^2 work dir test_full570_h22z, TruthfulQA baseline_x_E2 of the h22c4 work dirs): its net and answers on the
# same question sets as the four cells (cell 0 does not enter the "all four answer" set) and 4 - 0 (C's own effect when
# honesty is the sentence). Without the flag the output is the earlier one, line for line.
import json
import sys
from pathlib import Path

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks"]
import qaqa_pilot_table as Q   # noqa: E402
import iti_table as A          # noqa: E402

S = Q.S
MP, ps = A.MP, A.ps
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
ROOT = Path(S.NS)
PILOT = set(json.loads((ROOT / "splits/qaqa_pilot_20261122.json").read_text())["ids"])
BANK = ROOT / "bank/qaqa_full_bank_20261122.jsonl"
WORK = ("test_full570", "test_full570_tuned", "test_full570_strong")
GPT = {True: ((57, 16, 98), (3, -22, 30)), False: ((180, 104, 258), (60, 3, 115))}
ZERO = "--with-zero" in sys.argv
V = {"R": 1, "W": -1, "X": 0, "H": 0}
CELLS = ("1", "2", "3", "4")
CONTRASTS = (("E: 2 - 1", "21", lambda c: c["2"] - c["1"]), ("E: 3 - 4", "34", lambda c: c["3"] - c["4"]),
             ("C: 3 - 2", "32", lambda c: c["3"] - c["2"]), ("C: 4 - 1", "41", lambda c: c["4"] - c["1"]),
             ("main: 2 - 4", "24", lambda c: c["2"] - c["4"]),
             ("interaction (3-4)-(2-1)", "1234", lambda c: (c["3"] - c["4"]) - (c["2"] - c["1"])))


def qaqa(with4=True):
    """{model: {q: {cell: (value, answered)}}}, the bank's id order; and GPT's reproduction."""
    out, gpt = {}, {True: [], False: []}
    for model in ORDER:
        el, strength, cl = Q.SET[model][:3]
        raw = Q.Raw(model, "qaqafull", str(BANK), WORK)
        e = raw.e(strength)
        arms = {"1": "conf_E2", "2": "conf:" + e.split(":")[1], "3": raw.ec(strength, "0.25")}
        for a in arms.values():
            raw.need(a)
        if len(set(raw.ids)) != 570 or not PILOT <= set(raw.ids):
            raise SystemExit("wrong evaluation bank")
        d = {q: {k: (raw.cell(a, q), not raw.blank(a, q)) for k, a in arms.items()} for q in raw.ids}
        if with4:
            r4 = Q.Raw(model, "qaqafull", str(BANK), ["test_full570_h22"])
            r4.gdir = ROOT / "gen_prompt" / "h22" / "qaqafull" / model / "test"
            a4 = "x_E2:C%d.w.m0.25" % cl
            r4.need(a4)
            if r4.ids != raw.ids:
                raise SystemExit("cell 4 is on another bank")
            for q in raw.ids:
                d[q]["4"] = (r4.cell(a4, q), not r4.blank(a4, q))
        if with4 and ZERO:
            r0 = Q.Raw(model, "qaqafull", str(BANK), ["test_full570_h22z"])
            r0.gdir = ROOT / "gen_prompt" / "h22" / "qaqafull" / model / "test"
            r0.need("x_E2")
            if r0.ids != raw.ids:
                raise SystemExit("cell 0 is on another bank")
            for q in raw.ids:
                d[q]["0"] = (r0.cell("x_E2", q), not r0.blank("x_E2", q))
        out[model] = d
    ids = list(out[ORDER[0]])
    for pilot in (True, False):
        qs = [q for q in ids if (q in PILOT) == pilot]
        sh = lambda m, q: all(out[m][q][k][1] for k in "123")  # noqa: E731
        f_h = [sum(out[m][q]["2"][0] - out[m][q]["1"][0] for m in ORDER if sh(m, q)) for q in qs]
        c_f = [sum(out[m][q]["3"][0] - out[m][q]["2"][0] for m in ORDER if sh(m, q)) for q in qs]
        n = sum(sh(m, q) for m in ORDER for q in qs)
        got = (S.joint_ci([f_h]), S.joint_ci([c_f]))
        print("# GPT reproduction, (QA)^2 %s, %d questions, three cells answered: %d model x question | 2 - 1 %s | 3 - 2 %s | GPT: %s %s -> %s" % (
            "pilot" if pilot else "held-out", len(qs), n, got[0], got[1], GPT[pilot][0], GPT[pilot][1],
            "reproduced" if tuple(got) == GPT[pilot] else "NOT REPRODUCED"))
        if tuple(got) != GPT[pilot]:
            raise SystemExit("GPT's numbers are not reproduced: stop")
    return out, ids


def tqa(split):
    """{model: {q: {cell: (value, answered)}}} for TruthfulQA tune (i = 0) or test (i = 1); cell 2 only on test."""
    i = ("tune", "test").index(split)
    out = {}
    for model in ORDER:
        display, layer, s = A.KEY[model]
        D = MP.MT.Data(display, i)
        run = D.m["ec"][i].split("/eval_extra/")[0]
        cl = D.m["c"]
        tag = "l%d" % layer
        src = {"1": (D.m["d"][i], "baseline_conf_E2"),
               "3": (MP.work(run, split, "posprompt2" + tag), "X_x--E%d-w-p%s-and-C%d-w-m0-25-at-prompt" % (layer, s, cl)),
               "4": (MP.work(run, split, "h22c4"), "C_L%d_x_E2-w-m0-25" % cl)}
        if ZERO:
            src["0"] = (MP.work(run, split, "h22c4"), "baseline_x_E2")
        if i == 1:
            src["2"] = (MP.work(run, "test", "pospromptd" + tag), "E_L%d_conf-w-p%s" % (layer, s))
        loaded = {}
        for k, (w, cond) in src.items():
            if not MP.has(w):
                raise FileNotFoundError("%s: no Pro verdicts in %s" % (model, w))
            st, bl = ps.load(w, "pro")
            if not any(c == cond for _, c in st):
                raise SystemExit("%s: condition %s not in %s" % (model, cond, w))
            loaded[k] = (st, bl, cond)
        d, dropped = {}, 0
        for q in D.sp.qs:
            if any(loaded[k][0].get((q, loaded[k][2]), "?") == "?" for k in loaded):
                dropped += 1
                continue
            d[q] = {k: (V[st[(q, cond)]], not bl[(q, cond)]) for k, (st, bl, cond) in loaded.items()}
        print("# TruthfulQA %s, %s: %d questions of the main table's list, %d left out (a cell without a verdict)" % (split, model, len(d), dropped))
        out[model] = d
    common = sorted(set.intersection(*(set(v) for v in out.values())))
    return out, common


def table(title, data, qs, models):
    have = [k for k in CELLS if all(k in data[m][q] for m in models for q in qs)]
    print("\n## %s: %d questions per model, %s" % (title, len(qs), ", ".join(models) if len(models) > 1 else "one model"))
    if have != list(CELLS):
        print("    cells: %s; cell%s %s: none" % (" ".join(have), "" if len(CELLS) - len(have) == 1 else "s", " ".join(k for k in CELLS if k not in have)))
    for label in ("(a) all four answer", "(b) every question"):
        shared = label.startswith("(a)")
        use = lambda m, q: (not shared) or all(data[m][q][k][1] for k in have)  # noqa: E731
        n = sum(use(m, q) for m in models for q in qs)
        nets = {k: sum(data[m][q][k][0] for m in models for q in qs if use(m, q)) for k in have}
        ans = {k: sum(data[m][q][k][1] for m in models for q in qs if use(m, q)) for k in have}
        print("    %s, %d model x question: %s" % (label, n, " | ".join("cell %s net %+d answers %d" % (k, nets[k], ans[k]) for k in have)))
        parts = []
        for name, need, f in CONTRASTS:
            if all(k in have for k in need):
                per = [sum(f({k: data[m][q][k][0] for k in have}) for m in models if use(m, q)) for q in qs]
                parts.append("%s %+d [%+d, %+d]" % ((name,) + S.joint_ci([per])))
            else:
                parts.append("%s none" % name)
        print("        " + " | ".join(parts))
        if ZERO and all("0" in data[m][q] for m in models for q in qs):
            z = [sum(data[m][q]["4"][0] - data[m][q]["0"][0] for m in models if use(m, q)) for q in qs]
            print("        extra: cell 0 (plain + honest sentence, no write) net %+d answers %d | 4 - 0 %+d [%+d, %+d]" % (
                (sum(data[m][q]["0"][0] for m in models for q in qs if use(m, q)), sum(data[m][q]["0"][1] for m in models for q in qs if use(m, q)))
                + tuple(S.joint_ci([z]))))


def report(name, data, qs):
    for m in ORDER:
        table("%s, %s" % (name, m), data, qs, [m])
    table("%s, the five models together" % name, data, qs, list(ORDER))


def main():
    args = [a for a in sys.argv[1:] if a != "--with-zero"]
    part = args[0] if args else ""
    if part not in ("dev", "held", "repro"):
        raise SystemExit("usage: replacement_2x2.py dev|held|repro")
    q, ids = qaqa(part != "repro")
    if part == "repro":
        return
    if part == "dev":
        report("(QA)^2 pilot / tune, 100", q, [x for x in ids if x in PILOT])
        t, common = tqa("tune")
        report("TruthfulQA tune", t, common)
    else:
        report("(QA)^2 held-out, 470", q, [x for x in ids if x not in PILOT])
        t, common = tqa("test")
        report("TruthfulQA test", t, common)


if __name__ == "__main__":
    main()
