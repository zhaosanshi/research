# Iteration_01 analysis (Claude Fable 5.2, 2026-11-22), read-only, no model, no judge call. Exploration.
# (QA)^2 at the write rule "prompt". DeepSeek Pro, two-axis rubric, reference = the set's abstractive answer; value of
# an answer: A (true and informative) +1, B (false) -1, C / H (no information, hedged) 0, empty output 0.
# Systems: no write; prompt only (conf_E2); single C; single E; N1 = E on every question, a question blank under E
# (the rule detector's abstention or an empty answer) is re-asked with E + C at 0.25 / 0.5 / 1 / 2 up to the cap of its
# kind (the model's own blank / answered without a write and blank under E; a cap may be "none" = no re-ask) and the
# first non-blank answer is taken, for E's blanks only if it differs from the no-write answer (release_origin.norm).
# A question that is not released keeps E's cell.
# Three uses:
#   frozen   (default) every setting as chosen on TruthfulQA (SET below); the 100-question pilot (defaults) or the
#            570-question run (--task qaqafull --bank ... --work test_full570), per model and the models together.
#   --tune   the lead, 2026-11-22 01:58, on the frozen pilot: "the strength is wrong". The method chooses strengths on a
#            tune split of the dataset, so: the 100 pilot questions are the TUNE split. Per model: single E at every
#            strength found, single C at every strength found, N1 at every E strength with every pair of caps; the
#            choice (highest net, ties to the earlier candidate: lower strength, then the smaller caps, "none" first)
#            is printed and, with --write-choice FILE, saved.
#   --choice FILE   read the settings chosen on tune and report them on the questions that are NOT in the tune split
#            (give the tune id file as --skip-ids): N1 against single E at its own chosen strength, and the rest.
# --work takes a comma list of judge work dirs under newsets/judge/<task>/<model>/ (their verdicts are merged).
# --skip-ids FILE leaves out the questions of an id file.
# Runs in the scoring container (the verdict files record container paths):
#   docker exec -u 1000:1001 -w /workspace/Truth_Decoupling truth_dec python3 -B Iteration_01/diag/qaqa_pilot_table.py llama3.1-8b
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks"]
import newsets_gen as G                              # noqa: E402
import newsets_score as S                            # noqa: E402
import release_origin as ro                          # noqa: E402
from judge_two_axis_run import load_verified_single  # noqa: E402

VALUE = {"A": 1, "B": -1, "C": 0, "H": 0}
LAD = ("0.25", "0.5", "1", "2")
CAPS = (None,) + LAD
# model key: (E layer, strength, C layer, cap for E's blanks, cap for own blanks, single C): the TruthfulQA choices at
# "prompt" (results/write_positions/main_table_prompt_budget1_20261121.txt)
SET = {
    "llama3.1-8b": (16, "3.5", 19, "1", "2", "0.25"),
    "qwen3.5-9b": (15, "2", 16, "0.25", "0.5", "0.25"),
    "qwen3.8-27b": (28, "1.75", 33, "0.5", "1", "0.5"),
    "gemma3-12b": (24, "2.5", 25, "2", "2", "0.5"),
    "gemma3-27b": (28, "1", 31, "1", "2", "0.25"),
}
# E strengths looked at on the tune split (the sweeps of the TriviaQA / PopQA runs; they contain the main strength)
SWEEP = {"llama3.1-8b": ("1", "2", "3.5", "5"), "qwen3.5-9b": ("0.5", "1", "2", "3"), "qwen3.8-27b": ("0.5", "1", "1.75", "3"),
         "gemma3-12b": ("1", "2.5", "4"), "gemma3-27b": ("0.5", "1", "2")}
TYPES = (("no_failure", "valid assumption"), ("failure", "false assumption"), ("unverifiable", "unverifiable assumption"))
ORDER = ("no write", "prompt only", "single C", "single E", "N1")
PAIRS = (("N1", "single E"), ("N1", "no write"), ("N1", "single C"), ("N1", "prompt only"), ("single E", "no write"))


class Raw:
    """The answers and verdicts of one model on one bank: every arm that has a complete file and verdicts."""

    def __init__(self, model, task, bank_path, works):
        self.model = model
        el, _, cl = SET[model][:3]
        self.el, self.cl = el, cl
        with open(bank_path, encoding="utf-8") as f:
            bank = [json.loads(l) for l in f]
        self.ids = [b["id"] for b in bank]
        self.qtype = {b["id"]: b["qtype"] for b in bank}
        self.gdir = Path(S.NS) / "gen_prompt" / task / model / "test"
        self.args = SimpleNamespace(task=task, split="test", model=model, items=bank_path)
        self.rows, self.cond, self.val = {}, {}, {}
        for work_name in works:
            work = Path(S.NS) / "judge" / task / model / work_name
            # require_complete=False: an empty output has no vote; every other cell must be scored (checked here)
            _, cells, votes = load_verified_single(str(work), "pro", generation_path=str(work / "judge_input.json"), require_complete=False)
            for key, c in cells.items():
                v = votes[key]
                k = (c["pair_id"].split(":", 1)[1], c["condition"])
                if v["status"] == "scored":
                    new = VALUE[v["label"]]
                elif v.get("basis") == "empty_output_no_judge_vote":
                    new = 0
                else:
                    raise SystemExit("%s: unresolved vote for %s %s" % (model, k[0], k[1]))
                if self.val.get(k, new) != new:
                    raise SystemExit("%s: two work dirs disagree on %s %s" % (model, k[0], k[1]))
                self.val[k] = new

    def e(self, s):
        return "x:E%d.w.p%s" % (self.el, s)

    def c(self, k):
        return "x:C%d.w.m%s" % (self.cl, k)

    def ec(self, s, k):
        return "%s+C%d.w.m%s" % (self.e(s), self.cl, k)

    def has(self, arm):
        """True when the arm has a complete generation file of this bank and a verdict for every non-empty answer."""
        if arm in self.rows:
            return True
        p = self.gdir / (G.arm_token(arm) + ".json")
        if not p.is_file():
            return False
        g = G.load_generation(p, self.args, self.ids)
        k = g["contract"]
        if k["arm"] != arm or k.get("positions") != ("prompt" if ":" in arm else None):
            raise SystemExit("%s: %s is not the pass %s at the expected write rule" % (self.model, p.name, arm))
        rows = {r["id"]: r for r in g["rows"]}
        if any((q, k["condition"]) not in self.val and rows[q]["text"].strip() for q in self.ids):
            return False
        self.rows[arm], self.cond[arm] = rows, k["condition"]
        return True

    def need(self, arm):
        if not self.has(arm):
            raise FileNotFoundError("%s: no complete judged pass for %s" % (self.model, arm))

    def blank(self, arm, q):
        r = self.rows[arm][q]
        return bool(r["rule_abstain"]) or not r["text"].strip()

    def cell(self, arm, q):
        return self.val.get((q, self.cond[arm]), 0)

    def text(self, arm, q):
        return self.rows[arm][q]["text"]

    def arm_system(self, arm):
        self.need(arm)
        return {q: self.cell(arm, q) for q in self.ids}, {q: not self.blank(arm, q) for q in self.ids}

    def n1_system(self, s, cap_e, cap_own):
        """N1 at E strength s with the two caps (None = that kind of blank is not re-asked)."""
        e = self.e(s)
        for a in [e, "x"] + [self.ec(s, k) for k in LAD]:
            self.need(a)
        v, a, events = {}, {}, {}
        for q in self.ids:
            if not self.blank(e, q):
                v[q], a[q] = self.cell(e, q), True
                continue
            own = self.blank("x", q)
            cap = cap_own if own else cap_e
            lad = () if cap is None else LAD[:LAD.index(cap) + 1]
            k = next((k for k in lad if not self.blank(self.ec(s, k), q)), None)
            kind = "own" if own else "E's"
            if k is None:
                events[q] = (kind, "none")
            elif not own and ro.norm(self.text(self.ec(s, k), q)) == ro.norm(self.text("x", q)):
                events[q] = (kind, "vetoed")
                k = None
            if k is None:
                v[q], a[q] = self.cell(e, q), False
            else:
                v[q], a[q] = self.cell(self.ec(s, k), q), True
                events[q] = (kind, "right" if v[q] > 0 else "wrong" if v[q] < 0 else "no information")
        return v, a, events


def net(system, qs):
    return sum(system[0][q] for q in qs)


def show_cap(c):
    return "none" if c is None else c


def report(R, ids, systems, events, title, setting):
    print("# (QA)^2 %s, %s, %d questions, write rule prompt, %s; DeepSeek Pro" % (title, R.model, len(ids), setting))
    groups = [("all", ids)] + [(label, [q for q in ids if R.qtype[q] == t]) for t, label in TYPES]
    for label, qs in groups:
        print("\n## %s (%d questions)" % (label, len(qs)))
        for name in systems:
            v, a = systems[name][0], systems[name][1]
            print("    %-28s net %+4d | answers %3d | right %3d | wrong %3d" % (
                name, sum(v[q] for q in qs), sum(a[q] for q in qs), sum(v[q] > 0 for q in qs), sum(v[q] < 0 for q in qs)))
        pairs = [(x, y) for x, y in PAIRS if x in systems and y in systems]
        if "single E, N1's strength" in systems:
            pairs.insert(1, ("N1", "single E, N1's strength"))
        print("    " + " | ".join("%s minus %s %s" % (x, y, S.ps.ci([systems[x][0][q] - systems[y][0][q] for q in qs])) for x, y in pairs))
    cnt = lambda kind, what: sum(1 for q in ids if events.get(q) == (kind, what))  # noqa: E731
    print("\n## the re-ask: answered without a write and blank under E: released right %d, wrong %d, no information %d; same answer again and kept blank %d; not released %d | the model's own blanks: released right %d, wrong %d, no information %d; not released %d" % (
        cnt("E's", "right"), cnt("E's", "wrong"), cnt("E's", "no information"), cnt("E's", "vetoed"), cnt("E's", "none"),
        cnt("own", "right"), cnt("own", "wrong"), cnt("own", "no information"), cnt("own", "none")))


def changed_lines(R, ids, e_arm):
    ch = [q for q in ids if not R.blank("x", q) and not R.blank(e_arm, q) and ro.norm(R.text("x", q)) != ro.norm(R.text(e_arm, q))]
    for t, label in (("all", "all"),) + TYPES:
        qs = [q for q in ch if t == "all" or R.qtype[q] == t]
        b = sum(R.cell(e_arm, q) > R.cell("x", q) for q in qs)
        w = sum(R.cell(e_arm, q) < R.cell("x", q) for q in qs)
        print("    answered in both passes and E changed the answer, %s: %d, better %d, worse %d" % (label, len(qs), b, w))


def tune(R, ids):
    """Every setting on the tune questions; the choice as a dict."""
    strengths = [s for s in SWEEP[R.model] if all(R.has(a) for a in [R.e(s)] + [R.ec(s, k) for k in LAD])]
    cs = [k for k in LAD if R.has(R.c(k))]
    R.need("x")
    if not strengths or not cs:
        raise FileNotFoundError("%s: no complete E strength or no single C on the tune split" % R.model)
    base = R.arm_system("x")
    print("# (QA)^2 tune split, %s, %d questions: no write net %+d (answers %d)%s" % (
        R.model, len(ids), net(base, ids), sum(base[1][q] for q in ids),
        "; prompt only net %+d" % net(R.arm_system("conf_E2"), ids) if R.has("conf_E2") else ""))
    c_net = [(k, net(R.arm_system(R.c(k)), ids)) for k in cs]
    print("    single C: " + " | ".join("%s: %+d" % x for x in c_net))
    e_net, n_best = [], []
    for s in strengths:
        e = R.arm_system(R.e(s))
        grid = [((ce, co), net(R.n1_system(s, ce, co), ids)) for ce in CAPS for co in CAPS]
        best = max(grid, key=lambda x: (x[1], -grid.index(x)))
        e_net.append((s, net(e, ids)))
        n_best.append((s, best[0], best[1]))
        print("    E %-4s single E %+4d (answers %3d) | N1 best %+4d at caps E's %s / own %s | by caps (rows: cap for E's blanks none 0.25 0.5 1 2; columns: cap for own blanks): %s" % (
            s, net(e, ids), sum(e[1][q] for q in ids), best[1], show_cap(best[0][0]), show_cap(best[0][1]),
            " / ".join(" ".join("%+d" % net(R.n1_system(s, ce, co), ids) for co in CAPS) for ce in CAPS)))
    s_e = max(e_net, key=lambda x: (x[1], -e_net.index(x)))
    k_c = max(c_net, key=lambda x: (x[1], -c_net.index(x)))
    n1 = max(n_best, key=lambda x: (x[2], -n_best.index(x)))
    choice = {"model": R.model, "tune_questions": len(ids), "strengths_seen": strengths, "single_C_seen": cs,
              "single_E": s_e[0], "single_C": k_c[0], "n1_strength": n1[0], "n1_cap_e": n1[1][0], "n1_cap_own": n1[1][1],
              "tune_net": {"no write": net(base, ids), "single E": s_e[1], "single C": k_c[1], "N1": n1[2]}}
    print("    chosen on tune: single E at %s (%+d) | single C at %s (%+d) | N1 at E %s with caps E's %s / own %s (%+d)" % (
        s_e[0], s_e[1], k_c[0], k_c[1], n1[0], show_cap(n1[1][0]), show_cap(n1[1][1]), n1[2]))
    return choice


def main():
    argv = sys.argv[1:]
    opt = {"--task": "qaqa", "--bank": "Iteration_01/newsets/bank/qaqa_pilot_bank_20261122.jsonl", "--work": "test_pilot100",
           "--skip-ids": None, "--write-choice": None, "--choice": None}
    mode_tune = False
    while argv[:1] and (argv[0] in opt or argv[0] == "--tune"):
        if argv[0] == "--tune":
            mode_tune, argv = True, argv[1:]
        else:
            opt[argv[0]], argv = argv[1], argv[2:]
    if not argv:
        raise SystemExit("usage: qaqa_pilot_table.py [--tune [--write-choice FILE]] [--choice FILE] [--task T] [--bank FILE] [--work NAME[,NAME]] [--skip-ids FILE] MODEL [MODEL ...]")
    skip = set()
    if opt["--skip-ids"]:
        with open(opt["--skip-ids"], encoding="utf-8") as f:
            skip = set(json.load(f)["ids"])
    chosen = {}
    if opt["--choice"]:
        with open(opt["--choice"], encoding="utf-8") as f:
            chosen = json.load(f)
    title = "pilot" if opt["--task"] == "qaqa" else "full run"
    done, choices = {}, {}
    for model in argv:
        try:
            R = Raw(model, opt["--task"], opt["--bank"], opt["--work"].split(","))
            ids = [q for q in R.ids if q not in skip]
            if mode_tune:
                choices[model] = tune(R, ids)
                print()
                continue
            el, s, cl, cap_e, cap_own, kc = SET[model]
            s_e = s
            setting = "E%d at %s, caps E's %s / own %s, single C %s (all from TruthfulQA)" % (el, s, cap_e, cap_own, kc)
            if chosen:
                ch = chosen[model]
                s, cap_e, cap_own, kc, s_e = ch["n1_strength"], ch["n1_cap_e"], ch["n1_cap_own"], ch["single_C"], ch["single_E"]
                setting = "chosen on the tune split: N1 at E%d %s with caps E's %s / own %s; single E at %s; single C at %s" % (
                    el, s, show_cap(cap_e), show_cap(cap_own), s_e, kc)
            n1 = R.n1_system(s, cap_e, cap_own)
            systems = {"no write": R.arm_system("x"), "prompt only": R.arm_system("conf_E2"), "single C": R.arm_system(R.c(kc)),
                       "single E": R.arm_system(R.e(s_e))}
            if s_e != s:
                systems["single E, N1's strength"] = R.arm_system(R.e(s))
            systems["N1"] = (n1[0], n1[1])
        except (FileNotFoundError, ValueError) as err:
            print("\n# %s: not ready (%s)\n" % (model, str(err)[:200]))
            continue
        report(R, ids, systems, n1[2], title + (", without the tune questions" if skip else ""), setting)
        changed_lines(R, ids, R.e(s_e))
        print()
        done[model] = (ids, R.qtype, systems)
    if mode_tune and opt["--write-choice"] and choices:
        with open(opt["--write-choice"], "x", encoding="utf-8") as f:
            json.dump(choices, f, indent=1)
        print("choices written to %s" % opt["--write-choice"])
    if len(done) > 1:
        first = next(iter(done.values()))
        ids, qtype = first[0], first[1]
        if any(d[0] != ids for d in done.values()):
            raise SystemExit("the models do not have the same questions")
        print("# the %d models together (%s), questions resampled jointly" % (len(done), ", ".join(done)))
        for label, qs in [("all", ids)] + [(lab, [q for q in ids if qtype[q] == t]) for t, lab in TYPES]:
            print("\n## %s (%d questions per model)" % (label, len(qs)))
            print("    net: " + " | ".join("%s %+d" % (name, sum(d[2][name][0][q] for d in done.values() for q in qs)) for name in ORDER))
            for x, y in PAIRS:
                per = [[d[2][x][0][q] - d[2][y][0][q] for q in qs] for d in done.values()]
                t_, lo, hi = S.joint_ci(per)
                print("    %-28s %+d [%+d, %+d], %d of %d models positive" % ("%s minus %s" % (x, y), t_, lo, hi, sum(sum(v) > 0 for v in per), len(per)))


if __name__ == "__main__":
    main()
