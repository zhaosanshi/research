#!/usr/bin/env python3
# Iteration_01 exploration (main session, Claude Fable 5.2, 2026-11-24): an independent recomputation of the FalseQA
# development-set numbers. Nothing of diag/falseqa_read.py or of the project's reading code is imported: the systems
# are rebuilt here from the written definitions (header of falseqa_read.py; queue L section 11) and from the raw files:
#   bank      Iteration_01/newsets/falseqa/falseqa_full_bank_20261123.jsonl (id, set, premise)
#   answers   Iteration_01/newsets/gen_prompt/falseqa/<model>/test/*.json (contract.arm, contract.condition; rows: id,
#             text, rule_abstain), indexed by the arm name written inside each file
#   verdicts  Iteration_01/newsets/judge/falseqa/<model>/test_full700_dev/judge2/cells.jsonl + single_pro.jsonl
#             (the development work dir only; the held-out work dir is never opened)
# The settings (E layer, main strength, s_sel, C layer) are typed in here from the table of queue L (progress 00:09),
# not read from the files the reading program uses. Value of a cell: A +1, B -1, C 0, H 0, empty output 0; a cell is
# answered when its row is not blank by the rule flag and its text is not empty.
# The caps are chosen again by the written rule (highest development net; N1 / SN1 over every pair of none, 0.25, 0.5,
# 1, 2 with ties to the earlier pair; Conly / SC over 0.25, 0.5, 1, 2 with ties to the smaller cap). Compared with the
# archived output of falseqa_dev_read.py: the settings line, the chosen caps, the whole cap grids, every system's net
# and number of answers (premise false / valid / all, per model and the five-model row) and the right / wrong / no
# information counts of the five fill systems inside B.
# Not recomputed: the intervals (pair resampling) and the equal-answer-count lines (they need the self-judge margins);
# those are covered by the synthetic test of falseqa_read.py and by falseqa_table_dryrun.py.
# usage (from the project root, host or container): python3 -B Iteration_01/diag/falseqa_dev_recheck.py ARCHIVED_OUTPUT
import glob
import json
import re
import sys

NS = "Iteration_01/newsets"
BANK = NS + "/falseqa/falseqa_full_bank_20261123.jsonl"
SET = {"llama3.1-8b": (16, "3.5", "3.5", 19), "qwen3.5-9b": (15, "2", "2", 16), "qwen3.8-27b": (28, "1.75", "0.875", 33),
       "gemma3-12b": (24, "2.5", "1.25", 25), "gemma3-27b": (28, "1", "1", 31)}
LAD = ("0.25", "0.5", "1", "2")
CAPS = (None,) + LAD
VALUE = {"A": 1, "B": -1, "C": 0, "H": 0}
SYS = ("O", "E", "N1", "Conly", "S0", "SE", "SN1", "SC", "T2", "TE", "fill EC", "fill E", "fill C", "fill prompt", "fill pE2")
FILLS = ("fill EC", "fill E", "fill C", "fill prompt", "fill pE2")


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (t or "").lower())).strip()


class Model:
    def __init__(self, m, qs):
        self.m, self.qs = m, qs
        self.el, self.s, self.ss, self.cl = SET[m]
        self.rows, self.cond = {}, {}
        for p in sorted(glob.glob("%s/gen_prompt/falseqa/%s/test/*.json" % (NS, m))):
            with open(p, encoding="utf-8") as f:
                g = json.load(f)
            k = g.get("contract") if isinstance(g, dict) else None
            if not isinstance(k, dict) or "arm" not in k or "rows" not in g:
                continue
            if k["arm"] in self.rows:
                raise SystemExit("%s: two files of the arm %s" % (m, k["arm"]))
            self.rows[k["arm"]] = {r["id"]: r for r in g["rows"]}
            self.cond[k["arm"]] = k["condition"]
        w = "%s/judge/falseqa/%s/test_full700_dev/judge2/" % (NS, m)
        cells = {}
        with open(w + "cells.jsonl", encoding="utf-8") as f:
            for l in f:
                c = json.loads(l)
                cells[c["cell_id"]] = c
        self.val, self.pred = {}, {}
        with open(w + "single_pro.jsonl", encoding="utf-8") as f:
            for l in f:
                r = json.loads(l)
                c = cells[r["cell_id"]]
                key = (c["pair_id"].split(":", 1)[1], c["condition"])
                if r["status"] == "scored":
                    v = VALUE[r["label"]]
                elif r.get("basis") == "empty_output_no_judge_vote":
                    v = 0
                else:
                    raise SystemExit("%s: a development cell without a verdict: %s %s" % (m, key[0], key[1]))
                if key in self.val:
                    raise SystemExit("%s: two verdicts of %s %s" % (m, key[0], key[1]))
                self.val[key], self.pred[key] = v, c["prediction"]
        self.n_bound = 0

    def blank(self, arm, q):
        r = self.rows[arm][q]
        return bool(r["rule_abstain"]) or not r["text"].strip()

    def text(self, arm, q):
        return self.rows[arm][q]["text"]

    def cell(self, arm, q):
        """(value, answered) of the answer of the pass `arm`; the judged text must be the text of the answer file."""
        key = (q, self.cond[arm])
        t = self.text(arm, q)
        if key not in self.val:
            if t.strip():
                raise SystemExit("%s: no verdict for %s under %s" % (self.m, q, arm))
            return 0, False
        if self.pred[key].strip() != t.strip():
            raise SystemExit("%s: the judged text of %s under %s is not the text of the answer file" % (self.m, q, arm))
        self.n_bound += 1
        return self.val[key], not self.blank(arm, q)

    def single(self, arm):
        return {q: self.cell(arm, q) for q in self.qs}

    def n1(self, row, s, cap_e, cap_own):
        """E first; a question E leaves blank is asked again with E + C from 0.25 up to its cap (cap_own when the
        no-write pass of the row is blank too, else cap_e); the first answered rung is handed in, unless the question
        is one the no-write pass answers and the new answer is that same answer (then E's blank stays)."""
        e = "%s:E%d.w.p%s" % (row, self.el, s)
        out = {}
        for q in self.qs:
            if not self.blank(e, q):
                out[q] = self.cell(e, q)
                continue
            own = self.blank(row, q)
            cap = cap_own if own else cap_e
            pick = None
            if cap is not None:
                for k in LAD[:LAD.index(cap) + 1]:
                    a = "%s+C%d.w.m%s" % (e, self.cl, k)
                    if not self.blank(a, q):
                        pick = a
                        break
            if pick is not None and not own and norm(self.text(pick, q)) == norm(self.text(row, q)):
                pick = None
            out[q] = (self.cell(e, q)[0], False) if pick is None else self.cell(pick, q)
        return out

    def conly(self, row, cap):
        """The no-write pass of the row first; its blanks are asked again with C alone from 0.25 up to the cap."""
        out = {}
        for q in self.qs:
            if not self.blank(row, q):
                out[q] = self.cell(row, q)
                continue
            pick = None
            for k in LAD[:LAD.index(cap) + 1]:
                a = "%s:C%d.w.m%s" % (row, self.cl, k)
                if not self.blank(a, q):
                    pick = a
                    break
            out[q] = (self.cell(row, q)[0], False) if pick is None else self.cell(pick, q)
        return out


def net(system, qs):
    return sum(system[q][0] for q in qs)


def build(M):
    qs = M.qs
    caps, grids = {}, {}
    for name, row, s in (("N1", "x", M.s), ("SN1", "x_E2", M.ss)):
        grid = [((ce, co), net(M.n1(row, s, ce, co), qs)) for ce in CAPS for co in CAPS]
        best = grid[0]
        for g in grid[1:]:
            if g[1] > best[1]:
                best = g
        caps[name], grids[name] = best[0], [g[1] for g in grid]
    for name, row in (("Conly", "x"), ("SC", "x_E2")):
        nets = [net(M.conly(row, k), qs) for k in LAD]
        caps[name], grids[name] = LAD[max(range(len(LAD)), key=lambda i: (nets[i], -i))], nets
    te = "conf_E2:E%d.w.p%s" % (M.el, M.ss)
    d = {"O": M.single("x"), "E": M.single("x:E%d.w.p%s" % (M.el, M.s)), "N1": M.n1("x", M.s, *caps["N1"]),
         "Conly": M.conly("x", caps["Conly"]), "S0": M.single("x_E2"), "SE": M.single("x_E2:E%d.w.p%s" % (M.el, M.ss)),
         "SN1": M.n1("x_E2", M.ss, *caps["SN1"]), "SC": M.conly("x_E2", caps["SC"]), "T2": M.single("conf_E2"), "TE": M.single(te)}
    for f in FILLS:
        d[f] = {}
    for q in qs:
        s0 = d["S0"][q]
        if s0[1]:
            for f in FILLS:
                d[f][q] = s0
        else:
            d["fill EC"][q], d["fill E"][q], d["fill C"][q], d["fill prompt"][q] = d["SN1"][q], d["SE"][q], d["SC"][q], d["T2"][q]
            d["fill pE2"][q] = d["SE"][q] if d["SE"][q][1] else d["TE"][q]
    return caps, grids, d


def show(c):
    return "none" if c is None else c


def parse(path):
    """The archived output: settings, caps and grids per model; per group the net (answers) rows and the fill rows."""
    out = {"set": {}, "caps": {}, "grid": {}, "net": {}, "fill": {}}
    model, group, block = None, None, None
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    for l in lines:
        m = re.match(r"^# (\S+): E(\d+) main (\S+), s_sel (\S+), C(\d+); chosen on the 100 development questions:$", l)
        if m:
            model = m.group(1)
            out["set"][model] = (int(m.group(2)), m.group(3), m.group(4), int(m.group(5)))
            continue
        m = re.match(r"^    (N1|SN1) caps E's blanks (\S+) / own blanks (\S+) \(dev net ([+-]\d+)\); grid rows .*: (.*)$", l)
        if m:
            out["caps"][(model, m.group(1))] = (m.group(2), m.group(3), int(m.group(4)))
            out["grid"][(model, m.group(1))] = [int(x) for x in m.group(5).replace("/", " ").split()]
            continue
        m = re.match(r"^    (Conly|SC) cap (\S+) \(dev nets (.*)\)$", l)
        if m:
            out["caps"][(model, m.group(1))] = (m.group(2),)
            out["grid"][(model, m.group(1))] = [int(x.split()[1]) for x in m.group(3).split(" | ")]
            continue
        m = re.match(r"^## dev, (premise false|premise valid|all) \(", l)
        if m:
            group, block = m.group(1), None
            continue
        if l.startswith("## "):
            group, block = None, None
            continue
        if l.startswith("-- "):
            block = "net" if l.startswith("-- every system: net (answers)") else "fill" if l.startswith("-- fills inside B") else None
            continue
        if group is None or block is None or not l.startswith("  "):
            continue
        m = re.match(r"^  (\d+ models|\S+)\s+(.*)$", l)
        lab, rest = m.group(1), m.group(2)
        if block == "net":
            for part in rest.split(" | "):
                mm = re.match(r"^(.+) ([+-]\d+) \((\d+)\)$", part)
                out["net"][(group, lab, mm.group(1))] = (int(mm.group(2)), int(mm.group(3)))
        else:
            parts = rest.split(" | ")
            out["fill"][(group, lab, "B")] = int(parts[0].split()[1])
            for part in parts[1:]:
                mm = re.match(r"^(.+) (\d+): (\d+) / (\d+) / (\d+)$", part)
                out["fill"][(group, lab, mm.group(1))] = tuple(int(mm.group(i)) for i in (2, 3, 4, 5))
    return out


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: falseqa_dev_recheck.py ARCHIVED_OUTPUT")
    ref = parse(sys.argv[1])
    with open(BANK, encoding="utf-8") as f:
        bank = [json.loads(l) for l in f if l.strip()]
    dev = [r for r in bank if r["set"] == "dev"]
    ids = [r["id"] for r in dev]
    prem = {r["id"]: r["premise"] for r in dev}
    groups = (("premise false", [q for q in ids if prem[q] == "false"]), ("premise valid", [q for q in ids if prem[q] == "valid"]), ("all", ids))
    if [len(g[1]) for g in groups] != [50, 50, 100]:
        raise SystemExit("the development set is not 50 + 50 questions")
    bad, n = [], 0
    tot = {}

    def same(what, got, want):
        nonlocal n
        n += 1
        if got != want:
            bad.append("%s: recomputed %r, archived %r" % (what, got, want))
    for m in SET:
        M = Model(m, ids)
        caps, grids, d = build(M)
        same("%s settings" % m, SET[m], ref["set"].get(m))
        for name in ("N1", "SN1"):
            want = ref["caps"].get((m, name))
            same("%s %s caps and dev net" % (m, name), (show(caps[name][0]), show(caps[name][1]), max(grids[name])), want)
            same("%s %s grid (25 nets)" % (m, name), grids[name], ref["grid"].get((m, name)))
        for name in ("Conly", "SC"):
            same("%s %s cap" % (m, name), (caps[name],), ref["caps"].get((m, name)))
            same("%s %s nets by cap" % (m, name), grids[name], ref["grid"].get((m, name)))
        for label, qs in groups:
            for sname in SYS:
                got = (net(d[sname], qs), sum(d[sname][q][1] for q in qs))
                same("%s, %s, %s net (answers)" % (m, label, sname), got, ref["net"].get((label, m, sname)))
                t = tot.setdefault((label, sname), [0, 0])
                t[0] += got[0]
                t[1] += got[1]
            inb = [q for q in qs if not d["S0"][q][1]]
            same("%s, %s, B" % (m, label), len(inb), ref["fill"].get((label, m, "B")))
            t = tot.setdefault((label, "B"), [0])
            t[0] += len(inb)
            for f in FILLS:
                got = [d[f][q] for q in inb if d[f][q][1]]
                c = (len(got), sum(x[0] == 1 for x in got), sum(x[0] == -1 for x in got), sum(x[0] == 0 for x in got))
                same("%s, %s, %s inside B" % (m, label, f), c, ref["fill"].get((label, m, f)))
                t = tot.setdefault((label, f, "in B"), [0, 0, 0, 0])
                for i in range(4):
                    t[i] += c[i]
        print("%-12s recomputed: caps N1 %s/%s, SN1 %s/%s, Conly %s, SC %s | %d passes read, %d cells bound to their judged text" % (
            m, show(caps["N1"][0]), show(caps["N1"][1]), show(caps["SN1"][0]), show(caps["SN1"][1]), caps["Conly"], caps["SC"],
            len(M.rows), M.n_bound))
    lab = "%d models" % len(SET)
    for label, _ in groups:
        for sname in SYS:
            same("%s, %s, %s net (answers)" % (lab, label, sname), tuple(tot[(label, sname)]), ref["net"].get((label, lab, sname)))
        same("%s, %s, B" % (lab, label), tot[(label, "B")][0], ref["fill"].get((label, lab, "B")))
        for f in FILLS:
            same("%s, %s, %s inside B" % (lab, label, f), tuple(tot[(label, f, "in B")]), ref["fill"].get((label, lab, f)))
    for b in bad[:40]:
        print("  DIFFERENT: " + b)
    print("development recheck: %s (%d comparisons with the archived output, %d different)" % ("PASS" if not bad else "FAILED", n, len(bad)))
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
