#!/usr/bin/env python3
# Iteration_01 exploration (main session, Claude Fable 5.2, 2026-11-27; plan discussion/work_queue/2026-11-26_queue_o.md,
# after the GPT review 2026-11-26_gpt_gate_review.md section 6). Not reviewed by GPT. Read-only: existing answers,
# rule flags and DeepSeek Pro verdicts of the TruthfulQA TEST split (main-table settings, write rule "prompt").
# usage (container, from the project root): python3 -B Iteration_01/diag/n1_flow_rule_1127.py
# read-only: the flow of questions through N1 counted by the RULE flag (what the method itself sees: an answer is
# "blank" when the fixed text rule says so), next to the judge's verdicts of what is handed in.
# TruthfulQA test split, main settings, write rule "prompt".
import collections, sys
sys.path.insert(0, "Iteration_01/diag")
import iti_table as A
MP, ML, S, ps = A.MP, A.ML, A.S, A.ps
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
for model in ORDER:
    display, layer, s = A.KEY[model]
    test, err = MP.prompt_split(display, layer, 1, s)
    src = [x for x in ML.sources(display)[2] if x[0] == layer]
    cx = [q for q in test[3] if q in set(src[0][3].qs)]
    if cx != test[3]:
        test, err = MP.prompt_split(display, layer, 1, s, qs=cx)
    tune, _ = MP.prompt_split(display, layer, 0, s)
    pick = ML.n1_pick(tune[0], tune[1], s)
    sp, tx, singles, ids = test
    cE = sp.cE(s)
    b1 = {q: bool(sp.blank[(q, "baseline_x")]) for q in ids}
    b2 = {q: bool(sp.blank[(q, cE)]) for q in ids}
    v1 = {q: sp.st[(q, "baseline_x")] for q in ids}
    v2 = {q: sp.st[(q, cE)] for q in ids}
    n1v, n1ans, rec = S.system(sp, tx, pick)
    nv = dict(zip(ids, n1v)); r = dict(zip(ids, rec))
    notes = collections.Counter(str(r[q].get("note")) for q in ids if b2[q])
    own = [q for q in ids if b2[q] and b1[q]]; eb = [q for q in ids if b2[q] and not b1[q]]
    rel_own = [q for q in own if r[q].get("note") == "released"]; rel_eb = [q for q in eb if r[q].get("note") == "released"]
    def vd(qs, d):
        c = collections.Counter(d[q] for q in qs)
        return "right %d wrong %d hedge %d judged-no-answer %d" % (c["R"], c["W"], c["H"], c["X"])
    handed2 = [q for q in ids if not b2[q]]
    print("## %s (%d questions; caps E's blanks %s / own blanks %s)" % (model, len(ids), pick[2], pick[3]))
    print("   pass 1 by the rule: answered %d, blank %d | judge on the answered ones: %s" % (sum(not b1[q] for q in ids), sum(b1.values()), vd([q for q in ids if not b1[q]], v1)))
    print("   pass 2 by the rule: answered %d (handed in), blank %d = own blanks %d + E's blanks %d | judge on the handed-in ones: %s" % (
        len(handed2), sum(b2.values()), len(own), len(eb), vd(handed2, v2)))
    print("   pass 2 answered: pass 1 was blank by the rule in %d of them" % sum(1 for q in handed2 if b1[q]))
    print("   pass 3 notes on the %d blank questions: %s" % (sum(b2.values()), dict(notes)))
    print("   pass 3 released: own %d (%s) | E's %d (%s)" % (len(rel_own), vd(rel_own, nv), len(rel_eb), vd(rel_eb, nv)))
    final_ans = [q for q in ids if (not b2[q]) or r[q].get("note") == "released"]
    c = collections.Counter(nv[q] for q in ids)
    print("   final: handed in %d, not handed in %d | judge on all %d: right %d wrong %d hedge %d no-answer %d | net %+d" % (
        len(final_ans), len(ids) - len(final_ans), len(ids), c["R"], c["W"], c["H"], c["X"], c["R"] - c["W"]))
