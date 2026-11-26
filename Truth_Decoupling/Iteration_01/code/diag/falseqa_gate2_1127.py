#!/usr/bin/env python3
# Iteration_01 exploration (executor you, Claude Opus 5.5, 2026-11-27), work queue discussion/work_queue/2026-11-26_queue_p.md
# section 5, segment A: FalseQA held-out 600, behind the SAME frozen switch (stage 1, pool B, chosen layer, ON = p > 0.5),
# N1 at a larger E strength s2 against N1 at the original strength s. Not reviewed by GPT. CPU helper (container python3,
# from /workspace/Truth_Decoupling); it never loads a model and never calls a judge.
# Frozen (plan section 5): E layer / C layer / s / s2 in SET2 below (checked against falseqa_read.setting and
# gate_sim_1126.STRENGTH); the N1 caps and the Conly cap = falseqa_read.choose on the 100 development questions at the
# ORIGINAL strength; the switch decisions = newsets/gate_1126/<model>/fit_B_gate_1126.json (read through the audit's
# falseqa_matched.verify_gate, unchanged).
# New arms per model (gen dir newsets/gen_prompt/falseqa_gate2_1127/<model>/test, write rule "prompt", the generation
# program and settings of the existing FalseQA arms): E2 = x:E<el>.w.p<s2>; E2 + C rungs x:E<el>.w.p<s2>+C<cl>.w.m<c>
# (c = 0.25 0.5 1 2); TE2 = conf_E2:E<el>.w.p<s2> (must answer + honest sentence + E at s2).
# Cells that can be handed in (the only ones judged and self-judge scored), on held-out questions where the switch is ON:
#   E2's answer (non-empty text); where E2 is blank by the rule: the E2 + C rungs up to the cap of the question's class
#   (cap for own blanks when the original answer is blank, else the cap for E's blanks) and TE2's answer.
# Commands:
#   check-gen MODEL full|smoke          the six new files: contract arm, positions prompt, task, bank, items
#   judge-input MODEL full|smoke [EXPORT]  the judge input of the work dir (ON questions only; cells that cannot be
#                                       handed in get an empty output = no judge call); full: rows checked against the
#                                       unchanged ns_h22 export EXPORT of the same six arms
#   export-margins MODEL full|smoke     batch-1 self-judge item lists (sources fqa_gate2_1127[s]_*) of the new answered
#                                       cells that can be handed in, statements already scored left out
#   check-margins MODEL full|smoke      the 16 check statements scored twice in order and once shuffled agree exactly
#   smoke-ids MODEL                     writes newsets/gate2_1127/smoke_ids_gate2_1127.json (3 ids: ON answered by E at s,
#                                       OFF, ON with E(s) blank and filled by a rung) unless it exists; prints them
import collections
import glob
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks",
                "Iteration_01/gpt_tools_20261126/gate_audit"]
import falseqa_read as F          # noqa: E402
import gate_sim_1126 as GS        # noqa: E402

Q, G = F.Q, F.G
TAG = "gate2_1127"
SET2 = {"llama3.1-8b": (16, "3.5", 19, "5"), "qwen3.5-9b": (15, "2", 16, "3"), "qwen3.8-27b": (28, "1.75", 33, "3"),
        "gemma3-12b": (24, "2.5", 25, "4"), "gemma3-27b": (28, "1", 31, "2")}
ORDER = ("qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b", "llama3.1-8b")
LAD = F.LAD
NSD = F.NSD
JIN = "judge_input_%s.json" % TAG
SMOKE_IDS = NSD + "/gate2_1127/smoke_ids_gate2_1127.json"
SMOKE_LIST = NSD + "/gate2_1127/smoke_idlist_gate2_1127.json"
SRC = {"smoke": ("fqa_gate2_1127s_items", "fqa_gate2_1127s_chk", "fqa_gate2_1127s_chka", "fqa_gate2_1127s_chkb"),
       "full": ("fqa_gate2_1127_items", "fqa_gate2_1127_chk", "fqa_gate2_1127_chka", "fqa_gate2_1127_chkb")}


def setting2(m):
    el, s, cl, s2 = SET2[m]
    st = F.setting(m)
    if (st[0], st[1], st[2]) != (el, s, cl) or GS.STRENGTH[m] != (s, s2) or GS.EC[m] != (el, cl):
        raise SystemExit("%s: frozen setting differs from falseqa_read.setting / gate_sim_1126 (%s, %s)" % (m, st, GS.STRENGTH[m]))
    return el, s, cl, s2, st


def arms(m):
    el, s, cl, s2 = SET2[m]
    e2 = "x:E%d.w.p%s" % (el, s2)
    return {"E2": e2, "TE2": "conf_E2:E%d.w.p%s" % (el, s2), **{"EC2_" + k: "%s+C%d.w.m%s" % (e2, cl, k) for k in LAD}}


def gdir(m, mode):
    return Path(NSD) / "gen_prompt" / ("falseqa_%s%s" % (TAG, "_smoke" if mode == "smoke" else "")) / m / "test"


def work(m, mode):
    return Path(NSD) / "judge" / "falseqa" / m / ("test_%s_%s" % (TAG, "smoke" if mode == "smoke" else "heldout"))


def bank_rows():
    ok, rows = F.check_bank()
    if not ok:
        raise SystemExit("bank check FAILED")
    return rows


def smoke_ids():
    return json.load(open(SMOKE_IDS, encoding="utf-8"))["ids"]


def load_new(m, mode, rows):
    """{name: {id: row}} and {name: condition} of the six new files, checked against the bank and the plan."""
    ids = [r["id"] for r in rows]
    args = SimpleNamespace(task="falseqa", split="test", model=m, items=F.BANK)
    want = ids if mode == "full" else [q for q in ids if q in set(smoke_ids())]
    out, cond = {}, {}
    for name, arm in arms(m).items():
        tok = G.arm_token(arm)
        if mode == "full":
            p = gdir(m, mode) / (tok + ".json")
            g = G.load_generation(p, args, want)
        else:
            subs = sorted(glob.glob(str(gdir(m, mode) / ("s????????????.%s.json" % tok))))
            if len(subs) != 1:
                raise SystemExit("%s: %d smoke files for %s" % (m, len(subs), arm))
            p = Path(subs[0])
            g = G.load_generation(p, args, want, allow_subset=True)
        k = g["contract"]
        if k["arm"] != arm or k.get("positions") != "prompt" or k.get("max_new_tokens") != 48 or k.get("batch_size") != 8:
            raise SystemExit("%s: %s is not the pass %s at the prompt rule (48 tokens, batch 8)" % (m, p, arm))
        out[name] = {r["id"]: r for r in g["rows"]}
        cond[name] = k["condition"]
    return out, cond, want


def blank_row(r):
    return bool(r["rule_abstain"]) or not r["text"].strip()


def base_state(m, rows):
    """caps (dev, original strength), the held-out Raw rows R, R2 and the systems of falseqa_read, the ON list."""
    import falseqa_matched as FM
    el, s, cl, s2, st = setting2(m)
    dev = {r["id"] for r in rows if r["set"] == "dev"}
    held = {r["id"] for r in rows if r["set"] == "heldout"}
    tr, tr2 = F.load_rows(m, dev)
    caps, lines = F.choose(tr, tr2, st)
    R, R2 = F.load_rows(m, held)
    on = FM.verify_gate(m, "fqa_held", R.ids)
    return SimpleNamespace(st=st, caps=caps, cap_lines=lines, R=R, R2=R2, ids=list(R.ids), on=on, held=held)


def needed(m, B, new, qs):
    """{name: [ids]} of the new cells that can be handed in on the questions qs (ON questions)."""
    cap_e, cap_own = B.caps["N1"]
    out = collections.OrderedDict((n, []) for n in arms(m))
    for q in qs:
        e2 = new["E2"][q]
        if e2["text"].strip():
            out["E2"].append(q)
        if not blank_row(e2):
            continue
        own = B.R.blank("x", q)
        cap = cap_own if own else cap_e
        for k in (() if cap is None else LAD[:LAD.index(cap) + 1]):
            if new["EC2_" + k][q]["text"].strip():
                out["EC2_" + k].append(q)
        if new["TE2"][q]["text"].strip():
            out["TE2"].append(q)
    return out


def on_questions(m, mode, B):
    on = list(B.on)
    if mode == "smoke":
        keep = set(smoke_ids())
        on = [q for q in on if q in keep]
    return on


def export_rows(m, rows, new, cond, qs):
    """Rows of newsets_gen.run_export (unchanged layout) over the questions qs for the six new conditions."""
    bank = {r["id"]: r for r in rows}
    names = list(arms(m))
    out = []
    for q in qs:
        b = bank[q]
        out.append({"pair_id": "falseqa:%s" % q, "question": b["question"], "gold": list(b["gold"][:G.GOLD_CAP]),
                    "incorrect": [], "best_answer": b["gold"][0],
                    "outputs": {cond[n]: new[n][q]["text"] for n in names},
                    "finished": {cond[n]: bool(new[n][q]["finished"]) for n in names},
                    "rule_abstain": {cond[n]: bool(new[n][q]["rule_abstain"]) for n in names}})
    return out


def judge_input(m, mode, export_path=None):
    rows = bank_rows()
    B = base_state(m, rows)
    new, cond, _ = load_new(m, mode, rows)
    qs = on_questions(m, mode, B)
    need = needed(m, B, new, qs)
    full_rows = export_rows(m, rows, new, cond, qs)
    if export_path:
        ex = json.load(open(export_path, encoding="utf-8"))
        if ex.get("completed") is not True or sorted(ex.get("conditions", [])) != sorted(cond[n] for n in arms(m)) or ex.get("model") != m:
            raise SystemExit("%s: the export is not the six new arms of this model" % m)
        by = {e["pair_id"]: e for e in ex["evaluations"]}
        bad = [r["pair_id"] for r in full_rows if by.get(r["pair_id"]) != r]
        if bad:
            raise SystemExit("%s: %d rows differ from the unchanged export (%s)" % (m, len(bad), bad[:3]))
        print("%s: the %d ON rows equal the unchanged ns_h22 export %s" % (m, len(full_rows), export_path))
    keep = {(q, cond[n]) for n, l in need.items() for q in l}
    withheld = 0
    for r in full_rows:
        q = r["pair_id"].split(":", 1)[1]
        for c in list(r["outputs"]):
            if (q, c) not in keep and r["outputs"][c].strip():
                r["outputs"][c] = ""
                withheld += 1
    names = list(arms(m))
    first = load_contracts(m, mode)
    payload = {"completed": True, "model": m, "protocol": G.PROTOCOL, "export_protocol": G.PROTOCOL + "_export",
               "gen_version": G.GEN_VERSION, "split": "test", "task": "falseqa", "pro100_only": False, "gold_cap": G.GOLD_CAP,
               "identity_sha256": first[names[0]]["identity_sha256"], "conditions": [cond[n] for n in names],
               "condition_specs": [{"name": cond[n], "arm": first[n]["arm"], "parts": first[n]["parts"],
                                    "base_prompt": first[n]["base_prompt"], "kind": "intervention",
                                    "uses_reference_answer": False} for n in names],
               "question_ids": [r["pair_id"] for r in full_rows], "bank_sha256": F.sha_file(F.BANK), "evaluations": full_rows,
               TAG: {"mode": mode, "switch_on": len(qs), "caps": {k: v for k, v in B.caps.items()},
                     "needed": {cond[n]: l for n, l in need.items()}, "withheld_nonempty_cells": withheld,
                     "rule": "ON held-out questions; E2 answer; where E2 is blank by the rule: E2+C rungs up to the cap of "
                             "the class and TE2; every other cell has an empty output (no judge call)"}}
    payload = json.loads(json.dumps(payload))
    w = work(m, mode)
    out = w / JIN
    if out.exists():
        if json.load(open(out, encoding="utf-8")) != payload:
            raise SystemExit("%s exists and differs: stop" % out)
        print("%s exists and equals the build" % out)
        return
    w.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("%s %s: %d ON questions x %d conditions; cells that can be handed in %s; withheld non-empty %d -> %s" % (
        m, mode, len(qs), len(names), " ".join("%s %d" % (n, len(l)) for n, l in need.items()), withheld, out))


def load_contracts(m, mode):
    out = {}
    for name, arm in arms(m).items():
        tok = G.arm_token(arm)
        if mode == "full":
            p = gdir(m, mode) / (tok + ".json")
        else:
            p = Path(sorted(glob.glob(str(gdir(m, mode) / ("s????????????.%s.json" % tok))))[0])
        out[name] = json.loads(p.read_text(encoding="utf-8"))["contract"]
    return out


def check_gen(m, mode):
    rows = bank_rows()
    new, cond, want = load_new(m, mode, rows)
    k = load_contracts(m, mode)
    el, s, cl, s2, st = setting2(m)
    for n, c in k.items():
        parts = [(p["family"], p["layer"], p["kind"], float(p["beta"])) for p in c["parts"]]
        exp = [("E", el, "w", float(s2))] + ([("C", cl, "w", 1.0)] if n.startswith("EC2_") else [])
        if [x[:3] for x in parts] != [x[:3] for x in exp] or abs(parts[0][3] - float(s2)) > 1e-12:
            raise SystemExit("%s: %s parts %s, expected %s" % (m, n, parts, exp))
        base = "conf_E2" if n == "TE2" else "x"
        if c["base_prompt"] != base:
            raise SystemExit("%s: %s base prompt %s" % (m, n, c["base_prompt"]))
        rr = new[n]
        print("%s %s %s: %d rows, blank by the rule %d, unfinished %d, parts %s" % (
            m, mode, c["arm"], len(rr), sum(blank_row(r) for r in rr.values()), sum(not r["finished"] for r in rr.values()), parts))
    print("check-gen %s %s: PASS (%d questions)" % (m, mode, len(want)))


def items_of(m, mode):
    """[(name, id, text)] of the new answered cells that can be handed in."""
    rows = bank_rows()
    B = base_state(m, rows)
    new, cond, _ = load_new(m, mode, rows)
    qs = on_questions(m, mode, B)
    need = needed(m, B, new, qs)
    out = []
    for n, l in need.items():
        for q in l:
            if not blank_row(new[n][q]):
                out.append((n, q, new[n][q]["text"]))
    return rows, out


def sources_all():
    return SRC["smoke"] + SRC["full"]


def export_margins(m, mode):
    import numpy as np
    import selfjudge_probe as SP
    import selfjudge_stack_pick as SKP
    SP.SOURCES.update(F.SRC["smoke"] + F.SRC["full"] + sources_all())
    src = SRC[mode]
    root = os.path.join(SP.OUT, m)
    if os.path.exists(os.path.join(root, src[0] + "_items.json")):
        raise SystemExit("%s: %s already exported" % (m, src[0]))
    rows, cand = items_of(m, mode)
    bank = {r["id"]: r for r in rows}
    extra = (F.SRC["smoke"][0], F.SRC["full"][0]) + ((SRC["smoke"][0],) if mode == "full" else ())
    have = SKP.lookup(m, extra_sources=extra)
    items, seen, tally = [], set(), collections.Counter()
    for name, q, text in cand:
        h = F.statement_sha(bank[q]["question"], text)
        if h in have:
            tally["already scored"] += 1
            continue
        if h in seen:
            tally["same statement again"] += 1
            continue
        seen.add(h)
        tally[name] += 1
        items.append(dict(id=h, split="falseqa %s %s %s" % (bank[q]["set"], TAG, name), question=bank[q]["question"],
                          answer=text, answer_sha1=SP.sha1(text), value=None, answered=True, neg=bool(SP.NEG_RE.match(text))))
    print("%s %s: %d candidate answers | %s" % (m, mode, len(cand), " | ".join("%s %d" % kv for kv in sorted(tally.items()))))
    if not items:
        print("%s: nothing to score" % m)
        return
    SP.save_items(os.path.join(root, src[0] + "_items.json"), dict(model=m, source=src[0], items=items))
    k = min(16, len(items))
    pick = sorted(np.random.default_rng(20261122).choice(len(items), k, replace=False).tolist())
    chk = [items[j] for j in pick]
    perm = np.random.default_rng(20261123).permutation(k).tolist()
    for s_, xs in ((src[1], chk), (src[2], chk), (src[3], [chk[j] for j in perm])):
        SP.save_items(os.path.join(root, s_ + "_items.json"), dict(model=m, source=s_, items=xs))
    print("%s %s: %d new statements exported (%s)" % (m, mode, len(items), src[0]))


def check_margins(m, mode):
    import selfjudge_probe as SP
    SP.SOURCES.update(sources_all())
    src = SRC[mode]
    root = os.path.join(SP.OUT, m)
    if not os.path.exists(os.path.join(root, src[0] + "_items.json")):
        print("check-margins %s %s: no items exported (nothing to score)" % (m, mode))
        return
    maps = {}
    for s_ in src:
        _, meta, z = SP.read_probe(m, source=s_)
        if meta.get("batch_size") != 1:
            raise SystemExit("%s: %s is not batch 1" % (m, s_))
        maps[s_] = {str(h): float(z["baseline__margin"][j]) for j, h in enumerate(z["statement_sha1"])}
    a, b, c = (maps[s_] for s_ in src[1:])
    if set(a) != set(b) or set(a) != set(c) or not set(a) <= set(maps[src[0]]):
        raise SystemExit("%s: check sets differ" % m)
    d1 = max(abs(a[h] - b[h]) for h in a)
    d2 = max(abs(a[h] - c[h]) for h in a)
    d3 = max(abs(a[h] - maps[src[0]][h]) for h in a)
    v = sorted(maps[src[0]].values())
    print("margins %s %s: %d statements scored (batch 1); check %d statements: second pass max |diff| %.6f, shuffled %.6f, "
          "vs the main pass %.6f -> %s | margin min %.2f median %.2f max %.2f" % (
              m, mode, len(v), len(a), d1, d2, d3, "OK" if d1 == d2 == d3 == 0 else "MISMATCH", v[0], v[len(v) // 2], v[-1]))
    if not d1 == d2 == d3 == 0:
        raise SystemExit(1)


def write_id_list():
    """newsets_gen --only-ids reads a plain JSON list: SMOKE_LIST, written once from SMOKE_IDS."""
    p = Path(SMOKE_LIST)
    if p.exists():
        if json.load(open(p, encoding="utf-8")) != smoke_ids():
            raise SystemExit("%s differs from %s" % (SMOKE_LIST, SMOKE_IDS))
        return
    with open(p, "x", encoding="utf-8") as f:
        json.dump(smoke_ids(), f)


def make_smoke_ids(m):
    p = Path(SMOKE_IDS)
    if p.exists():
        write_id_list()
        print("smoke ids (existing file): %s; list file %s" % (smoke_ids(), SMOKE_LIST))
        return
    rows = bank_rows()
    B = base_state(m, rows)
    el, s, cl, s2, st = setting2(m)
    sysd = F.systems_of(B.R, B.R2, st, B.caps, B.ids)
    on = set(B.on)
    a = next(q for q in B.ids if q in on and sysd["E"][q][1])
    b = next(q for q in B.ids if q not in on)
    c = next(q for q in B.ids if q in on and not sysd["E"][q][1] and sysd["N1"][q][1] and sysd["N1"][q][3] not in (None, "0"))
    ids = [q for q in B.ids if q in (a, b, c)]
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "x", encoding="utf-8") as f:
        json.dump({"model": m, "ids": ids, "roles": {a: "ON, E(s) answers", b: "OFF", c: "ON, E(s) blank, filled by a rung at s"}}, f, indent=1)
    write_id_list()
    print("smoke ids written: %s (ON answered %s, OFF %s, ON E(s) blank + rung %s); list file %s" % (ids, a, b, c, SMOKE_LIST))


def main():
    a = sys.argv[1:]
    if len(a) >= 2 and a[1] not in SET2:
        raise SystemExit("unknown model %s" % a[1])
    if len(a) >= 3 and a[2] not in ("full", "smoke"):
        raise SystemExit("mode full or smoke")
    if a[:1] == ["check-gen"] and len(a) == 3:
        check_gen(a[1], a[2])
    elif a[:1] == ["judge-input"] and len(a) in (3, 4):
        judge_input(a[1], a[2], a[3] if len(a) == 4 else None)
    elif a[:1] == ["export-margins"] and len(a) == 3:
        export_margins(a[1], a[2])
    elif a[:1] == ["check-margins"] and len(a) == 3:
        check_margins(a[1], a[2])
    elif a[:1] == ["smoke-ids"] and len(a) == 2:
        make_smoke_ids(a[1])
    else:
        raise SystemExit("usage: falseqa_gate2_1127.py check-gen|judge-input|export-margins|check-margins MODEL full|smoke [EXPORT] | smoke-ids MODEL")


if __name__ == "__main__":
    main()
