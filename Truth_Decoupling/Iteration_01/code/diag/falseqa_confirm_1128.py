#!/usr/bin/env python3
# Iteration_01 exploration (executor hai, Claude Opus 5.5, 2026-11-28), work queue discussion/work_queue/2026-11-28_queue_q.md:
# the frozen simple rule on the FalseQA confirmation bank (the 386 test.csv pairs outside the held-out 300). Not reviewed
# by GPT. CPU helper (container python3, from /workspace/Truth_Decoupling); it never loads a model and never calls a judge.
# Frozen per model (plan section 1): E layer el, E strength s, C layer cl, c (the C strength of the fill), cC (single C).
# Arms (gen dir newsets/gen_prompt/falseqa_confirm_1128/<model>/test; write rule "prompt"; the generation program and
# settings of the held-out files): O = x | E = x:E<el>.w.p<s> | C = x:C<cl>.w.m<cC> | EC_k = x:E<el>.w.p<s>+C<cl>.w.m<k>,
# k = 0.25 0.5 1 2. All arm names are arms of the held-out run (same names, same writes).
# Answered = the generation row's rule flag false and non-empty text. Cells judged (all others get an empty output, no
# judge call): O, E, C where answered; EC_k for k <= c where answered, only on questions where O and E are both blank.
# Commands:
#   check-gen MODEL full|smoke                 the eight files: arm, write rule prompt (x: none), 48 tokens, batch 8, parts
#   judge-input MODEL q100|all|smoke [EXPORT]  the judge input judge_input_confirm_1128.json of the stage's work dir
#                                              newsets/judge/falseqa/<model>/test_confirm_1128_<stage>; full files: rows
#                                              checked against the unchanged ns_h22 export EXPORT of the eight arms
#   smoke-ids                                  writes newsets/falseqa/falseqa_confirm_1128_smoke_ids.json (3 ids) unless
#                                              it exists; prints them
#   stage-ids                                  prints the stage-1 ids count and the bank checks
#   setting MODEL                              prints "el s cl c cC" (checked against falseqa_read.setting)
import collections
import glob
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks"]
import falseqa_read as F          # noqa: E402

G = F.G
TAG = "confirm_1128"
NSD = F.NSD
BANK = NSD + "/falseqa/falseqa_confirm_bank_20261128.jsonl"
IDS = NSD + "/falseqa/falseqa_confirm_1128_ids_20261128.json"
SMOKE_IDS = NSD + "/falseqa/falseqa_confirm_1128_smoke_ids.json"
JIN = "judge_input_%s.json" % TAG
KS = ("0.25", "0.5", "1", "2")
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
# model: (E layer, E strength, C layer, c, single-C strength cC)   plan section 1 table
SETQ = {"llama3.1-8b": (16, "3.5", 19, "2", "2"), "qwen3.5-9b": (15, "2", 16, "0.5", "0.5"),
        "qwen3.8-27b": (28, "1.75", 33, "1", "0.25"), "gemma3-12b": (24, "2.5", 25, "2", "2"),
        "gemma3-27b": (28, "1", 31, "2", "0.25")}


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def setting(m):
    el, s, cl, c, cc = SETQ[m]
    st = F.setting(m)
    if (st[0], st[1], st[2]) != (el, s, cl):
        raise SystemExit("%s: plan setting E%s p%s C%s differs from falseqa_read.setting %s" % (m, el, s, cl, st))
    return el, s, cl, c, cc


def arms(m):
    el, s, cl, c, cc = setting(m)
    e = "x:E%d.w.p%s" % (el, s)
    out = collections.OrderedDict([("O", "x"), ("E", e), ("C", "x:C%d.w.m%s" % (cl, cc))])
    for k in KS:
        out["EC" + k] = "%s+C%d.w.m%s" % (e, cl, k)
    return out


def gdir(m, mode):
    return Path(NSD) / "gen_prompt" / ("falseqa_%s%s" % (TAG, "_smoke" if mode == "smoke" else "")) / m / "test"


def work(m, stage):
    return Path(NSD) / "judge" / "falseqa" / m / ("test_%s_%s" % (TAG, stage))


def load_bank():
    with open(BANK, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    idf = json.load(open(IDS, encoding="utf-8"))
    bad = []
    if idf["bank_sha256"] != sha_file(BANK):
        bad.append("the id file records another bank")
    if len(rows) != idf["n_questions"] or len({r["pair_id"] for r in rows}) != idf["n_pairs"]:
        bad.append("counts differ from the id file")
    if any(r.get("set") != "confirm" or r.get("split") != "test" or r.get("pro100") is not True for r in rows):
        bad.append("a row not of set confirm / split test / pro100")
    per = collections.defaultdict(list)
    for r in rows:
        per[r["pair_id"]].append(r["premise"])
        want = {"false": "f", "valid": "v"}.get(r["premise"])
        if want is None or r["id"] != r["pair_id"] + want:
            bad.append("row %s: id / premise" % r["id"])
    if any(sorted(v) != ["false", "valid"] for v in per.values()):
        bad.append("a pair without both rows")
    s1 = idf["stage1"]["ids"]
    if len(s1) != 100 or not set(s1) <= {r["id"] for r in rows}:
        bad.append("stage-1 ids")
    if bad:
        raise SystemExit("confirm bank check FAILED: " + "; ".join(bad[:10]))
    return rows, idf


def smoke_ids():
    return json.load(open(SMOKE_IDS, encoding="utf-8"))


def stage_ids(stage, rows, idf):
    ids = [r["id"] for r in rows]
    if stage == "q100":
        keep = set(idf["stage1"]["ids"])
    elif stage == "smoke":
        keep = set(smoke_ids())
    else:
        keep = set(ids)
    return [q for q in ids if q in keep]


def gen_args(m):
    return SimpleNamespace(task="falseqa", split="test", model=m, items=BANK)


def gen_file(m, mode, arm):
    tok = G.arm_token(arm)
    if mode == "full":
        return gdir(m, mode) / (tok + ".json")
    subs = sorted(glob.glob(str(gdir(m, mode) / ("s????????????.%s.json" % tok))))
    if len(subs) != 1:
        raise SystemExit("%s: %d smoke files for %s" % (m, len(subs), arm))
    return Path(subs[0])


def load_gen(m, mode, rows):
    """{name: {id: row}}, {name: condition}, {name: contract}, the ids covered; every file checked against the bank."""
    ids = [r["id"] for r in rows]
    want = ids if mode == "full" else [q for q in ids if q in set(smoke_ids())]
    out, cond, con = {}, {}, {}
    for name, arm in arms(m).items():
        p = gen_file(m, mode, arm)
        g = G.load_generation(p, gen_args(m), want, allow_subset=(mode != "full"))
        k = g["contract"]
        pos = "prompt" if ":" in arm else None
        if k["arm"] != arm or k.get("positions") != pos or k.get("max_new_tokens") != 48 or k.get("batch_size") != 8:
            raise SystemExit("%s: %s is not the pass %s at write rule %s (48 tokens, batch 8)" % (m, p, arm, pos))
        if [r["id"] for r in g["rows"]] != want:
            raise SystemExit("%s: %s does not cover the questions in bank order" % (m, p))
        out[name] = {r["id"]: r for r in g["rows"]}
        cond[name] = k["condition"]
        con[name] = k
    return out, cond, con, want


def answered(r):
    return (not r["rule_abstain"]) and bool(r["text"].strip())


def needed(m, gen, qs):
    """{name: [ids]} of the cells to judge on the questions qs (plan section 2.3)."""
    el, s, cl, c, cc = setting(m)
    out = collections.OrderedDict((n, []) for n in arms(m))
    for q in qs:
        for n in ("O", "E", "C"):
            if answered(gen[n][q]):
                out[n].append(q)
        if answered(gen["O"][q]) or answered(gen["E"][q]):
            continue
        for k in KS[:KS.index(c) + 1]:
            if answered(gen["EC" + k][q]):
                out["EC" + k].append(q)
    return out


def export_rows(m, rows, gen, cond, qs):
    """Rows of newsets_gen.run_export (unchanged layout, as falseqa_gate2_1127.export_rows) over the questions qs."""
    bank = {r["id"]: r for r in rows}
    names = list(arms(m))
    out = []
    for q in qs:
        b = bank[q]
        out.append({"pair_id": "falseqa:%s" % q, "question": b["question"], "gold": list(b["gold"][:G.GOLD_CAP]),
                    "incorrect": [], "best_answer": b["gold"][0],
                    "outputs": {cond[n]: gen[n][q]["text"] for n in names},
                    "finished": {cond[n]: bool(gen[n][q]["finished"]) for n in names},
                    "rule_abstain": {cond[n]: bool(gen[n][q]["rule_abstain"]) for n in names}})
    return out


def judge_input(m, stage, export_path=None):
    rows, idf = load_bank()
    mode = "smoke" if stage == "smoke" else "full"
    gen, cond, con, _ = load_gen(m, mode, rows)
    qs = stage_ids(stage, rows, idf)
    need = needed(m, gen, qs)
    full_rows = export_rows(m, rows, gen, cond, qs)
    if export_path:
        ex = json.load(open(export_path, encoding="utf-8"))
        if (ex.get("completed") is not True or sorted(ex.get("conditions", [])) != sorted(cond.values()) or ex.get("model") != m
                or ex.get("bank_sha256") != sha_file(BANK)):
            raise SystemExit("%s: the export is not the eight arms of this model on the confirm bank" % m)
        by = {e["pair_id"]: e for e in ex["evaluations"]}
        bad = [r["pair_id"] for r in full_rows if by.get(r["pair_id"]) != r]
        if bad:
            raise SystemExit("%s: %d rows differ from the unchanged export (%s)" % (m, len(bad), bad[:3]))
        print("%s: the %d rows equal the unchanged ns_h22 export %s" % (m, len(full_rows), export_path))
    elif mode == "full":
        raise SystemExit("full files need the export to check against")
    keep = {(q, cond[n]) for n, l in need.items() for q in l}
    withheld = 0
    for r in full_rows:
        q = r["pair_id"].split(":", 1)[1]
        for cnd in list(r["outputs"]):
            if (q, cnd) not in keep and r["outputs"][cnd].strip():
                r["outputs"][cnd] = ""
                withheld += 1
    names = list(arms(m))
    payload = {"completed": True, "model": m, "protocol": G.PROTOCOL, "export_protocol": G.PROTOCOL + "_export",
               "gen_version": G.GEN_VERSION, "split": "test", "task": "falseqa", "pro100_only": False, "gold_cap": G.GOLD_CAP,
               "identity_sha256": con[names[0]]["identity_sha256"], "conditions": [cond[n] for n in names],
               "condition_specs": [{"name": cond[n], "arm": con[n]["arm"], "parts": con[n]["parts"],
                                    "base_prompt": con[n]["base_prompt"], "kind": "intervention" if ":" in con[n]["arm"] else "baseline",
                                    "uses_reference_answer": False} for n in names],
               "question_ids": [r["pair_id"] for r in full_rows], "bank_sha256": sha_file(BANK), "evaluations": full_rows,
               TAG: {"stage": stage, "questions": len(qs), "setting": list(SETQ[m]),
                     "needed": {cond[n]: l for n, l in need.items()}, "withheld_nonempty_cells": withheld,
                     "rule": "answered = rule flag false and non-empty text; O, E, C answered cells; EC_k (k <= c) answered "
                             "cells on questions where O and E are both blank; every other cell has an empty output (no judge call)"}}
    if export_path:     # full files: everything of the unchanged export kept, as falseqa_judge_split.py does
        header = {k: v for k, v in ex.items() if k not in ("evaluations", "question_ids")}
        payload = dict(header, evaluations=full_rows, question_ids=[r["pair_id"] for r in full_rows],
                       subset={"of": Path(export_path).name, "of_sha256": sha_file(export_path), "n_questions": len(full_rows)},
                       **{TAG: payload[TAG]})
    payload = json.loads(json.dumps(payload))
    w = work(m, stage)
    out = w / JIN
    if out.exists():
        if json.load(open(out, encoding="utf-8")) != payload:
            raise SystemExit("%s exists and differs: stop" % out)
        print("%s exists and equals the build" % out)
        return
    w.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("%s %s: %d questions x %d conditions; cells to judge %s; withheld non-empty %d -> %s" % (
        m, stage, len(qs), len(names), " ".join("%s %d" % (n, len(l)) for n, l in need.items()), withheld, out))


def check_gen(m, mode):
    rows, _ = load_bank()
    gen, cond, con, want = load_gen(m, mode, rows)
    el, s, cl, c, cc = setting(m)
    for n, k in con.items():
        parts = [(p["family"], p["layer"], p["kind"], float(p["beta"])) for p in (k.get("parts") or [])]
        # beta of a C write ".w.m<k>" is -k (seen in the archived files: n1s_check parts, amount = beta x unit amount)
        if n.startswith("EC"):
            exp = [("E", el, "w", float(s)), ("C", cl, "w", -float(n[2:]))]
        else:
            exp = {"O": [], "E": [("E", el, "w", float(s))], "C": [("C", cl, "w", -float(cc))]}[n]
        if parts != exp:
            raise SystemExit("%s: %s parts %s, expected %s" % (m, n, parts, exp))
        if k["base_prompt"] != "x":
            raise SystemExit("%s: %s base prompt %s" % (m, n, k["base_prompt"]))
        rr = gen[n]
        print("%s %s %s: %d rows, answered %d, blank by the rule %d, unfinished %d, parts %s" % (
            m, mode, k["arm"], len(rr), sum(answered(r) for r in rr.values()), sum(not answered(r) for r in rr.values()),
            sum(not r["finished"] for r in rr.values()), parts))
    print("check-gen %s %s: PASS (%d questions)" % (m, mode, len(want)))


def make_smoke_ids():
    p = Path(SMOKE_IDS)
    if p.exists():
        print("smoke ids (existing file): %s" % smoke_ids())
        return
    rows, idf = load_bank()
    first = rows[0]["pair_id"]
    ids = [r["id"] for r in rows if r["pair_id"] == first] + [rows[1]["id"]]
    ids = [r["id"] for r in rows if r["id"] in set(ids)]
    with open(p, "x", encoding="utf-8") as f:
        json.dump(ids, f)
    print("smoke ids written: %s -> %s" % (ids, p))


def main():
    a = sys.argv[1:]
    if len(a) >= 2 and a[0] != "smoke-ids" and a[1] not in SETQ:
        raise SystemExit("unknown model %s" % a[1])
    if a[:1] == ["check-gen"] and len(a) == 3 and a[2] in ("full", "smoke"):
        check_gen(a[1], a[2])
    elif a[:1] == ["judge-input"] and len(a) in (3, 4) and a[2] in ("q100", "all", "smoke"):
        judge_input(a[1], a[2], a[3] if len(a) == 4 else None)
    elif a[:1] == ["setting"] and len(a) == 2:
        print("%d %s %d %s %s" % setting(a[1]))
    elif a == ["smoke-ids"]:
        make_smoke_ids()
    elif a == ["stage-ids"]:
        rows, idf = load_bank()
        print("confirm bank %d rows, %d pairs; stage 1 %d ids; sha256 %s" % (len(rows), idf["n_pairs"], len(idf["stage1"]["ids"]),
                                                                           idf["bank_sha256"][:16]))
    else:
        raise SystemExit("usage: falseqa_confirm_1128.py check-gen MODEL full|smoke | judge-input MODEL q100|all|smoke [EXPORT] | "
                         "smoke-ids | stage-ids | setting MODEL")


if __name__ == "__main__":
    main()
