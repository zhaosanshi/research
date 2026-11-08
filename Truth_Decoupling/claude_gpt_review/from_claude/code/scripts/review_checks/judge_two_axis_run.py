#!/usr/bin/env python3
"""Two-axis grading of a generation file with three judges and 赵磊 (rules: core/judge_two_axis.py).

usage (RUN_DIR holds the generation file, e.g. an e42 tune.json; everything is written to RUN_DIR/judge2/):
  judge_two_axis_run.py cells      RUN_DIR [--generation tune.json] [--audit-n 50 --audit-seed 0]
  (GPU)  scripts/allenai_tqa_judge.py --cells RUN_DIR/judge2/cells.jsonl --out RUN_DIR/judge2/allenai.jsonl ...
  judge_two_axis_run.py llm        RUN_DIR --judge pro                   DeepSeek V4 Pro, every cell
  judge_two_axis_run.py disputes   RUN_DIR                               cells AllenAI + Pro did not settle, + audit
  judge_two_axis_run.py llm        RUN_DIR --judge gpt --only disputes   GPT-5.5 on those cells only
  judge_two_axis_run.py adjudicate RUN_DIR                               final.jsonl, human_queue.jsonl, summary
  judge_two_axis_run.py human      RUN_DIR list
  judge_two_axis_run.py human      RUN_DIR set CELL_ID LETTER --by WHO [--note ..]   then re-run adjudicate

Files in RUN_DIR/judge2/: cells.jsonl (one row per question x condition), allenai.jsonl, pro.jsonl, gpt.jsonl,
disputes.jsonl, final.jsonl, human_queue.jsonl, human_verdicts.jsonl, summary.json / summary.md, meta.json,
cache/judge2_cache_<model>.jsonl (API replies). API keys come from DEEPSEEK_API_KEY_FILE / XIAOAI_API_KEY_FILE.
Config: configs/judges/two_axis_v5.json (override with --config or TD_JUDGE2_CONFIG).
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, OrderedDict

CODE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, CODE_ROOT)
from core.io_utils import read_jsonl, write_jsonl, append_jsonl, atomic_json_dump  # noqa: E402
from core.judge_two_axis import (TWO_AXIS_VERSION, LABELS, LABEL_NAMES, TRUTH_VALUES, INFO_VALUES, TwoAxisLLMJudge,  # noqa: E402
                                 adjudicate_cell, allenai_votes, info_binary, judge_own_label, coarse)
from core.qa_scoring import reference_answers  # noqa: E402

J2 = "judge2"
JUDGES = ("allenai", "pro", "gpt")


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def j2dir(run_dir):
    p = os.path.join(run_dir, J2)
    os.makedirs(p, exist_ok=True)
    return p


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_cfg(path=None):
    path = path or os.environ.get("TD_JUDGE2_CONFIG") or os.path.join(CODE_ROOT, "configs", "judges", "two_axis_v5.json")
    cfg = json.load(open(path, encoding="utf-8"))
    if cfg.get("version") != TWO_AXIS_VERSION:
        raise SystemExit("config version %r does not match the code's rubric %r" % (cfg.get("version"), TWO_AXIS_VERSION))
    return cfg


def _rows(path):
    return OrderedDict((r["cell_id"], r) for r in read_jsonl(path)) if os.path.exists(path) else OrderedDict()


def _cells(run_dir):
    path = os.path.join(j2dir(run_dir), "cells.jsonl")
    if not os.path.exists(path):
        raise SystemExit("no cells.jsonl: run `cells` first")
    return list(read_jsonl(path))


def _human_verdicts(run_dir):
    """Last verdict per cell wins (append-only file)."""
    out = {}
    path = os.path.join(j2dir(run_dir), "human_verdicts.jsonl")
    if os.path.exists(path):
        for r in read_jsonl(path):
            out[r["cell_id"]] = r
    return out


# ----------------------------------------------------------------------------- cells
def build_cells(run_dir, generation="tune.json", audit_n=50, audit_seed=0):
    gen_path = os.path.join(run_dir, generation)
    data = json.load(open(gen_path, encoding="utf-8"))
    if not data.get("completed"):
        raise SystemExit("generation incomplete: %s" % gen_path)
    conds = list(data["conditions"])
    if len(set(conds)) != len(conds):
        raise SystemExit("duplicate condition names")
    cells = []
    for row in data["evaluations"]:
        refs, only_refusal = reference_answers(row["gold"])
        for c in conds:
            pred = row["outputs"][c]
            if not isinstance(pred, str):
                raise SystemExit("non-string output for %s/%s" % (row["pair_id"], c))
            cells.append({"cell_id": "%s|%s" % (row["pair_id"], c), "pair_id": row["pair_id"], "condition": c,
                          "question": row["question"], "gold": list(row["gold"]), "gold_refs": refs,
                          "gold_is_abstention": only_refusal, "incorrect": list(row.get("incorrect") or []),
                          "best_answer": row.get("best_answer"), "prediction": pred,
                          "rule_abstain": bool((row.get("rule_abstain") or {}).get(c, False)), "audit": False})
    ids = [c["cell_id"] for c in cells]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate cell ids (duplicate pair_id?)")
    pool = sorted(c["cell_id"] for c in cells if not c["gold_is_abstention"] and c["prediction"].strip())
    audit = set(random.Random(audit_seed).sample(pool, min(audit_n, len(pool))))
    for c in cells:
        c["audit"] = c["cell_id"] in audit
    out = j2dir(run_dir)
    write_jsonl(cells, os.path.join(out, "cells.jsonl"))
    meta = {"version": TWO_AXIS_VERSION, "generation": generation, "generation_sha256": sha256_file(gen_path),
            "model": data.get("model"), "conditions": conds, "n_questions": len(data["evaluations"]), "n_cells": len(cells),
            "n_unique_question_answer": len({(c["question"], c["prediction"]) for c in cells}),
            "n_empty_outputs": sum(not c["prediction"].strip() for c in cells),
            "n_abstention_only_gold_questions": sum(1 for r in data["evaluations"] if reference_answers(r["gold"])[1]),
            "audit": {"n": len(audit), "seed": audit_seed, "cell_ids": sorted(audit)}, "ts": _now()}
    atomic_json_dump(meta, os.path.join(out, "meta.json"))
    return meta


# ----------------------------------------------------------------------------- LLM judges
def run_llm(run_dir, judge_name, cfg, only=None, judge=None, log_every=50):
    """Grade cells with one LLM judge (two questions each). Resumable: cells already holding two valid letters under
    the current rubric are kept; a D is valid. `only='disputes'` restricts to disputes.jsonl plus the audit sample."""
    out = j2dir(run_dir)
    cells = _cells(run_dir)
    jcfg = cfg[judge_name]
    judge = judge or TwoAxisLLMJudge(jcfg, cache_dir=os.path.join(out, "cache"), name=judge_name)
    target = cells
    if only == "disputes":
        dpath = os.path.join(out, "disputes.jsonl")
        if not os.path.exists(dpath):
            raise SystemExit("no disputes.jsonl: run `disputes` first")
        disp = {r["cell_id"] for r in read_jsonl(dpath)}
        target = [c for c in cells if c["cell_id"] in disp or c["audit"]]
    path = os.path.join(out, judge_name + ".jsonl")
    done = _rows(path)
    n_new, t0 = 0, time.time()
    for i, c in enumerate(target):
        prev = done.get(c["cell_id"])
        if (prev and prev.get("rubric") == TWO_AXIS_VERSION and prev.get("truth") in ("T", "F", "D")
                and prev.get("info") in ("S", "H", "R", "D")):
            continue
        if not c["prediction"].strip():
            done[c["cell_id"]] = {"cell_id": c["cell_id"], "judge": judge_name, "model": jcfg["model"], "rubric": TWO_AXIS_VERSION,
                                  "truth_raw": "<empty_output>", "truth": "T", "info_raw": "<empty_output>", "info": "R",
                                  "empty_output": True}
            continue
        g = judge.grade(c["question"], c["gold_refs"], c["prediction"], c["incorrect"] or None)
        g["cell_id"] = c["cell_id"]
        done[c["cell_id"]] = g
        n_new += 1
        if n_new % log_every == 0:
            write_jsonl(list(done.values()), path)
            print("[%s] %s %d/%d graded, %d new calls, %d cache hits (%.0fs)" % (
                time.strftime("%H:%M:%S"), judge_name, i + 1, len(target), judge.llm.n_calls, judge.llm.n_cache_hits,
                time.time() - t0), flush=True)
    write_jsonl(list(done.values()), path)
    tr = Counter(done[c["cell_id"]].get("truth") for c in target if c["cell_id"] in done)
    ir = Counter(done[c["cell_id"]].get("info") for c in target if c["cell_id"] in done)
    summary = {"judge": judge_name, "model": jcfg["model"], "n_target": len(target), "n_graded": sum(c["cell_id"] in done for c in target),
               "truth_letters": dict(tr), "info_letters": dict(ir), "api_calls": judge.llm.n_calls,
               "cache_hits": judge.llm.n_cache_hits, "seconds": round(time.time() - t0, 1)}
    return summary


# ----------------------------------------------------------------------------- disputes / adjudication
def _require(run_dir, names):
    out = j2dir(run_dir)
    got = {}
    for n in names:
        rows = _rows(os.path.join(out, n + ".jsonl"))
        if not rows:
            raise SystemExit("missing or empty %s.jsonl" % n)
        got[n] = rows
    return got


def compute_disputes(run_dir, cfg):
    cells = _cells(run_dir)
    j = _require(run_dir, ("allenai", "pro"))
    thr = cfg["allenai"]["threshold"]
    missing = [c["cell_id"] for c in cells if c["cell_id"] not in j["allenai"] or c["cell_id"] not in j["pro"]]
    if missing:
        raise SystemExit("%d cells lack an AllenAI or Pro grade, e.g. %s" % (len(missing), missing[:3]))
    disputes, stats = [], Counter()
    for c in cells:
        rec = adjudicate_cell(c, allenai=j["allenai"][c["cell_id"]], pro=j["pro"][c["cell_id"]], gpt=None, threshold=thr)
        stats["truth_" + rec["truth_basis"]] += 1
        stats["info_" + rec["info_basis"]] += 1
        stats["status_" + rec["status"]] += 1
        if rec["status"] == "needs_gpt" or c["audit"]:
            disputes.append({"cell_id": c["cell_id"], "reason": "disagreement" if rec["status"] == "needs_gpt" else "audit",
                             "audit": c["audit"], "votes_truth": rec["votes_truth"], "votes_info": rec["votes_info"],
                             "info_subtypes": rec["info_subtypes"]})
    write_jsonl(disputes, os.path.join(j2dir(run_dir), "disputes.jsonl"))
    summary = {"n_cells": len(cells), "n_to_gpt": len(disputes),
               "n_disagreement": sum(d["reason"] == "disagreement" for d in disputes),
               "n_audit_only": sum(d["reason"] == "audit" for d in disputes), **dict(stats)}
    return summary


def _kappa(pairs):
    """Cohen's kappa for a list of (a, b) label pairs (any number of categories)."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def _agreement(cells, judges, humans, thr):
    """Pairwise judge agreement per axis (cells where both voted) and judge / pipeline agreement with 赵磊 (cells with
    a human verdict), each as {n, agreement, kappa}."""
    def votes(name, cid):
        r = judges.get(name, {}).get(cid)
        if r is None:
            return None, None, None
        if name == "allenai":
            t, i = allenai_votes(r, thr)
            return t, i, None
        t = r.get("truth") if r.get("truth") in TRUTH_VALUES else None
        return t, info_binary(r.get("info")), (r.get("info") if r.get("info") in ("S", "H", "R") else None)
    out = {"pairwise": {}, "vs_human": {}}
    names = [n for n in JUDGES if judges.get(n)]
    for a_i in range(len(names)):
        for b_i in range(a_i + 1, len(names)):
            a, b = names[a_i], names[b_i]
            pt, pi = [], []
            for c in cells:
                ta, ia, _ = votes(a, c["cell_id"])
                tb, ib, _ = votes(b, c["cell_id"])
                if ta and tb:
                    pt.append((ta, tb))
                if ia and ib:
                    pi.append((ia, ib))
            out["pairwise"]["%s_vs_%s" % (a, b)] = {
                "truth": {"n": len(pt), "agreement": sum(x == y for x, y in pt) / len(pt) if pt else None, "kappa": _kappa(pt)},
                "info": {"n": len(pi), "agreement": sum(x == y for x, y in pi) / len(pi) if pi else None, "kappa": _kappa(pi)}}
    if humans:
        hcells = [c for c in cells if c["cell_id"] in humans]
        for name in names:
            pairs = []
            for c in hcells:
                t, i, sub = votes(name, c["cell_id"])
                own = judge_own_label(t, sub if name != "allenai" else i)
                if own is not None:
                    pairs.append((coarse(own) if name == "allenai" else own,
                                  coarse(humans[c["cell_id"]]["label"]) if name == "allenai" else humans[c["cell_id"]]["label"]))
            out["vs_human"][name] = {"n": len(pairs), "agreement": sum(x == y for x, y in pairs) / len(pairs) if pairs else None,
                                     "kappa": _kappa(pairs), "coarse_labels_for_allenai": name == "allenai"}
        pairs = []
        for c in hcells:
            rec = adjudicate_cell(c, allenai=judges.get("allenai", {}).get(c["cell_id"]), pro=judges.get("pro", {}).get(c["cell_id"]),
                                  gpt=judges.get("gpt", {}).get(c["cell_id"]), human=None, threshold=thr)
            if rec["label"] is not None:
                pairs.append((rec["label"], humans[c["cell_id"]]["label"]))
        out["vs_human"]["pipeline_without_human"] = {"n": len(pairs), "agreement": sum(x == y for x, y in pairs) / len(pairs) if pairs else None,
                                                     "kappa": _kappa(pairs), "n_human_cells": len(hcells)}
    return out


def adjudicate(run_dir, cfg):
    out = j2dir(run_dir)
    cells = _cells(run_dir)
    meta = json.load(open(os.path.join(out, "meta.json"), encoding="utf-8"))
    judges = {n: _rows(os.path.join(out, n + ".jsonl")) for n in JUDGES}
    if not judges["allenai"] or not judges["pro"]:
        raise SystemExit("adjudication needs allenai.jsonl and pro.jsonl")
    humans = _human_verdicts(run_dir)
    thr = cfg["allenai"]["threshold"]
    finals, queue = [], []
    for c in cells:
        cid = c["cell_id"]
        rec = adjudicate_cell(c, allenai=judges["allenai"].get(cid), pro=judges["pro"].get(cid), gpt=judges["gpt"].get(cid),
                              human=humans.get(cid), threshold=thr)
        rec.update(pair_id=c["pair_id"], condition=c["condition"], audit=c["audit"], gold_is_abstention=c["gold_is_abstention"])
        finals.append(rec)
        needs = rec["status"] in ("needs_gpt", "needs_human")
        if (needs or c["audit"]) and cid not in humans:
            queue.append({"cell_id": cid, "reason": "audit" if not needs else rec["status"], "pair_id": c["pair_id"],
                          "condition": c["condition"], "question": c["question"], "gold": c["gold"], "incorrect": c["incorrect"],
                          "best_answer": c.get("best_answer"), "prediction": c["prediction"],
                          "votes_truth": rec["votes_truth"], "votes_info": rec["votes_info"], "info_subtypes": rec["info_subtypes"],
                          "judges_label": rec["judges_label"]})
    write_jsonl(finals, os.path.join(out, "final.jsonl"))
    write_jsonl(queue, os.path.join(out, "human_queue.jsonl"))
    # summary: per condition label counts over cells with substantive gold (the analysis denominator)
    per_cond = OrderedDict()
    for cond in meta["conditions"]:
        rs = [r for r in finals if r["condition"] == cond and not r["gold_is_abstention"]]
        cnt = Counter(r["label"] for r in rs)
        n = len(rs)
        per_cond[cond] = {"n": n, **{LABEL_NAMES[k] + "_" + k: cnt.get(k, 0) for k in LABELS}, "unresolved": cnt.get(None, 0),
                          "true_rate": (cnt.get("A", 0) + cnt.get("C", 0) + cnt.get("H", 0)) / n if n else None,
                          "info_rate": (cnt.get("A", 0) + cnt.get("B", 0)) / n if n else None,
                          "true_info_rate": cnt.get("A", 0) / n if n else None}
    src = Counter(r["source"] for r in finals)
    status = Counter(r["status"] for r in finals)
    summary = {"version": TWO_AXIS_VERSION, "model": meta.get("model"), "generation_sha256": meta["generation_sha256"],
               "n_cells": len(cells), "n_final": status.get("final", 0), "n_needs_gpt": status.get("needs_gpt", 0),
               "n_needs_human": status.get("needs_human", 0), "n_human_queue": len(queue),
               "n_human_verdicts": len(humans), "judges_present": [n for n in JUDGES if judges[n]],
               "label_sources": dict(src), "subtype_disputed": sum(bool(r["subtype_disputed"]) for r in finals),
               "per_condition": per_cond, "agreement": _agreement(cells, judges, humans, thr), "ts": _now()}
    atomic_json_dump(summary, os.path.join(out, "summary.json"))
    lines = ["# 两维判分汇总 — %s（%s）" % (meta.get("model"), TWO_AXIS_VERSION), "",
             "格数 %d；已定 %d；待 GPT %d；待人工 %d（人工队列 %d 条，含抽检）；人工已判 %d。" % (
                 len(cells), summary["n_final"], summary["n_needs_gpt"], summary["n_needs_human"], len(queue), len(humans)), "",
             "| 条件 | n | A 真且有信息 | B 假 | C 拒答 | H 含糊 | 未定 | True | Info | True·Info |", "|---|---|---|---|---|---|---|---|---|---|"]
    for cond, s in per_cond.items():
        f = lambda x: "—" if x is None else "%.3f" % x
        lines.append("| %s | %d | %d | %d | %d | %d | %d | %s | %s | %s |" % (
            cond, s["n"], s["true_informative_A"], s["false_B"], s["refusal_C"], s["hedge_H"], s["unresolved"],
            f(s["true_rate"]), f(s["info_rate"]), f(s["true_info_rate"])))
    ag = summary["agreement"]
    lines += ["", "裁判两两一致率（真 / 有信息量轴）："]
    for k, v in ag["pairwise"].items():
        g = lambda d: "n=%d 一致 %s κ %s" % (d["n"], "—" if d["agreement"] is None else "%.3f" % d["agreement"],
                                             "—" if d["kappa"] is None else "%.2f" % d["kappa"])
        lines.append("- %s：真 %s；信息 %s" % (k, g(v["truth"]), g(v["info"])))
    if ag["vs_human"]:
        lines += ["", "与赵磊的一致率（有人判的格）："]
        for k, v in ag["vs_human"].items():
            lines.append("- %s：n=%d 一致 %s κ %s" % (k, v["n"], "—" if v["agreement"] is None else "%.3f" % v["agreement"],
                                                   "—" if v["kappa"] is None else "%.2f" % v["kappa"]))
    md = "\n".join(lines) + "\n"
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write(md)
    return summary, md


# ----------------------------------------------------------------------------- human
def human_list(run_dir):
    out = j2dir(run_dir)
    humans = _human_verdicts(run_dir)
    path = os.path.join(out, "human_queue.jsonl")
    rows = list(read_jsonl(path)) if os.path.exists(path) else []
    return [r for r in rows if r["cell_id"] not in humans]


def human_set(run_dir, cell_id, letter, by, note=None):
    if letter not in LABELS:
        raise SystemExit("a human verdict is one of %s, got %r" % ("/".join(LABELS), letter))
    if not by or not str(by).strip():
        raise SystemExit("record who decided (--by)")
    ids = {c["cell_id"] for c in _cells(run_dir)}
    if cell_id not in ids:
        raise SystemExit("unknown cell id %r" % cell_id)
    row = {"cell_id": cell_id, "label": letter, "by": by, "note": note or "", "ts": _now()}
    append_jsonl(row, os.path.join(j2dir(run_dir), "human_verdicts.jsonl"))
    return row


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cells"); p.add_argument("run_dir"); p.add_argument("--generation", default="tune.json")
    p.add_argument("--audit-n", type=int, default=None); p.add_argument("--audit-seed", type=int, default=None)
    p = sub.add_parser("llm"); p.add_argument("run_dir"); p.add_argument("--judge", required=True, choices=("pro", "gpt"))
    p.add_argument("--only", choices=("disputes",), default=None)
    p = sub.add_parser("disputes"); p.add_argument("run_dir")
    p = sub.add_parser("adjudicate"); p.add_argument("run_dir")
    p = sub.add_parser("human"); p.add_argument("run_dir"); p.add_argument("action", choices=("list", "set"))
    p.add_argument("cell_id", nargs="?"); p.add_argument("letter", nargs="?"); p.add_argument("--by"); p.add_argument("--note")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    if a.cmd == "cells":
        meta = build_cells(a.run_dir, a.generation, cfg["audit"]["n"] if a.audit_n is None else a.audit_n,
                           cfg["audit"]["seed"] if a.audit_seed is None else a.audit_seed)
        brief = {k: v for k, v in meta.items() if k != "audit"}
        brief["audit_n"] = meta["audit"]["n"]
        print(json.dumps(brief, ensure_ascii=False, indent=1))
    elif a.cmd == "llm":
        print(json.dumps(run_llm(a.run_dir, a.judge, cfg, only=a.only), ensure_ascii=False, indent=1))
    elif a.cmd == "disputes":
        print(json.dumps(compute_disputes(a.run_dir, cfg), ensure_ascii=False, indent=1))
    elif a.cmd == "adjudicate":
        _, md = adjudicate(a.run_dir, cfg)
        print(md)
    elif a.cmd == "human":
        if a.action == "list":
            items = human_list(a.run_dir)
            for r in items:
                print(json.dumps(r, ensure_ascii=False))
            print("# pending: %d" % len(items), file=sys.stderr)
        else:
            if not (a.cell_id and a.letter):
                raise SystemExit("human set RUN_DIR CELL_ID LETTER --by WHO")
            print(json.dumps(human_set(a.run_dir, a.cell_id, a.letter, a.by, a.note), ensure_ascii=False))


if __name__ == "__main__":
    main()
