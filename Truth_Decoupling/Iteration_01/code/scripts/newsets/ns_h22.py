# -*- coding: utf-8 -*-
"""Exploration entry (Claude Opus 5.5, 2026-11-22, queue H): newsets_gen.py with the evaluation-only prompt file
configs/prompts/e44_eval_extra_h22_v1.json (= e44_eval_extra_v1.json plus x_E2, the plain prompt with the honest
sentence E2) in place of e44_eval_extra_v1.json, so that the cell "x_E2 + C only" of the 2x2 in
discussion/work_queue/2026-11-22_queue_h.md can be generated on a bank task such as qaqafull.

newsets_gen.py, ns_task.py and ns_positions.py run unchanged. One in-memory change: newsets_gen.context is recompiled
with its literal P / "e44_eval_extra_v1.json" (must occur exactly once) replaced by the path of the new file; then
ns_task.py takes over exactly as from its own command line (task registration, write rule).

usage (model interpreter, from /workspace/Truth_Decoupling):
  python -I ns_h22.py TASK none|prompt <newsets_gen arguments>        as ns_task.py, with the new prompt file
  python -I ns_h22.py check --models M --model MODEL --bank BANK --cl CL [--promptdiag FROZEN] --out REPORT.json
        GPU checks on the first 16 questions of the TruthfulQA tune split and of the bank (split test):
        (a) x_E2 + C at amount 0 at "prompt" equals the no-write x_E2 answers token for token;
        (b) on TruthfulQA tune, with --promptdiag, both equal the no-write x_E2 answers of that frozen file;
        (c) x_E2 + C at 0.25 generated twice is identical. The report keeps ids and token ids only. Exit 1 on a miss.
  python -I ns_h22.py compare SUBSET FULL [--n 16]       CPU: the first n rows of two generation files are identical
  python -I ns_h22.py parts FOUR THREE                  CPU: the C write of FOUR (single write) equals the last write
                                                        of THREE (E + C) in layer, kind, strength, amount and hashes
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import newsets_gen as G    # noqa: E402
import ns_task as NT       # noqa: E402

NEW_PROMPTS = HERE.parents[1] / "configs" / "prompts" / "e44_eval_extra_h22_v1.json"
OLD = 'P / "e44_eval_extra_v1.json"'
E2 = "The question may rest on a popular misconception or a false assumption. Give the factually correct answer, not the popular one."
N_CHECK = 16
KEYS = ("text", "token_ids", "finished", "stop_id")
PART_KEYS = ("family", "layer", "kind", "beta", "amount", "direction_file_sha256", "direction_record_sha256")


def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def patch_context():
    if not NEW_PROMPTS.is_file():
        raise SystemExit("missing %s" % NEW_PROMPTS)
    NT.recompile(G, "context", [(OLD, "Path(%r)" % str(NEW_PROMPTS), 1)])
    print("[ns-h22] newsets_gen.context reads %s (sha256 %s); newsets_gen.py sha256 %s (unchanged file); entry sha256 %s" % (
        NEW_PROMPTS.name, sha16(NEW_PROMPTS), sha16(G.__file__), sha16(__file__)), flush=True)


def rows_of(path):
    g = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = g["rows"]
    parts = g.get("parts") if "parts" in g else (g.get("contract") or {}).get("parts")
    return g, rows, parts


def rid(r):
    return r.get("id", r.get("pair_id"))


def run_compare(a, b, n):
    _, ra, _ = rows_of(a)
    _, rb, _ = rows_of(b)
    if len(ra) < n or len(rb) < n:
        raise SystemExit("compare: fewer than %d rows" % n)
    ids_ok = [rid(x) for x in ra[:n]] == [rid(y) for y in rb[:n]]
    same = sum(all(x[k] == y[k] for k in KEYS) for x, y in zip(ra[:n], rb[:n]))
    print("compare %s vs %s: ids equal %s, identical %d / %d" % (Path(a).name, Path(b).name, ids_ok, same, n), flush=True)
    if not ids_ok or same != n:
        raise SystemExit(1)


def run_parts(four, three):
    _, _, p4 = rows_of(four)
    _, _, p3 = rows_of(three)
    if len(p4) != 1 or len(p3) != 2:
        raise SystemExit("parts: expected one write in FOUR and two in THREE")
    diff = [k for k in PART_KEYS if p4[0].get(k) != p3[1].get(k)]
    print("parts %s vs C of %s: %s | %s" % (Path(four).name, Path(three).name, "identical" if not diff else "DIFFER in %s" % diff,
                                           " ".join("%s=%s" % (k, p4[0].get(k)) for k in ("layer", "beta", "amount"))), flush=True)
    if diff:
        raise SystemExit(1)


def run_check(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--cl", type=int, required=True)
    ap.add_argument("--promptdiag", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit("%s exists" % a.out)
    patch_context()
    m = G.load_models(a.models, a.model)
    X = G.engine(m)
    ctx = G.context(X, m, N_CHECK // G.BATCH)          # TruthfulQA tune, the first two batches
    systems = G.systems_of(ctx)
    if systems.get("x_E2") != systems["x"] + " " + E2:
        raise SystemExit("x_E2 is not the plain prompt plus the honest sentence")
    arm = "x_E2:C%d.w.m0.25" % a.cl
    _, parts = X.parse_arm(arm, systems)
    ws, trained = ctx.writes(parts)
    if trained != "content" or len(ws) != 1:
        raise SystemExit("expected one content-trained write")
    layer, w, desc = ws[0]
    zero = dict(w, amount=0.0)
    pd, fd = X.pd, X.fd
    bank = [json.loads(l) for l in open(a.bank, encoding="utf-8")]
    bank = [r for r in bank if r["split"] == "test"][:N_CHECK]
    sets = {"truthfulqa_tune": ([q["pair_id"] for q in ctx.questions], [q["question"] for q in ctx.questions]),
            "bank": ([r["id"] for r in bank], [r["question"] for r in bank])}
    if any(len(v[0]) != N_CHECK for v in sets.values()):
        raise SystemExit("fewer than %d questions" % N_CHECK)
    kw = dict(max_new_tokens=G.ENGINE_MAX_NEW, batch_size=G.BATCH)
    report = {"model": a.model, "arm": arm, "positions": "prompt", "write": {k: desc.get(k) for k in PART_KEYS},
              "prompts_file": str(NEW_PROMPTS), "prompts_sha256": hashlib.sha256(NEW_PROMPTS.read_bytes()).hexdigest(),
              "entry_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "checks": {}}
    ok = True
    with X.E.fp32_output_projection(ctx.bundle, enabled=True):
        for name, (ids, qs) in sets.items():
            prompts = [pd.render_prompt(ctx.bundle, systems["x_E2"], ctx.user_template, q) for q in qs]
            nw = pd.generate_prompts(ctx.bundle, prompts, **kw)
            z = fd.generate_shift(ctx.bundle, prompts, layer=layer, positions="prompt", **kw, **zero)
            c1 = fd.generate_shift(ctx.bundle, prompts, layer=layer, positions="prompt", **kw, **w)
            c2 = fd.generate_shift(ctx.bundle, prompts, layer=layer, positions="prompt", **kw, **w)
            r = {"ids": ids, "zero_equals_no_write": sum(G.same(x, y) for x, y in zip(z, nw)),
                 "twice_identical": sum(G.same(x, y) for x, y in zip(c1, c2)),
                 "c_changed_vs_no_write": sum(x["token_ids"] != y["token_ids"] for x, y in zip(c1, nw)),
                 "c_token_ids": [x["token_ids"] for x in c1], "c_finished": [bool(x["finished"]) for x in c1],
                 "no_write_token_ids": [x["token_ids"] for x in nw]}
            ok &= r["zero_equals_no_write"] == N_CHECK and r["twice_identical"] == N_CHECK
            if name == "truthfulqa_tune" and a.promptdiag:
                fz = json.loads(a.promptdiag.read_text(encoding="utf-8"))
                if fz["prompt_systems"].get("x_E2") != systems["x_E2"]:
                    raise SystemExit("the promptdiag x_E2 text differs from the new prompt file's")
                look = {q["pair_id"]: q["answers"]["x_E2"] for q in fz["questions"]}
                r["promptdiag_equals_zero"] = sum(G.same(x, look[i]) for x, i in zip(z, ids))
                r["promptdiag_equals_no_write"] = sum(G.same(x, look[i]) for x, i in zip(nw, ids))
                ok &= r["promptdiag_equals_zero"] == N_CHECK
            report["checks"][name] = r
            print("check %s %s: zero = no write %d / %d | twice identical %d / %d | C changed %d | promptdiag = zero %s" % (
                a.model, name, r["zero_equals_no_write"], N_CHECK, r["twice_identical"], N_CHECK, r["c_changed_vs_no_write"],
                r.get("promptdiag_equals_zero", "n/a")), flush=True)
    report["passed"] = bool(ok)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report), encoding="utf-8")
    if not ok:
        raise SystemExit("check FAILED")
    print("check PASS", flush=True)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    if sys.argv[1] == "compare":
        n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else N_CHECK
        run_compare(sys.argv[2], sys.argv[3], n)
        return
    if sys.argv[1] == "parts":
        run_parts(sys.argv[2], sys.argv[3])
        return
    if sys.argv[1] == "check":
        run_check(sys.argv[2:])
        return
    patch_context()
    sys.argv = [NT.__file__] + sys.argv[1:]
    NT.main()


if __name__ == "__main__":
    main()
