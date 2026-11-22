# -*- coding: utf-8 -*-
"""Checks for queue I (Claude Opus 5.5, 2026-11-22): E and E + C written at "prompt" on the prompt x_E2.
  gpu    (model interpreter) on the first 16 questions of the TruthfulQA tune split, through newsets_gen.context with the
         h22 prompt file (scripts/newsets/ns_h22.patch_context; engine unchanged): x_E2 + E(EL) at amount 0 = the no-write
         x_E2 answers = the first 16 rows of eval_extra/base_x_E2_tune.json; x_E2 + E(s) twice identical; x_E2 + E(s) +
         C(CL) 0.25 (core generate_two, one rule for both hooks) twice identical. The report keeps ids and token ids only.
  parts  (CPU) the writes of a generation file of this queue against the main table's file of the same family / layer
         (E: x--E<EL>-w-p<main>-at-prompt_<split>.json; C: the C part of x--E<EL>-w-p<main>-and-C<CL>-w-m<c>-at-prompt):
         family, layer, kind, direction file and record hashes equal, positions "prompt"; the amount equal to the main
         file's amount times strength / main strength.
  cross  (CPU) the token ids of the gpu check's runs = the first 16 rows of the full generation files.
usage:
  python -I n1s_check.py gpu --models M --model MODEL --el EL --cl CL --s S --base-file BASE_X_E2_TUNE --out REPORT
  python -I n1s_check.py parts NEW.json MAIN.json STRENGTH MAIN_STRENGTH [E|C]
  python -I n1s_check.py cross REPORT NEW_E.json NEW_EC.json
"""
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
N = 16
KEYS = ("text", "token_ids", "finished", "stop_id")
PK = ("family", "layer", "kind", "direction_file_sha256", "direction_record_sha256")


def gpu(argv):
    import argparse
    import ns_h22 as H
    G = H.G
    ap = argparse.ArgumentParser()
    for k in ("--models", "--model", "--el", "--cl", "--s", "--base-file", "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit("%s exists" % out)
    H.patch_context()
    m = G.load_models(Path(a.models), a.model)
    X = G.engine(m)
    ctx = G.context(X, m, N // G.BATCH)
    systems = G.systems_of(ctx)
    if systems.get("x_E2") != systems["x"] + " " + H.E2:
        raise SystemExit("x_E2 is not the plain prompt plus the honest sentence")
    arm = "x_E2:E%s.w.p%s+C%s.w.m0.25" % (a.el, a.s, a.cl)
    _, parts = X.parse_arm(arm, systems)
    ws, trained = ctx.writes(parts)
    if trained != "content" or len(ws) != 2:
        raise SystemExit("expected two content-trained writes")
    (le, we, de), (lc, wc, dc) = ws
    pd, fd = X.pd, X.fd
    kw = dict(max_new_tokens=G.ENGINE_MAX_NEW, batch_size=G.BATCH)
    ids = [q["pair_id"] for q in ctx.questions]
    base = json.loads(Path(a.base_file).read_text(encoding="utf-8"))
    brow = base["rows"][:N]
    if [r["pair_id"] for r in brow] != ids or base.get("prompt") != "x_E2":
        raise SystemExit("the base file is not the x_E2 tune pass of these questions")
    with X.E.fp32_output_projection(ctx.bundle, enabled=True):
        prompts = [pd.render_prompt(ctx.bundle, systems["x_E2"], ctx.user_template, q["question"]) for q in ctx.questions]
        nw = pd.generate_prompts(ctx.bundle, prompts, **kw)
        z = fd.generate_shift(ctx.bundle, prompts, layer=le, positions="prompt", **kw, **dict(we, amount=0.0))
        e1 = fd.generate_shift(ctx.bundle, prompts, layer=le, positions="prompt", **kw, **we)
        e2 = fd.generate_shift(ctx.bundle, prompts, layer=le, positions="prompt", **kw, **we)
        c1 = X.generate_two(ctx.bundle, prompts, [(le, we), (lc, wc)], positions="prompt", **kw)
        c2 = X.generate_two(ctx.bundle, prompts, [(le, we), (lc, wc)], positions="prompt", **kw)
    r = {"model": a.model, "arm": arm, "ids": ids, "writes": [{k: d.get(k) for k in PK + ("beta", "amount")} for d in (de, dc)],
         "E0_equals_no_write": sum(G.same(x, y) for x, y in zip(z, nw)),
         "E0_equals_saved_P": sum(all(x[k] == y[k] for k in KEYS) for x, y in zip(z, brow)),
         "E_twice": sum(G.same(x, y) for x, y in zip(e1, e2)), "EC_twice": sum(G.same(x, y) for x, y in zip(c1, c2)),
         "E_changed": sum(x["token_ids"] != y["token_ids"] for x, y in zip(e1, nw)),
         "EC_changed": sum(x["token_ids"] != y["token_ids"] for x, y in zip(c1, nw)),
         "E_token_ids": [x["token_ids"] for x in e1], "EC_token_ids": [x["token_ids"] for x in c1],
         "entry_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    r["passed"] = all(r[k] == N for k in ("E0_equals_no_write", "E0_equals_saved_P", "E_twice", "EC_twice"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r), encoding="utf-8")
    print("gpu check %s: E0 = no write %d/%d, E0 = saved P %d/%d, E twice %d/%d, E+C twice %d/%d, changed by E %d, by E+C %d" % (
        a.model, r["E0_equals_no_write"], N, r["E0_equals_saved_P"], N, r["E_twice"], N, r["EC_twice"], N, r["E_changed"], r["EC_changed"]), flush=True)
    if not r["passed"]:
        raise SystemExit("gpu check FAILED")


def parts(new, main, s, s_main, fam):
    gn, gm = json.loads(Path(new).read_text()), json.loads(Path(main).read_text())
    pn = [p for p in gn["parts"] if p["family"] == fam]
    pm = [p for p in gm["parts"] if p["family"] == fam]
    if len(pn) != 1 or len(pm) != 1:
        raise SystemExit("parts: one %s write expected in each file" % fam)
    pn, pm = pn[0], pm[0]
    bad = [k for k in PK if pn.get(k) != pm.get(k)]
    if gn.get("positions") != "prompt" or gm.get("positions") != "prompt":
        bad.append("positions")
    ratio = float(s) / float(s_main) if fam == "E" else 1.0
    if not math.isclose(float(pn["amount"]), float(pm["amount"]) * ratio, rel_tol=1e-6, abs_tol=1e-9):
        bad.append("amount")
    print("parts %s %s vs %s: %s (layer %s, amount %.6g, main %.6g)" % (fam, Path(new).name, Path(main).name,
          "identical" if not bad else "DIFFER in %s" % bad, pn["layer"], pn["amount"], pm["amount"]), flush=True)
    if bad:
        raise SystemExit(1)


def cross(report, fe, fec):
    r = json.loads(Path(report).read_text())
    ok = True
    for key, f in (("E_token_ids", fe), ("EC_token_ids", fec)):
        rows = json.loads(Path(f).read_text())["rows"][:N]
        same = sum(t == x["token_ids"] and i == x["pair_id"] for t, i, x in zip(r[key], r["ids"], rows))
        print("cross %s: check run vs %s first %d rows identical %d" % (key, Path(f).name, N, same), flush=True)
        ok &= same == N
    if not ok:
        raise SystemExit(1)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    if sys.argv[1] == "gpu":
        gpu(sys.argv[2:])
    elif sys.argv[1] == "parts":
        parts(*sys.argv[2:7])
    elif sys.argv[1] == "cross":
        cross(*sys.argv[2:5])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
