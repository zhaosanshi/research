# -*- coding: utf-8 -*-
"""Mechanism probe: does the E write move the model along the C direction? (Claude Opus 5.5, 2026-11-16)

赵磊 11-16 plan, item 6: "机制探测：推 E 之后，C 方向上的读数掉多少，用来解释两类模型". On the plain prompt, E alone
turns many answers of Llama and Qwen into "I don't know"; Gemma 4 almost never does (it turns doubt into a flat denial).
If the E write moves the state of a DOWNSTREAM layer along the commitment direction C toward caution, the new blanks are
C's doing, and C toward confident can release them; if it does not, E's doubt is expressed by other means.

What is run (read-only; forward passes on the prompt only, no generation, no judge). Every question of the tune split, on
the plain prompt x, under
   none       no write
   E<s>       the learned E write at layer EL, strength s (the driver's own write: amount = s x gap of E, content positions)
   rand<s>    the driver's norm-matched random write at EL, strength s (same size, no E information)
and, without any write, the cautious and the confident prompts (caut, conf) as the sign and scale check. At each C layer
CL > EL (a write can only reach later layers) the block output is read at the last prompt token (where the C direction
was trained to read the prompt difference) and, as a second readout, averaged over the prompt's content positions (where
the writes act). The readout is the coordinate on the unit C direction w_C:
   scale      gap_C, the record's gap: a strength-1 C write moves the state by gap_C along w_C. By construction +w_C points
              from the confident toward the cautious prompt and gap_C > 0 (e44_factorial_das.py: plus = caut, minus = conf).
              Check on these questions: mean[coordinate(caut) - coordinate(conf)] / gap_C at the read position must be
              > 0 (else the run stops); its size says how much of the prompt contrast the direction carries here
   delta      (coordinate under the write - coordinate without it) / gap_C, > 0 = toward caution
   direct     the part of delta the write carries by itself: the residual stream passes the added vector unchanged to
              every later layer, so amount x cos(write direction, w_C) / gap_C is in delta before any layer reacts;
              delta - direct is the reaction of the layers between EL and CL. Both are reported
The questions are grouped by what E alone did to them in a judged grid of the same split, read from the grid's export
(judge_input.json: rule_abstain, or an empty output; no verdict is used). The grid must have the run's identity and its
E-alone conditions must carry the same E file and amount as the probe's writes (checked):
   newly blank     answered without any write, blank under E<s>
   still answered  answered without any write and under E<s>
   already blank   blank without any write
For rand<s> the grouping of E<s> is used, so that the two writes are compared on the same questions. Per group: the mean
delta with a bootstrap interval (4000 resamples, seed 20261116; a group under 10 questions is flagged, read its rows);
per question every coordinate. Nothing here is a method result; it only says whether C is the channel of E's blanks.
Writes RUN/eval_extra/probe_c_under_e_<NAME>.json and .md (a partial run: the t<N>. prefix); refuses to overwrite.
usage (from the code copy's root, with the model's own interpreter):
  python scripts/review_checks/ff_ec_probe.py NAME --source-run S --run-dir E_RUN --e-on E2 [--c-run-dir C_RUN --c-e-on E2]
         --el 24 --cls 25 27 --es 1 2 --grid E_RUN/eval_extra/judge_tune_ec21_tune [--first-batches K]"""
import argparse
import contextlib
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE):          # <code copy>/scripts (engine, driver), then review_checks (the confidence script)
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import torch  # noqa: E402
import e44_eval_extra as X  # noqa: E402
from core.model_io import forward_logits  # noqa: E402
import ff_explore_ec_confidence as CF  # noqa: E402  (its constants; its main() runs only as a script)
pd, fd = X.pd, X.fd
PROTOCOL = "ff_ec_probe_v1"
SEED = 20261116


def log(msg):
    print("%s %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tok(s):
    return s.replace(".", "-")


def e_alone(el, s):
    return "E_L%d_x-w-p%s" % (el, tok(s))


def load_grid(grid, el, es):
    """(blank flags {pair_id: {"none": bool, "E<s>": bool}}, the export, its sha256) from a grid's judge_input.json:
    a question is blank under a condition when the rule detector flagged it or the output is empty."""
    path = Path(grid) / "judge_input.json"
    raw = path.read_bytes()
    d = json.loads(raw)
    if d.get("completed") is not True or d.get("split") != "tune":
        raise SystemExit("%s is not a completed export of the tune split" % path)
    want = {"baseline_x": "none"}
    want.update({e_alone(el, s): "E%s" % s for s in es})
    missing = sorted(set(want) - set(d["conditions"]))
    if missing:
        raise SystemExit("the grid has no condition %s" % missing)
    out = {}
    for ev in d["evaluations"]:
        flags = {}
        for cond, key in want.items():
            text = ev["outputs"][cond]
            flags[key] = bool(ev["rule_abstain"][cond]) or not str(text).strip()
        out[ev["pair_id"]] = flags
    return out, d, hashlib.sha256(raw).hexdigest()


def one_write(ctx, arm):
    """(layer, write kwargs, description) of a one-part arm through the driver's verified path."""
    _, parsed = X.parse_arm(arm, ctx.all_systems)
    ws, positions = ctx.writes(parsed)
    if positions != "content" or len(ws) != 1:
        raise SystemExit("a one-part content-position write expected for %s" % arm)
    return ws[0]


@torch.no_grad()
def readouts(ctx, prompts, write, cls, wc):
    """Per question {CL: (coordinate at the last prompt token, mean coordinate over the prompt's content positions)}.
    write = None or (layer, write kwargs): a fd.ShiftHook on the prompt's content positions, as on the prefill of
    generation (fd.generate_shift / generate_two)."""
    res, bs = [], ctx.args.batch_size
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        inputs = ctx.bundle.encode([p["prefix_text"] for p in chunk])
        ids, att = inputs["input_ids"], inputs["attention_mask"]
        width = ids.shape[1]
        for b, p in enumerate(chunk):
            if ids[b][att[b].bool()].tolist() != p["prefix_ids"]:
                raise RuntimeError("batched prompt tokens differ from the single-prompt rendering")
            if not bool(att[b, width - 1]):
                raise RuntimeError("the prompt batch is not left-padded: the last position is not the last prompt token")
        mask = fd._prefill_mask(inputs, chunk, "content")
        caps = {cl: X.Capture() for cl in cls}
        hook = None
        with contextlib.ExitStack() as stack:
            if write is not None:
                layer, w = write
                hook = fd.ShiftHook(w["direction"], amounts=torch.full((len(chunk),), float(w["amount"])), mask=mask,
                                    decode_all=False, reference_direction=w.get("reference_direction"))
                stack.enter_context(pd.clamp(ctx.bundle, layer, hook))
            for cl in cls:
                stack.enter_context(pd.clamp(ctx.bundle, cl, caps[cl]))
            forward_logits(ctx.bundle, {"input_ids": ids, "attention_mask": att}, grad=False)
        if hook is not None and (hook.applied != 1 or hook.prefill is None or hook.prefill["sequence_width"] != width):
            raise RuntimeError("the write did not fire exactly once on the prompt")
        m = mask.detach().bool().cpu()
        for cl in cls:
            h = caps[cl].h
            if h is None or tuple(h.shape[:2]) != (len(chunk), width) or not bool(torch.isfinite(h).all()):
                raise RuntimeError("no finite states captured at layer %d" % cl)
        for b in range(len(chunk)):
            if not bool(m[b, width - 1]):
                raise RuntimeError("the last prompt token is not a content position")
            row = {}
            for cl in cls:
                proj = caps[cl].h[b] @ wc[cl]
                row[cl] = (float(proj[width - 1]), float(proj[m[b]].mean()))
            res.append(row)
    if len(res) != len(prompts):
        raise RuntimeError("readouts returned the wrong number of rows")
    return res


def mean_ci(v):
    if not v:
        return None
    rng = random.Random(SEED)
    n = len(v)
    bs = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(4000))
    return {"n": n, "mean": sum(v) / n, "lo": bs[100], "hi": bs[3899]}


def fmt(c):
    if c is None:
        return "—"
    return "%+.3f [%+.3f, %+.3f] (n %d%s)" % (c["mean"], c["lo"], c["hi"], c["n"], ", under 10: read the rows" if c["n"] < 10 else "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True, help="the run of the E direction")
    p.add_argument("--e-on", default="E2")
    p.add_argument("--c-run-dir", type=Path, default=None, help="the run of the C direction (default: --run-dir)")
    p.add_argument("--c-e-on", default="E2")
    p.add_argument("--el", type=int, required=True)
    p.add_argument("--cls", type=int, nargs="+", required=True)
    p.add_argument("--es", nargs="+", default=["1", "2"])
    p.add_argument("--grid", type=Path, required=True, help="a judged E+C (or E) grid of the tune split of the E run (its export)")
    p.add_argument("--first-batches", type=int, default=0)
    args = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.name):
        raise SystemExit("NAME: letters, digits and _ only")
    if len(set(args.cls)) != len(args.cls) or any(cl <= args.el for cl in args.cls):
        raise SystemExit("every C layer must be distinct and deeper than the E layer %d" % args.el)
    if len(set(args.es)) != len(args.es):
        raise SystemExit("an E strength is repeated")
    for s in args.es:
        if not re.fullmatch(r"[0-9]+(\.[0-9]+)?", s) or float(s) <= 0:
            raise SystemExit("E strength %r is not a positive number" % s)
    for need in (args.run_dir / "eval_extra", args.grid / "judge_input.json"):   # before the model is loaded
        if not need.exists():
            raise SystemExit("%s does not exist" % need)
    P = Path(X.EXP_ROOT) / "configs" / "prompts"
    a = argparse.Namespace(                                    # the confidence script's Context arguments, split tune
        phase="generate", source_run=args.source_run, run_dir=args.run_dir, prompts=P / "default_v1.json",
        prompts_e44=P / "e44_factorial_v1.json", prompts_extra=P / "e44_eval_extra_v1.json", e_on=args.e_on, e_off="none",
        c_run_dir=args.c_run_dir, c_e_on=args.c_e_on, c_e_off="none", arms=[], bases=[], family=None, layers=[], pairs=[],
        answers=["minus", "plus"], split="tune", seed=42, batch_size=CF.BATCH, max_new_tokens=CF.MAX_TOKENS,
        max_target_tokens=24, max_ref_tokens=48, output_projection="float32", first_batches=args.first_batches,
        gen=None, prompts_out=None, name="judge_input", overwrite=False, n_check=0, pair_ids=None)
    ctx = X.Context(a)
    stem = ctx.name("probe_c_under_e_%s" % args.name)
    out_json, out_md = ctx.out_dir / (stem + ".json"), ctx.out_dir / (stem + ".md")
    if out_json.exists() or out_md.exists():
        raise SystemExit("%s exists" % out_json)
    ids = [q["pair_id"] for q in ctx.questions]
    bl, grid_export, grid_sha = load_grid(args.grid, args.el, args.es)
    if grid_export.get("identity_sha256") != ctx.ident_sha:
        raise SystemExit("the grid was made by another run identity (%s; this run %s)" % (grid_export.get("identity_sha256"), ctx.ident_sha))
    miss = [q for q in ids if q not in bl]
    if miss:
        raise SystemExit("the grid lacks %d questions of this split, e.g. %s" % (len(miss), miss[:3]))

    wc, cdesc = {}, {}
    for cl in args.cls:
        rec, path, fsha = ctx.record("C", cl)
        w = torch.as_tensor(rec["w"]).float()
        if w.ndim != 1 or not float(w.norm()) > 0 or not float(rec["gap"]) > 0:
            raise SystemExit("the C record of layer %d has no usable direction or gap" % cl)
        wc[cl] = w / w.norm()
        cdesc[cl] = {"layer": cl, "gap": float(rec["gap"]), "direction_file": path.name, "direction_file_sha256": fsha,
                     "direction_record_sha256": rec["artifact_sha256"], "e_on": rec["e_on"],
                     "positions": rec["training_contract"]["positions"]}

    arms = {"none": None}
    edesc = {}
    for s in args.es:
        for kind, key in (("w", "E%s" % s), ("rand", "rand%s" % s)):
            layer, w, desc = one_write(ctx, "x:E%d.%s.p%s" % (args.el, kind, s))
            arms[key] = (layer, w)
            edesc[key] = desc
    specs = {c["name"]: c for c in grid_export["condition_specs"]}
    for s in args.es:                                          # the grid's E-alone answers must come from the probe's write
        parts, mine = specs[e_alone(args.el, s)]["parts"], edesc["E%s" % s]
        if len(parts) != 1 or parts[0]["direction_file_sha256"] != mine["direction_file_sha256"] or \
                abs(float(parts[0]["amount"]) - float(mine["amount"])) > 1e-9:
            raise SystemExit("the grid's %s was made with another E write than the probe's" % e_alone(args.el, s))
    direct = {}                                                # what each write carries to C by itself, in units of gap_C
    for key, wr in arms.items():
        if wr is not None:
            u = torch.as_tensor(wr[1]["direction"]).detach().float().cpu()
            u = u / u.norm()
            direct[key] = {cl: float(wr[1]["amount"]) * float(u @ wc[cl]) / cdesc[cl]["gap"] for cl in args.cls}
    prompts_x = ctx.render("x")
    log("%d questions; C layers %s; arms %s" % (len(ids), args.cls, list(arms)))
    rows = {key: readouts(ctx, prompts_x, wr, args.cls, wc) for key, wr in arms.items()}
    for name in ("caut", "conf"):
        rows["prompt_" + name] = readouts(ctx, ctx.render(name), None, args.cls, wc)
    ctx.still_current()

    summary, lines = {}, []
    lines.append("# Probe: does the E write move the state along C? (%s, run %s, E%d, tune %d questions)\n" % (
        args.name, args.run_dir.name, args.el, len(ids)))
    lines.append("delta = (coordinate under the write - without it) / gap_C, > 0 = toward caution; read = last prompt token, "
                 "content = mean over the prompt's content positions; direct = what the write carries to C by itself. "
                 "Groups by what E alone did (export of %s).\n" % args.grid.name)
    for cl in args.cls:
        g = cdesc[cl]["gap"]
        chk = {r: [(rows["prompt_caut"][j][cl][r] - rows["prompt_conf"][j][cl][r]) / g for j in range(len(ids))] for r in (0, 1)}
        if not sum(chk[0]) > 0:
            raise SystemExit("C layer %d: on these questions the cautious prompt does not sit above the confident one along +w_C" % cl)
        summary[cl] = {"gap": g, "caut_minus_conf_over_gap": {"read": mean_ci(chk[0]), "content": mean_ci(chk[1])},
                       "direct": {k: v[cl] for k, v in direct.items()}, "arms": {}}
        lines.append("\n## C layer %d (gap %.4g)\n" % (cl, g))
        lines.append("- check, caut minus conf over gap: read %s; content %s\n" % (fmt(mean_ci(chk[0])), fmt(mean_ci(chk[1]))))
        lines.append("| arm | direct | group | delta, read position | delta - direct, read position | delta, content positions |")
        lines.append("|---|---|---|---|---|---|")
        for key in arms:
            if key == "none":
                continue
            s = key[1:] if key.startswith("E") else key[4:]
            groups = {"all": [], "newly blank": [], "still answered": [], "already blank": []}
            for j, q in enumerate(ids):
                d_read = (rows[key][j][cl][0] - rows["none"][j][cl][0]) / g
                d_cont = (rows[key][j][cl][1] - rows["none"][j][cl][1]) / g
                b0, b1 = bl[q]["none"], bl[q]["E%s" % s]
                grp = "already blank" if b0 else ("newly blank" if b1 else "still answered")
                for gname in ("all", grp):
                    groups[gname].append((d_read, d_read - direct[key][cl], d_cont))
            summary[cl]["arms"][key] = {}
            for grp, v in groups.items():
                c = tuple(mean_ci([x[r] for x in v]) for r in range(3))
                summary[cl]["arms"][key][grp] = {"read": c[0], "read_minus_direct": c[1], "content": c[2]}
                lines.append("| %s | %+.3f | %s | %s | %s | %s |" % (key, direct[key][cl], grp, fmt(c[0]), fmt(c[1]), fmt(c[2])))
        for s in args.es:
            v = [((rows["E%s" % s][j][cl][0] - rows["none"][j][cl][0]) - (rows["rand%s" % s][j][cl][0] - rows["none"][j][cl][0])) / g
                 for j in range(len(ids))]
            summary[cl]["E%s_minus_rand%s_read" % (s, s)] = mean_ci(v)
            lines.append("\n- E%s minus rand%s, all questions, read position: %s" % (s, s, fmt(mean_ci(v))))

    report = {"protocol": PROTOCOL, "split": "tune", "partial": ctx.partial, "question_ids": ids,
              "identity_sha256": ctx.ident_sha, "frozen_file_sha256": ctx.frozen_sha, "script_sha256": sha(__file__),
              "eval_extra_sha256": sha(X.__file__), "spec": {"e_layer": args.el, "c_layers": args.cls, "e_strengths": args.es,
              "grid": str(args.grid), "grid_judge_input_sha256": grid_sha, "grid_identity_sha256": grid_export.get("identity_sha256"),
              "bootstrap": {"resamples": 4000, "seed": SEED}},
              "c_directions": {str(k): v for k, v in cdesc.items()}, "e_writes": edesc,
              "summary": {str(k): v for k, v in summary.items()},
              "rows": [{"pair_id": q, "blank": bl[q], "coordinates": {key: {str(cl): rows[key][j][cl] for cl in args.cls} for key in rows}}
                       for j, q in enumerate(ids)]}
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("wrote %s and %s" % (out_json.name, out_md.name))


if __name__ == "__main__":
    main()
