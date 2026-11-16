# -*- coding: utf-8 -*-
"""Mechanism probe: after the E write, does the downstream C readout of the prompt forward change, and is that change
related to the newly-blank group? (Claude Opus 5.5, 2026-11-16; revised the same day after the GPT review
discussion/mechanism_probe/2026-11-16_gpt_review.md: B1, S1-S3 and its two suggestions)

赵磊 11-16 plan, item 6: "机制探测：推 E 之后，C 方向上的读数掉多少，用来解释两类模型". On the plain prompt, E alone
turns many answers of Llama and Qwen into "I don't know"; Gemma 4 almost never does (it turns doubt into a flat denial).
This probe asks only: after the E write at layer EL, does the C coordinate that DOWNSTREAM layers carry on the prompt
forward change, and is that change related to the questions E newly turns blank? A change is evidence of that and no
more: it does not by itself show that the new blanks are caused by C (no causal mediation is identified here), and no
change holds only for the positions read (the prompt forward; the decode positions of generation are not read).

What is run (read-only; forward passes on the prompt only, no generation, no judge). Every question of the tune split, on
the plain prompt x, under
   none       no write
   E<s>       the learned E write at layer EL, strength s (the driver's own write: amount = s x gap of E, content positions)
   rand<s>    the driver's norm-matched random write at EL, strength s (same size, no E information)
and, without any write, the cautious and the confident prompts (caut, conf) as the sign and scale check. At each C layer
CL > EL (a write can only reach later layers) the block output is read at the last prompt token (where the C direction
was trained to read the prompt difference) and, as a second readout, averaged over all prefill write positions (every
prompt position the write acts on: the user content, the chat-template / assistant-header tokens after it and the last
prompt token; not the question text alone). The readout is the coordinate on the unit C direction w_C:
   scale      gap_C, the record's gap: a strength-1 C write moves the state by gap_C along w_C. By construction +w_C points
              from the confident toward the cautious prompt and gap_C > 0 (e44_factorial_das.py: plus = caut, minus = conf).
              Check on these questions: mean[coordinate(caut) - coordinate(conf)] / gap_C at the read position must be
              > 0, else the run stops and writes nothing (the sign is never flipped to pass); its size says how much of
              the prompt contrast the direction carries here
   delta      (coordinate under the write - coordinate without it) / gap_C, > 0 = toward caution
   actual direct  the part of delta the write carries by itself, measured in the write's own call: the change the hook
              actually stored at EL, h_after.float() - h_before.float(), projected on w_C at every position and read like
              delta (last prompt token; mean over all prefill write positions), / gap_C, per question. The residual stream
              passes it unchanged to every later layer, so it is in delta before any layer reacts. Rounding to the model's
              dtype and the random arm's per-position stored-norm matching make it differ from the nominal write
   delta - actual direct  the remaining coordinate change at CL (two position readouts); not an identified mediation
              effect
   intended direct  the nominal amount x cos(write direction, w_C) / gap_C; reported for reference, never subtracted
The questions are grouped by what E alone did to them in a judged grid of the same split, read from the grid's export
(judge_input.json: rule_abstain, or an empty output; no verdict is used):
   newly blank     answered without any write, blank under E<s>
   still answered  answered without any write and under E<s>
   already blank   blank without any write, whatever E<s> did (it includes questions E then answers: not "still blank")
Checked before the model is loaded: question IDs, evaluation pair_ids and condition names are each unique, the grid's full
question set is the full tune split (also for a partial run), and its stored rule flags are what the rule detector
(core.readout.rule_abstains) gives now. After it is loaded: the grid has the run's identity, its questions and baseline_x
answers (text, finished) are the current frozen ones, each E-alone condition stands on baseline_x and was made with the
probe's own write (family, layer, kind, beta, amount, direction file, record and training-freeze hashes, E factor), and,
followed through the export's sources, the frozen file is the current one and each E-alone generation file is unchanged
and has the probe's arm, positions, split, seed, token budget, script version, run identity and frozen file. The grid is
hashed again before the report is written.
For rand<s> the grouping of E<s> is used, so that the two writes are compared on the same questions. Per group: the mean
delta with a bootstrap interval (4000 resamples, seed 20261116; a group under 10 questions is flagged, read its rows);
per question every coordinate and the actual direct. Nothing here is a method result.
Writes RUN/eval_extra/probe_c_under_e_<NAME>.json and .md (a partial run: the t<N>. prefix); refuses to overwrite, checked
before the model is loaded.
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
PROTOCOL = "ff_ec_probe_v2"         # v2 (GPT review 2026-11-16): actual direct, grid checks, renamed position readout
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
    a question is blank under a condition when the rule detector flagged it or the output is empty. Question IDs,
    evaluation pair_ids and condition_specs names must each be unique: a repeated pair_id would silently replace a row."""
    path = Path(grid) / "judge_input.json"
    raw = path.read_bytes()
    d = json.loads(raw)
    if d.get("completed") is not True or d.get("split") != "tune":
        raise SystemExit("%s is not a completed export of the tune split" % path)
    for what, names in (("question_ids", list(d["question_ids"])), ("evaluation pair_ids", [ev["pair_id"] for ev in d["evaluations"]]),
                        ("condition_specs names", [c["name"] for c in d["condition_specs"]])):
        if len(set(names)) != len(names):
            raise SystemExit("the grid repeats %s, e.g. %s" % (what, sorted({n for n in names if names.count(n) > 1})[:3]))
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


def check_split(d, tune):
    """The grid's full question set (question_ids and evaluations) is the full tune split; checked on the whole grid, also
    when only the first batches are probed."""
    want = {q["pair_id"] for q in tune}
    if set(d["question_ids"]) != want or {ev["pair_id"] for ev in d["evaluations"]} != want:
        raise SystemExit("the grid's question set is not the full tune split (grid %d questions, %d evaluations; split %d)" % (
            len(d["question_ids"]), len(d["evaluations"]), len(want)))


def recheck_flags(d, el, es, rule):
    """Recompute the rule flags of the conditions that group the questions with the export's own detector (rule =
    core.readout.rule_abstains) on every question of the grid; each must equal the stored flag, so the blank flags of
    load_grid (rule flag or empty output) are the recomputed ones."""
    conds = ["baseline_x"] + [e_alone(el, s) for s in es]
    bad = [(ev["pair_id"], c) for ev in d["evaluations"] for c in conds
           if bool(rule(ev["outputs"][c])) != bool(ev["rule_abstain"][c])]
    if bad:
        raise SystemExit("%d stored rule flags differ from the rule detector, e.g. %s" % (len(bad), bad[:3]))


def check_frozen(d, look):
    """Question by question, the grid's question and baseline_x answer (text and finished) are the current frozen ones
    (look = the run's frozen lookup of the split)."""
    bad = [ev["pair_id"] for ev in d["evaluations"]
           if ev["pair_id"] not in look or ev.get("question") != look[ev["pair_id"]]["question"]
           or ev["outputs"]["baseline_x"] != look[ev["pair_id"]]["answers"]["x"]["text"]
           or ev["finished"]["baseline_x"] != look[ev["pair_id"]]["answers"]["x"]["finished"]]
    if bad:
        raise SystemExit("%d questions of the grid differ from the current frozen question / baseline_x answer, e.g. %s" % (
            len(bad), bad[:3]))


E_PART_KEYS = ("family", "layer", "kind", "beta", "direction_file_sha256", "direction_record_sha256",
               "training_frozen_file_sha256", "e_on", "e_off")


def check_e_alone(d, el, es, edesc):
    """Each E-alone condition stands on baseline_x and its single part is the write one_write builds for the probe
    (edesc["E<s>"]): family, layer, kind, beta, direction file / record / training-freeze hashes, E factor, amount."""
    specs = {c["name"]: c for c in d["condition_specs"]}                  # names are unique (load_grid)
    for s in es:
        name, mine = e_alone(el, s), edesc["E%s" % s]
        spec = specs.get(name)
        if spec is None or spec.get("base_condition") != "baseline_x":
            raise SystemExit("the grid's %s has no spec standing on baseline_x" % name)
        parts = spec.get("parts") or []
        if len(parts) != 1 or any(parts[0].get(k) != mine[k] for k in E_PART_KEYS) or \
                abs(float(parts[0]["amount"]) - float(mine["amount"])) > 1e-9:
            raise SystemExit("the grid's %s was made with another E write than the probe's" % name)


def check_sources(d, ctx, el, es):
    """Follow the export's sources: the frozen parent is the current frozen file, and each E-alone generation file is the
    exported one (same sha256) whose generation contract has the probe's arm, content positions, split, seed, token
    budget and script version (own_sha256), on this run identity and frozen file: the generation identity itself, not the
    condition name or the parent training identity."""
    src = d.get("sources") or {}
    sums, run, a = src.get("sha256") or {}, ctx.args.run_dir.resolve(), ctx.args
    if src.get("frozen") != ctx.frozen_name or sums.get(str(run / ctx.frozen_name)) != ctx.frozen_sha:
        raise SystemExit("the grid was exported from another frozen file than the current %s" % ctx.frozen_name)
    for s in es:
        arm = "x:E%d.w.p%s" % (el, s)
        name = "%s_tune.json" % X.arm_token(arm)
        path = run / "eval_extra" / "gen" / name
        if name not in (src.get("gen") or []) or not path.is_file() or sums.get(str(path)) != sha(path):
            raise SystemExit("%s is not the generation file the grid's %s was exported from, or it changed" % (name, e_alone(el, s)))
        c = X.e43.validate_artifact(json.loads(path.read_text(encoding="utf-8")), "extra generation").get("generation_contract") or {}
        want = {"arm": arm, "base_prompt": "x", "positions": "content", "split": "tune", "seed": a.seed,
                "max_new_tokens": a.max_new_tokens, "own_sha256": ctx.own_sha, "identity_sha256": ctx.ident_sha,
                "frozen_file_sha256": ctx.frozen_sha}
        bad = sorted(k for k, v in want.items() if c.get(k) != v)
        if bad:
            raise SystemExit("the generation contract of %s differs from this probe in %s" % (name, bad))


def one_write(ctx, arm):
    """(layer, write kwargs, description) of a one-part arm through the driver's verified path."""
    _, parsed = X.parse_arm(arm, ctx.all_systems)
    ws, positions = ctx.writes(parsed)
    if positions != "content" or len(ws) != 1:
        raise SystemExit("a one-part content-position write expected for %s" % arm)
    return ws[0]


class ActualWrite:
    """Forward hook around the driver's fd.ShiftHook: calls it once, returns its output unchanged (the write operator is
    not touched), and keeps what that same call actually stored, h_after.float() - h_before.float() [B, T, hidden], on
    the CPU (B1 of the GPT review: never re-drawn, never inferred from the nominal amount or from mean norms)."""

    def __init__(self, hook):
        self.hook, self.calls, self.delta = hook, 0, None

    def __call__(self, module, inp, out):
        before = (out[0] if isinstance(out, tuple) else out).detach().float().clone()
        new = self.hook(module, inp, out)
        after = new[0] if isinstance(new, tuple) else new
        self.calls += 1
        if self.delta is None:
            self.delta = (after.detach().float() - before).cpu()
        return new


@torch.no_grad()
def readouts(ctx, prompts, write, cls, wc):
    """(rows, actual): per question {CL: (coordinate at the last prompt token, mean coordinate over all prefill write
    positions)}, and for a write the same two readouts of the change the write itself stored at its layer projected on
    w_C (None without a write). write = None or (layer, write kwargs): a fd.ShiftHook on the prompt's write positions
    (the content mask), as on the prefill of generation (fd.generate_shift / generate_two). Every projection is done on
    the CPU in float32, the same way for the readouts and for the stored change."""
    res, act, bs = [], [], ctx.args.batch_size
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
        hook = rec = None
        with contextlib.ExitStack() as stack:
            if write is not None:
                layer, w = write
                hook = fd.ShiftHook(w["direction"], amounts=torch.full((len(chunk),), float(w["amount"])), mask=mask,
                                    decode_all=False, reference_direction=w.get("reference_direction"))
                rec = ActualWrite(hook)
                stack.enter_context(pd.clamp(ctx.bundle, layer, rec))
            for cl in cls:
                stack.enter_context(pd.clamp(ctx.bundle, cl, caps[cl]))
            forward_logits(ctx.bundle, {"input_ids": ids, "attention_mask": att}, grad=False)
        if hook is not None and (rec.calls != 1 or hook.applied != 1 or hook.prefill is None or hook.prefill["sequence_width"] != width):
            raise RuntimeError("the write did not fire exactly once on the prompt")
        m = mask.detach().bool().cpu()
        for cl in cls:
            h = caps[cl].h
            if h is None or tuple(h.shape[:2]) != (len(chunk), width) or not bool(torch.isfinite(h).all()):
                raise RuntimeError("no finite states captured at layer %d" % cl)
        for b in range(len(chunk)):
            if not bool(m[b, width - 1]):
                raise RuntimeError("the last prompt token is not a content position")
            row, arow = {}, {}
            for cl in cls:
                proj = caps[cl].h[b] @ wc[cl]
                row[cl] = (float(proj[width - 1]), float(proj[m[b]].mean()))
                if rec is not None:                      # the stored change on w_C, read at the same two places
                    dproj = rec.delta[b] @ wc[cl]
                    arow[cl] = (float(dproj[width - 1]), float(dproj[m[b]].mean()))
            res.append(row)
            act.append(arow)
    if len(res) != len(prompts):
        raise RuntimeError("readouts returned the wrong number of rows")
    return res, (act if write is not None else None)


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
    for s in args.es:
        if not re.fullmatch(r"[0-9]+(\.[0-9]+)?", s) or float(s) <= 0:
            raise SystemExit("E strength %r is not a positive number" % s)
    if len({float(s) for s in args.es}) != len(args.es):       # by value: 1 and 1.0 are the same strength
        raise SystemExit("an E strength is repeated")
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
    # Before the model is loaded. The output names (Context.name: t<number of questions>. for a partial run, whose
    # questions are the first --first-batches batches of the split) from the read-only question list of the source run.
    _, prepared = X.E.load_source(a)
    full = list(prepared["splits"]["tune"])
    partial = args.first_batches > 0
    stem = ("t%d." % len(full[:args.first_batches * a.batch_size]) if partial else "") + "probe_c_under_e_%s" % args.name
    out_dir = args.run_dir / "eval_extra"
    out_json, out_md = out_dir / (stem + ".json"), out_dir / (stem + ".md")
    if out_json.exists() or out_md.exists():
        raise SystemExit("%s exists" % (out_json if out_json.exists() else out_md))
    # The whole grid, also for a partial run: unique IDs and names, the full tune split, the stored rule flags.
    bl, grid_export, grid_sha = load_grid(args.grid, args.el, args.es)
    check_split(grid_export, full)
    recheck_flags(grid_export, args.el, args.es, X.rule_abstains)
    ctx = X.Context(a)
    if ctx.out_dir != out_dir or ctx.name("probe_c_under_e_%s" % args.name) != stem:
        raise RuntimeError("the output names derived before the model load differ from the run's")
    if out_json.exists() or out_md.exists():
        raise SystemExit("%s exists" % out_json)
    ids = [q["pair_id"] for q in ctx.questions]
    if grid_export.get("identity_sha256") != ctx.ident_sha:
        raise SystemExit("the grid was made by another run identity (%s; this run %s)" % (grid_export.get("identity_sha256"), ctx.ident_sha))
    miss = [q for q in ids if q not in bl]
    if miss:
        raise SystemExit("the grid lacks %d questions of this split, e.g. %s" % (len(miss), miss[:3]))
    check_frozen(grid_export, ctx.look)

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
    check_e_alone(grid_export, args.el, args.es, edesc)        # the grid's E-alone answers must come from the probe's write
    check_sources(grid_export, ctx, args.el, args.es)
    intended = {}                     # the nominal amount x cos(write direction, w_C) / gap_C: reported, never subtracted
    for key, wr in arms.items():
        if wr is not None:
            u = torch.as_tensor(wr[1]["direction"]).detach().float().cpu()
            u = u / u.norm()
            intended[key] = {cl: float(wr[1]["amount"]) * float(u @ wc[cl]) / cdesc[cl]["gap"] for cl in args.cls}
    prompts_x = ctx.render("x")
    log("%d questions; C layers %s; arms %s" % (len(ids), args.cls, list(arms)))
    rows, actual = {}, {}             # actual[key][j][cl]: the change the write stored on w_C (read, prefill-write mean)
    for key, wr in arms.items():
        rows[key], actual[key] = readouts(ctx, prompts_x, wr, args.cls, wc)
    for name in ("caut", "conf"):
        rows["prompt_" + name], _ = readouts(ctx, ctx.render(name), None, args.cls, wc)
    ctx.still_current()

    summary, lines = {}, []
    lines.append("# Probe: after the E write, does the prompt-forward C readout downstream change? (%s, run %s, E%d, tune %d questions)\n" % (
        args.name, args.run_dir.name, args.el, len(ids)))
    lines.append("delta = (coordinate under the write - without it) / gap_C, > 0 = toward caution; read = last prompt token; "
                 "prefill-write mean = mean over all prefill write positions (user content, chat template / assistant header, "
                 "last prompt token); actual direct = the change the write itself stored along w_C at the E layer (from the "
                 "hook's own call), read the same way; delta - actual direct = the remaining coordinate change at this layer, "
                 "not a mediation estimate; intended direct = the nominal amount x cos / gap_C, reference only. Prompt forward "
                 "only: decode positions are not read, so a null result holds for these positions only. Groups by what E alone "
                 "did (export of %s); already blank = blank without any write, including questions E then answers.\n" % args.grid.name)
    for cl in args.cls:
        g = cdesc[cl]["gap"]
        chk = {r: [(rows["prompt_caut"][j][cl][r] - rows["prompt_conf"][j][cl][r]) / g for j in range(len(ids))] for r in (0, 1)}
        if not sum(chk[0]) > 0:                  # a failed calibration stops the run: the C sign is never flipped to pass
            raise SystemExit("C layer %d: on these questions the cautious prompt does not sit above the confident one along +w_C" % cl)
        summary[cl] = {"gap": g, "caut_minus_conf_over_gap": {"read": mean_ci(chk[0]), "prefill_write_mean": mean_ci(chk[1])},
                       "intended_direct": {k: v[cl] for k, v in intended.items()}, "arms": {}}
        lines.append("\n## C layer %d (gap %.4g)\n" % (cl, g))
        lines.append("- check, caut minus conf over gap: read %s; prefill-write mean %s\n" % (fmt(mean_ci(chk[0])), fmt(mean_ci(chk[1]))))
        lines.append("| arm | intended direct | group | delta, read | actual direct, read | delta - actual direct, read | "
                     "delta, prefill-write mean | delta - actual direct, prefill-write mean |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for key in arms:
            if key == "none":
                continue
            s = key[1:] if key.startswith("E") else key[4:]
            groups = {"all": [], "newly blank": [], "still answered": [], "already blank": []}
            for j, q in enumerate(ids):
                d_read = (rows[key][j][cl][0] - rows["none"][j][cl][0]) / g
                d_mean = (rows[key][j][cl][1] - rows["none"][j][cl][1]) / g
                a_read, a_mean = actual[key][j][cl][0] / g, actual[key][j][cl][1] / g     # this question's actual direct
                b0, b1 = bl[q]["none"], bl[q]["E%s" % s]
                grp = "already blank" if b0 else ("newly blank" if b1 else "still answered")
                for gname in ("all", grp):
                    groups[gname].append((d_read, d_read - a_read, d_mean, d_mean - a_mean, a_read, a_mean))
            summary[cl]["arms"][key] = {}
            for grp, v in groups.items():
                c = tuple(mean_ci([x[r] for x in v]) for r in range(6))
                summary[cl]["arms"][key][grp] = {"read": c[0], "read_minus_direct": c[1], "prefill_write_mean": c[2],
                                                 "prefill_write_mean_minus_direct": c[3],
                                                 "actual_direct": {"read": c[4], "prefill_write_mean": c[5]}}
                lines.append("| %s | %+.3f | %s | %s | %s | %s | %s | %s |" % (key, intended[key][cl], grp, fmt(c[0]), fmt(c[4]),
                                                                           fmt(c[1]), fmt(c[2]), fmt(c[3])))
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
              "row_layout": {"coordinates": "[last prompt token, mean over all prefill write positions] of the block output on unit w_C",
                             "actual_direct": "[last prompt token, mean over all prefill write positions] of the change the write "
                                              "stored at the E layer (h_after - h_before) on unit w_C, divided by gap_C"},
              "c_directions": {str(k): v for k, v in cdesc.items()}, "e_writes": edesc,
              "summary": {str(k): v for k, v in summary.items()},
              "rows": [{"pair_id": q, "blank": bl[q], "coordinates": {key: {str(cl): rows[key][j][cl] for cl in args.cls} for key in rows},
                        "actual_direct": {key: {str(cl): [v / cdesc[cl]["gap"] for v in actual[key][j][cl]] for cl in args.cls}
                                          for key in actual if actual[key] is not None}}
                       for j, q in enumerate(ids)]}
    if sha(Path(args.grid) / "judge_input.json") != grid_sha:
        raise SystemExit("the grid changed during the run; nothing is written")
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("wrote %s and %s" % (out_json.name, out_md.name))


if __name__ == "__main__":
    main()
