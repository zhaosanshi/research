#!/usr/bin/env python3
"""Evaluation-only additions to a finished E44 run (reply 32, sections 2.4 and 3; GPT round 33, sections 3-4).

Nothing here trains and no reviewed file changes: scripts/e44_factorial_das.py and core/ are imported as they are,
so the identity of the run (frozen answers, directions) stays the one it was made with, and every parent file is
read through the driver's own verified loaders. This script only
  - generates with writes the driver has no arm for (parts of the learned write; two writes at two layers),
  - generates on evaluation-only prompts (configs/prompts/e44_eval_extra_v1.json, e.g. conf_F),
  - reads (no intervention) the state difference between two prompts at the write positions, and
  - exports its own judge input (same layout as e44_export_judge.py).
Its files live in <run-dir>/eval_extra/ and record the parent identity, the sha256 of every parent file they used,
and the sha256 of this script and of its prompts file.

Phases
  selfcheck  the degenerate equalities of the two-write generation on the first batches of a split
  generate   one generation file per arm (--arms)
  natural    read-only: prompt-pair state differences at the write positions, projected on a trained direction
  export     one judge input over generation files of this script

Arm grammar (a base prompt, then one or two writes; two writes act at two different layers, lower layer first)
  <base>:<F><layer>.<kind>.<p|m><strength>[+<F><layer>.<kind>.<p|m><strength>]
  F = E or C (which trained direction of that layer); kind =
    w      learned direction,          amount = strength * gap             (the driver's fixed-shift arm)
    md     mean-difference direction,  amount = strength * md_norm         (the driver's md arm: the mean difference itself)
    mdg    mean-difference direction,  amount = strength * gap             (same norm as the w arm)
    wpar   mean-difference direction,  amount = strength * gap * cos       (the part of the w write parallel to the mean difference)
    wperp  unit(w - cos * w_md),       amount = strength * gap * |w - cos * w_md|   (the part orthogonal to it)
    rand   random direction, amount = strength * gap, stored norm matched to the w arm (the driver's random arm)
  wpar + wperp is exactly the w write. Examples:  conf:E10.wperp.p2   conf_F:E10.w.p2   x:E10.w.p2+C31.w.m0.5

The C direction may come from another finished run of the same identity (--c-run-dir with its own --c-e-on /
--c-e-off): a side branch trains only the E family, so its E direction is combined with the main run's C direction.
generate --bases NAME ... makes the no-write answers of evaluation-only prompts without any arm (prompt controls).

The model must be opened with the same --batch-size / --max-new-tokens / --max-target-tokens / --output-projection as
the parent run: they are part of its identity, and a mismatch stops at the first load.
"""
import argparse
import json
import math
import re
from pathlib import Path

import torch

import common  # noqa: F401
from common import EXP_ROOT
import e43_qa_prompt_das as e43
import e44_factorial_das as E
from core import qa_factorial_das as fd
from core import qa_prompt_das as pd
from core.model_io import forward_logits, greedy_generate
from core.readout import rule_abstains

PROTOCOL = "e44_eval_extra_v1"
KINDS = ("w", "md", "mdg", "wpar", "wperp", "rand")
PART_RE = re.compile(r"^(?P<fam>[EC])(?P<layer>\d+)\.(?P<kind>w|md|mdg|wpar|wperp|rand)\.(?P<sign>[pm])(?P<beta>\d+(?:\.\d+)?)$")
read, save, log, seal, file_sha = E.read, E.save, E.log, E.seal, E.file_sha


# ----------------------------------------------------------------------------- arms
def parse_arm(arm, known_prompts):
    """'<base>:<part>[+<part>]' -> (base, [ {family, layer, kind, beta}, ... ])."""
    if arm.count(":") != 1:
        raise SystemExit("arm %r is not <base>:<part>[+<part>]" % arm)
    base, body = arm.split(":")
    if base not in known_prompts:
        raise SystemExit("arm %r: unknown base prompt %s" % (arm, base))
    parts = []
    for text in body.split("+"):
        m = PART_RE.match(text)
        if not m:
            raise SystemExit("arm %r: part %r is not <E|C><layer>.<%s>.<p|m><strength>" % (arm, text, "|".join(KINDS)))
        beta = float(m["beta"]) * (1.0 if m["sign"] == "p" else -1.0)
        if beta == 0:
            raise SystemExit("arm %r: zero strength is the no-write control" % arm)
        parts.append({"family": m["fam"], "layer": int(m["layer"]), "kind": m["kind"], "beta": beta})
    if len(parts) > 2:
        raise SystemExit("arm %r: at most two writes" % arm)
    if len(parts) == 2 and not parts[0]["layer"] < parts[1]["layer"]:
        raise SystemExit("arm %r: two writes need two different layers, lower layer first" % arm)
    return base, parts


def arm_token(arm):
    return arm.replace(":", "--").replace("+", "-and-").replace(".", "-")


def split_learned_write(record):
    """The learned write per unit strength, gap * w, as (parallel, orthogonal) to the mean-difference direction:
    gap * w = (gap * cos) * w_md + gap * (w - cos * w_md). Returns unit vectors and amounts in float64."""
    if record.get("mean_difference_defined") is False:
        raise ValueError("this direction has no defined mean difference")
    w, w_md = torch.as_tensor(record["w"]).double(), torch.as_tensor(record["w_md"]).double()
    if w.ndim != 1 or w.shape != w_md.shape or not float(w.norm()) > 0 or not float(w_md.norm()) > 0:
        raise ValueError("direction record has malformed vectors")
    w, w_md = w / w.norm(), w_md / w_md.norm()
    gap = float(record["gap"])
    cos = float(w @ w_md)
    rest = w - cos * w_md
    rest_norm = float(rest.norm())
    if not rest_norm > 1e-6:
        raise ValueError("the learned direction has no part orthogonal to the mean difference")
    return {"w_md": w_md, "perp": rest / rest_norm, "cos": cos, "perp_norm": rest_norm,
            "par_amount": gap * cos, "perp_amount": gap * rest_norm}


def part_write(record, kind, beta, hidden_size, seed, layer):
    """Keyword arguments of one write for fd.generate_shift / generate_two."""
    if kind in ("w", "md", "rand"):
        return E.arm_write(record, kind, beta, hidden_size, seed, layer)            # the driver's own definition
    s = split_learned_write(record)
    if kind == "mdg":
        return {"direction": s["w_md"].float(), "amount": beta * float(record["gap"])}
    if kind == "wpar":
        return {"direction": s["w_md"].float(), "amount": beta * s["par_amount"]}
    if kind == "wperp":
        return {"direction": s["perp"].float(), "amount": beta * s["perp_amount"]}
    raise ValueError(kind)


# ----------------------------------------------------------------------------- two writes at two layers
@torch.no_grad()
def generate_two(bundle, prompts, parts, *, max_new_tokens, batch_size=8, positions="content"):
    """Greedy generation with two fixed writes, each at its own layer.

    parts = [(layer, write), (layer, write)], write = {"direction", "amount"[, "reference_direction"]} exactly as
    fd.generate_shift takes them. Masks, the decode rule (finished rows are no longer written) and the record
    layout are those of fd.generate_shift; rec["write"] is a list with one entry per part. The two hooks are
    independent fd.ShiftHook objects: nothing is averaged, renormalised or orthogonalised, and the lower layer
    acts first because the model runs it first (the upper write is added to a state the lower one already moved)."""
    if len(parts) != 2 or parts[0][0] == parts[1][0]:
        raise ValueError("generate_two needs two writes at two different layers")
    stops = pd.stop_ids(bundle)
    amounts = [fd._row_amounts(w["amount"], len(prompts)) for _, w in parts]
    if any(a is None for a in amounts):
        raise ValueError("each write needs an amount")
    results = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        inputs = bundle.encode([p["prefix_text"] for p in chunk])
        for b, p in enumerate(chunk):
            got = inputs["input_ids"][b][inputs["attention_mask"][b].bool()].tolist()
            if got != p["prefix_ids"]:
                raise RuntimeError("batched prompt tokens differ from the single-prompt rendering")
        mask = fd._prefill_mask(inputs, chunk, positions)
        decode = positions != "prompt"
        hooks = [fd.ShiftHook(w["direction"], amounts=torch.tensor(a[i:i + len(chunk)]), mask=mask, decode_all=decode,
                              reference_direction=w.get("reference_direction")) for (_, w), a in zip(parts, amounts)]
        active = torch.ones(len(chunk), dtype=torch.bool, device=bundle.device)

        def track_active_rows(model, args, kwargs, hooks=hooks, active=active, n=len(chunk)):
            # The previously generated token says which rows ended; both hooks share the one record of it.
            ids = kwargs.get("input_ids", args[0] if args else None)
            if hooks[0].calls and ids is not None:
                if ids.ndim != 2 or ids.shape[0] != n:
                    raise ValueError("shifted generation requires an unexpanded greedy batch")
                stopped = torch.zeros(n, dtype=torch.bool, device=ids.device)
                for token in stops:
                    stopped |= ids[:, -1] == token
                active.logical_and_(~stopped.to(active.device))
                for h in hooks:
                    h.active_rows = active

        pre = bundle.model.register_forward_pre_hook(track_active_rows, with_kwargs=True)
        try:
            with pd.clamp(bundle, parts[0][0], hooks[0]), pd.clamp(bundle, parts[1][0], hooks[1]):
                _, out = greedy_generate(bundle, inputs, max_new_tokens)
        finally:
            pre.remove()
        for h in hooks:
            if h.prefill is None or h.prefill["sequence_width"] != inputs["input_ids"].shape[1]:
                raise RuntimeError("a write did not fire on the prefill")
        if hooks[0].calls != hooks[1].calls or hooks[0].applied != hooks[1].applied:
            raise RuntimeError("the two writes did not fire on the same forwards")
        stats = [h.stats() for h in hooks]
        prompt_len = inputs["input_ids"].shape[1]
        for b, p in enumerate(chunk):
            ids, finished, stop_id = pd._cut_new_tokens(out[b][prompt_len:], stops)
            rec = {"text": bundle.tokenizer.decode(ids, skip_special_tokens=True).strip(), "token_ids": ids,
                   "finished": finished, "stop_id": stop_id, "n_tokens": len(ids), "write": []}
            own = len(ids) if finished else max(len(ids) - 1, 0)
            for (layer, _), st in zip(parts, stats):
                rels = [s[b] for s in st.get("decode_rel_steps", [])][:own]
                rec["write"].append({"layer": layer, "amount": st["prefill"]["amount"][b], "prefill_rel": st["prefill"]["rel"][b],
                                     "prefill_stored_norm": st["prefill"]["stored_norm"][b],
                                     "prefill_positions": st["prefill"]["positions"][b],
                                     "decode_steps_total": st.get("decode_steps", 0), "decode_steps_own": len(rels),
                                     "decode_rel_mean": (sum(rels) / len(rels)) if rels else None,
                                     "write_measurement": st["write_measurement"]})
            results.append(rec)
    return results


def same_answers(a, b):
    return len(a) == len(b) and all(x["token_ids"] == y["token_ids"] and x["finished"] == y["finished"]
                                    and x["stop_id"] == y["stop_id"] and x["text"] == y["text"] for x, y in zip(a, b))


# ----------------------------------------------------------------------------- context
class Context:
    """Everything a phase needs, loaded and verified once."""

    def __init__(self, args):
        self.args = args
        self.meta, self.prepared = E.load_source(args)
        self.cfg, self.systems = E.load_prompts(self.meta, args.prompts, args.prompts_e44)
        extra_cfg = read(args.prompts_extra)
        if extra_cfg.get("version") != PROTOCOL:
            raise SystemExit("extra prompts file version %r is not %s" % (extra_cfg.get("version"), PROTOCOL))
        qa = read(args.prompts)["qa"]
        bases = {"neutral": self.meta["config"]["qa_system"], "confident": qa["system_confident"], "cautious": qa["system_cautious"]}
        self.extra = {}
        for name, spec in extra_cfg["prompts"].items():
            if not re.fullmatch(r"[A-Za-z0-9_]+", name) or name in self.systems:
                raise SystemExit("extra prompt name %r is malformed or already a prompt of the run" % name)
            if spec["base"] not in bases or any(s not in self.cfg["sentences"] for s in spec["add"]):
                raise SystemExit("extra prompt %s refers to an unknown base or sentence" % name)
            self.extra[name] = " ".join([bases[spec["base"]]] + [self.cfg["sentences"][s] for s in spec["add"]])
        every = dict(self.systems, **self.extra)
        if len(set(every.values())) != len(every):
            raise SystemExit("an extra prompt renders the same system text as another prompt")
        self.all_systems = every
        self.user_template = self.meta["config"]["qa_user"]
        torch.manual_seed(args.seed)
        self.bundle = e43.open_model(self.meta)
        self.eot = pd.turn_terminator_ids(self.bundle)
        # the parent identity is computed with the run's own prompt set: extra prompts are not part of it
        self.ident, self.ident_sha = E.run_identity(self.meta, self.systems, self.cfg, args, self.eot, self.bundle)
        self.frozen, self.frozen_name = E.load_frozen(args, args.split, self.systems, self.ident_sha, self.prepared, allow_pilot=False)
        self.frozen_sha = file_sha(args.run_dir / self.frozen_name)
        self.look = e43.frozen_lookup(self.frozen)
        full = list(self.prepared["splits"][args.split])
        self.partial = args.first_batches > 0
        # a partial run keeps the batch composition of the full run, so the frozen answers still reproduce exactly
        self.questions = full[:args.first_batches * args.batch_size] if self.partial else full
        if not self.questions:
            raise SystemExit("no questions selected")
        self.out_dir = args.run_dir / "eval_extra"
        self.own_sha = {"script": file_sha(Path(__file__).resolve()), "prompts_extra": file_sha(args.prompts_extra)}
        self.rendered, self.base_cache, self.records = {}, {}, {}
        log("identity %s (parent run verified); %d questions of %s%s" % (
            self.ident_sha[:12], len(self.questions), args.split, " [partial: first batches only]" if self.partial else ""))

    def render(self, name):
        if name not in self.rendered:
            self.rendered[name] = [pd.render_prompt(self.bundle, self.all_systems[name], self.user_template, q["question"])
                                   for q in self.questions]
        return self.rendered[name]

    def record(self, family, layer):
        """(record, path, sha256) of a trained direction, through the driver's verified loader. The C family may live
        in another finished run of the same identity (--c-run-dir, with that run's own truth-level flags)."""
        if (family, layer) not in self.records:
            a = self.args
            if family == "C" and a.c_run_dir is not None:
                a = argparse.Namespace(**dict(vars(a), run_dir=a.c_run_dir, e_on=a.c_e_on, e_off=a.c_e_off))
            rec, path = E.load_direction(a, family, layer, self.ident_sha)
            self.records[(family, layer)] = (rec, path, file_sha(path))
        return self.records[(family, layer)]

    def name(self, stem):
        return ("t." if self.partial else "") + stem

    def base_answers(self, name):
        """No-write answers of a base prompt. A prompt of the run must reproduce its frozen answers exactly (the
        driver's own check). An evaluation-only prompt is generated once, saved, and must reproduce that file."""
        if name in self.base_cache:
            return self.base_cache[name]
        a = self.args
        if name in self.systems:
            recs = E.verified_base(a, self.bundle, {name: self.render(name)}, self.look, self.questions, name, {})
            rows = [{"text": r["text"], "token_ids": r["token_ids"], "finished": r["finished"], "stop_id": r["stop_id"],
                     "n_tokens": r["n_tokens"]} for r in recs]
            kind = "frozen"
        else:
            recs = pd.generate_prompts(self.bundle, self.render(name), max_new_tokens=a.max_new_tokens, batch_size=a.batch_size)
            rows = [{"text": r["text"], "token_ids": r["token_ids"], "finished": r["finished"], "stop_id": r["stop_id"],
                     "n_tokens": r["n_tokens"]} for r in recs]
            path = self.out_dir / self.name("base_%s_%s.json" % (name, a.split))
            payload = seal({"protocol": PROTOCOL, "phase": "base", "prompt": name, "system": self.all_systems[name],
                            "split": a.split, "partial": self.partial, "identity": self.ident, "identity_sha256": self.ident_sha,
                            "own_sha256": self.own_sha["prompts_extra"], "cohort_sha256": pd.fingerprint(self.questions),
                            "max_new_tokens": a.max_new_tokens, "question_ids": [q["pair_id"] for q in self.questions],
                            "rows": [dict(r, pair_id=q["pair_id"]) for r, q in zip(rows, self.questions)],
                            "summary": E.text_stats([r["text"] for r in rows]), "completed": not self.partial})
            if path.exists():
                if e43.validate_artifact(read(path), "extra base") != json.loads(json.dumps(payload)):
                    raise RuntimeError("no-write generation does not reproduce the saved %s answers" % name)
            else:
                save(path, payload)
                s = payload["summary"]
                log("base %s (evaluation-only prompt): refusals %d | sentence-end %d | words %.1f" % (
                    name, s["refusal_rule"], s["ends_with_period"], s["mean_words"]))
            kind = "extra"
        self.base_cache[name] = (rows, kind)
        return self.base_cache[name]

    def writes(self, parts):
        """[(layer, write kwargs, description dict)] for the parts of one arm; one write position rule for all."""
        out, positions = [], None
        for p in parts:
            rec, path, sha = self.record(p["family"], p["layer"])
            pos = rec["training_contract"]["positions"]
            if positions not in (None, pos):
                raise ValueError("the two directions were trained with different write positions")
            positions = pos
            w = part_write(rec, p["kind"], p["beta"], self.bundle.hidden_size, self.args.seed, p["layer"])
            desc = dict(p, amount=float(w["amount"]), gap=float(rec["gap"]), md_norm=float(rec["md_norm"]),
                        cosine_w_md=float(rec["cosine_w_md"]), direction_run=path.parent.parent.name, direction_file=path.name,
                        direction_file_sha256=sha,
                        direction_record_sha256=rec["artifact_sha256"], e_on=rec["e_on"], e_off=rec.get("e_off"))
            out.append((p["layer"], w, desc))
        return out, positions

    def generate(self, base, parts):
        ws, positions = self.writes(parts)
        a = self.args
        if len(ws) == 1:
            layer, w, _ = ws[0]
            recs = fd.generate_shift(self.bundle, self.render(base), max_new_tokens=a.max_new_tokens,
                                     batch_size=a.batch_size, layer=layer, positions=positions, **w)
        else:
            recs = generate_two(self.bundle, self.render(base), [(l, w) for l, w, _ in ws],
                                max_new_tokens=a.max_new_tokens, batch_size=a.batch_size, positions=positions)
        if len(recs) != len(self.questions):
            raise RuntimeError("generation returned the wrong number of answers")
        return recs, [d for _, _, d in ws], positions

    def still_current(self):
        E.require_current(self.ident, self.args)
        if file_sha(self.args.run_dir / self.frozen_name) != self.frozen_sha:
            raise RuntimeError("the frozen answers changed during execution")
        for _, path, sha in self.records.values():
            if file_sha(path) != sha:
                raise RuntimeError("a direction file changed during execution")
        if self.own_sha != {"script": file_sha(Path(__file__).resolve()), "prompts_extra": file_sha(self.args.prompts_extra)}:
            raise RuntimeError("this script or its prompts file changed during execution")


# ----------------------------------------------------------------------------- generate
def run_generate(ctx):
    a = ctx.args
    arms = [(arm,) + parse_arm(arm, ctx.all_systems) for arm in a.arms]
    for name in a.bases:                                   # prompt controls: no-write answers only
        if name not in ctx.all_systems:
            raise SystemExit("unknown prompt %s" % name)
        ctx.base_answers(name)
    for arm, base, parts in arms:
        out_path = ctx.out_dir / "gen" / ctx.name("%s_%s.json" % (arm_token(arm), a.split))
        base_rows, base_kind = ctx.base_answers(base)
        ws, positions = ctx.writes(parts)
        descs = [d for _, _, d in ws]
        contract = {"arm": arm, "base_prompt": base, "parts": descs, "split": a.split, "partial": ctx.partial,
                    "seed": a.seed, "positions": positions, "cohort_sha256": pd.fingerprint(ctx.questions),
                    "frozen_file_sha256": ctx.frozen_sha, "identity_sha256": ctx.ident_sha, "own_sha256": ctx.own_sha,
                    "max_new_tokens": a.max_new_tokens}
        if out_path.exists() and not a.overwrite:
            previous = e43.validate_artifact(read(out_path), "extra generation")
            if previous.get("generation_contract") != contract:
                raise ValueError("%s was made with another seed, subset, direction or script version" % out_path.name)
            log("skip %s (verified)" % out_path.name)
            continue
        recs, descs2, _ = ctx.generate(base, parts)
        if descs2 != descs:
            raise RuntimeError("write description changed between planning and generation")
        rows = []
        for q, rec, b in zip(ctx.questions, recs, base_rows):
            rec = dict(rec)
            rec.update({"pair_id": q["pair_id"], "question": q["question"], "refusal_rule": bool(rule_abstains(rec["text"])),
                        "baseline_text": b["text"], "baseline_refusal_rule": bool(rule_abstains(b["text"])),
                        "changed": rec["text"] != b["text"],
                        "changed_normalized": E.normalize_text(rec["text"]) != E.normalize_text(b["text"])})
            rows.append(rec)
        before, after = E.text_stats([r["baseline_text"] for r in rows]), E.text_stats([r["text"] for r in rows])
        summary = {"n": len(rows), "changed_normalized": sum(r["changed_normalized"] for r in rows), "before": before,
                   "after": after, "unfinished": sum(1 for r in rows if not r["finished"])}
        ctx.still_current()
        save(out_path, seal({"protocol": PROTOCOL, "phase": "generate", "arm": arm, "base_prompt": base, "base_kind": base_kind,
                             "parts": descs, "split": a.split, "positions": positions, "partial": ctx.partial,
                             "parent_protocol": fd.PROTOCOL, "identity": ctx.ident, "identity_sha256": ctx.ident_sha, "frozen_source": ctx.frozen_name,
                             "frozen_file_sha256": ctx.frozen_sha, "own_sha256": ctx.own_sha,
                             "cohort_sha256": pd.fingerprint(ctx.questions), "generation_contract": contract,
                             "baseline_reproduction_verified": True, "max_new_tokens": a.max_new_tokens,
                             "question_ids": [q["pair_id"] for q in ctx.questions], "summary": summary, "rows": rows,
                             "completed": not ctx.partial, "execution_complete": True}))
        log("%s: %s | changed %d | refusals %d -> %d | sentence-end %d -> %d | words %.1f -> %.1f" % (
            arm, "; ".join("%s%d %s %+.3f" % (d["family"], d["layer"], d["kind"], d["amount"]) for d in descs),
            summary["changed_normalized"], before["refusal_rule"], after["refusal_rule"],
            before["ends_with_period"], after["ends_with_period"], before["mean_words"], after["mean_words"]))


# ----------------------------------------------------------------------------- selfcheck
def run_selfcheck(ctx):
    """Acceptance of generate_two before any composition result is read (GPT round 33, section 4):
       both writes zero = the no-write answers; second zero = the single first write; first zero = the single second
       write - token for token - plus the same forwards, positions and finished-row rule for both hooks."""
    a = ctx.args
    if len(a.arms) != 1:
        raise SystemExit("selfcheck takes exactly one two-write arm")
    arm = a.arms[0]
    base, parts = parse_arm(arm, ctx.all_systems)
    if len(parts) != 2 or any(p["kind"] == "rand" for p in parts):
        raise SystemExit("selfcheck needs a two-write arm without a random part")
    base_rows, _ = ctx.base_answers(base)
    ws, positions = ctx.writes(parts)
    prompts = ctx.render(base)
    kw = {"max_new_tokens": a.max_new_tokens, "batch_size": a.batch_size, "positions": positions}
    zero = lambda w: {"direction": w["direction"], "amount": 0.0}  # noqa: E731
    (l0, w0, _), (l1, w1, _) = ws
    both_zero = generate_two(ctx.bundle, prompts, [(l0, zero(w0)), (l1, zero(w1))], **kw)
    if not same_answers(both_zero, base_rows):
        raise RuntimeError("selfcheck: two zero writes do not reproduce the no-write answers")
    single0 = fd.generate_shift(ctx.bundle, prompts, layer=l0, **w0, **kw)
    first_only = generate_two(ctx.bundle, prompts, [(l0, w0), (l1, zero(w1))], **kw)
    if not same_answers(first_only, single0):
        raise RuntimeError("selfcheck: (first write, zero) differs from the single first write")
    single1 = fd.generate_shift(ctx.bundle, prompts, layer=l1, **w1, **kw)
    second_only = generate_two(ctx.bundle, prompts, [(l0, zero(w0)), (l1, w1)], **kw)
    if not same_answers(second_only, single1):
        raise RuntimeError("selfcheck: (zero, second write) differs from the single second write")
    both = generate_two(ctx.bundle, prompts, [(l0, w0), (l1, w1)], **kw)
    for i, (r2, r0, r1) in enumerate(zip(first_only, single0, both)):
        for k in ("amount", "prefill_positions", "decode_steps_own"):
            if r2["write"][0][k] != r0["write"][k]:
                raise RuntimeError("selfcheck: row %d, the first hook's %s differs from the single-write run" % (i, k))
        if abs(r2["write"][0]["prefill_stored_norm"] - r0["write"]["prefill_stored_norm"]) > 1e-6:
            raise RuntimeError("selfcheck: row %d, the first hook stored a different write" % i)
        if r1["write"][0]["prefill_positions"] != r1["write"][1]["prefill_positions"] or \
                r1["write"][0]["decode_steps_own"] != r1["write"][1]["decode_steps_own"]:
            raise RuntimeError("selfcheck: row %d, the two hooks wrote at different positions or steps" % i)
        if any(abs(w["amount"] - d["amount"]) > 1e-6 * max(1.0, abs(d["amount"])) for w, (_, _, d) in zip(r1["write"], ws)):
            raise RuntimeError("selfcheck: row %d, a recorded amount differs from the planned one" % i)
    n = len(prompts)
    changed = lambda rows: sum(1 for r, b in zip(rows, base_rows) if r["token_ids"] != b["token_ids"])  # noqa: E731
    ctx.still_current()
    log("selfcheck passed on %d questions of %s: zero/zero = no write; (first, zero) = single first; (zero, second) = single second" % (n, a.split))
    log("answers changed vs no write: first alone %d, second alone %d, both %d (of %d); write norms %s" % (
        changed(single0), changed(single1), changed(both), n,
        ", ".join("L%d %.3f" % (l, abs(d["amount"])) for l, _, d in ws)))


# ----------------------------------------------------------------------------- natural differences (read-only)
class Capture:
    def __init__(self):
        self.h = None

    def __call__(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        self.h = h.detach().float().cpu()


def layer_states(bundle, layer, encoded):
    """Block outputs of `layer` for a packed teacher-forced batch -> ([B, T, hidden] float32 cpu, packed)."""
    packed = pd.pack_with_content_mask(encoded, bundle.tokenizer.pad_token_id, bundle.device)
    cap = Capture()
    with pd.clamp(bundle, layer, cap):
        forward_logits(bundle, packed["inputs"], grad=False)
    if cap.h is None or not bool(torch.isfinite(cap.h).all()):
        raise RuntimeError("no finite states captured at layer %d" % layer)
    return cap.h, packed


def natural_rows(bundle, layer, prompts_plus, prompts_minus, answers, eot, max_target_tokens, batch_size):
    """Per question: the state difference (plus prompt minus minus prompt) at every content position, the two
    prompts carrying the SAME question and the SAME teacher-forced answer tokens. The content span (first
    user-content token .. end) is the write span of the trained directions; it is right-aligned in both batches.
    Returns [(D [K, hidden] float32, n_prompt_content)] with None where a question cannot be used, and reasons."""
    out, skipped = [], {}
    items = []
    for pp, pm, ans in zip(prompts_plus, prompts_minus, answers):
        try:
            if ans is None:
                raise ValueError("unusable answer")
            ep = pd.build_target(bundle, pp, answer_ids=ans["token_ids"], finished=ans["finished"], max_target_tokens=max_target_tokens, eot_ids=eot)
            em = pd.build_target(bundle, pm, answer_ids=ans["token_ids"], finished=ans["finished"], max_target_tokens=max_target_tokens, eot_ids=eot)
            tail_p, tail_m = ep["full_ids"][ep["content_start"]:], em["full_ids"][em["content_start"]:]
            if tail_p != tail_m:
                raise ValueError("content spans differ")
            items.append((ep, em, len(tail_p), ep["prompt_length"] - ep["content_start"]))
        except ValueError as e:
            skipped[str(e)] = skipped.get(str(e), 0) + 1
            items.append(None)
    for i in range(0, len(items), batch_size):
        chunk = [(j, it) for j, it in enumerate(items[i:i + batch_size], start=i) if it is not None]
        res = {}
        if chunk:
            hp, _ = layer_states(bundle, layer, [it[0] for _, it in chunk])
            hm, _ = layer_states(bundle, layer, [it[1] for _, it in chunk])
            for row, (j, it) in enumerate(chunk):
                k = it[2]
                res[j] = (hp[row, hp.shape[1] - k:, :] - hm[row, hm.shape[1] - k:, :], it[3])
        out.extend(res.get(j) for j in range(i, min(i + batch_size, len(items))))
    return out, skipped


def segment_stats(rows, vectors):
    """rows: [(D [K, hidden], n_prompt_content) or None]. Segments: question (content before the read position),
    read (last prompt token = where training read the difference), answer (every answer position).
    Per segment and per named unit vector: the per-question mean coordinate, then mean / sd / share positive over
    questions; plus the mean difference norm and the cosine between the pooled mean difference and each vector."""
    segs = {"question": lambda n, k: slice(0, n - 1), "read": lambda n, k: slice(n - 1, n), "answer": lambda n, k: slice(n, k)}
    out = {}
    for name, cut in segs.items():
        per_q = {v: [] for v in vectors}
        norms, pooled, count = [], None, 0
        for item in rows:
            if item is None:
                continue
            D, n = item
            part = D[cut(n, D.shape[0])]
            if part.shape[0] == 0:
                continue
            for v, u in vectors.items():
                per_q[v].append(float((part @ u).mean()))
            norms.append(float(part.norm(dim=-1).mean()))
            s = part.sum(0).double()
            pooled = s if pooled is None else pooled + s
            count += part.shape[0]
        entry = {"questions": len(norms), "positions": count, "difference_norm_mean": (sum(norms) / len(norms)) if norms else None}
        for v, values in per_q.items():
            if values:
                t = torch.tensor(values, dtype=torch.float64)
                entry[v] = {"mean": float(t.mean()), "sd": float(t.std()) if len(values) > 1 else 0.0,
                            "min": float(t.min()), "max": float(t.max()), "share_positive": float((t > 0).double().mean())}
        if pooled is not None and float(pooled.norm()) > 0:
            m = (pooled / count).float()
            entry["pooled_mean_norm"] = float(m.norm())
            entry["cosine_of_pooled_mean"] = {v: float(pd.unit(m) @ u) for v, u in vectors.items()}
        out[name] = entry
    return out


def run_natural(ctx):
    a = ctx.args
    if not (a.family and a.layers and a.pairs):
        raise SystemExit("natural needs --family, --layers and --pairs plus:minus ...")
    pairs = []
    for text in a.pairs:
        if text.count(":") != 1 or any(n not in ctx.systems for n in text.split(":")):
            raise SystemExit("pair %r is not <plus prompt>:<minus prompt> over prompts of the run" % text)
        pairs.append(tuple(text.split(":")))
    report = {"protocol": PROTOCOL, "phase": "natural", "family": a.family, "split": a.split, "partial": ctx.partial,
              "identity": ctx.ident, "identity_sha256": ctx.ident_sha, "own_sha256": ctx.own_sha,
              "frozen_file_sha256": ctx.frozen_sha, "question_ids": [q["pair_id"] for q in ctx.questions], "layers": {}}
    for layer in a.layers:
        rec, path, sha = ctx.record(a.family, layer)
        s = split_learned_write(rec)
        vectors = {"w": pd.unit(torch.as_tensor(rec["w"]).float()), "w_md": s["w_md"].float(), "perp": s["perp"].float()}
        spec = E.family_spec(a.family, rec["e_on"], rec.get("e_off"))
        train_ids = set(rec["training_contract"].get("train_ids") or rec.get("planned_train_ids") or [])
        entry = {"gap": float(rec["gap"]), "md_norm": float(rec["md_norm"]), "cosine_w_md": s["cos"],
                 "write_parallel_amount": s["par_amount"], "write_orthogonal_amount": s["perp_amount"],
                 "direction_file_sha256": sha, "pairs": {}}
        sanity = []
        for plus, minus in pairs:
            for source in a.answers:
                name = plus if source == "plus" else minus
                answers = []
                for q in ctx.questions:
                    r = ctx.look[q["pair_id"]]["answers"][name]
                    answers.append(None if E.usable_answer(r, ctx.eot) else r)
                rows, skipped = natural_rows(ctx.bundle, layer, ctx.render(plus), ctx.render(minus), answers, ctx.eot,
                                             a.max_target_tokens, a.batch_size)
                stats = segment_stats(rows, vectors)
                entry["pairs"]["%s:%s|answer=%s" % (plus, minus, source)] = {"skipped": skipped, "segments": stats}
                if (plus, minus) in spec["plus_minus"] and a.split == "train" and not ctx.partial and train_ids:
                    kept = [r for q, r in zip(ctx.questions, rows) if q["pair_id"] in train_ids and r is not None]
                    sanity.append(segment_stats(kept, {"w": vectors["w"]})["read"]["w"]["mean"])
                for seg in ("question", "read", "answer"):
                    e = stats[seg]
                    if e["questions"]:
                        log("L%d %s:%s answer=%s %-8s n=%d | d.w %+.3f (sd %.3f, >0 %.2f) = %.2f x gap | d.w_md %+.3f | d.perp %+.3f = %.2f x written perp | |d| %.3f | cos(mean d, w) %+.3f" % (
                            layer, plus, minus, source, seg, e["questions"], e["w"]["mean"], e["w"]["sd"], e["w"]["share_positive"],
                            e["w"]["mean"] / entry["gap"] if entry["gap"] else float("nan"), e["w_md"]["mean"], e["perp"]["mean"],
                            e["perp"]["mean"] / s["perp_amount"], e["difference_norm_mean"], e.get("cosine_of_pooled_mean", {}).get("w", float("nan"))))
        if sanity and len(sanity) == len(spec["plus_minus"]) * len(a.answers):
            got = sum(sanity) / len(sanity)
            entry["read_position_check"] = {"mean_coordinate": got, "gap": entry["gap"]}
            if abs(got - entry["gap"]) > 0.02 * abs(entry["gap"]) + 1e-3:
                raise RuntimeError("L%d: the read-position coordinate %.4f does not reproduce the stored gap %.4f" % (layer, got, entry["gap"]))
            log("L%d read-position check: mean coordinate %.4f = stored gap %.4f" % (layer, got, entry["gap"]))
        elif a.split == "train" and not ctx.partial:
            log("L%d read-position check not run: give every training pair of the family (%s)" % (
                layer, ", ".join("%s:%s" % p for p in spec["plus_minus"])))
        report["layers"][str(layer)] = entry
    ctx.still_current()
    out = ctx.out_dir / ctx.name("natural_%s_%s.json" % (a.family, a.split))
    if out.exists() and not a.overwrite:
        raise SystemExit("%s exists (use --overwrite)" % out.name)
    save(out, seal(dict(report, completed=not ctx.partial)))
    log("saved %s" % out.name)


# ----------------------------------------------------------------------------- export (no model)
def run_export(args):
    """One judge input over generation files of this script; layout of e44_export_judge.py."""
    from core.io_utils import atomic_json_dump
    run_dir = args.run_dir.resolve()
    runs = {run_dir.name: run_dir}
    if args.c_run_dir is not None:
        runs[args.c_run_dir.resolve().name] = args.c_run_dir.resolve()
    out_dir = run_dir / "eval_extra"
    questions = read(args.source_run / "questions.json")["splits"][args.split]
    frozen_path = run_dir / ("frozen_%s.json" % args.split)
    frozen = read(frozen_path)
    E.validate_frozen(frozen, questions, frozen["prompt_systems"], require_full=True)
    if frozen["split"] != args.split:
        raise SystemExit("frozen artifact declares a different split")
    if frozen["identity"].get("source_files_sha256") != {n: file_sha(args.source_run / n) for n in ("bank.json", "questions.json")}:
        raise SystemExit("frozen answers use different source questions or bank")
    files = sorted((out_dir / "gen").glob("*_%s.json" % args.split)) if not args.gen else [out_dir / "gen" / n for n in args.gen]
    files = [p for p in files if not p.name.startswith("t.")]
    if not files or any(not p.is_file() for p in files):
        raise SystemExit("no (or missing) full generation files to export")
    before = {str(p): file_sha(p) for p in [frozen_path, *files]}
    gens, bases = [], {}
    for path in files:
        d = e43.validate_artifact(read(path), "extra generation")
        if d.get("completed") is not True or d.get("partial") is not False or d.get("phase") != "generate" or d.get("split") != args.split:
            raise SystemExit("partial / foreign file cannot enter a judge export: " + path.name)
        if d["identity_sha256"] != frozen["identity_sha256"] or d["frozen_file_sha256"] != before[str(frozen_path)]:
            raise SystemExit("generation and frozen answers belong to different runs: " + path.name)
        if d["question_ids"] != [q["pair_id"] for q in questions]:
            raise SystemExit("generation covers other questions: " + path.name)
        for part in d["parts"]:
            if part["direction_run"] not in runs:
                raise SystemExit("%s uses a direction of run %s; pass that run as --c-run-dir" % (path.name, part["direction_run"]))
            dpath = runs[part["direction_run"]] / "dirs" / part["direction_file"]
            if file_sha(dpath) != part["direction_file_sha256"]:
                raise SystemExit("direction file changed after generation: " + path.name)
        gens.append(d)
        if d["base_kind"] == "extra" and d["base_prompt"] not in bases:
            bpath = out_dir / ("base_%s_%s.json" % (d["base_prompt"], args.split))
            b = e43.validate_artifact(read(bpath), "extra base")
            if b.get("completed") is not True or b["identity_sha256"] != frozen["identity_sha256"] or b["question_ids"] != d["question_ids"]:
                raise SystemExit("extra base answers are partial or foreign: " + bpath.name)
            before[str(bpath)] = file_sha(bpath)
            bases[d["base_prompt"]] = {r["pair_id"]: r for r in b["rows"]}
    prompt_names = sorted({d["base_prompt"] for d in gens} | set(args.prompts_out or []))
    for p in prompt_names:
        if p not in frozen["prompt_systems"] and p not in bases:
            bpath = out_dir / ("base_%s_%s.json" % (p, args.split))
            b = e43.validate_artifact(read(bpath), "extra base")
            if b.get("completed") is not True or b["identity_sha256"] != frozen["identity_sha256"]:
                raise SystemExit("extra base answers are partial or foreign: " + bpath.name)
            before[str(bpath)] = file_sha(bpath)
            bases[p] = {r["pair_id"]: r for r in b["rows"]}
    conditions = ["baseline_" + p for p in prompt_names]
    specs = [{"name": "baseline_" + p, "base_condition": "baseline_" + p, "kind": "prompt_control",
              "uses_reference_answer": False} for p in prompt_names]
    lookups = []
    for d in gens:
        condition = "X_" + arm_token(d["arm"])
        conditions.append(condition)
        specs.append({"name": condition, "base_condition": "baseline_" + d["base_prompt"], "arm": d["arm"], "parts": d["parts"],
                      "kind": "intervention", "uses_reference_answer": False})
        lookups.append((condition, d["base_prompt"], e43._unique_questions(d["rows"])))
    if len(set(conditions)) != len(conditions):
        raise SystemExit("duplicate conditions")
    rows = []
    for question, fq in zip(questions, frozen["questions"]):
        pid = question["pair_id"]
        outputs, finished, rule = {}, {}, {}
        base_rec = {}
        for p in prompt_names:
            rec = fq["answers"][p] if p in frozen["prompt_systems"] else bases[p][pid]
            base_rec[p] = rec
            outputs["baseline_" + p], finished["baseline_" + p], rule["baseline_" + p] = rec["text"], rec["finished"], bool(rule_abstains(rec["text"]))
        for condition, base, lookup in lookups:
            row = lookup[pid]
            if row.get("baseline_text") != base_rec[base]["text"]:
                raise SystemExit("generation comparison baseline differs from the base answers: " + condition)
            outputs[condition], finished[condition], rule[condition] = row["text"], row["finished"], bool(rule_abstains(row["text"]))
        rows.append({"pair_id": pid, "question": question["question"], "gold": list(question["gold"]),
                     "incorrect": list(question.get("incorrect") or []), "best_answer": question.get("best_answer"),
                     "outputs": outputs, "finished": finished, "rule_abstain": rule})
    payload = {"completed": True, "model": frozen["model"], "protocol": frozen["protocol"], "export_protocol": PROTOCOL,
               "split": args.split, "identity": frozen["identity"], "identity_sha256": frozen["identity_sha256"],
               "conditions": conditions, "condition_specs": specs, "question_ids": [q["pair_id"] for q in questions],
               "sources": {"frozen": frozen_path.name, "gen": [p.name for p in files], "sha256": before}, "evaluations": rows}
    from scripts.review_checks.judge_two_axis_run import _derive_cells      # the exact downstream schema check
    _derive_cells(payload)
    for path, digest in before.items():
        if file_sha(path) != digest:
            raise RuntimeError("source artifact changed during judge export: " + path)
    out = out_dir / ("judge_%s" % args.split) / (args.name + ".json")
    if out.exists():
        if read(out) != payload:
            raise SystemExit("existing judge input differs; use a new --name, never overwrite judged inputs")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(payload, str(out))
    log("exported %s: %d questions, %d conditions" % (out, len(rows), len(conditions)))


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=("selfcheck", "generate", "natural", "export"))
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True, help="the finished E44 run directory (read; this script writes only eval_extra/)")
    p.add_argument("--prompts", type=Path, default=Path(EXP_ROOT) / "configs" / "prompts" / "default_v1.json")
    p.add_argument("--prompts-e44", type=Path, default=Path(EXP_ROOT) / "configs" / "prompts" / "e44_factorial_v1.json")
    p.add_argument("--prompts-extra", type=Path, default=Path(EXP_ROOT) / "configs" / "prompts" / "e44_eval_extra_v1.json")
    p.add_argument("--e-on", default="E1")
    p.add_argument("--e-off", default="none", choices=("none", "M"))
    p.add_argument("--c-run-dir", type=Path, default=None, help="take C directions from this finished run (same identity) instead of --run-dir")
    p.add_argument("--c-e-on", default="E1", help="truth-level flags of the run given by --c-run-dir")
    p.add_argument("--c-e-off", default="none", choices=("none", "M"))
    p.add_argument("--arms", nargs="+", default=[])
    p.add_argument("--bases", nargs="+", default=[], help="generate: also make the no-write answers of these prompts (prompt controls)")
    p.add_argument("--family", choices=E.FAMILIES, help="natural: the trained direction to project on")
    p.add_argument("--layers", nargs="+", type=int, default=[], help="natural: layers")
    p.add_argument("--pairs", nargs="+", default=[], help="natural: <plus prompt>:<minus prompt> ...")
    p.add_argument("--answers", nargs="+", default=["minus", "plus"], choices=("minus", "plus"),
                   help="natural: whose frozen answer is teacher-forced under both prompts")
    p.add_argument("--split", default="tune", choices=("train", "tune"))
    p.add_argument("--seed", type=int, default=42, help="random-direction seed (the driver's --seed)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--max-target-tokens", type=int, default=24)
    p.add_argument("--max-ref-tokens", type=int, default=48)
    p.add_argument("--output-projection", default="float32", choices=("float32", "native"))
    p.add_argument("--first-batches", type=int, default=0, help="partial run on the first K batches (writes t.* files, never exported)")
    p.add_argument("--gen", nargs="*", default=None, help="export: generation file names (default: every full file of the split)")
    p.add_argument("--prompts-out", nargs="*", default=None, help="export: extra prompt controls to include")
    p.add_argument("--name", default="judge_input", help="export: output stem")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    args.n_check, args.pair_ids = 0, None                 # the driver's pilot switches: never used here
    if min(args.batch_size, args.max_new_tokens, args.max_target_tokens, args.max_ref_tokens) < 1 or args.first_batches < 0:
        raise SystemExit("counts must be positive; --first-batches nonnegative")
    if len(args.arms) != len(set(args.arms)) or len(args.layers) != len(set(args.layers)):
        raise SystemExit("duplicate arms or layers")
    if not math.isfinite(float(args.seed)):
        raise SystemExit("bad seed")
    if args.phase == "export":
        run_export(args)
        return
    if (args.phase == "selfcheck" and not args.arms) or (args.phase == "generate" and not (args.arms or args.bases)):
        raise SystemExit("%s needs --arms%s" % (args.phase, " or --bases" if args.phase == "generate" else ""))
    ctx = Context(args)
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    (ctx.out_dir / "gen").mkdir(exist_ok=True)
    with E.fp32_output_projection(ctx.bundle, enabled=args.output_projection == "float32"):
        {"selfcheck": run_selfcheck, "generate": run_generate, "natural": run_natural}[args.phase](ctx)


if __name__ == "__main__":
    main()
