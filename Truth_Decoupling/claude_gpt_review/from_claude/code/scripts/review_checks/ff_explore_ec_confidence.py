# -*- coding: utf-8 -*-
"""Confidence of the model in ITS OWN answers, for the selection comparators of from_claude/39, for any model and any
pair of E / C layers (Claude, 2026-11-13; from_claude/44). The Llama version is
results/09_E44_两因子交换训练_Llama/scripts/e44_confidence.py; this copy changes only what is model- or run-specific:
the code location (this file's own code copy), the run directories and truth-level flags (arguments; the C direction may
come from another run, as in e44_eval_extra.py --c-run-dir), and the target list, which is built from the E / C layers
and strengths instead of being written out for Llama's E10 / C31. Scoring, the three checks and the output format are
unchanged.

For every target (a prompt, zero / one / two writes, and a set of already generated answers) this re-runs the answers
teacher-forced under the SAME prompt and the SAME writes (core ShiftHook on the content positions, as fd.score_answers;
two writes = two hooks on their own layers, nested), and records per question the log-probability of every answer body
token (the end-of-turn token closing a complete answer is appended but not scored). Exact generated token IDs are used,
never re-tokenised text. Reads only; writes one JSON file under the E run's eval_extra/ directory.

Checks (enforced; red team 39 item 2):
  1. greedy: with the writes, the teacher-forced arg-max reproduces >= 98% of all generated answer tokens;
  2. first difference: against the answers of the same prompt WITHOUT the target's last write (the frozen no-write
     answers, or the E-alone answers for an E+C target), at the first token where the two answers differ (where the
     write tipped greedy decoding, often a near tie), the margin log p(target token) - log p(reference token) must be
     larger with the full writes than with the last write removed in >= 90% of those questions, and on average - a
     missing write leaves the margin unchanged and a wrong one moves it at random, so neither passes; the arg-max
     reproduction rates at those positions are reported (they are sensitive to bf16 noise at near ties);
  3. every write's family / layer / kind / beta / amount / direction file and record hashes equal the generation's.

Targets (answers from eval_extra/gen of the E run, or the frozen no-write answers):
  conf, x (no write); conf_E<s>, x_E<s> (E alone, each E strength s); x_C<k> (C alone toward confident, each k);
  x_E<s>C<k> (E + C, every pair; reference = E alone, so check 2 tests the C write).
Run with the model's own interpreter:
  ff_explore_ec_confidence.py OUT_NAME --source-run S --run-dir E_RUN --e-on E2 [--c-run-dir C_RUN --c-e-on E1]
                              --el 15 --cl 16 --es 1 2 --ks 0.25 0.5 1 2 [--first-batches K]"""
import argparse, contextlib, hashlib, json, sys, time
from pathlib import Path
CODE = Path(__file__).resolve().parents[2]          # <code copy>/scripts/review_checks/<this file>
sys.path.insert(0, str(CODE / "scripts"))
import torch  # noqa: E402
import e44_eval_extra as X  # noqa: E402
from core.model_io import forward_logits  # noqa: E402
pd, E, fd = X.pd, X.E, X.fd

def part_token(parts):
    """File-name token of the writes of an arm (e44_eval_extra.arm_token without the base prompt)."""
    return "+".join(parts).replace("+", "-and-").replace(".", "-")


def ordered(el, cl, e_part, c_part):
    """The two writes of an E+C arm, lower layer first (the grammar of e44_eval_extra.py)."""
    return [e_part, c_part] if el < cl else [c_part, e_part]


def build_targets(el, cl, es, ks):
    """(name, base prompt, writes, answers ("frozen" or an eval_extra generation file), reference for check 2)."""
    T = [("conf", "conf", [], "frozen", None), ("x", "x", [], "frozen", None)]
    for s in es:
        e = "E%d.w.p%s" % (el, s)
        T.append(("conf_E%s" % s, "conf", [e], "conf--%s_tune.json" % part_token([e]), ("frozen", [])))
        T.append(("x_E%s" % s, "x", [e], "x--%s_tune.json" % part_token([e]), ("frozen", [])))
    for k in ks:
        c = "C%d.w.m%s" % (cl, k)
        T.append(("x_C%s" % k, "x", [c], "x--%s_tune.json" % part_token([c]), ("frozen", [])))
    for s in es:
        e = "E%d.w.p%s" % (el, s)
        for k in ks:
            parts = ordered(el, cl, e, "C%d.w.m%s" % (cl, k))
            T.append(("x_E%sC%s" % (s, k), "x", parts, "x--%s_tune.json" % part_token(parts),
                      ("x--%s_tune.json" % part_token([e]), [e])))
    return T


MAX_TOKENS = 48          # the generation cap (max_new_tokens): no generated answer is longer, so nothing is truncated
BATCH = 8
GREEDY_MIN, FIRSTDIFF_MIN, FIRSTDIFF_N = 0.98, 0.90, 5
DESC_KEYS = ("family", "layer", "kind", "beta", "direction_file", "direction_file_sha256", "direction_record_sha256")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_answers(ctx, base, source, ids):
    if source == "frozen":
        return [ctx.look[pid]["answers"][base] for pid in ids], {"kind": "frozen", "prompt": base}, None
    path = ctx.out_dir / "gen" / source
    g = X.e43.validate_artifact(json.loads(path.read_text(encoding="utf-8")), "extra generation")
    if g.get("completed") is not True or g.get("partial") is not False or g.get("base_prompt") != base or g.get("split") != "tune":
        raise SystemExit("%s is not a complete tune generation on %s" % (source, base))
    gen_ids = [r["pair_id"] for r in g["rows"]]
    if len(set(gen_ids)) != len(gen_ids) or gen_ids[:len(ids)] != ids or (not ctx.partial and len(gen_ids) != len(ids)):
        raise SystemExit("%s covers other questions (or repeats one)" % source)
    rows = {r["pair_id"]: r for r in g["rows"]}
    return [rows[pid] for pid in ids], {"kind": "generation", "file": source, "file_sha256": sha(path), "arm": g["arm"]}, g


def make_writes(ctx, base, parts, gen):
    """[(layer, direction, amount)] and their descriptions, checked against the generation's parts when given."""
    if not parts:
        if gen is not None and gen.get("parts"):
            raise SystemExit("a no-write target cannot come from a generation with writes")
        return [], []
    _, parsed = X.parse_arm("%s:%s" % (base, "+".join(parts)), ctx.all_systems)
    ws, positions = ctx.writes(parsed)
    if positions != "content" or len(ws) != len(parts):
        raise SystemExit("content-position writes expected for %s" % parts)
    if gen is not None:
        if len(gen["parts"]) != len(ws):
            raise SystemExit("the generation has %d writes, the target %d" % (len(gen["parts"]), len(ws)))
        for (_, _, desc), gp in zip(ws, gen["parts"]):
            for k in DESC_KEYS:
                if desc.get(k) != gp.get(k):
                    raise SystemExit("write %s differs from the generation in %s: %r vs %r" % (parts, k, desc.get(k), gp.get(k)))
            if abs(float(desc["amount"]) - float(gp["amount"])) > 1e-9:
                raise SystemExit("write %s differs from the generation in amount" % parts)
    out = [(layer, w["direction"], float(w["amount"])) for layer, w, _ in ws]
    return out, [{k: d.get(k) for k in DESC_KEYS + ("amount",)} for _, _, d in ws]


def score(ctx, prompts, answers, writes, probes=None, batch=8):
    """Per question: (body log-probs [n], arg-max over the whole target incl. end-of-turn, target IDs incl. end-of-turn,
    probe) where probe = (log-prob of the target token, log-prob of the alternative token) at the probe position j for
    probes[i] = (j, alternative token) (None when there is no probe)."""
    res = []
    for i in range(0, len(answers), batch):
        chunk = [pd.build_target(ctx.bundle, prompts[j], answer_ids=answers[j]["token_ids"], finished=answers[j]["finished"],
                                 max_target_tokens=MAX_TOKENS, eot_ids=ctx.eot) for j in range(i, min(i + batch, len(answers)))]
        if any(c["truncated"] for c in chunk):
            raise SystemExit("a generated answer is longer than the cap")
        packed = pd.pack_with_content_mask(chunk, ctx.bundle.tokenizer.pad_token_id, ctx.bundle.device)
        hooks = [(layer, fd.ShiftHook(direction, amounts=torch.full((len(chunk),), amount),
                                      mask=fd.position_mask(packed, "content"), decode_all=False))
                 for layer, direction, amount in writes]
        with contextlib.ExitStack() as stack:
            for layer, hook in hooks:
                stack.enter_context(pd.clamp(ctx.bundle, layer, hook))
            logits = forward_logits(ctx.bundle, packed["inputs"], grad=False)
        if any(hook.applied != 1 for _, hook in hooks):
            raise RuntimeError("every write must fire exactly once per scoring forward")
        logp, rows_b, target_ids = fd.answer_token_logprobs(logits, packed)
        tok = logp.gather(1, target_ids[:, None]).squeeze(1)
        arg = logp.argmax(dim=-1)
        for b, item in enumerate(chunk):
            sel = rows_b == b
            own, own_arg, own_tgt = tok[sel], arg[sel].tolist(), target_ids[sel].tolist()
            body = item["answer_body_tokens"]
            if len(own_tgt) != item["answer_length"] or body < 1 or own_tgt[:body] != list(answers[i + b]["token_ids"])[:body]:
                raise RuntimeError("scored tokens do not match the answer")
            probe = None
            if probes is not None and probes[i + b] is not None:
                j, alt = probes[i + b]
                row_logp = logp[sel]
                probe = (float(row_logp[j, own_tgt[j]]), float(row_logp[j, alt]))
            res.append((own[:body].double(), own_arg, own_tgt, probe))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_name")
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True, help="the run of the E direction (its eval_extra/ holds the answers)")
    p.add_argument("--e-on", default="E1")
    p.add_argument("--c-run-dir", type=Path, default=None, help="the run of the C direction (default: --run-dir)")
    p.add_argument("--c-e-on", default="E1")
    p.add_argument("--el", type=int, required=True)
    p.add_argument("--cl", type=int, required=True)
    p.add_argument("--es", nargs="+", required=True)
    p.add_argument("--ks", nargs="+", required=True)
    p.add_argument("--first-batches", type=int, default=0)
    p.add_argument("--batch", type=int, default=BATCH, help="scoring batch size (default 8 = the generation's and Llama's); chain k: 1 only if a target misses the 98%% greedy check at 8, decided before any run")
    args = p.parse_args()
    if args.el == args.cl:
        raise SystemExit("the E and C writes act at two different layers")
    TARGETS = build_targets(args.el, args.cl, args.es, args.ks)
    if len({t[0] for t in TARGETS}) != len(TARGETS):
        raise SystemExit("duplicate target names (repeated strengths?)")
    P = Path(X.EXP_ROOT) / "configs" / "prompts"
    a = argparse.Namespace(
        phase="generate", source_run=args.source_run,
        run_dir=args.run_dir, prompts=P / "default_v1.json",
        prompts_e44=P / "e44_factorial_v1.json", prompts_extra=P / "e44_eval_extra_v1.json", e_on=args.e_on, e_off="none",
        c_run_dir=args.c_run_dir, c_e_on=args.c_e_on, c_e_off="none", arms=[], bases=[], family=None, layers=[], pairs=[],
        answers=["minus", "plus"], split="tune", seed=42, batch_size=BATCH, max_new_tokens=MAX_TOKENS,
        max_target_tokens=24, max_ref_tokens=48, output_projection="float32", first_batches=args.first_batches,
        gen=None, prompts_out=None, name="judge_input", overwrite=False, n_check=0, pair_ids=None)
    ctx = X.Context(a)
    out_path = ctx.out_dir / ctx.name("confidence_%s.json" % args.out_name)   # a partial run gets the t<N>. prefix
    if out_path.exists():
        raise SystemExit("%s exists" % out_path)
    ids = [q["pair_id"] for q in ctx.questions]
    eot = list(ctx.eot)
    report = {"protocol": "e44_confidence_v1", "split": "tune", "partial": ctx.partial, "question_ids": ids,
              "identity_sha256": ctx.ident_sha, "frozen_file_sha256": ctx.frozen_sha,
              "script_sha256": sha(__file__), "eval_extra_sha256": sha(X.__file__),
              "max_target_tokens": MAX_TOKENS, "scoring_batch_size": args.batch,
              "spec": {"e_layer": args.el, "c_layer": args.cl, "e_strengths": args.es, "c_strengths": args.ks,
                       "run_dir": str(args.run_dir), "c_run_dir": str(args.c_run_dir) if args.c_run_dir else None,
                       "e_on": args.e_on, "c_e_on": args.c_e_on},
              "targets": {}}
    with E.fp32_output_projection(ctx.bundle, enabled=True):
        for name, base, parts, source, reference in TARGETS:
            t0 = time.time()
            prompts = ctx.render(base)
            answers, src, gen = load_answers(ctx, base, source, ids)
            if gen is not None and gen.get("identity_sha256") != ctx.ident_sha:
                raise SystemExit("%s belongs to another run" % source)
            writes, descs = make_writes(ctx, base, parts, gen)
            probes, ref_answers = None, None
            if reference is not None:
                # first token where the target answer and the reference system's own answer differ (both incl. end-of-turn)
                ref_source, ref_parts = reference
                ref_answers, _, ref_gen = load_answers(ctx, base, ref_source, ids)
                probes = []
                for ans, ra in zip(answers, ref_answers):
                    tf = list(ans["token_ids"]) + (eot if ans["finished"] else [])
                    rf = list(ra["token_ids"]) + (eot if ra["finished"] else [])
                    j = next((k for k in range(min(len(tf), len(rf))) if tf[k] != rf[k]), None)
                    probes.append(None if j is None else (j, rf[j]))   # None: same answer, or one a capped prefix of the other
            full = score(ctx, prompts, answers, writes, probes, batch=args.batch)
            same = sum(sum(1 for x, y in zip(fa[:len(lp)], ft[:len(lp)]) if x == y) for lp, fa, ft, _ in full)
            total = sum(len(lp) for lp, _, _, _ in full)
            if same / total < GREEDY_MIN:
                raise RuntimeError("%s: teacher-forced arg-max reproduces only %.3f of the generated tokens" % (name, same / total))
            check = None
            if reference is not None:
                abl_writes, _ = make_writes(ctx, base, ref_parts, ref_gen)
                abl = score(ctx, prompts, answers, abl_writes, probes, batch=args.batch)
                n = toward = arg_full = arg_abl = 0
                shifts = []
                for pr, (_, fa, ft, pf), (_, aa, _, pa) in zip(probes, full, abl):
                    if pr is None:
                        continue
                    j = pr[0]
                    n += 1
                    shift = (pf[0] - pf[1]) - (pa[0] - pa[1])   # how much the removed write moves the target-vs-reference margin
                    shifts.append(shift)
                    toward += int(shift > 0)
                    arg_full += int(fa[j] == ft[j])
                    arg_abl += int(aa[j] == ft[j])
                check = {"reference": ref_source, "ablated_writes": ref_parts, "questions": n,
                         "share_toward_target": toward / n if n else None,
                         "mean_margin_shift": sum(shifts) / n if n else None,
                         "argmax_full_rate": arg_full / n if n else None, "argmax_ablated_rate": arg_abl / n if n else None}
                if n < FIRSTDIFF_N:
                    if not ctx.partial:
                        raise RuntimeError("%s: only %d answers differ from the reference; the write check needs %d" % (name, n, FIRSTDIFF_N))
                    check["note"] = "partial run: too few differing answers to enforce the check"
                elif toward / n < FIRSTDIFF_MIN or check["mean_margin_shift"] <= 0:
                    raise RuntimeError("%s: first-difference check failed %s" % (name, check))
            rows = []
            for pid, (lp, fa, ft, _), ans in zip(ids, full, answers):
                rows.append({"pair_id": pid, "n_tokens": len(lp), "finished": bool(ans["finished"]),
                             "refusal_rule": bool(X.rule_abstains(ans["text"])),
                             "text_sha1": hashlib.sha1(ans["text"].strip().encode("utf-8")).hexdigest(),
                             "sum_logp": float(lp.sum()), "mean_logp": float(lp.mean()),
                             "first_logp": float(lp[0]), "min_logp": float(lp.min())})
            report["targets"][name] = {"base_prompt": base, "writes": parts, "write_descriptions": descs, "source": src,
                                       "rows": rows, "greedy_reproduction": {"tokens": total, "same": same, "rate": same / total},
                                       "first_difference_check": check}
            print("%s %-7s base %-4s writes %-24s | greedy %d / %d (%.4f) | first-diff %s | mean of mean_logp %.4f | %.0f s" % (
                time.strftime("%H:%M:%S"), name, base, "+".join(parts) or "-", same, total, same / total,
                "-" if check is None or not check["questions"] else "%d q, toward target %.3f, margin shift %+.3f, argmax full %.3f / ablated %.3f" % (
                    check["questions"], check["share_toward_target"], check["mean_margin_shift"], check["argmax_full_rate"], check["argmax_ablated_rate"]),
                sum(r["mean_logp"] for r in rows) / len(rows), time.time() - t0), flush=True)
    ctx.still_current()
    out_path.write_text(json.dumps(report, ensure_ascii=False) + "\n", encoding="utf-8")
    print("saved", out_path)


if __name__ == "__main__":
    main()
