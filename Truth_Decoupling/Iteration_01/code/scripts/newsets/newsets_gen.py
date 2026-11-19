"""Generation on a new question list with the frozen TruthfulQA directions (work queue C1 / C3 and F0 / F1).

Nothing here trains, and no core or engine file changes. The model, its directions and their identity checks are
loaded through the engine's own Context (scripts/e44_eval_extra.py of the model's code copy), built exactly as
scripts/review_checks/ff_explore_ec_confidence.py builds it, on the run's own TruthfulQA tune split. The new prompts
are rendered with the same renderer (pd.render_prompt) and generated with the same functions the engine uses
(pd.generate_prompts, fd.generate_shift, X.generate_two), inside the same float32 output projection, batch size 8.
Writes act on the content positions (the user turn, the assistant header and every generated token).

Phases (run with the model's own interpreter; paths are relative to the project root /workspace/Truth_Decoupling):
  repro   on the first --first-batches batches of the TruthfulQA tune split: regenerate the no-write answers of
          prompt x, the E-alone arm and the E+C arm at C 0.25, and require token-for-token equality with the run's
          frozen answers and its saved eval_extra/gen files (same batch composition). Run once per model before gen.
  gen     one generation file per arm over a task's items:
            --task triviaqa|popqa  items = the neutral bank (newsets_bank.py) rows of --split, user turn = the
                                   run's own template "Question: {question}\\nAnswer:" (read from bank.json)
            --task mmlu|gsm8k      items = the raw parquet rows listed in --ids, user turn = cap_prompts.py
          arms: a base prompt name for no-write answers (x, caut, conf, conf_E2, conf_FR, neutral), or the engine's
          arm grammar <base>:<part>[+<part>] (e.g. x:E11.w.p2+C19.w.m0.25).
          --only-ids FILE restricts the items (used for N1 re-asks on MMLU / GSM8K; batches then differ from a
          full pass, which is recorded in the file).
  export  (newsets only) one judge input per split in the engine's export layout, condition names as the engine
          names them (baseline_<base>, <F>_L<layer>_<base>-<kind>-<p|m><beta>, X_<arm token>); --pro100 keeps only
          the fixed 100 test questions (the DeepSeek Pro cohort). gold = the first GOLD_CAP (10) of the alias list
          (canonical first), incorrect = []; the judge is the unchanged two-axis rubric (two_axis_v6).

Output files record the parent identity, frozen-file and direction hashes, the items file hash, this script's and the
engine's sha256, max_new_tokens and batch size. Existing files are never overwritten.
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/Truth_Decoupling")
PROTOCOL = "newsets_gen_v1"
BATCH = 8
ENGINE_MAX_NEW = 48          # the parent runs' identity value; Context must be built with it
FR_SENTENCE = ("The answer that first comes to mind may belong to a different but similarly named person, place or work. "
               "Give the answer for the one the question asks about.")
NEUTRAL_SYSTEM = "You are a helpful assistant."
GOLD_CAP = 10              # references shown to the Pro judge: canonical answer first, then aliases in file order


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def arm_token(arm):
    return arm.replace(":", "--").replace("+", "-and-").replace(".", "-")


def condition_name(arm, parts):
    """The engine's names (e44_eval_extra.condition_of / run_export): no write -> baseline_<base>; one write ->
    <F>_L<layer>_<base>-<kind>-<p|m><beta as written>; two writes -> X_<arm token>."""
    if ":" not in arm:
        return "baseline_" + arm
    base, body = arm.split(":")
    if len(parts) == 1:
        p = body
        fam, rest = p[0], p[1:]
        layer, kind, sb = rest.split(".", 2)
        return "%s_L%s_%s" % (fam, layer, ("%s.%s.%s" % (base, kind, sb)).replace(".", "-"))
    return "X_" + arm_token(arm)


def load_models(path, name):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    if name not in cfg.get("models", {}):
        raise SystemExit("model %s not in %s" % (name, path))
    m = dict(cfg["models"][name])
    for k in ("code", "source_run", "run_dir", "e_on", "el", "cl"):
        if k not in m:
            raise SystemExit("model %s: missing %s" % (name, k))
    if not m["el"] < m["cl"]:
        raise SystemExit("the engine's two-write grammar needs the E layer below the C layer")
    return m


def engine(m):
    code = ROOT / m["code"]
    sys.path.insert(0, str(code / "scripts"))
    import e44_eval_extra as X   # noqa: E402  (the model's own code copy)
    if Path(X.__file__).resolve().parent != (code / "scripts").resolve():
        raise SystemExit("imported the engine from %s, expected %s" % (X.__file__, code))
    return X


def context(X, m, first_batches):
    P = Path(X.EXP_ROOT) / "configs" / "prompts"
    a = argparse.Namespace(
        phase="generate", source_run=ROOT / m["source_run"], run_dir=ROOT / m["run_dir"],
        prompts=P / "default_v1.json", prompts_e44=P / "e44_factorial_v1.json", prompts_extra=P / "e44_eval_extra_v1.json",
        e_on=m["e_on"], e_off="none",
        c_run_dir=(ROOT / m["c_run_dir"]) if m.get("c_run_dir") else None, c_e_on=m.get("c_e_on", "E1"), c_e_off="none",
        arms=[], bases=[], family=None, layers=[], pairs=[], answers=["minus", "plus"], split="tune", seed=42,
        batch_size=BATCH, max_new_tokens=ENGINE_MAX_NEW, max_target_tokens=24, max_ref_tokens=48,
        output_projection="float32", first_batches=first_batches, gen=None, prompts_out=None, name="judge_input",
        overwrite=False, n_check=0, pair_ids=None)
    return X.Context(a)


def systems_of(ctx):
    s = dict(ctx.all_systems)
    for k, v in (("conf_FR", s["conf"] + " " + FR_SENTENCE), ("neutral", NEUTRAL_SYSTEM)):
        if k in s and s[k] != v:
            raise SystemExit("prompt name %s already means something else in this run" % k)
        s[k] = v
    return s


GEN_VERSION = 2          # bump whenever the generated answers could change (rendering, writes, rows kept)
SUBSET_RE = re.compile(r"^s[0-9a-f]{12}\.")


def ids_digest(ids):
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def plan_arm(X, ctx, systems, arm):
    """(base prompt, parsed parts, write descriptions, positions) of one arm, before any generation."""
    base = arm.split(":")[0]
    if base not in systems:
        raise SystemExit("unknown base prompt %s" % base)
    if ":" not in arm:
        return base, [], [], None
    _, parts = X.parse_arm(arm, systems)
    ws, positions = ctx.writes(parts)
    if positions != "content":
        raise SystemExit("content-position directions expected, got %s" % positions)
    return base, parts, [d for _, _, d in ws], positions


def gen_arm(X, ctx, systems, arm, prompts, max_new, batch=BATCH):
    """(records, write descriptions, positions) of one arm over already rendered prompts; the count is checked on
    every branch."""
    pd, fd = X.pd, X.fd
    base, parts, descs, positions = plan_arm(X, ctx, systems, arm)
    if not parts:
        recs = pd.generate_prompts(ctx.bundle, prompts, max_new_tokens=max_new, batch_size=batch)
    else:
        ws, _ = ctx.writes(parts)
        if len(ws) == 1:
            layer, w, _ = ws[0]
            recs = fd.generate_shift(ctx.bundle, prompts, max_new_tokens=max_new, batch_size=batch, layer=layer,
                                     positions=positions, **w)
        else:
            recs = X.generate_two(ctx.bundle, prompts, [(l, w) for l, w, _ in ws], max_new_tokens=max_new,
                                  batch_size=batch, positions=positions)
    if len(recs) != len(prompts):
        raise RuntimeError("generation returned %d answers for %d prompts" % (len(recs), len(prompts)))
    return recs, descs, positions


KEEP = ("text", "token_ids", "finished", "stop_id", "n_tokens")


def same(a, b):
    return all(a[k] == b[k] for k in ("text", "token_ids", "finished", "stop_id"))


def repro_reference(ctx, m, arm, descs):
    """Saved tune answers of the parent run for one arm: its eval_extra generation, or for must-answer + E alone the
    driver's own generation file; write files and strengths must equal the planned writes."""
    cand = [ctx.out_dir / "gen" / ("%s_tune.json" % arm_token(arm))]
    if arm.startswith("conf:") and "+" not in arm:
        cand.append(ctx.args.run_dir / "gen" / ("E_L%d_conf-w-p2_tune.json" % m["el"]))
    path = next((p for p in cand if p.exists()), None)
    if path is None:
        raise SystemExit("no saved tune generation for %s" % arm)
    g = json.loads(path.read_text(encoding="utf-8"))
    if g.get("completed") is not True or g.get("identity_sha256") != ctx.ident_sha:
        raise SystemExit("%s is not a complete generation of this run" % path.name)
    saved = g["parts"] if "parts" in g else [{k: g.get(k) for k in ("family", "layer", "direction_file_sha256", "direction_record_sha256", "beta")}]
    if len(saved) != len(descs):
        raise SystemExit("%s: %d saved writes, %d planned" % (arm, len(saved), len(descs)))
    for d, gp in zip(descs, saved):
        for k in ("family", "layer", "direction_file_sha256", "direction_record_sha256"):
            if d.get(k) != gp.get(k):
                raise SystemExit("%s: write differs from the saved generation in %s" % (arm, k))
        if abs(float(d["beta"]) - float(gp["beta"])) > 1e-12:
            raise SystemExit("%s: strength differs from the saved generation" % arm)
        if "amount" in gp and abs(float(d["amount"]) - float(gp["amount"])) > 1e-9:
            raise SystemExit("%s: write amount differs from the saved generation" % arm)
    return {r["pair_id"]: r for r in g["rows"]}, path.name


def run_repro(X, m, args):
    ctx = context(X, m, args.first_batches)
    systems = systems_of(ctx)
    ids = [q["pair_id"] for q in ctx.questions]
    e, c = "E%d.w.p2" % m["el"], "C%d.w" % m["cl"]
    arms = ["x", "x:" + e, "x:%s+%s.m0.25" % (e, c), "conf:" + e, "x:%s.m0.25" % c, "x:%s+%s.m1" % (e, c)]
    report = {"protocol": PROTOCOL + "_repro", "gen_version": GEN_VERSION, "model": args.model, "questions": len(ids), "arms": {}}
    with X.E.fp32_output_projection(ctx.bundle, enabled=True):
        for arm in arms:
            base = arm.split(":")[0]
            prompts = [X.pd.render_prompt(ctx.bundle, systems[base], ctx.user_template, q["question"]) for q in ctx.questions]
            if [p["prefix_text"] for p in prompts] != [p["prefix_text"] for p in ctx.render(base)]:
                raise SystemExit("our rendering of %s differs from the engine's" % base)
            recs, descs, _ = gen_arm(X, ctx, systems, arm, prompts, ENGINE_MAX_NEW)
            if arm == "x":
                ref, src = [ctx.look[pid]["answers"]["x"] for pid in ids], "frozen"
            else:
                rows, src = repro_reference(ctx, m, arm, descs)
                ref = [rows[pid] for pid in ids]
            n_same = sum(same(r, f) for r, f in zip(recs, ref))
            n_fin = sum(bool(r["finished"]) == bool(f["finished"]) for r, f in zip(recs, ref))
            report["arms"][arm] = {"reference": src, "identical": n_same, "finished_equal": n_fin, "of": len(ids),
                                   "writes": [{k: d.get(k) for k in ("family", "layer", "kind", "beta", "amount", "direction_file_sha256")} for d in descs]}
            print("repro %-40s identical %d / %d, end state equal %d (%s)" % (arm, n_same, len(ids), n_fin, src), flush=True)
    report["passed"] = all(v["identical"] == v["of"] for v in report["arms"].values())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit("%s exists" % out)
    out.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("reproduction FAILED: do not use this model with newsets_gen.py")
    print("reproduction PASS", flush=True)


def load_items(args):
    """[(item id, question-or-user-text)], user template (None = the run's own) for the task, in the frozen order."""
    if args.task in ("triviaqa", "popqa"):
        rows = [json.loads(l) for l in open(args.items, encoding="utf-8")]
        rows = [r for r in rows if r["split"] == args.split]
        ids = [r["id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise SystemExit("the bank repeats an item")
        return [(r["id"], r["question"]) for r in rows], None
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "capability"))
    import cap_prompts as CP  # noqa: E402
    import pyarrow.parquet as pq  # noqa: E402
    t = pq.read_table(args.items).to_pylist()
    with open(args.ids, encoding="utf-8") as f:
        idl = json.load(f)
    if idl.get("source_file_sha256") != sha_file(args.items):
        raise SystemExit("the ID list was made from another file")
    if len(set(idl["ids"])) != len(idl["ids"]):
        raise SystemExit("the ID list repeats an item")
    out = []
    if args.task == "mmlu":
        per = {}
        for r in t:
            per.setdefault(r["subject"], []).append(r)
        for i in idl["ids"]:
            _, subj, k = i.split(":")
            r = per[subj][int(k)]
            out.append((i, CP.mmlu_user(r["subject"], r["question"], r["choices"])))
    else:
        for i in idl["ids"]:
            r = t[int(i.split(":")[1])]
            out.append((i, CP.gsm8k_user(r["question"])))
    return out, "{question}"


def select_items(items, args):
    """The items one call generates: all, the ones named by --only-ids (in task order, exactly those), or the first
    batches (smoke tests)."""
    if args.only_ids:
        keep = json.load(open(args.only_ids, encoding="utf-8"))
        if len(set(keep)) != len(keep):
            raise SystemExit("--only-ids repeats an item")
        pos = {iid: k for k, (iid, _) in enumerate(items)}
        if any(i not in pos for i in keep):
            raise SystemExit("--only-ids names items that are not in the task")
        items = [items[k] for k in sorted(pos[i] for i in keep)]
    if args.first_batches:
        items = items[:args.first_batches * args.batch_size]
    if not items:
        raise SystemExit("no items")
    return items


def run_gen(X, m, args):
    ctx = context(X, m, 0)
    systems = systems_of(ctx)
    items, template = load_items(args)
    items = select_items(items, args)
    ids = [i for i, _ in items]
    template = template or ctx.user_template
    full = not (args.first_batches or args.only_ids)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        base, _, descs, positions = plan_arm(X, ctx, systems, arm)
        contract = json.loads(json.dumps({
            "protocol": PROTOCOL, "gen_version": GEN_VERSION, "task": args.task, "split": args.split, "model": args.model,
            "model_config": m, "arm": arm, "condition": condition_name(arm, descs), "base_prompt": base,
            "system_text": systems[base], "user_template": template, "parts": descs, "positions": positions,
            "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size, "full_pass": full, "ids": ids,
            "identity_sha256": ctx.ident_sha, "frozen_file_sha256": ctx.frozen_sha, "items_file_sha256": sha_file(args.items),
            "ids_file_sha256": sha_file(args.ids) if args.ids else None,
            "only_ids_sha256": sha_file(args.only_ids) if args.only_ids else None, "engine_sha256": sha_file(X.__file__)}))
        name = arm_token(arm) + ".json" if full else "s%s.%s.json" % (ids_digest(ids)[:12], arm_token(arm))
        out = out_dir / name
        if out.exists():
            old = json.loads(out.read_text(encoding="utf-8"))
            if old.get("contract") != contract or old.get("completed") is not True or len(old.get("rows", [])) != len(ids):
                raise SystemExit("%s exists but was made from other inputs or is incomplete: stop (move it away to regenerate)" % out)
            print("skip %s (same inputs, complete)" % out.name, flush=True)
            continue
        t0 = time.time()
        with X.E.fp32_output_projection(ctx.bundle, enabled=True):
            prompts = [X.pd.render_prompt(ctx.bundle, systems[base], template, q) for _, q in items]
            recs, descs2, _ = gen_arm(X, ctx, systems, arm, prompts, args.max_new_tokens, args.batch_size)
        if descs2 != descs:
            raise RuntimeError("write description changed between planning and generation")
        rows = []
        for (iid, _), rec in zip(items, recs):
            r = {k: rec[k] for k in KEEP}
            r.update(id=iid, rule_abstain=bool(X.rule_abstains(rec["text"])))
            rows.append(r)
        if [r["id"] for r in rows] != ids:
            raise RuntimeError("rows do not follow the items")
        ctx.still_current()
        payload = dict(contract, contract=contract, script_sha256=sha_file(__file__), rows=rows, completed=True,
                       summary={"n": len(rows), "rule_abstain": sum(r["rule_abstain"] for r in rows),
                                "unfinished": sum(1 for r in rows if not r["finished"]), "seconds": round(time.time() - t0, 1)})
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.rename(out)
        s = payload["summary"]
        print("%s: %d items, rule abstain %d, unfinished %d, %.0fs" % (arm, s["n"], s["rule_abstain"], s["unfinished"], s["seconds"]), flush=True)


def load_generation(path, args, want_ids, allow_subset=False):
    """A generation file checked against the plan: protocol and version, completion, task, split, model, bank,
    exact item list in order, condition name."""
    g = json.loads(Path(path).read_text(encoding="utf-8"))
    c = g.get("contract") or {}
    bad = []
    if g.get("protocol") != PROTOCOL or c.get("gen_version") != GEN_VERSION:
        bad.append("protocol or version")
    if g.get("completed") is not True:
        bad.append("not completed")
    if not allow_subset and c.get("full_pass") is not True:
        bad.append("not a full pass")
    for k in ("task", "split", "model"):
        if c.get(k) != getattr(args, k):
            bad.append(k)
    if c.get("items_file_sha256") != sha_file(args.items):
        bad.append("bank")
    ids = [r.get("id") for r in g.get("rows", [])]
    if len(set(ids)) != len(ids):
        bad.append("an item appears twice")
    if want_ids is not None and ids != want_ids:
        bad.append("items differ from the plan")
    if c.get("ids") != ids:
        bad.append("rows differ from the contract")
    if c.get("condition") != condition_name(c.get("arm", ""), c.get("parts") or []):
        bad.append("condition name")
    if bad:
        raise SystemExit("%s: %s" % (Path(path).name, "; ".join(bad)))
    return g


def run_conf(X, m, args):
    """Mean log-probability of the model's own answers in one generation file, teacher-forced under the same prompt
    and the same writes, with the scorer of scripts/review_checks/ff_explore_ec_confidence.py (score(), unchanged).
    Enforced: (1) the teacher-forced arg-max reproduces >= 98% of the generated answer tokens; (2) with --ref-gen-file
    (the same arm without its last write), at the first token where the two answers differ, the margin toward the
    target token must be larger with the last write than without it in >= 90% of those answers and on average (that
    script's check 2), when at least 5 answers differ. Every row keeps the sha1 of its answer text and the file keeps
    the generation's sha256, so the scorer can bind a confidence to the exact answer a judge saw."""
    sys.path.insert(0, str(ROOT / m["code"] / "scripts" / "review_checks"))
    import ff_explore_ec_confidence as FC  # noqa: E402
    gpre = json.loads(Path(args.gen_file).read_text(encoding="utf-8"))
    c = gpre.get("contract") or {}
    args.task, args.split = c.get("task"), c.get("split")
    items, template = load_items(args)
    items = select_items(items, args) if args.first_batches else items
    want = [i for i, _ in items]
    g = load_generation(args.gen_file, args, want, allow_subset=bool(args.first_batches))
    if c.get("max_new_tokens") != ENGINE_MAX_NEW or c.get("batch_size") != BATCH:
        raise SystemExit("confidence is defined for the 48-token, batch-8 answers only")
    ctx = context(X, m, 0)
    if c.get("identity_sha256") != ctx.ident_sha or c.get("engine_sha256") != sha_file(X.__file__):
        raise SystemExit("the generation belongs to another run or engine version")
    systems = systems_of(ctx)
    template = template or ctx.user_template
    rows = g["rows"]
    ref_rows, ref_writes = None, None
    if args.ref_gen_file:
        r = load_generation(args.ref_gen_file, args, want, allow_subset=bool(args.first_batches))
        rc = r["contract"]
        if rc["base_prompt"] != c["base_prompt"] or rc["parts"] != c["parts"][:-1] or not c["parts"]:
            raise SystemExit("--ref-gen-file must be the same arm without its last write")
        ref_rows = r["rows"]
    report = {"protocol": PROTOCOL + "_conf", "gen_version": GEN_VERSION, "model": args.model, "task": c["task"],
              "split": c["split"], "arm": c["arm"], "condition": c["condition"], "generation_sha256": sha_file(args.gen_file),
              "reference_sha256": sha_file(args.ref_gen_file) if args.ref_gen_file else None,
              "identity_sha256": ctx.ident_sha, "script_sha256": sha_file(__file__), "scorer_sha256": sha_file(FC.__file__)}
    eot = list(ctx.eot)
    with X.E.fp32_output_projection(ctx.bundle, enabled=True):
        prompts = [X.pd.render_prompt(ctx.bundle, systems[c["base_prompt"]], template, q) for _, q in items]
        writes = []
        if c["parts"]:
            _, parts = X.parse_arm(c["arm"], systems)
            ws, positions = ctx.writes(parts)
            if positions != "content" or json.loads(json.dumps([d for _, _, d in ws])) != c["parts"]:
                raise SystemExit("the writes differ from the generation's")
            writes = [(layer, w["direction"], float(w["amount"])) for layer, w, _ in ws]
        keep = [i for i, r in enumerate(rows) if r["token_ids"] and r["text"].strip()]
        probes = None
        if ref_rows is not None:
            probes = []
            for i in keep:
                tf = list(rows[i]["token_ids"]) + (eot if rows[i]["finished"] else [])
                rf = list(ref_rows[i]["token_ids"]) + (eot if ref_rows[i]["finished"] else [])
                j = next((k for k in range(min(len(tf), len(rf))) if tf[k] != rf[k]), None)
                probes.append(None if j is None else (j, rf[j]))
        res = FC.score(ctx, [prompts[i] for i in keep], [rows[i] for i in keep], writes, probes, batch=BATCH)
        check = None
        if probes is not None:
            abl = FC.score(ctx, [prompts[i] for i in keep], [rows[i] for i in keep], writes[:-1], probes, batch=BATCH)
            shifts = [(f[3][0] - f[3][1]) - (a[3][0] - a[3][1]) for pr, f, a in zip(probes, res, abl) if pr is not None]
            n = len(shifts)
            toward = sum(s > 0 for s in shifts)
            check = {"questions": n, "share_toward_target": toward / n if n else None,
                     "mean_margin_shift": sum(shifts) / n if n else None}
            if n >= 5 and (toward / n < 0.90 or check["mean_margin_shift"] <= 0):
                raise SystemExit("first-difference check failed %s" % check)
            if n < 5:
                check["note"] = "fewer than 5 answers differ from the reference: not enforced"
    same_n = sum(sum(1 for x, y in zip(fa[:len(lp)], ft[:len(lp)]) if x == y) for lp, fa, ft, _ in res)
    total = sum(len(lp) for lp, _, _, _ in res)
    if not total or same_n / total < 0.98:
        raise SystemExit("teacher-forced arg-max reproduces only %.3f of the generated tokens" % (same_n / total if total else 0.0))
    out_rows = [{"id": r["id"], "mean_logp": None, "n_tokens": 0, "empty_answer": True,
                 "text_sha1": hashlib.sha1(r["text"].strip().encode("utf-8")).hexdigest()} for r in rows]
    for i, (lp, _, _, _) in zip(keep, res):
        out_rows[i] = {"id": rows[i]["id"], "n_tokens": len(lp), "sum_logp": float(lp.sum()), "mean_logp": float(lp.mean()),
                       "first_logp": float(lp[0]), "min_logp": float(lp.min()),
                       "text_sha1": hashlib.sha1(rows[i]["text"].strip().encode("utf-8")).hexdigest()}
    report.update(rows=out_rows, greedy_reproduction={"tokens": total, "same": same_n, "rate": same_n / total},
                  first_difference_check=check, empty_answers=len(rows) - len(keep))
    out = Path(args.out)
    if out.exists():
        raise SystemExit("%s exists" % out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report), encoding="utf-8")
    print("confidence %s: greedy %d / %d (%.4f), first-difference %s, empty %d" % (
        c["condition"], same_n, total, same_n / total, check, len(rows) - len(keep)), flush=True)


def run_export(args):
    """One judge input: the exact --only-arms, every one a complete full-pass generation of this model over the whole
    split in bank order; --pro100 then keeps the frozen 100 test questions."""
    rows_bank = [json.loads(l) for l in open(args.items, encoding="utf-8")]
    split_rows = [r for r in rows_bank if r["split"] == args.split]
    want = [r["id"] for r in split_rows]
    if len(set(want)) != len(want):
        raise SystemExit("the bank repeats an item")
    bank = {r["id"]: r for r in split_rows}
    keep = [i for i in want if bank[i]["pro100"]] if args.pro100 else want
    if args.pro100 and (args.split != "test" or len(keep) != 100):
        raise SystemExit("the Pro cohort is the 100 frozen test questions")
    if not args.only_arms or len(set(args.only_arms)) != len(args.only_arms):
        raise SystemExit("export needs the exact, distinct --only-arms to judge")
    files = {}
    for p in sorted(Path(args.gen_dir).glob("*.json")):
        if SUBSET_RE.match(p.name):
            continue                       # subset (smoke or re-ask) files are never exported
        g = json.loads(p.read_text(encoding="utf-8"))
        arm = (g.get("contract") or {}).get("arm", g.get("arm"))
        if arm not in args.only_arms:
            continue
        if arm in files:
            raise SystemExit("two files for arm %s" % arm)
        files[arm] = (p, load_generation(p, args, want))
    missing = sorted(set(args.only_arms) - set(files))
    if missing:
        raise SystemExit("no full generation for %s" % missing)
    gens = [files[a][1] for a in args.only_arms]
    for k in ("identity_sha256", "frozen_file_sha256", "engine_sha256", "max_new_tokens", "batch_size"):
        if len({json.dumps(g["contract"][k]) for g in gens}) != 1:
            raise SystemExit("generation files disagree on %s" % k)
    conds = [g["contract"]["condition"] for g in gens]
    if len(set(conds)) != len(conds):
        raise SystemExit("duplicate conditions")
    by = [{r["id"]: r for r in g["rows"]} for g in gens]
    rows = []
    for iid in keep:
        b = bank[iid]
        rows.append({"pair_id": "%s:%s" % (args.task, iid), "question": b["question"], "gold": list(b["gold"][:GOLD_CAP]),
                     "incorrect": [], "best_answer": b["gold"][0],
                     "outputs": {c: d[iid]["text"] for c, d in zip(conds, by)},
                     "finished": {c: bool(d[iid]["finished"]) for c, d in zip(conds, by)},
                     "rule_abstain": {c: bool(d[iid]["rule_abstain"]) for c, d in zip(conds, by)}})
    payload = {"completed": True, "model": args.model, "protocol": PROTOCOL, "export_protocol": PROTOCOL + "_export",
               "gen_version": GEN_VERSION, "split": args.split, "task": args.task, "pro100_only": bool(args.pro100),
               "gold_cap": GOLD_CAP, "identity_sha256": gens[0]["contract"]["identity_sha256"], "conditions": conds,
               "condition_specs": [{"name": g["contract"]["condition"], "arm": g["contract"]["arm"], "parts": g["contract"]["parts"],
                                    "base_prompt": g["contract"]["base_prompt"],
                                    "kind": "intervention" if g["contract"]["parts"] else "prompt_control",
                                    "uses_reference_answer": False} for g in gens],
               "source_files": {g["contract"]["arm"]: {"file": files[g["contract"]["arm"]][0].name,
                                                       "sha256": sha_file(files[g["contract"]["arm"]][0])} for g in gens},
               "question_ids": [r["pair_id"] for r in rows], "bank_sha256": sha_file(args.items), "evaluations": rows}
    out = Path(args.out)
    if out.exists():
        raise SystemExit("%s exists; never overwrite a judge input" % out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("exported %s: %d questions x %d conditions" % (out, len(rows), len(conds)), flush=True)


def selftest():
    import tempfile
    from types import SimpleNamespace
    assert condition_name("x", []) == "baseline_x"
    assert condition_name("x:E11.w.p2", [{}]) == "E_L11_x-w-p2"
    assert condition_name("conf:E15.w.p2", [{}]) == "E_L15_conf-w-p2"
    assert condition_name("x:C19.w.m0.25", [{}]) == "C_L19_x-w-m0-25"
    assert condition_name("x:E11.w.p2+C19.w.m0.25", [{}, {}]) == "X_x--E11-w-p2-and-C19-w-m0-25"
    assert arm_token("x:E11.w.p2+C19.w.m1") == "x--E11-w-p2-and-C19-w-m1"
    assert "\\" not in FR_SENTENCE and FR_SENTENCE.endswith("asks about.")
    assert ids_digest(["a"]) != ids_digest(["b"]) and SUBSET_RE.match("s0123456789ab.x.json") and not SUBSET_RE.match("x.json")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bank = root / "bank.jsonl"
        bank.write_text("\n".join(json.dumps(dict(id=i, question="Q" + i, gold=["g"], split="test", pro100=(i == "q1")))
                                   for i in ("q1", "q2")) + "\n")
        gdir = root / "gen"
        gdir.mkdir()

        def art(rows, **over):
            c = dict(protocol=PROTOCOL, gen_version=GEN_VERSION, task="popqa", split="test", model="m1", arm="x",
                     condition="baseline_x", parts=[], base_prompt="x", full_pass=True, ids=[r["id"] for r in rows],
                     items_file_sha256=sha_file(bank), identity_sha256="i", frozen_file_sha256="f", engine_sha256="e",
                     max_new_tokens=48, batch_size=8)
            c.update(over.pop("contract", {}))
            g = dict(c, contract=c, rows=rows, completed=True)
            g.update(over)
            return g
        good = [dict(id="q1", text="a", finished=True, rule_abstain=False), dict(id="q2", text="b", finished=True, rule_abstain=False)]
        args = SimpleNamespace(items=bank, gen_dir=gdir, only_arms=["x"], out=str(root / "out.json"), task="popqa",
                               split="test", pro100=False, model="m1")
        bads = {"not completed": art(good, completed=False),
                "item twice": art([good[0], dict(good[0], text="c")], contract=dict(ids=["q1", "q1"])),
                "foreign item": art([good[0], dict(id="zz", text="o", finished=True, rule_abstain=False)], contract=dict(ids=["q1", "zz"])),
                "other model": art(good, contract=dict(model="m2")),
                "missing row": art(good[:1], contract=dict(ids=["q1"]))}
        for why, g in bads.items():
            (gdir / "x.json").write_text(json.dumps(g))
            try:
                run_export(args)
                raise AssertionError("export accepted: " + why)
            except SystemExit:
                pass
        (gdir / "x.json").write_text(json.dumps(art(good)))
        run_export(args)
        out = json.loads(Path(args.out).read_text())
        assert out["question_ids"] == ["popqa:q1", "popqa:q2"] and out["evaluations"][1]["outputs"]["baseline_x"] == "b"
        args.out, args.pro100 = str(root / "out100.json"), True
        try:
            run_export(args)
            raise AssertionError("a Pro cohort of one question was accepted")
        except SystemExit:
            pass
    print("selftest PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["repro", "gen", "conf", "export", "selftest"])
    ap.add_argument("--gen-file", type=Path)
    ap.add_argument("--ref-gen-file", type=Path, help="conf: the same arm without its last write (first-difference check)")
    ap.add_argument("--models", type=Path)
    ap.add_argument("--model")
    ap.add_argument("--task", choices=["triviaqa", "popqa", "mmlu", "gsm8k"])
    ap.add_argument("--items", type=Path)
    ap.add_argument("--ids", type=Path)
    ap.add_argument("--split", choices=["tune", "test", "all"], default="all")
    ap.add_argument("--arms", nargs="+", default=[])
    ap.add_argument("--max-new-tokens", type=int, default=ENGINE_MAX_NEW)
    ap.add_argument("--first-batches", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=BATCH, choices=(8, 16, 32),
                    help="generation batch size; newsets and conf always use 8 (the TruthfulQA runs' value); mmlu / gsm8k may use 16 or 32 for speed, the same value for every condition of a model")
    ap.add_argument("--only-ids", type=Path)
    ap.add_argument("--gen-dir", type=Path)
    ap.add_argument("--pro100", action="store_true")
    ap.add_argument("--only-arms", nargs="+", default=None, help="export: exactly these arms (required)")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.phase == "selftest":
        selftest()
        return
    if args.phase == "export":
        if not (args.task in ("triviaqa", "popqa") and args.items and args.gen_dir and args.out and args.model
                and args.split in ("tune", "test") and args.only_arms):
            ap.error("export needs --task triviaqa|popqa --model --items --split tune|test --gen-dir --only-arms --out")
        run_export(args)
        return
    if not (args.models and args.model and args.out):
        ap.error("--models, --model and --out are required")
    m = load_models(args.models, args.model)
    X = engine(m)
    if args.phase == "repro":
        if args.first_batches < 1:
            ap.error("repro needs --first-batches >= 1")
        run_repro(X, m, args)
        return
    if args.phase == "conf":
        if not (args.gen_file and args.items):
            ap.error("conf needs --gen-file and --items (the bank or parquet the generation used)")
        run_conf(X, m, args)
        return
    if not (args.task and args.items and args.arms):
        ap.error("gen needs --task, --items and --arms")
    if args.task in ("triviaqa", "popqa") and args.split not in ("tune", "test"):
        ap.error("newsets need --split tune|test")
    if args.task in ("mmlu", "gsm8k") and not args.ids:
        ap.error("mmlu / gsm8k need --ids")
    if args.task in ("triviaqa", "popqa") and args.batch_size != BATCH:
        ap.error("newsets generate with batch size 8, as the TruthfulQA runs did")
    run_gen(X, m, args)


if __name__ == "__main__":
    main()
