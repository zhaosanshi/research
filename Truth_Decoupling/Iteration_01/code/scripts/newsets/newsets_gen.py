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


def gen_arm(X, ctx, systems, arm, prompts, max_new, batch=BATCH):
    """(records, write descriptions, positions) of one arm over already rendered prompts."""
    pd, fd = X.pd, X.fd
    if ":" not in arm:
        if arm not in systems:
            raise SystemExit("unknown base prompt %s" % arm)
        return pd.generate_prompts(ctx.bundle, prompts, max_new_tokens=max_new, batch_size=batch), [], None
    _, parts = X.parse_arm(arm, systems)
    ws, positions = ctx.writes(parts)
    if positions != "content":
        raise SystemExit("content-position directions expected, got %s" % positions)
    if len(ws) == 1:
        layer, w, _ = ws[0]
        recs = fd.generate_shift(ctx.bundle, prompts, max_new_tokens=max_new, batch_size=batch, layer=layer,
                                 positions=positions, **w)
    else:
        recs = X.generate_two(ctx.bundle, prompts, [(l, w) for l, w, _ in ws], max_new_tokens=max_new,
                              batch_size=batch, positions=positions)
    if len(recs) != len(prompts):
        raise RuntimeError("generation returned the wrong number of answers")
    return recs, [d for _, _, d in ws], positions


KEEP = ("text", "token_ids", "finished", "stop_id", "n_tokens")


def same(a, b):
    return all(a[k] == b[k] for k in ("text", "token_ids", "finished", "stop_id"))


def run_repro(X, m, args):
    ctx = context(X, m, args.first_batches)
    systems = systems_of(ctx)
    ids = [q["pair_id"] for q in ctx.questions]
    e = "E%d.w.p2" % m["el"]
    arms = ["x", "x:" + e, "x:%s+C%d.w.m0.25" % (e, m["cl"])]
    report = {"protocol": PROTOCOL + "_repro", "model": args.model, "questions": len(ids), "arms": {}}
    with X.E.fp32_output_projection(ctx.bundle, enabled=True):
        prompts = [X.pd.render_prompt(ctx.bundle, systems["x"], ctx.user_template, q["question"]) for q in ctx.questions]
        if [p["prefix_text"] for p in prompts] != [p["prefix_text"] for p in ctx.render("x")]:
            raise SystemExit("our rendering differs from the engine's")
        for arm in arms:
            recs, descs, _ = gen_arm(X, ctx, systems, arm, prompts, ENGINE_MAX_NEW)
            if arm == "x":
                ref = [ctx.look[pid]["answers"]["x"] for pid in ids]
                src = "frozen"
            else:
                path = ctx.out_dir / "gen" / ("%s_tune.json" % arm_token(arm))
                g = json.loads(path.read_text(encoding="utf-8"))
                if g.get("completed") is not True or g.get("identity_sha256") != ctx.ident_sha:
                    raise SystemExit("%s is not a complete generation of this run" % path.name)
                for d, gp in zip(descs, g["parts"]):
                    for k in ("family", "layer", "kind", "beta", "direction_file_sha256", "direction_record_sha256"):
                        if d.get(k) != gp.get(k):
                            raise SystemExit("%s: write differs from the saved generation in %s" % (arm, k))
                    if abs(float(d["amount"]) - float(gp["amount"])) > 1e-9:
                        raise SystemExit("%s: write amount differs from the saved generation" % arm)
                rows = {r["pair_id"]: r for r in g["rows"]}
                ref = [rows[pid] for pid in ids]
                src = path.name
            n_same = sum(same(r, f) for r, f in zip(recs, ref))
            report["arms"][arm] = {"reference": src, "identical": n_same, "of": len(ids)}
            print("repro %-40s identical %d / %d (%s)" % (arm, n_same, len(ids), src), flush=True)
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
    """[(item id, question-or-user-text, base user template)] for the task."""
    if args.task in ("triviaqa", "popqa"):
        rows = [json.loads(l) for l in open(args.items, encoding="utf-8")]
        rows = [r for r in rows if r["split"] == args.split]
        return [(r["id"], r["question"]) for r in rows], None
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "capability"))
    import cap_prompts as CP  # noqa: E402
    import pyarrow.parquet as pq  # noqa: E402
    t = pq.read_table(args.items).to_pylist()
    with open(args.ids, encoding="utf-8") as f:
        idl = json.load(f)
    if idl.get("source_file_sha256") != sha_file(args.items):
        raise SystemExit("the ID list was made from another file")
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


def run_gen(X, m, args):
    ctx = context(X, m, 0)
    systems = systems_of(ctx)
    items, template = load_items(args)
    if args.only_ids:
        keep = set(json.load(open(args.only_ids, encoding="utf-8")))
        items = [it for it in items if it[0] in keep]
        if len(items) != len(keep):
            raise SystemExit("--only-ids names items that are not in the task")
    if args.first_batches:
        items = items[:args.first_batches * args.batch_size]
    if not items:
        raise SystemExit("no items")
    template = template or ctx.user_template
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        base = arm.split(":")[0]
        if base not in systems:
            raise SystemExit("unknown base prompt %s" % base)
        out = out_dir / ("%s%s.json" % ("t%d." % len(items) if (args.first_batches or args.only_ids) else "", arm_token(arm)))
        if out.exists():
            print("skip %s (exists)" % out.name, flush=True)
            continue
        t0 = time.time()
        with X.E.fp32_output_projection(ctx.bundle, enabled=True):
            prompts = [X.pd.render_prompt(ctx.bundle, systems[base], template, q) for _, q in items]
            recs, descs, positions = gen_arm(X, ctx, systems, arm, prompts, args.max_new_tokens, args.batch_size)
        rows = []
        for (iid, _), rec in zip(items, recs):
            r = {k: rec[k] for k in KEEP}
            r.update(id=iid, rule_abstain=bool(X.rule_abstains(rec["text"])))
            rows.append(r)
        ctx.still_current()
        payload = {"protocol": PROTOCOL, "task": args.task, "split": args.split, "model": args.model, "arm": arm,
                   "condition": condition_name(arm, descs), "base_prompt": base, "system_text": systems[base],
                   "user_template": template, "parts": descs, "positions": positions, "max_new_tokens": args.max_new_tokens,
                   "batch_size": args.batch_size, "full_pass": not (args.first_batches or args.only_ids),
                   "identity_sha256": ctx.ident_sha, "frozen_file_sha256": ctx.frozen_sha,
                   "items_file_sha256": sha_file(args.items), "ids_file_sha256": sha_file(args.ids) if args.ids else None,
                   "only_ids_sha256": sha_file(args.only_ids) if args.only_ids else None,
                   "script_sha256": sha_file(__file__), "engine_sha256": sha_file(X.__file__),
                   "rows": rows, "completed": True,
                   "summary": {"n": len(rows), "rule_abstain": sum(r["rule_abstain"] for r in rows),
                               "unfinished": sum(1 for r in rows if not r["finished"]), "seconds": round(time.time() - t0, 1)}}
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.rename(out)
        s = payload["summary"]
        print("%s: %d items, rule abstain %d, unfinished %d, %.0fs" % (arm, s["n"], s["rule_abstain"], s["unfinished"], s["seconds"]), flush=True)


def run_conf(X, m, args):
    """Mean log-probability of the model's own answers in one generation file, teacher-forced under the same prompt
    and the same writes, with the scorer of scripts/review_checks/ff_explore_ec_confidence.py (score(), unchanged).
    Enforced: the teacher-forced arg-max reproduces >= 98% of the generated answer tokens (that script's check 1).
    Its first-difference check (check 2) is not repeated here; generation itself is checked by the repro phase."""
    sys.path.insert(0, str(ROOT / m["code"] / "scripts" / "review_checks"))
    import ff_explore_ec_confidence as FC  # noqa: E402
    g = json.loads(Path(args.gen_file).read_text(encoding="utf-8"))
    if g.get("protocol") != PROTOCOL or g.get("model") != args.model or not (g.get("full_pass") or args.first_batches):
        raise SystemExit("not a full generation file of this model (a partial one needs --first-batches, for smoke tests)")
    if g["max_new_tokens"] != ENGINE_MAX_NEW:
        raise SystemExit("confidence is only defined for the 48-token answers")
    ctx = context(X, m, 0)
    if g["identity_sha256"] != ctx.ident_sha or g["engine_sha256"] != sha_file(X.__file__):
        raise SystemExit("the generation belongs to another run or engine version")
    systems = systems_of(ctx)
    args.task, args.split, args.items = g["task"], g["split"], args.items
    if sha_file(args.items) != g["items_file_sha256"]:
        raise SystemExit("--items is not the bank this generation used")
    items, template = load_items(args)
    if args.first_batches:
        items = items[:args.first_batches * BATCH]
    rows = g["rows"]
    if [i for i, _ in items] != [r["id"] for r in rows]:
        raise SystemExit("the items do not match the generation rows")
    template = template or ctx.user_template
    report = {"protocol": PROTOCOL + "_conf", "model": args.model, "task": g["task"], "split": g["split"], "arm": g["arm"],
              "condition": g["condition"], "generation_sha256": sha_file(args.gen_file), "identity_sha256": ctx.ident_sha,
              "script_sha256": sha_file(__file__), "scorer_sha256": sha_file(FC.__file__)}
    with X.E.fp32_output_projection(ctx.bundle, enabled=True):
        prompts = [X.pd.render_prompt(ctx.bundle, systems[g["base_prompt"]], template, q) for _, q in items]
        if ":" in g["arm"]:
            _, parts = X.parse_arm(g["arm"], systems)
            ws, positions = ctx.writes(parts)
            if positions != "content" or [d for _, _, d in ws] != g["parts"]:
                raise SystemExit("the writes differ from the generation's")
            writes = [(layer, w["direction"], float(w["amount"])) for layer, w, _ in ws]
        else:
            writes = []
        keep = [i for i, r in enumerate(rows) if r["token_ids"] and r["text"].strip()]
        res = FC.score(ctx, [prompts[i] for i in keep], [rows[i] for i in keep], writes, None, batch=BATCH)
    same = sum(sum(1 for x, y in zip(fa[:len(lp)], ft[:len(lp)]) if x == y) for lp, fa, ft, _ in res)
    total = sum(len(lp) for lp, _, _, _ in res)
    if not total or same / total < 0.98:
        raise SystemExit("teacher-forced arg-max reproduces only %.3f of the generated tokens" % (same / total if total else 0.0))
    out_rows = [{"id": r["id"], "mean_logp": None, "n_tokens": 0, "empty_answer": True} for r in rows]
    for i, (lp, _, _, _) in zip(keep, res):
        out_rows[i] = {"id": rows[i]["id"], "n_tokens": len(lp), "sum_logp": float(lp.sum()), "mean_logp": float(lp.mean()),
                       "first_logp": float(lp[0]), "min_logp": float(lp.min()),
                       "text_sha1": hashlib.sha1(rows[i]["text"].strip().encode("utf-8")).hexdigest()}
    report.update(rows=out_rows, greedy_reproduction={"tokens": total, "same": same, "rate": same / total},
                  empty_answers=len(rows) - len(keep))
    out = Path(args.out)
    if out.exists():
        raise SystemExit("%s exists" % out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report), encoding="utf-8")
    print("confidence %s: greedy %d / %d (%.4f), empty %d" % (g["condition"], same, total, same / total, len(rows) - len(keep)), flush=True)


def run_export(args):
    bank = {}
    for l in open(args.items, encoding="utf-8"):
        r = json.loads(l)
        if r["split"] == args.split and (not args.pro100 or r["pro100"]):
            bank[r["id"]] = r
    gens = []
    for p in sorted(Path(args.gen_dir).glob("*.json")):
        if p.name.startswith("t") and p.name.split(".", 1)[0][1:].isdigit():
            continue                       # partial files are never exported
        g = json.loads(p.read_text(encoding="utf-8"))
        if g.get("protocol") != PROTOCOL or g.get("task") != args.task or g.get("split") != args.split or not g.get("full_pass"):
            raise SystemExit("%s is not a full %s %s generation" % (p.name, args.task, args.split))
        if g.get("items_file_sha256") != sha_file(args.items):
            raise SystemExit("%s was made from another bank file" % p.name)
        gens.append(g)
    if args.only_arms:
        gens = [g for g in gens if g["arm"] in set(args.only_arms)]
        if sorted(g["arm"] for g in gens) != sorted(set(args.only_arms)):
            raise SystemExit("--only-arms names arms without a full generation file")
    if not gens:
        raise SystemExit("no generation files")
    for k in ("model", "identity_sha256", "frozen_file_sha256", "script_sha256", "engine_sha256"):
        if len({g[k] for g in gens}) != 1:
            raise SystemExit("generation files disagree on %s" % k)
    conds = [g["condition"] for g in gens]
    if len(set(conds)) != len(conds):
        raise SystemExit("duplicate conditions")
    by = [{r["id"]: r for r in g["rows"]} for g in gens]
    rows = []
    for iid, b in bank.items():
        if any(iid not in d for d in by):
            raise SystemExit("question %s missing from a generation file" % iid)
        rows.append({"pair_id": "%s:%s" % (args.task, iid), "question": b["question"], "gold": list(b["gold"][:GOLD_CAP]),
                     "incorrect": [], "best_answer": b["gold"][0],
                     "outputs": {c: d[iid]["text"] for c, d in zip(conds, by)},
                     "finished": {c: bool(d[iid]["finished"]) for c, d in zip(conds, by)},
                     "rule_abstain": {c: bool(d[iid]["rule_abstain"]) for c, d in zip(conds, by)}})
    payload = {"completed": True, "model": gens[0]["model"], "protocol": PROTOCOL, "export_protocol": PROTOCOL + "_export",
               "split": args.split, "task": args.task, "pro100_only": bool(args.pro100), "gold_cap": GOLD_CAP,
               "identity_sha256": gens[0]["identity_sha256"], "conditions": conds,
               "condition_specs": [{"name": g["condition"], "arm": g["arm"], "parts": g["parts"], "base_prompt": g["base_prompt"],
                                    "kind": "intervention" if g["parts"] else "prompt_control", "uses_reference_answer": False}
                                   for g in gens],
               "question_ids": [r["pair_id"] for r in rows], "bank_sha256": sha_file(args.items), "evaluations": rows}
    out = Path(args.out)
    if out.exists():
        raise SystemExit("%s exists; never overwrite a judge input" % out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("exported %s: %d questions x %d conditions" % (out, len(rows), len(conds)), flush=True)


def selftest():
    assert condition_name("x", []) == "baseline_x"
    assert condition_name("x:E11.w.p2", [{}]) == "E_L11_x-w-p2"
    assert condition_name("conf:E15.w.p2", [{}]) == "E_L15_conf-w-p2"
    assert condition_name("x:C19.w.m0.25", [{}]) == "C_L19_x-w-m0-25"
    assert condition_name("x:E11.w.p2+C19.w.m0.25", [{}, {}]) == "X_x--E11-w-p2-and-C19-w-m0-25"
    assert arm_token("x:E11.w.p2+C19.w.m1") == "x--E11-w-p2-and-C19-w-m1"
    assert "\\" not in FR_SENTENCE and FR_SENTENCE.endswith("asks about.")
    print("selftest PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["repro", "gen", "conf", "export", "selftest"])
    ap.add_argument("--gen-file", type=Path)
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
    ap.add_argument("--only-arms", nargs="+", default=None, help="export: only these arms (default: every full generation file)")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.phase == "selftest":
        selftest()
        return
    if args.phase == "export":
        if not (args.task in ("triviaqa", "popqa") and args.items and args.gen_dir and args.out and args.split in ("tune", "test")):
            ap.error("export needs --task triviaqa|popqa --items --split tune|test --gen-dir --out")
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
