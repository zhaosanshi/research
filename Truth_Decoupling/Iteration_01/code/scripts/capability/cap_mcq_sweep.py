"""MMLU by option probability, a sweep of write variants: why does writing E lower MMLU accuracy (2026-11-20)?

Same 1,000 items, prompt (plain system prompt x, cap_prompts.mmlu_user), letter spellings and pick rule as
cap_mcq_prob.py (protocol v2). Differences:
  - one forward per batch: the next-token distribution at the last prompt position gives every letter at once (v2
    teacher-forced each letter token separately; the predictor position is the same and the appended token cannot
    reach it, so the numbers agree up to bf16 noise; --ref checks this against a v2 file);
  - any single- or two-write arm of the e44_eval_extra grammar, with E# / C# standing for the model's frozen E / C
    layer: dose (x:E#.w.p0.5 ...), reversed sign (x:E#.w.m2), the random axis with the same stored write norm
    (x:E#.rand.p2), the mean-difference direction at the same nominal amount (x:E#.mdg.p2), the two parts of the
    learned write (x:E#.wpar.p2, x:E#.wperp.p2: parallel / orthogonal to the mean difference), and C;
  - an arm may end in @gen: the write then starts at the last prompt token (core position rule "gen": the question
    tokens are untouched) instead of the trained rule "content" (first user-content token .. end). The directions were
    trained under "content"; @gen is a diagnostic, not a method setting.
An arm whose direction cannot be built (for example no defined mean difference) is skipped loudly and listed in the
output.

usage: <model python> -I cap_mcq_sweep.py run --models CFG --model NAME --items MMLU_PARQUET --ids MMLU_IDS --out OUT_JSON
                                        --arms ARM [ARM ...] [--ref V2_JSON] [--first-batches K]
       python3 -I cap_mcq_sweep.py report OUT_JSON...
"""
import argparse
import contextlib
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "newsets"))
sys.path.insert(0, str(HERE))
LETTERS = "ABCD"


def first_token_logp(ctx, X, FC, prompts, writes, positions, letter_ids, batch):
    """Per prompt: {letter: log of the summed probability of its single-token spellings as the first answer token}."""
    import torch
    pd, fd = X.pd, X.fd
    dummy = letter_ids["A"][0]           # any answer token: the first answer position is predicted from before it
    pad = ctx.bundle.tokenizer.pad_token_id
    out = []
    for i in range(0, len(prompts), batch):
        chunk = [pd.build_target(ctx.bundle, p, answer_ids=[dummy], finished=False, max_target_tokens=FC.MAX_TOKENS,
                                 eot_ids=ctx.eot) for p in prompts[i:i + batch]]
        packed = pd.pack_with_content_mask(chunk, pad, ctx.bundle.device)
        mask = fd.position_mask(packed, positions)
        hooks = [(layer, fd.ShiftHook(w["direction"], amounts=torch.full((len(chunk),), float(w["amount"])), mask=mask,
                                      decode_all=False, reference_direction=w.get("reference_direction")))
                 for layer, w in writes]
        with contextlib.ExitStack() as stack:
            for layer, hook in hooks:
                stack.enter_context(pd.clamp(ctx.bundle, layer, hook))
            logits = FC.forward_logits(ctx.bundle, packed["inputs"], grad=False)
        if any(hook.applied != 1 for _, hook in hooks):
            raise RuntimeError("every write must fire exactly once per forward")
        logp, rows_b, _ = fd.answer_token_logprobs(logits, packed)
        if rows_b.tolist() != list(range(len(chunk))):
            raise RuntimeError("one answer position per row expected")
        for b in range(len(chunk)):
            row = logp[b].double()
            out.append({L: float(torch.logsumexp(row[torch.tensor(ids, device=row.device)], 0))
                        for L, ids in letter_ids.items()})
    return out


def run(a):
    import newsets_gen as NG
    import cap_prompts as CP  # noqa: F401  (load_items formats the items with it)
    m = NG.load_models(a.models, a.model)
    X = NG.engine(m)
    sys.path.insert(0, str(NG.ROOT / m["code"] / "scripts" / "review_checks"))
    import ff_explore_ec_confidence as FC  # noqa: E402
    ctx = NG.context(X, m, 0)
    systems = NG.systems_of(ctx)
    a.task, a.split, a.only_ids = "mmlu", "all", None
    items, template = NG.load_items(a)
    if a.first_batches:
        items = items[:a.first_batches * NG.BATCH]
    tok = ctx.bundle.tokenizer
    letter_ids = {}
    for L in LETTERS:                    # as cap_mcq_prob.py v2: every single-token spelling of the letter
        cands = []
        for spelling in (L, " " + L):
            ids = tok.encode(spelling, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in cands:
                cands.append(ids[0])
        if not cands:
            raise SystemExit("letter %s has no single-token spelling for this tokenizer" % L)
        letter_ids[L] = cands
    if len({t for v in letter_ids.values() for t in v}) != sum(len(v) for v in letter_ids.values()):
        raise SystemExit("two letters share a token")
    import pyarrow.parquet as pq
    per = {}
    for r in pq.read_table(a.items).to_pylist():
        per.setdefault(r["subject"], []).append(r)
    gold = {i: LETTERS[per[i.split(":")[1]][int(i.split(":")[2])]["answer"]] for i, _ in items}
    ref = json.loads(Path(a.ref).read_text(encoding="utf-8")) if a.ref else None
    out = {"protocol": "cap_mcq_sweep_v1", "model": a.model, "items_file_sha256": NG.sha_file(a.items),
           "ids_file_sha256": NG.sha_file(a.ids), "identity_sha256": ctx.ident_sha, "script_sha256": NG.sha_file(__file__),
           "letter_token_ids": letter_ids, "n": len(items), "arms": {}, "skipped": {}}
    with X.E.fp32_output_projection(ctx.bundle, enabled=True):
        prompts = [X.pd.render_prompt(ctx.bundle, systems["x"], template, q) for _, q in items]
        for spec in a.arms:
            arm = spec.replace("E#", "E%d" % m["el"]).replace("C#", "C%d" % m["cl"])
            positions, core = ("gen", arm[:-4]) if arm.endswith("@gen") else ("content", arm)
            writes, descs = [], []
            if ":" in core:
                try:
                    _, parts = X.parse_arm(core, systems)
                    ws, trained = ctx.writes(parts)
                except (Exception, SystemExit) as e:      # noqa: BLE001  (an undefined direction: skip this arm, loudly)
                    out["skipped"][arm] = "%s: %s" % (type(e).__name__, e)
                    print("SKIP %s %s: %s" % (a.model, arm, out["skipped"][arm]), flush=True)
                    continue
                if trained != "content":
                    raise SystemExit("content-position directions expected")
                writes = [(layer, w) for layer, w, _ in ws]
                descs = [{k: d[k] for k in ("family", "layer", "kind", "beta", "amount", "gap")} for _, _, d in ws]
            lp = first_token_logp(ctx, X, FC, prompts, writes, positions, letter_ids, NG.BATCH)
            recs = []
            for (iid, _), v in zip(items, lp):
                pick = max(LETTERS, key=lambda L: (v[L], -LETTERS.index(L)))
                recs.append({"id": iid, "logp": v, "pick": pick, "gold": gold[iid],
                             "letter_mass": sum(math.exp(x) for x in v.values())})
            acc = sum(r["pick"] == r["gold"] for r in recs) / len(recs)
            mass = sorted(r["letter_mass"] for r in recs)[len(recs) // 2]
            out["arms"][arm] = {"rows": recs, "accuracy": acc, "median_letter_mass": mass, "positions": positions, "writes": descs}
            note = ""
            if ref is not None and core in ref["arms"] and positions == "content":
                old = {r["id"]: r for r in ref["arms"][core]["rows"]}
                same = sum(r["pick"] == old[r["id"]]["pick"] for r in recs)
                gap = max(abs(r["logp"][L] - old[r["id"]]["logp"][L]) for r in recs for L in LETTERS if old[r["id"]]["logp"][L] > -5)
                note = " | vs the v2 file: same pick %d/%d, largest log-prob difference %.4f" % (same, len(recs), gap)
            print("%s %-22s accuracy %.3f | median mass on A-D %.3f%s" % (a.model, arm, acc, mass, note), flush=True)
    ctx.still_current()
    o = Path(a.out)
    if o.exists():
        raise SystemExit("%s exists" % o)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(out), encoding="utf-8")


def ci(v, seed=20261119, n_boot=4000):
    rng = random.Random(seed)
    n = len(v)
    bs = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return sum(v) / n, bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot) - 1]


def report(paths):
    print("MMLU by option probability, write variants: accuracy; change vs no write in points [95%, items resampled]; picks changed")
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        arms = list(d["arms"])
        base = {r["id"]: r for r in d["arms"]["x"]["rows"]}
        print("\n== %s (%d items), no write %.3f" % (d["model"], d["n"], d["arms"]["x"]["accuracy"]))
        for arm in arms:
            if arm == "x":
                continue
            rows = d["arms"][arm]["rows"]
            diff = [(r["pick"] == r["gold"]) - (base[r["id"]]["pick"] == base[r["id"]]["gold"]) for r in rows]
            m_, lo, hi = ci(diff)
            moved = sum(r["pick"] != base[r["id"]]["pick"] for r in rows)
            amt = "; ".join("L%d %+.2f" % (w["layer"], w["amount"]) for w in d["arms"][arm]["writes"])
            print("    %-22s %.3f (%+.1f [%+.1f, %+.1f]) | picks changed %3d | mass %.2f | write %s" % (
                arm.split(":", 1)[1], d["arms"][arm]["accuracy"], 100 * m_, 100 * lo, 100 * hi, moved,
                d["arms"][arm]["median_letter_mass"], amt))
        for arm, why in d.get("skipped", {}).items():
            print("    skipped %s: %s" % (arm, why))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "report"])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--models", type=Path)
    ap.add_argument("--model")
    ap.add_argument("--items", type=Path)
    ap.add_argument("--ids", type=Path)
    ap.add_argument("--out")
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--ref")
    ap.add_argument("--first-batches", type=int, default=0)
    a = ap.parse_args()
    if a.phase == "report":
        report(a.paths)
        return
    if not (a.models and a.model and a.items and a.ids and a.out and a.arms):
        ap.error("run needs --models --model --items --ids --out --arms")
    if a.arms[0] != "x":
        ap.error("the first arm must be x (the no-write reference)")
    run(a)


if __name__ == "__main__":
    main()
