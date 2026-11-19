"""MMLU scored by option probability (the protocol of ITI / LoFiT through lm-evaluation-harness: the model writes nothing
and cannot abstain; the option whose letter is most probable is its answer). Same 1,000 items, same prompt (the plain
system prompt x and the user text of cap_prompts.mmlu_user), same writes as the generation protocol of
newsets_gen.py; only the scoring changes (asked for by the project lead on 2026-11-19, 19:5x Beijing time).

For every item and arm: the log-probability of each letter A-D as the first answer token, teacher-forced with the
writes on the content positions, through the unchanged scorer of scripts/review_checks/ff_explore_ec_confidence.py
(score(); the letter's own position is predicted from the positions before it, so the write on the letter itself plays
no part). Arms: x (no write), x:E<l>.w.p2 (E as frozen), x:C<l>.w.m2 (C alone at its largest strength). N1 is not
defined here: nothing can be blank.
Output: one JSON per model with the four log-probabilities per item and arm, the pick, the mass on the four letters.

usage: <model python> -I cap_mcq_prob.py run --models CFG --model NAME --items MMLU_PARQUET --ids MMLU_IDS --out OUT_JSON
                                      [--first-batches K]
       python3 -I cap_mcq_prob.py report OUT_JSON...        (accuracy and paired changes vs no write, 95% intervals)
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "newsets"))
sys.path.insert(0, str(HERE))
LETTERS = "ABCD"


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
    # every single-token spelling of a letter counts as that letter ("D" and " D"): Qwen3.5-9B answers with the
    # space-prefixed token, the other four models with the bare one (2026-11-19 22:5x: the bare token alone left
    # Qwen3.5-9B with 1% of its first-token mass on the letters)
    letter_ids = {}
    for L in LETTERS:
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
    rows = pq.read_table(a.items).to_pylist()
    per = {}
    for r in rows:
        per.setdefault(r["subject"], []).append(r)
    gold = {i: LETTERS[per[i.split(":")[1]][int(i.split(":")[2])]["answer"]] for i, _ in items}
    e, c = "E%d.w.p2" % m["el"], "C%d.w.m2" % m["cl"]
    out = {"protocol": "cap_mcq_prob_v2", "model": a.model, "items_file_sha256": NG.sha_file(a.items),
           "ids_file_sha256": NG.sha_file(a.ids), "identity_sha256": ctx.ident_sha, "script_sha256": NG.sha_file(__file__),
           "scorer_sha256": NG.sha_file(FC.__file__), "letter_token_ids": letter_ids, "n": len(items), "arms": {}}
    with X.E.fp32_output_projection(ctx.bundle, enabled=True):
        prompts = [X.pd.render_prompt(ctx.bundle, systems["x"], template, q) for _, q in items]
        for arm in ("x", "x:" + e, "x:" + c):
            writes = []
            if ":" in arm:
                _, parts = X.parse_arm(arm, systems)
                ws, positions = ctx.writes(parts)
                if positions != "content":
                    raise SystemExit("content-position directions expected")
                writes = [(layer, w["direction"], float(w["amount"])) for layer, w, _ in ws]
            lp = {}
            for L in LETTERS:
                per_tok = []
                for t in letter_ids[L]:
                    ans = [{"token_ids": [t], "finished": False} for _ in items]
                    res = FC.score(ctx, prompts, ans, writes, None, batch=NG.BATCH)
                    per_tok.append([float(r[0][0]) for r in res])
                # log of the summed probability of the letter's spellings
                lp[L] = [max(v) + math.log(sum(math.exp(x - max(v)) for x in v)) for v in zip(*per_tok)]
            recs = []
            for k, (iid, _) in enumerate(items):
                v = {L: lp[L][k] for L in LETTERS}
                pick = max(LETTERS, key=lambda L: (v[L], -LETTERS.index(L)))
                recs.append({"id": iid, "logp": v, "pick": pick, "gold": gold[iid],
                             "letter_mass": sum(math.exp(x) for x in v.values())})
            acc = sum(r["pick"] == r["gold"] for r in recs) / len(recs)
            mass = sorted(r["letter_mass"] for r in recs)[len(recs) // 2]
            out["arms"][arm] = {"rows": recs, "accuracy": acc, "median_letter_mass": mass}
            print("%s %-14s accuracy %.3f | median mass on A-D %.3f" % (a.model, arm, acc, mass), flush=True)
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
    print("MMLU by option probability: accuracy per arm; change vs no write in points (95%, items resampled)")
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        arms = list(d["arms"])
        base = {r["id"]: r["pick"] == r["gold"] for r in d["arms"]["x"]["rows"]}
        line = ["%-12s no write %.3f" % (d["model"], d["arms"]["x"]["accuracy"])]
        for arm in arms[1:]:
            rows = d["arms"][arm]["rows"]
            diff = [(r["pick"] == r["gold"]) - base[r["id"]] for r in rows]
            m_, lo, hi = ci(diff)
            line.append("%s %.3f (%+.1f [%+.1f, %+.1f])" % ("E 2" if ":E" in arm else "C = 2", d["arms"][arm]["accuracy"], 100 * m_, 100 * lo, 100 * hi))
        line.append("mass on A-D %.2f" % min(v["median_letter_mass"] for v in d["arms"].values()))
        print(" | ".join(line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "report"])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--models", type=Path)
    ap.add_argument("--model")
    ap.add_argument("--items", type=Path)
    ap.add_argument("--ids", type=Path)
    ap.add_argument("--out")
    ap.add_argument("--first-batches", type=int, default=0)
    a = ap.parse_args()
    if a.phase == "report":
        report(a.paths)
        return
    if not (a.models and a.model and a.items and a.ids and a.out):
        ap.error("run needs --models --model --items --ids --out")
    run(a)


if __name__ == "__main__":
    main()
