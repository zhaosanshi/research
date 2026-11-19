"""D2, minimal version (work queue 2.4; novelty/ours/2026-11-19_new_datasets_pilot_design.md section 7): the gap on a new
task's tune split, measured with the direction that is written (the October lesson: the read-out that measures the gap
must be the one that writes).

  readout AUROC     the block output of the E layer at the last prompt token, must-answer prompt (conf), no write, projected
                    on E's unit write direction (the direction newsets_gen.py writes); AUROC of the projection for right vs
                    wrong of the model's own no-write must-answer answer. Sign fixed in advance: larger projection = right.
  confidence AUROC  the mean log-probability of that same answer (newsets_gen.py conf output for arm conf), same labels.
  gap               readout AUROC - confidence AUROC.
Labels: DeepSeek Pro verdicts of the tune work dir for condition baseline_conf, verified by newsets_votes.py (one vote
per cell, bound to the current answer text; the confidence row must carry the sha1 of that same text); right (A) = 1,
wrong (B) = 0; hedges, refusals, empty answers, excluded cells and non-finite values are counted and left out; with only
one class the gap is recorded as not estimable. Reads only (no write, no generation); one JSON file out.
AUROC = Mann-Whitney with average ranks for ties.

usage: <model python> -I newsets_gap.py --models CFG --model NAME --task triviaqa|popqa --items BANK --pro-work WORK
                                        --gen-dir TUNE_GEN_DIR --conf-file CONF_JSON --out OUT_JSON
       python3 -I newsets_gap.py --selftest
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import newsets_gen as NG  # noqa: E402


def auroc(scores, labels):
    """P(score of a random positive > score of a random negative), ties counted one half."""
    pairs = sorted(zip(scores, labels))
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2.0 + 1.0
        i = j + 1
    pos = sum(1 for _, y in pairs if y == 1)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        raise ValueError("AUROC needs both classes")
    r = sum(rk for rk, (_, y) in zip(ranks, pairs) if y == 1)
    return (r - pos * (pos + 1) / 2.0) / (pos * neg)


def labels_of(work, gen_dir, task, want_ids):
    """Verified Pro labels of the no-write must-answer answers (newsets_votes.load_votes: one vote per cell, bound to the
    current generation's text). Returns ({pair_id: 1 right / 0 wrong}, {pair_id: judged text}, category counts)."""
    import math  # noqa: F401
    import newsets_votes as NV
    table, _, _ = NV.load_votes(work, gen_dir)
    lab, text, cnt = {}, {}, {"right": 0, "wrong": 0, "hedge": 0, "refusal or empty": 0, "excluded": 0}
    have = {p for p, c in table if c == "baseline_conf"}
    want = {"%s:%s" % (task, i) for i in want_ids}
    if have != want:
        raise SystemExit("the work dir judges %d baseline_conf answers, the tune split has %d (or other questions)" % (len(have), len(want)))
    for (p, c), v in table.items():
        if c != "baseline_conf":
            continue
        text[p] = v["text"]
        if v["state"] == "R":
            lab[p] = 1
            cnt["right"] += 1
        elif v["state"] == "W":
            lab[p] = 0
            cnt["wrong"] += 1
        elif v["state"] == "H":
            cnt["hedge"] += 1
        elif v["state"] == "X":
            cnt["refusal or empty"] += 1
        else:
            cnt["excluded"] += 1
    return lab, text, cnt


def run(args):
    import math
    import torch
    m = NG.load_models(args.models, args.model)
    X = NG.engine(m)
    ctx = NG.context(X, m, 0)
    systems = NG.systems_of(ctx)
    args.split, args.ids, args.only_ids, args.first_batches = "tune", None, None, 0
    items, _ = NG.load_items(args)
    want_ids = [i for i, _ in items]
    lab, judged_text, cnt = labels_of(args.pro_work, args.gen_dir, args.task, want_ids)
    judge_input = json.loads((Path(args.pro_work) / "judge_input.json").read_text(encoding="utf-8"))
    conf = json.loads(Path(args.conf_file).read_text(encoding="utf-8"))
    if (conf.get("arm"), conf.get("task"), conf.get("split"), conf.get("model")) != ("conf", args.task, "tune", args.model):
        raise SystemExit("--conf-file is not the tune confidence of arm conf for this model and task")
    if conf.get("generation_sha256") != judge_input["source_files"]["conf"]["sha256"]:
        raise SystemExit("the confidence was computed on another generation than the one judged")
    logp = {}
    for r in conf["rows"]:
        p = "%s:%s" % (args.task, r["id"])
        if p not in judged_text:
            raise SystemExit("confidence row %s is not a tune question" % p)
        if r["text_sha1"] != hashlib.sha1(judged_text[p].strip().encode("utf-8")).hexdigest():
            raise SystemExit("the confidence of %s belongs to another answer text" % p)
        logp[p] = r.get("mean_logp")
    ws, positions = ctx.writes([{"family": "E", "layer": m["el"], "kind": "w", "beta": 1.0}])
    layer, w, desc = ws[0]
    d = w["direction"].float().cpu()
    proj = {}
    prompts = [X.pd.render_prompt(ctx.bundle, systems["conf"], ctx.user_template, q) for _, q in items]
    for i in range(0, len(prompts), NG.BATCH):
        chunk = prompts[i:i + NG.BATCH]
        inputs = ctx.bundle.encode([p["prefix_text"] for p in chunk])
        for b, p in enumerate(chunk):
            got = inputs["input_ids"][b][inputs["attention_mask"][b].bool()].tolist()
            if got != p["prefix_ids"]:
                raise RuntimeError("batched prompt tokens differ from the single-prompt rendering")
            if int(inputs["attention_mask"][b][-1]) != 1:
                raise RuntimeError("expected left padding: the last position must be the last prompt token")
        cap = X.Capture()
        with torch.no_grad(), X.pd.clamp(ctx.bundle, layer, cap):
            X.forward_logits(ctx.bundle, inputs, grad=False)
        if cap.h is None:
            raise RuntimeError("no state captured")
        h = cap.h[:, -1, :]
        for (iid, _), v in zip(items[i:i + NG.BATCH], (h @ d).tolist()):
            proj["%s:%s" % (args.task, iid)] = float(v)
    bad_logp = [p for p in lab if logp.get(p) is None or not math.isfinite(logp[p])]
    bad_proj = [p for p in lab if not math.isfinite(proj[p])]
    cnt.update({"no finite confidence": len(bad_logp), "no finite projection": len(bad_proj)})
    ids = [p for p in sorted(lab) if p not in set(bad_logp) | set(bad_proj)]
    y = [lab[p] for p in ids]
    if 0 < sum(y) < len(y):
        ra, ca = auroc([proj[p] for p in ids], y), auroc([logp[p] for p in ids], y)
        gap, note = ra - ca, None
    else:
        ra = ca = gap = None
        note = "cannot be estimated: only one class among the labelled answers"
    out = {"protocol": "newsets_gap_v2", "model": args.model, "task": args.task, "split": "tune", "layer": layer,
           "direction_file_sha256": desc["direction_file_sha256"], "identity_sha256": ctx.ident_sha,
           "labels_from": str(args.pro_work), "judge_input_sha256": NG.sha_file(Path(args.pro_work) / "judge_input.json"),
           "conf_file_sha256": hashlib.sha256(Path(args.conf_file).read_bytes()).hexdigest(),
           "script_sha256": NG.sha_file(__file__), "counts": cnt, "n": len(ids), "n_right": sum(y), "n_wrong": len(y) - sum(y),
           "readout_auroc": ra, "confidence_auroc": ca, "gap": gap, "note": note,
           "rows": [{"id": p, "projection": proj[p], "mean_logp": logp[p], "right": lab[p]} for p in ids]}
    o = Path(args.out)
    if o.exists():
        raise SystemExit("%s exists" % o)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(out), encoding="utf-8")
    print("%s %s: %s | n %d (right %d) | readout AUROC %s | confidence AUROC %s | gap %s" % (
        args.task, args.model, json.dumps(cnt), len(ids), sum(y), ra, ca, gap if note is None else note), flush=True)


def selftest():
    assert abs(auroc([1, 2, 3, 4], [0, 0, 1, 1]) - 1.0) < 1e-12
    assert abs(auroc([4, 3, 2, 1], [0, 0, 1, 1]) - 0.0) < 1e-12
    assert abs(auroc([1, 1, 1, 1], [0, 1, 0, 1]) - 0.5) < 1e-12
    assert abs(auroc([1, 2, 2, 3], [0, 1, 0, 1]) - 0.875) < 1e-12
    print("selftest PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--models", type=Path)
    ap.add_argument("--model")
    ap.add_argument("--task", choices=["triviaqa", "popqa"])
    ap.add_argument("--items", type=Path)
    ap.add_argument("--pro-work", type=Path)
    ap.add_argument("--gen-dir", type=Path, help="the tune generation dir the work dir judged")
    ap.add_argument("--conf-file", type=Path)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not all((a.models, a.model, a.task, a.items, a.pro_work, a.gen_dir, a.conf_file, a.out)):
        ap.error("all of --models --model --task --items --pro-work --gen-dir --conf-file --out are required")
    run(a)


if __name__ == "__main__":
    main()
