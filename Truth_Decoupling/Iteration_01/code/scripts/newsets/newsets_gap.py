"""D2, minimal version (work queue 2.4; novelty/ours/2026-11-19_new_datasets_pilot_design.md section 7): the gap on a new
task's tune split, measured with the direction that is written (the October lesson: the read-out that measures the gap
must be the one that writes).

  readout AUROC     the block output of the E layer at the last prompt token, must-answer prompt (conf), no write, projected
                    on E's unit write direction (the direction newsets_gen.py writes); AUROC of the projection for right vs
                    wrong of the model's own no-write must-answer answer. Sign fixed in advance: larger projection = right.
  confidence AUROC  the mean log-probability of that same answer (newsets_gen.py conf output for arm conf), same labels.
  gap               readout AUROC - confidence AUROC.
Labels: DeepSeek Pro verdicts of the tune work dir for condition baseline_conf; right (A) = 1, wrong (B) = 0; hedges,
refusals and cells without a verdict are left out. Reads only (no write, no generation); one JSON file out.
AUROC = Mann-Whitney with average ranks for ties.

usage: <model python> -I newsets_gap.py --models CFG --model NAME --task triviaqa|popqa --items BANK --pro-work WORK
                                        --conf-file CONF_JSON --out OUT_JSON
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


def labels_of(work, task):
    out = {}
    for line in open(Path(work) / "judge2" / "single_pro.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["condition"] != "baseline_conf" or r["status"] != "scored":
            continue
        if r["label"] in ("A", "B"):
            out[r["pair_id"]] = 1 if r["label"] == "A" else 0
    if not out:
        raise SystemExit("no right / wrong verdicts for baseline_conf in %s" % work)
    if any(not p.startswith(task + ":") for p in out):
        raise SystemExit("the work dir is not a %s work dir" % task)
    return out


def run(args):
    import torch
    m = NG.load_models(args.models, args.model)
    X = NG.engine(m)
    ctx = NG.context(X, m, 0)
    systems = NG.systems_of(ctx)
    args.split, args.ids, args.only_ids, args.first_batches = "tune", None, None, 0
    items, _ = NG.load_items(args)
    lab = labels_of(args.pro_work, args.task)
    conf = json.loads(Path(args.conf_file).read_text(encoding="utf-8"))
    if conf.get("arm") != "conf" or conf.get("task") != args.task or conf.get("split") != "tune" or conf.get("model") != args.model:
        raise SystemExit("--conf-file is not the tune confidence of arm conf for this model and task")
    logp = {"%s:%s" % (args.task, r["id"]): r.get("mean_logp") for r in conf["rows"]}
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
    ids = [p for p in sorted(lab) if p in proj and logp.get(p) is not None]
    y = [lab[p] for p in ids]
    ra = auroc([proj[p] for p in ids], y)
    ca = auroc([logp[p] for p in ids], y)
    out = {"protocol": "newsets_gap_v1", "model": args.model, "task": args.task, "split": "tune", "layer": layer,
           "direction_file_sha256": desc["direction_file_sha256"], "identity_sha256": ctx.ident_sha,
           "labels_from": str(args.pro_work), "conf_file_sha256": hashlib.sha256(Path(args.conf_file).read_bytes()).hexdigest(),
           "script_sha256": NG.sha_file(__file__), "n": len(ids), "n_right": sum(y), "n_wrong": len(y) - sum(y),
           "readout_auroc": ra, "confidence_auroc": ca, "gap": ra - ca,
           "rows": [{"id": p, "projection": proj[p], "mean_logp": logp[p], "right": lab[p]} for p in ids]}
    o = Path(args.out)
    if o.exists():
        raise SystemExit("%s exists" % o)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(out), encoding="utf-8")
    print("%s %s: n %d (right %d) | readout AUROC %.3f | confidence AUROC %.3f | gap %+.3f" % (
        args.task, args.model, len(ids), sum(y), ra, ca, ra - ca), flush=True)


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
    ap.add_argument("--conf-file", type=Path)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not all((a.models, a.model, a.task, a.items, a.pro_work, a.conf_file, a.out)):
        ap.error("all of --models --model --task --items --pro-work --conf-file --out are required")
    run(a)


if __name__ == "__main__":
    main()
