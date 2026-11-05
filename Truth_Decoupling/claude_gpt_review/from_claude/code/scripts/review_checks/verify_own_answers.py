#!/usr/bin/env python3
"""Answer first, then verify the OWN answer with the TF-protocol E direction, abstain if judged false (Claude, 2026-11-05).

usage: verify_own_answers.py MODEL V1_RUN_DIR E23_DIR OUT_JSON [--code-dir DIR] [--n-boot N]

Nothing is re-selected and nothing is generated or judged anew:
  * own answers = the baseline free-generation answers of the QA-DAS v1 run (192 tuning, 529 test questions) with their
    semantic judge labels (tune_scored_items.json, test_scored_items.json); refusal-only-gold test items are excluded
    exactly as in the main table (n = 523);
  * E configurations = the frozen E23 selections of the TF protocol (96 estimation pairs; selection on the 192 tuning
    TF pairs with the exact-integer rule e23_compare_estimators.rule_pick), for DAS and mean difference, single layer and
    (when the sweep exists) the multi-layer window;
  * directions, centres and the write rule are rebuilt exactly as e23_knob_sweep.py does for pass B: MD direction and
    centre cB from the 192 estimation statements at the last prompt token; DAS = load_das_direction(..., align_to=MD);
    SteerSpec(mode="amplify", center=cB) at the last prompt token; native output precision as in E23.
Each own answer is written as the TF statement 'The answer to the question "Q?" is: A.' and read with the TF prompt.
Reported on tuning and test: AUROC of log P(True) - log P(False) for correct vs wrong own answers, says-True rates, and
two abstention gates -- the model's own verdict (abstain when it says False) and a threshold chosen on tuning -- as the
change of (correct - wrong)/n over the main-table questions, with paired bootstrap CIs vs the unsteered verifier.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("v1_run")
ap.add_argument("e23_dir")
ap.add_argument("out")
ap.add_argument("--code-dir", default="/workspace/Truth_Decoupling/review_e25/claude_gpt_review/from_claude/code")
ap.add_argument("--n-boot", type=int, default=2000)
args = ap.parse_args()
sys.path[:0] = [args.code_dir + "/scripts", args.code_dir]
import common  # noqa: F401,E402
from common import load_split, tf_prompt  # noqa: E402
from core.config import load_experiment, load_model_config  # noqa: E402
from core.directions import load_das_direction, mean_diff_direction  # noqa: E402
from core.intervene import SteerSpec, next_token_logits  # noqa: E402
from core.model_io import capture_last_token, load_model  # noqa: E402
from core.readout import build_label_token_map, label_logprobs_from_logits  # noqa: E402
from e23_compare_estimators import rule_pick  # noqa: E402

exp = load_experiment("E2b_tqa")
prompts = exp["prompts"]
st = prompts["statement"]
e23 = Path(args.e23_dir)
run = Path(args.v1_run)

# ---- frozen E23 selections ------------------------------------------------- #
configs = {}
for tag in ("das_tf_full", "meandiff_full", "das_tf_spans", "meandiff_spans", "das_tfce_full"):
    jp, zp = e23 / ("knob_sweep_%s.json" % tag), e23 / ("knob_sweep_%s.npz" % tag)
    if not jp.exists() or not zp.exists():
        continue
    j = json.loads(jp.read_text())
    z = np.load(zp, allow_pickle=True)
    lab, tune = z["labels"].astype(bool), z["half_a"].astype(bool)
    scale = int(lab[tune].sum()) * int((~lab[tune]).sum())
    names = [c for c in j["conds"] if c.startswith("E_")]
    chosen = rule_pick(j["conds"], names, scale)
    if chosen is None:
        configs[tag] = None
        continue
    if not chosen.startswith("E_L"):
        raise SystemExit("unexpected E condition name %r" % chosen)
    body, alpha = chosen[len("E_L"):].split("_a")
    layers = [int(x) for x in body.split("-")]
    est = "das" if j.get("e_estimator") == "das" else "md"
    configs[tag] = {"name": chosen, "estimator": est, "layers": layers, "alpha": float(alpha),
                    "stored_selection": (j.get("selection") or {}).get("E", {}).get("chosen"),
                    "tf_test_dJ": j["conds"][chosen]["B"]["J"] - j["conds"]["baseline"]["B"]["J"],
                    "tf_test_dAUROC": j["conds"][chosen]["B"]["AUROC"] - j["conds"]["baseline"]["B"]["AUROC"],
                    "das_dir": (j.get("das") or {}).get("dir"), "das_variant": (j.get("das") or {}).get("variant", ""),
                    "n_est": j["n_est"]}
print("frozen E23 selections:", json.dumps({k: (v and {kk: v[kk] for kk in ("name", "stored_selection", "tf_test_dJ")}) for k, v in configs.items()}), flush=True)

# ---- model, estimation statements, directions and centres ------------------ #
bundle = load_model(load_model_config(args.model))
tok = bundle.tokenizer
est = []
for ds in exp["eval_datasets"]:
    est += load_split(ds, "calib") + load_split(ds, "dev")
y_est = np.array([bool(r["label"]) for r in est])
for v in configs.values():
    if v and v["n_est"] != len(est):
        raise SystemExit("estimation set size %d differs from the sweep's n_est %d" % (len(est), v["n_est"]))
need = sorted({l for v in configs.values() if v for l in v["layers"]})
eB = [tf_prompt(bundle, prompts, r["text"], allow_idk=False) for r in est]
aB, _ = capture_last_token(bundle, eB, need, components=("resid",), batch_size=8)
dirs = {}
for l in need:
    HB = aB[("resid", l)].numpy().astype(np.float64)
    wmd, _ = mean_diff_direction(HB[y_est], HB[~y_est])
    cB = (HB[y_est].mean(0) + HB[~y_est].mean(0)) / 2
    dirs[l] = {"md": wmd, "cB": cB}
for v in configs.values():
    if v and v["estimator"] == "das":
        for l in v["layers"]:
            if "das" not in dirs[l]:
                wd, meta = load_das_direction(v["das_dir"], "E", l, variant=v["das_variant"], align_to=dirs[l]["md"])
                if wd is None:
                    raise SystemExit("missing DAS direction L%d in %s" % (l, v["das_dir"]))
                dirs[l]["das"] = wd


def specs_for(v):
    if v is None:
        return None
    return [SteerSpec(dirs[l][v["estimator"]], l, mode="amplify", alpha=v["alpha"], center=dirs[l]["cB"]) for l in v["layers"]]


# ---- own answers -------------------------------------------------------------- #
def statement(q, a):
    a = " ".join(a.split())
    return 'The answer to the question "%s" is: %s' % (q.strip().rstrip("?").strip() + "?", a.rstrip(".") + ".")


def load_items(fname):
    items = json.loads((run / fname).read_text())["evaluations"]
    rows = []
    for e in items:
        if e.get("gold_is_abstention"):
            continue
        out, ans = e["outcome"]["baseline"], e["outputs"]["baseline"]
        rows.append({"pair_id": e["pair_id"], "outcome": out, "text": statement(e["question"], ans) if ans.strip() else None})
    return rows


data = {s: load_items(f) for s, f in (("tune", "tune_scored_items.json"), ("test", "test_scored_items.json"))}
first = next(r["text"] for r in data["tune"] if r["text"])
mB = build_label_token_map(tok, tf_prompt(bundle, prompts, first, allow_idk=False),
                           {"True": st["label_variants"]["True"], "False": st["label_variants"]["False"]})
readers = {"baseline": None, **{k: v for k, v in configs.items() if v}}
margins = {s: {} for s in data}
with torch.no_grad():
    for s, rows in data.items():
        idx = [i for i, r in enumerate(rows) if r["outcome"] in ("correct", "wrong") and r["text"]]
        texts = [tf_prompt(bundle, prompts, rows[i]["text"], allow_idk=False) for i in idx]
        for name, v in readers.items():
            m = np.full(len(rows), np.nan)
            out = []
            for b in range(0, len(texts), 8):
                lg = next_token_logits(bundle, bundle.encode(texts[b:b + 8]), specs_for(v)).numpy()
                for x in lg:
                    lp = label_logprobs_from_logits(x, mB, use_abstain=False)
                    out.append(lp["True"] - lp["False"])
            m[idx] = out
            margins[s][name] = m
        print(s, "answered", len(idx), "of", len(rows), flush=True)


def auroc(sc, y):
    pos, neg = sc[y == 1], sc[y == 0]
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def outcomes(s):
    return np.array([{"correct": 1, "wrong": -1, "abstain": 0}[r["outcome"]] if r["outcome"] in ("correct", "wrong", "abstain") else 0 for r in data[s]])


o_tune, o_test = outcomes("tune"), outcomes("test")
rng = np.random.default_rng(0)
boot = [rng.integers(0, len(o_test), len(o_test)) for _ in range(args.n_boot)]


def cmw(o, keep):
    return float((o * keep).sum()) / len(o)


def tuned_threshold(m, o):
    ans = ~np.isnan(m)
    cand = np.unique(np.concatenate([[-np.inf], m[ans]]))
    best_t, best_v = -np.inf, -1e9
    for t in cand:
        keep = np.where(ans, m >= t, False) | (o == 0)
        v = cmw(o, keep)
        if v > best_v + 1e-12:
            best_t, best_v = t, v
    return float(best_t)


res = {"model": args.model, "configs": configs, "precision": "native output projection, as E23",
       "n": {s: {"items": len(data[s]), "answered": int((~np.isnan(margins[s]["baseline"])).sum()),
                 "correct": int((outcomes(s) == 1).sum()), "wrong": int((outcomes(s) == -1).sum())} for s in data},
       "readers": {}}
base_keep = {}
for name in readers:
    r = {}
    for s, o in (("tune", o_tune), ("test", o_test)):
        m = margins[s][name]
        ans = ~np.isnan(m)
        y = (o[ans] == 1).astype(int)
        r[s] = {"auroc": auroc(m[ans], y), "says_true_correct": float((m[ans][y == 1] > 0).mean()),
                "says_true_wrong": float((m[ans][y == 0] > 0).mean())}
        r[s]["J_at_0"] = r[s]["says_true_correct"] - r[s]["says_true_wrong"]
    t = tuned_threshold(margins["tune"][name], o_tune)
    gates = {}
    for gname, thr in (("own_verdict", 0.0), ("tuned_threshold", t)):
        m = margins["test"][name]
        keep = np.where(~np.isnan(m), m >= thr, False) | (o_test == 0)
        gates[gname] = {"threshold": thr, "keep": keep}
    r["gates"] = {}
    for gname, g in gates.items():
        keep = g["keep"]
        before, after = cmw(o_test, np.ones_like(keep)), cmw(o_test, keep)
        d = [cmw(o_test[i], keep[i]) - cmw(o_test[i], np.ones(len(i), bool)) for i in boot]
        r["gates"][gname] = {"threshold": g["threshold"], "test_cmw_before": before, "test_cmw_after": after,
                             "test_delta": after - before, "test_delta_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                             "abstained": int((~keep).sum()), "correct_lost": int(((o_test == 1) & ~keep).sum()),
                             "wrong_removed": int(((o_test == -1) & ~keep).sum())}
        if name == "baseline":
            base_keep[gname] = keep
        else:
            dd = [cmw(o_test[i], keep[i]) - cmw(o_test[i], base_keep[gname][i]) for i in boot]
            r["gates"][gname]["vs_unsteered_verifier"] = {
                "delta": after - cmw(o_test, base_keep[gname]), "ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}
    res["readers"][name] = r
    tt = r["test"]
    line = "%-15s %-16s TEST AUROC %.3f (tune %.3f) J@0 %+.3f says True c/w %.2f/%.2f" % (
        name, (readers[name] or {}).get("name", "-"), tt["auroc"], r["tune"]["auroc"], tt["J_at_0"], tt["says_true_correct"], tt["says_true_wrong"])
    for gname in ("own_verdict", "tuned_threshold"):
        g = r["gates"][gname]
        line += " | %s: (c-w)/n %+.3f -> %+.3f, abstain %d (lost %d c, removed %d w)%s" % (
            gname, g["test_cmw_before"], g["test_cmw_after"], g["abstained"], g["correct_lost"], g["wrong_removed"],
            ("" if name == "baseline" else ", vs unsteered %+.3f [%+.3f, %+.3f]" % (
                g["vs_unsteered_verifier"]["delta"], *g["vs_unsteered_verifier"]["ci95"])))
    print(line, flush=True)
json.dump({**res, "margins": {s: {k: [None if np.isnan(x) else float(x) for x in v] for k, v in d.items()} for s, d in margins.items()},
           "pair_ids": {s: [r["pair_id"] for r in data[s]] for s in data}}, open(args.out, "w"), indent=1)
print("saved", args.out)
