#!/usr/bin/env python3
"""The TF-protocol E coordinate as a detector of the model's OWN wrong answers (Claude, 2026-11-05).

usage: own_answer_tfdetect.py MODEL V1_RUN_DIR OUT_JSON [--code-dir DIR] [--n-boot N]
Own answers = the v1 baseline free-generation answers with their semantic judge labels (192 tuning, 523 main-table test
questions; abstentions and refusal-only-gold items excluded). Each answer is written as the TF statement and read under the
TF prompt (exactly as verify_own_answers.py). Scores:
  coord_<est>_L  projection of the last-prompt-token block output on the layer-L E direction of estimator <est>:
                 dasce (gold-label CE DAS, results_das_tfce E1), daskl (round-4 KL DAS, results_das_tf E1), md (mean
                 difference from the 192 estimation statements, rebuilt as e23 does); sign oriented on tuning data;
  verdict        the unsteered TF verdict margin log P(True) - log P(False) (the model's own judgement).
Per estimator the layer with the best TUNING AUROC is picked; TEST AUROC with paired bootstrap differences against the
verdict and between estimators.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("v1_run")
ap.add_argument("out")
ap.add_argument("--code-dir", default="/workspace/Truth_Decoupling/review_e25/claude_gpt_review/from_claude/code")
ap.add_argument("--n-boot", type=int, default=2000)
ap.add_argument("--direction-root", default="/workspace/Truth_Decoupling",
                help="root containing results_das_tfce and results_das_tf (existing defaults unchanged)")
args = ap.parse_args()
sys.path[:0] = [args.code_dir + "/scripts", args.code_dir]
import common  # noqa: F401,E402
from common import load_split, tf_prompt  # noqa: E402
from core.config import load_experiment, load_model_config  # noqa: E402
from core.directions import load_das_direction, mean_diff_direction  # noqa: E402
from core.model_io import capture_last_token, load_model  # noqa: E402
from core.readout import build_label_token_map, label_logprobs_from_logits  # noqa: E402

ROOT = args.direction_root
exp = load_experiment("E2b_tqa")
prompts = exp["prompts"]
st = prompts["statement"]
run = Path(args.v1_run)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def vector_sha256(value):
    """Hash the actual loaded/aligned float64 vector, with shape in the byte stream."""
    a = np.asarray(value, dtype="<f8")
    return hashlib.sha256(json.dumps(list(a.shape)).encode() + b"\0" + a.tobytes(order="C")).hexdigest()


def checkpoint_identity(directory, variant, meta, vector):
    files = (meta or {}).get("files") or []
    if not files:
        raise ValueError("DAS loader did not report the checkpoint files it used")
    return {"directory": str(Path(directory).resolve()), "variant": variant,
            "checkpoint_files": {name: file_sha256(Path(directory) / name) for name in files},
            "loaded_aligned_vector_sha256": vector_sha256(vector), "loader_metadata": meta}


source_files = {s: {"path": str((run / fname).resolve()), "sha256": file_sha256(run / fname)}
                for s, fname in (("tune", "tune_scored_items.json"), ("test", "test_scored_items.json"))}
bundle = load_model(load_model_config(args.model))
tok = bundle.tokenizer
L = list(range(bundle.num_layers))

est = []
for ds in exp["eval_datasets"]:
    est += load_split(ds, "calib") + load_split(ds, "dev")
y_est = np.array([bool(r["label"]) for r in est])
estimation_identity = {"records": [{k: r.get(k) for k in ("id", "pair_id", "label", "text")} for r in est],
                       "prompts": prompts, "model_config": bundle.cfg}
estimation_sha256 = json_sha256(estimation_identity)
aE, _ = capture_last_token(bundle, [tf_prompt(bundle, prompts, r["text"], allow_idk=False) for r in est], L, batch_size=8)
dirs = {"md": {}, "dasce": {}, "daskl": {}}
direction_artifacts = {"md": {}, "dasce": {}, "daskl": {}}
for l in L:
    HB = aE[("resid", l)].numpy().astype(np.float64)
    wmd, _ = mean_diff_direction(HB[y_est], HB[~y_est])
    dirs["md"][l] = wmd
    direction_artifacts["md"][str(l)] = {"estimation_identity_sha256": estimation_sha256,
                                        "vector_sha256": vector_sha256(wmd)}
    for name, d, var in (("dasce", "%s/results_das_tfce/E1/%s" % (ROOT, args.model), "tf-ce-bidir"),
                         ("daskl", "%s/results_das_tf/E1/%s" % (ROOT, args.model), "tf")):
        w, meta = load_das_direction(d, "E", l, variant=var, align_to=wmd)
        if w is not None:
            dirs[name][l] = np.asarray(w, dtype=np.float64)
            direction_artifacts[name][str(l)] = checkpoint_identity(d, var, meta, dirs[name][l])


def statement(q, a):
    a = " ".join(a.split())
    return 'The answer to the question "%s" is: %s' % (q.strip().rstrip("?").strip() + "?", a.rstrip(".") + ".")


data = {}
for split, fname in (("tune", "tune_scored_items.json"), ("test", "test_scored_items.json")):
    source_bytes = (run / fname).read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != source_files[split]["sha256"]:
        raise ValueError("source scoring file changed during the run: %s" % fname)
    rows = [e for e in json.loads(source_bytes)["evaluations"]
            if not e.get("gold_is_abstention") and e["outcome"]["baseline"] in ("correct", "wrong") and e["outputs"]["baseline"].strip()]
    texts = [tf_prompt(bundle, prompts, statement(e["question"], e["outputs"]["baseline"]), allow_idk=False) for e in rows]
    acts, lg = capture_last_token(bundle, texts, L, batch_size=8, return_logits=True)
    mB = build_label_token_map(tok, texts[0], {"True": st["label_variants"]["True"], "False": st["label_variants"]["False"]})
    verdict = []
    for x in lg.numpy():
        lp = label_logprobs_from_logits(x, mB, use_abstain=False)
        verdict.append(lp["True"] - lp["False"])
    data[split] = {"y": np.array([1 if e["outcome"]["baseline"] == "correct" else 0 for e in rows]),
                   "H": {l: acts[("resid", l)].numpy().astype(np.float64) for l in L}, "verdict": np.array(verdict),
                   "ids": [e["pair_id"] for e in rows]}
    print(split, "answered", len(rows), "correct", int(data[split]["y"].sum()), flush=True)


def auroc(s, y):
    pos, neg = s[y == 1], s[y == 0]
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def detector_scores(data, dirs):
    """Orient and select on tuning labels only; preserve every candidate for independent replay."""
    yt = data["tune"]["y"]
    scores = {"verdict": (data["tune"]["verdict"], data["test"]["verdict"])}
    candidates = {"verdict": {"family": "verdict", "layer": None, "sign": 1.0,
                               "tune_auroc": auroc(scores["verdict"][0], yt)}}
    for est_name, by_layer in dirs.items():
        for l, w in by_layer.items():
            st_, ss_ = data["tune"]["H"][l] @ w, data["test"]["H"][l] @ w
            raw_auc = auroc(st_, yt)
            sign = 1.0 if raw_auc >= 0.5 else -1.0
            name = "%s_L%d" % (est_name, l)
            scores[name] = (sign * st_, sign * ss_)
            candidates[name] = {"family": est_name, "layer": int(l), "sign": sign,
                                "tune_auroc_before_orientation": raw_auc,
                                "tune_auroc": auroc(scores[name][0], yt)}
    picks = {"verdict": "verdict"}
    for est_name in dirs:
        names = [k for k in scores if k.startswith(est_name + "_L")]
        if names:
            picks[est_name] = max(names, key=lambda k: (candidates[k]["tune_auroc"], -candidates[k]["layer"]))
    return scores, picks, candidates


def audit_payload(data, scores, picks, candidates, direction_artifacts, source_files, estimation_identity):
    per_split, selected = {}, {}
    for index, split in enumerate(("tune", "test")):
        ids = list(data[split]["ids"])
        labels = np.asarray(data[split]["y"])
        if len(ids) != len(set(ids)) or labels.shape != (len(ids),) or not np.isin(labels, [0, 1]).all():
            raise ValueError("invalid or duplicate %s question identities/labels" % split)
        columns = {}
        for name, by_split in scores.items():
            values = np.asarray(by_split[index], dtype=np.float64)
            if values.shape != (len(ids),) or not np.isfinite(values).all():
                raise ValueError("unaligned/nonfinite scores: %s/%s" % (split, name))
            columns[name] = values.tolist()
        per_split[split] = {"ids": ids, "labels": labels.astype(int).tolist(), "scores": columns}
        selected[split] = {family: columns[name] for family, name in picks.items()}
    if set(per_split["tune"]["ids"]) & set(per_split["test"]["ids"]):
        raise ValueError("tuning and test question identities overlap")
    return {"schema": "own_answer_tfdetect_v2", "data": per_split, "selected_scores": selected,
            "selection": {"split": "tune", "orientation": "sign from tuning AUROC >= 0.5",
                          "tie_break": "smaller layer", "picks": picks, "candidates": candidates},
            "direction_artifacts": direction_artifacts, "source_files": source_files,
            "estimation_identity": estimation_identity,
            "estimation_identity_sha256": json_sha256(estimation_identity)}


def legacy_payload_fields(out, data, scores, dirs, candidates, root, model, n_est):
    """Retain Claude's 17b2a768 NPZ schema and JSON aliases; add a payload digest."""
    payload = Path(out).with_suffix(".items.npz")
    yt, ys = data["tune"]["y"], data["test"]["y"]
    arrays = {"ids_tune": np.array(data["tune"]["ids"]), "ids_test": np.array(data["test"]["ids"]),
              "y_tune": yt, "y_test": ys,
              "verdict_tune": data["tune"]["verdict"], "verdict_test": data["test"]["verdict"]}
    signs, tune = {}, {}
    for family, by_layer in dirs.items():
        layers = sorted(by_layer)
        arrays["layers_" + family] = np.array(layers)
        for split, index, n in (("tune", 0, len(yt)), ("test", 1, len(ys))):
            arrays["scores_%s_%s" % (split, family)] = (
                np.stack([scores["%s_L%d" % (family, l)][index] for l in layers]) if layers else np.zeros((0, n)))
        signs[family] = {str(l): candidates["%s_L%d" % (family, l)]["sign"] for l in layers}
        tune[family] = {str(l): candidates["%s_L%d" % (family, l)]["tune_auroc"] for l in layers}
    np.savez_compressed(payload, **arrays)
    sources = {"dasce": {"dir": "%s/results_das_tfce/E1/%s" % (root, model), "variant": "tf-ce-bidir"},
               "daskl": {"dir": "%s/results_das_tf/E1/%s" % (root, model), "variant": "tf"},
               "md": {"dir": None, "variant": "mean difference of the %d estimation statements (calib + dev)" % n_est}}
    return {"per_layer_tune_auroc": tune, "signs_from_tuning": signs, "direction_sources": sources,
            "selection_rule": "per estimator: max tuning AUROC, ties -> lower layer; sign fixed on tuning data",
            "items_payload": payload.name, "items_payload_sha256": file_sha256(payload)}


yt, ys = data["tune"]["y"], data["test"]["y"]
scores, picks, candidates = detector_scores(data, dirs)
table = {}
rng = np.random.default_rng(0)
boot = [rng.integers(0, len(ys), len(ys)) for _ in range(args.n_boot)]
for fam, name in picks.items():
    a = auroc(scores[name][1], ys)
    bd = [auroc(scores[name][1][i], ys[i]) for i in boot]
    table[fam] = {"pick": name, "tune_auroc": auroc(scores[name][0], yt), "test_auroc": a,
                  "test_auroc_ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]}
paired = {}
for a_f, b_f in (("dasce", "verdict"), ("daskl", "verdict"), ("md", "verdict"), ("dasce", "md"), ("dasce", "daskl")):
    if a_f in picks and b_f in picks:
        sa, sb = scores[picks[a_f]][1], scores[picks[b_f]][1]
        d = [auroc(sa[i], ys[i]) - auroc(sb[i], ys[i]) for i in boot]
        paired["%s_minus_%s" % (a_f, b_f)] = {"delta": auroc(sa, ys) - auroc(sb, ys),
                                              "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
per_layer_test = {est_name: {str(l): auroc(scores["%s_L%d" % (est_name, l)][1], ys) for l in dirs[est_name]} for est_name in dirs}
audit = audit_payload(data, scores, picks, candidates, direction_artifacts, source_files, estimation_identity)
compatibility = legacy_payload_fields(args.out, data, scores, dirs, candidates, ROOT, args.model, len(est))
json.dump({"model": args.model, "v1_run": str(run),
           "n": {"tune": int(len(yt)), "test": int(len(ys)), "test_correct": int(ys.sum())},
           "picked": table, "paired_test_auroc_differences": paired, "per_layer_test_auroc": per_layer_test,
           "n_boot": args.n_boot, "bootstrap_seed": 0, **compatibility, **audit},
          open(args.out, "w"), indent=1)
for fam, t in table.items():
    print("%-8s %-10s tune AUROC %.3f | TEST AUROC %.3f [%.3f, %.3f]" % (fam, t["pick"], t["tune_auroc"], t["test_auroc"], *t["test_auroc_ci95"]))
for k, v in paired.items():
    print("paired TEST AUROC %-20s %+.3f [%+.3f, %+.3f]" % (k, v["delta"], *v["ci95"]))
print("saved", args.out)
