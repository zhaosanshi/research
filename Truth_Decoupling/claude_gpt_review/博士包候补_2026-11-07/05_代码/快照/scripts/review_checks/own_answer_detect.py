#!/usr/bin/env python3
"""Does a QA-DAS (or mean-difference) direction know when the model's OWN answer is wrong? (Claude, 2026-11-05)

usage: own_answer_detect.py MODEL RUN_DIR OUT_JSON [--code-dir DIR] [--n-boot N]
Uses only finished artifacts of a v1 run: the baseline free-generation answers and their semantic judge labels on the
192 tuning and 529 test questions (tune_scored_items.json, test_scored_items.json), the bank (MD directions) and the
trained DAS checkpoints. No new generation, no judge calls.

For every answered question (judge label correct or wrong; refusal-only-gold test questions excluded as in the main
table) the model's own answer is re-encoded exactly as v1 encodes answers (encode_answer: QA prompt + raw answer text),
and the block outputs at the last answer token are read for every layer. Scores compared:
  das_L / md_L   projection h·w of the layer-L direction (sign oriented on tuning data),
  probe          L2 logistic regression on the same states (layer and C chosen by 5-fold CV on tuning only),
  selfcheck_F0   the v1 same-turn verdict margin log P(" True") - log P(" False") on the model's own answer,
  selfcheck_F1   the same question asked in a new user turn (readout_formats.py F1).
Selection on TUNING only: per score family the layer with the best tuning AUROC. Report: TEST AUROC (bootstrap CI)
for correct vs wrong, and a selective-answering gate (abstain when the score is below a threshold chosen on tuning to
maximise (correct - wrong)/n): TEST (correct - wrong)/n before/after, correct answers lost, wrong answers removed.
"""
import argparse
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("run_dir")
p.add_argument("out")
p.add_argument("--code-dir", default="/workspace/Truth_Decoupling/review_e25/claude_gpt_review/from_claude/code")
p.add_argument("--n-boot", type=int, default=2000)
args = p.parse_args()
sys.path[:0] = [args.code_dir + "/scripts", args.code_dir]
import common  # noqa: F401,E402
import e25_qa_das as pipeline  # noqa: E402
from core.model_io import capture_hooks, forward_logits  # noqa: E402
from core.qa_context import _token_ids, encode_answer  # noqa: E402
from core.qa_das import encode_selfcheck  # noqa: E402
from core.qa_precision import fp32_output_projection  # noqa: E402
from core.readout import DEFAULT_LABEL_VARIANTS, build_label_token_map  # noqa: E402

run = Path(args.run_dir)
ns = SimpleNamespace(run_dir=run, model=args.model)
meta, bank, prepared = pipeline.load_bank(ns)
meta, bank = pipeline.runtime_bank(ns, meta, bank)
cfg = meta["config"]
# The F0 comparison readout is the fixed v1 same-turn self-check, whatever readout the run itself trained on
# (a user_turn run has no selfcheck_suffix; 2026-11-05 fix).
from core.config import load_json_config, resolve_config_path  # noqa: E402
F0 = load_json_config(resolve_config_path("experiments", "E25_qa_das_v1"))
layers = list(meta["layers"])
das_layers = list(meta["das_candidate_layers"])
bundle = pipeline.open_bundle(meta, prepared)
tok = bundle.tokenizer
stub = (bundle.cfg.get("chat") or {}).get("thinking_stub", "") or ""
SC_USER = "Is the answer above factually correct?\nReply with exactly one word: True or False."
VARIANTS = {"True": DEFAULT_LABEL_VARIANTS["True"], "False": DEFAULT_LABEL_VARIANTS["False"]}
DUMMY = "QQZZDUMMYANSWERZZQQ"


def followup_rest(system, user_text):
    chat = bundle.cfg.get("chat", {})
    if system and chat.get("supports_system", True):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_text}]
    else:
        msgs = [{"role": "user", "content": (system.strip() + "\n\n" + user_text) if system else user_text}]
    msgs += [{"role": "assistant", "content": DUMMY}, {"role": "user", "content": SC_USER}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return text[text.index(DUMMY) + len(DUMMY):] + stub


def margin_at_end(ids, text):
    lmap = build_label_token_map(tok, text, VARIANTS)
    x = torch.tensor([ids], dtype=torch.long, device=bundle.device)
    lp = torch.log_softmax(forward_logits(bundle, {"input_ids": x, "attention_mask": torch.ones_like(x)}, logits_to_keep=1)[0, -1].float(), -1)
    return float(torch.logsumexp(lp[sorted(set(lmap["True"]))], 0) - torch.logsumexp(lp[sorted(set(lmap["False"]))], 0))


def collect(split_file):
    items = json.loads((run / split_file).read_text())
    rows, skipped = [], {"abstain": 0, "refusal_only_gold": 0, "encode_failed": 0}
    for e in items["evaluations"]:
        if e.get("gold_is_abstention"):
            skipped["refusal_only_gold"] += 1
            continue
        out = e["outcome"]["baseline"]
        if out == "abstain":
            skipped["abstain"] += 1
            continue
        ans = e["outputs"]["baseline"]
        try:
            enc = encode_answer(bundle, cfg["qa_system"], e["question"], ans, cfg["qa_user"])
            sc0 = encode_selfcheck(bundle, cfg["qa_system"], e["question"], ans, cfg["qa_user"], F0["selfcheck_suffix"],
                                   F0["selfcheck_label_true"], F0["selfcheck_label_false"])
        except ValueError:
            skipped["encode_failed"] += 1
            continue
        rows.append({"pair_id": e["pair_id"], "label": 1 if out == "correct" else 0, "enc": enc, "sc0": sc0, "question": e["question"]})
    return rows, skipped


data = {}
with fp32_output_projection(bundle), torch.no_grad():
    for split, fname in (("tune", "tune_scored_items.json"), ("test", "test_scored_items.json")):
        rows, skipped = collect(fname)
        H = np.zeros((len(rows), len(layers), bundle.hidden_size), dtype=np.float32)
        f0, f1 = [], []
        for i, r in enumerate(rows):
            ids = r["enc"]["full_ids"]
            x = torch.tensor([ids], dtype=torch.long, device=bundle.device)
            with capture_hooks(bundle, layers, ("resid",), position="all") as st:
                forward_logits(bundle, {"input_ids": x, "attention_mask": torch.ones_like(x)}, logits_to_keep=1)
            for j, l in enumerate(layers):
                H[i, j] = st[("resid", l)][0, len(ids) - 1].float().cpu().numpy()
            sc = r["sc0"]
            xs = torch.tensor([sc["input_ids"]], dtype=torch.long, device=bundle.device)
            lp = torch.log_softmax(forward_logits(bundle, {"input_ids": xs, "attention_mask": torch.ones_like(xs)}, logits_to_keep=1)[0, -1].float(), -1)
            f0.append(float(lp[sc["label_ids"]["true"][0]] - lp[sc["label_ids"]["false"][0]]))
            rest = followup_rest(cfg["qa_system"], cfg["qa_user"].format(question=r["question"]))
            ids1 = ids + _token_ids(tok, rest)
            f1.append(margin_at_end(ids1, r["enc"]["full_text"] + rest))
        data[split] = {"y": np.array([r["label"] for r in rows]), "H": H, "f0": np.array(f0), "f1": np.array(f1),
                       "ids": [r["pair_id"] for r in rows], "skipped": skipped,
                       "n_total": len(json.loads((run / fname).read_text())["evaluations"])}
        print(split, "answered", len(rows), "skipped", skipped, flush=True)


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


scores = {}   # name -> (tune_scores, test_scores, meta)
for j, l in enumerate(layers):
    for est, key in (("md", "E_qa_md_L%d"), ("das", "E_qa_das_L%d")):
        if est == "das" and l not in das_layers:
            continue
        w = np.asarray(bank[key % l], dtype=np.float64).reshape(-1)
        st, ss = data["tune"]["H"][:, j] @ w, data["test"]["H"][:, j] @ w
        sign = 1.0 if auroc(st, data["tune"]["y"]) >= 0.5 else -1.0
        scores["%s_L%d" % (est, l)] = (sign * st, sign * ss, {"family": est, "layer": l, "sign": sign})
scores["selfcheck_F0"] = (data["tune"]["f0"], data["test"]["f0"], {"family": "selfcheck_F0"})
scores["selfcheck_F1"] = (data["tune"]["f1"], data["test"]["f1"], {"family": "selfcheck_F1"})

# Probe: layer and C by 5-fold CV on tuning only.
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
yt = data["tune"]["y"]
best = None
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for j, l in enumerate(layers):
    X = data["tune"]["H"][:, j]
    for C in (1e-3, 1e-2, 1e-1):
        cv = []
        for tr, va in skf.split(X, yt):
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X[tr]), yt[tr])
            cv.append(auroc(m.decision_function(sc.transform(X[va])), yt[va]))
        if best is None or np.mean(cv) > best[0]:
            best = (float(np.mean(cv)), l, j, C)
_, pl, pj, pC = best
sc = StandardScaler().fit(data["tune"]["H"][:, pj])
m = LogisticRegression(C=pC, max_iter=2000).fit(sc.transform(data["tune"]["H"][:, pj]), yt)
scores["probe"] = (m.decision_function(sc.transform(data["tune"]["H"][:, pj])), m.decision_function(sc.transform(data["test"]["H"][:, pj])),
                   {"family": "probe", "layer": pl, "C": pC, "cv_auroc_tune": best[0], "note": "tune scores are in-sample"})

y_tune, y_test = data["tune"]["y"], data["test"]["y"]
rng = np.random.default_rng(0)
nt = len(y_test)
boot = [rng.integers(0, nt, nt) for _ in range(args.n_boot)]
N_TEST = data["test"]["n_total"] - data["test"]["skipped"]["refusal_only_gold"]   # main-table denominator (523)
N_TUNE = data["tune"]["n_total"]


def gate(s_tune, s_test):
    """Threshold on tuning maximising (correct - wrong)/n_tune when items below it abstain; applied to test."""
    cand = np.unique(np.concatenate([[-np.inf], s_tune]))
    best_t, best_v = -np.inf, -1e9
    for t in cand:
        keep = s_tune >= t
        v = (int(((y_tune == 1) & keep).sum()) - int(((y_tune == 0) & keep).sum())) / N_TUNE
        if v > best_v + 1e-12:
            best_t, best_v = t, v
    keep = s_test >= best_t
    lost = int(((y_test == 1) & ~keep).sum())
    removed = int(((y_test == 0) & ~keep).sum())
    before = (int((y_test == 1).sum()) - int((y_test == 0).sum())) / N_TEST
    after = (int(((y_test == 1) & keep).sum()) - int(((y_test == 0) & keep).sum())) / N_TEST
    d = []
    for i in boot:
        k, yy = keep[i], y_test[i]
        d.append(((int(((yy == 1) & k).sum()) - int(((yy == 0) & k).sum())) - (int((yy == 1).sum()) - int((yy == 0).sum()))) / N_TEST)
    return {"threshold": float(best_t), "tune_value": best_v, "test_cmw_before": before, "test_cmw_after": after,
            "test_delta": after - before, "test_delta_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "test_abstained": int((~keep).sum()), "test_correct_lost": lost, "test_wrong_removed": removed}


table = {}
for name, (s_tune, s_test, info) in scores.items():
    a = auroc(s_test, y_test)
    bd = [auroc(s_test[i], y_test[i]) for i in boot]
    table[name] = {**info, "tune_auroc": auroc(s_tune, y_tune), "test_auroc": a,
                   "test_auroc_ci95": [float(np.nanpercentile(bd, 2.5)), float(np.nanpercentile(bd, 97.5))]}
picks = {}
for fam in ("das", "md"):
    names = [k for k, v in table.items() if v["family"] == fam]
    picks[fam] = max(names, key=lambda k: (table[k]["tune_auroc"], -table[k]["layer"]))
picks["probe"], picks["selfcheck_F0"], picks["selfcheck_F1"] = "probe", "selfcheck_F0", "selfcheck_F1"
gates = {fam: gate(scores[name][0], scores[name][1]) for fam, name in picks.items()}

# Paired bootstrap of TEST AUROC differences between the picked score families (same resampled items for both).
paired = {}
for a_fam, b_fam in (("das", "selfcheck_F0"), ("das", "selfcheck_F1"), ("das", "md"), ("das", "probe"),
                     ("md", "selfcheck_F0"), ("md", "selfcheck_F1")):
    sa, sb = scores[picks[a_fam]][1], scores[picks[b_fam]][1]
    d = [auroc(sa[i], y_test[i]) - auroc(sb[i], y_test[i]) for i in boot]
    paired["%s_minus_%s" % (a_fam, b_fam)] = {"delta": auroc(sa, y_test) - auroc(sb, y_test),
                                              "ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}


def selective(s, cov):
    """Accuracy among the `cov` fraction of TEST answers with the highest score (threshold from the score ranks only)."""
    k = max(1, int(round(cov * len(s))))
    keep = np.argsort(-s, kind="stable")[:k]
    return float(y_test[keep].mean())


coverage = {fam: {str(c): selective(scores[name][1], c) for c in (0.8, 0.5)} for fam, name in picks.items()}
coverage["all_answered"] = float(y_test.mean())
out = {"model": args.model, "n": {"tune_answered": int(len(y_tune)), "test_answered": int(len(y_test)),
                                  "tune_correct": int(y_tune.sum()), "test_correct": int(y_test.sum()),
                                  "N_TEST_main_table": N_TEST, "N_TUNE": N_TUNE,
                                  "skipped": {s: data[s]["skipped"] for s in data}},
       "picks": picks, "picked": {fam: table[name] for fam, name in picks.items()}, "gates": gates,
       "paired_auroc_differences_test": paired, "selective_accuracy_test": coverage, "all": table,
       "test_scores": {fam: scores[name][1].tolist() for fam, name in picks.items()}, "test_labels": y_test.tolist(),
       "test_ids": data["test"]["ids"]}
for k, v in paired.items():
    print("paired TEST AUROC %-24s %+.3f [%+.3f, %+.3f]" % (k, v["delta"], *v["ci95"]))
print("selective accuracy on TEST (keep top 80%% / 50%% by score; all answered %.3f): %s" % (
    coverage["all_answered"], "; ".join("%s %.3f / %.3f" % (fam, v["0.8"], v["0.5"]) for fam, v in coverage.items() if fam != "all_answered")))
json.dump(out, open(args.out, "w"), indent=1)
print("answered: tune %d (correct %d), test %d (correct %d); main-table n=%d" % (len(y_tune), y_tune.sum(), len(y_test), y_test.sum(), N_TEST))
for fam, name in picks.items():
    t, g = table[name], gates[fam]
    print("%-13s %-10s tune AUROC %.3f | TEST AUROC %.3f [%.3f, %.3f] | gate: (c-w)/n %+.3f -> %+.3f  d %+.3f [%+.3f, %+.3f]  abstain %d (lost %d correct, removed %d wrong)" % (
        fam, name, t["tune_auroc"], t["test_auroc"], *t["test_auroc_ci95"], g["test_cmw_before"], g["test_cmw_after"], g["test_delta"], *g["test_delta_ci95"],
        g["test_abstained"], g["test_correct_lost"], g["test_wrong_removed"]))
print("saved", args.out)
