"""E25 E-path checks on the real TruthfulQA data and the real Llama/Qwen tokenizers (CPU, no model weights).

usage: chk_e25_data_tokens.py CODE_DIR OUT.json
  1. split_questions on the real data: cleaning counts, exclusions, positive/negative sources; pilot behaviour.
  2. tune ids == the E23 draw (default_rng(seed).choice(sorted unique test pair ids, 192)).
  3. E25 positive/negative answers vs the production TF statements of the same pairs; reconstructed statement text vs
     production text.
  4. encode_answer / pack_answers with the real tokenizers on every estimation + tuning answer: boundary failures,
     round trip, first answer token, prefix == generation prompt, mask geometry.
"""
import collections
import json
import sys
import types
from pathlib import Path

CODE = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(CODE / "scripts"), str(CODE)]
import numpy as np  # noqa: E402

from common import load_split  # noqa: E402
from core.config import load_experiment, load_json_config, load_model_config, resolve_config_path  # noqa: E402
from core.model_io import ModelBundle  # noqa: E402
from core.qa_context import encode_answer, pack_answers  # noqa: E402
import e25_qa_context as e25  # noqa: E402

rep = {}
cfg = e25.validate_config(load_json_config(resolve_config_path("experiments", "E25_qa_context")))
prep = e25.split_questions(cfg)
rep["full_split_counts"] = prep["full_split_counts"]
for name in ("train", "tune"):
    a = prep["pair_audit"][name]
    rep[name] = {"n_input": a["n_input"], "n_retained": a["n_retained"], "n_excluded": a["n_excluded"],
                 "excluded": a["excluded"],
                 "positive_source": dict(collections.Counter(r["positive_source"] for r in a["records"] if r["status"] == "retained")),
                 "negative_source": dict(collections.Counter(r["negative_source"] for r in a["records"] if r["status"] == "retained")),
                 "positive_not_best_answer": [{"pair_id": r["pair_id"], "candidates": r["positive_candidates"][:3]}
                                              for r in a["records"] if r["status"] == "retained" and r["positive_source"] != "best_answer"]}
rep["pilot3"] = {k: len(v) for k, v in e25.split_questions(cfg, 3)["splits"].items()}
rep["pilot3_e_pairs"] = {k: len(e25.split_questions(cfg, 3)[k]) for k in ("e_train", "e_tune")}

exp = load_experiment(cfg["source_experiment"])
recs = []
for ds in exp["eval_datasets"]:
    recs += load_split(ds, exp.get("eval_split", "test"))
uniq = sorted(set(str(r.get("pair_id") or r["id"]) for r in recs))
e23_tune = set(np.random.default_rng(exp.get("seed", 42)).choice(uniq, cfg["tune_pairs"], replace=False).tolist())
e25_tune = {r["pair_id"] for r in prep["splits"]["tune"]}
rep["tune_ids_equal_e23_draw"] = e23_tune == e25_tune
rep["tune_sym_diff"] = len(e23_tune ^ e25_tune)
rep["test_n"] = len(prep["splits"]["test"])

tf_by_pair = {}
for split in ("calib", "dev", "test"):
    for r in load_split("truthfulqa_tf", split):
        tf_by_pair.setdefault(str(r["pair_id"]), {})[bool(r["label"])] = r


def compare(pairs, label):
    same_pos = same_neg = 0
    diff = []
    for p in pairs:
        t = tf_by_pair.get(p["pair_id"], {})
        pa = ((t.get(True) or {}).get("meta") or {}).get("answer")
        na = ((t.get(False) or {}).get("meta") or {}).get("answer")
        sp = (pa or "").strip() == p["positive_answer"].strip()
        sn = (na or "").strip() == p["negative_answer"].strip()
        same_pos += sp
        same_neg += sn
        if not (sp and sn):
            diff.append({"pair_id": p["pair_id"], "tf_pos": pa, "e25_pos": p["positive_answer"], "tf_neg": na, "e25_neg": p["negative_answer"]})
    rep[label + "_vs_production_tf"] = {"n": len(pairs), "same_positive": same_pos, "same_negative": same_neg,
                                        "n_diff": len(diff), "differences": diff[:40]}


compare(prep["e_train"], "train")
compare(prep["e_tune"], "tune")
stm = ['The answer to the question "%s" is: %s' % (r["question"], r["positive_answer"]) for r in prep["e_train"]]
prod = [tf_by_pair[r["pair_id"]][True]["text"] for r in prep["e_train"]]
rep["train_pos_statement_identical_to_production"] = int(sum(s == t for s, t in zip(stm, prod)))
rep["train_pos_statement_identical_if_period_added"] = int(sum(s + "." == t for s, t in zip(stm, prod)))
rep["train_statement_examples"] = [{"e25": s, "production": t} for s, t in list(zip(stm, prod))[:2]]

from transformers import AutoTokenizer  # noqa: E402

for model in ("llama3.1-8b", "qwen3-8b"):
    mcfg = load_model_config(model)
    tok = AutoTokenizer.from_pretrained(mcfg["path"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    bundle = types.SimpleNamespace(tokenizer=tok, cfg=mcfg)
    bundle.build_chat = types.MethodType(ModelBundle.build_chat, bundle)
    m = {"errors": collections.Counter(), "n_encoded": 0, "prompt_len": [], "answer_len": [], "roundtrip_mismatch": 0,
         "first_token_leading_space": 0, "prefix_equals_generation_prompt": 0, "samples": [], "pad_token_id": tok.pad_token_id,
         "pad_is_eos": tok.pad_token_id == tok.eos_token_id, "special_in_answer": 0}
    enc_all = []
    for split in ("e_train", "e_tune"):
        for r in prep[split]:
            gen_prompt = e25.qa_text(bundle, cfg, r["question"])
            for key in ("positive_answer", "negative_answer"):
                try:
                    e = encode_answer(bundle, cfg["qa_system"], r["question"], r[key], cfg["qa_user"])
                except ValueError as ex:
                    m["errors"][str(ex)[:90]] += 1
                    continue
                m["n_encoded"] += 1
                m["prompt_len"].append(e["prompt_length"])
                m["answer_len"].append(e["answer_length"])
                m["roundtrip_mismatch"] += tok.decode(e["answer_ids"]) != r[key]
                m["first_token_leading_space"] += tok.decode(e["answer_ids"][:1]).startswith(" ")
                m["prefix_equals_generation_prompt"] += e["prefix_text"] == gen_prompt
                enc_all.append(e)
                if len(m["samples"]) < 3:
                    m["samples"].append({"prefix_tail": e["prefix_text"][-70:], "answer": e["answer_text"],
                                         "answer_tokens": [tok.decode([i]) for i in e["answer_ids"][:10]],
                                         "prefix_last_tokens": [tok.decode([i]) for i in e["prefix_ids"][-4:]]})
    # Mask geometry on EVERY encoded answer (review round 13: the first version checked only the first 8 per model and the
    # summary over-stated the coverage). One pack of all answers; the longest answer sets the width.
    packed = pack_answers(enc_all, tok.pad_token_id, "cpu")
    ids, att = packed["inputs"]["input_ids"], packed["inputs"]["attention_mask"]
    tgt, prd, pe = packed["target_mask"], packed["predictor_mask"], packed["prompt_end_mask"]
    width = ids.shape[1]
    geom = []
    m["special_in_answer"] = int(sum(any(i in set(tok.all_special_ids) for i in e["answer_ids"]) for e in enc_all))
    for i, e in enumerate(enc_all):
        P, M = e["prompt_length"], e["answer_length"]
        geom.append({"P": P, "M": M, "prompt_end_idx": int(pe[i].nonzero()[0]), "expected_prompt_end_idx": width - M - 1,
                     "n_target": int(tgt[i].sum()), "n_predictor": int(prd[i].sum()), "overlap": int((tgt[i] & prd[i]).sum()),
                     "first_target_idx": int(tgt[i].nonzero()[0]), "last_predictor_idx": int(prd[i].nonzero()[-1]),
                     "pad_in_masks": int(((tgt[i] | prd[i]) & ~att[i].bool()).sum()),
                     "prompt_end_token": tok.decode([int(ids[i, int(pe[i].nonzero()[0])])])})
    m["mask_geometry_ok"] = all(g["prompt_end_idx"] == g["expected_prompt_end_idx"] and g["n_target"] == g["M"] == g["n_predictor"]
                                and g["overlap"] == g["M"] - 1 and g["first_target_idx"] == g["prompt_end_idx"] + 1
                                and g["last_predictor_idx"] == width - 2 and g["pad_in_masks"] == 0 for g in geom)
    m["mask_geometry"] = geom[:3]
    m["n_geometry_checked"] = len(geom)
    m["prompt_len_minmax"] = [min(m["prompt_len"]), max(m["prompt_len"])]
    m["answer_len_minmax_mean"] = [min(m["answer_len"]), max(m["answer_len"]), float(np.mean(m["answer_len"]))]
    m["errors"] = dict(m["errors"])
    del m["prompt_len"], m["answer_len"]
    rep[model] = m

print(json.dumps({k: v for k, v in rep.items() if k not in ("train", "tune")}, indent=1, ensure_ascii=False)[:6000])
for name in ("train", "tune"):
    print(name, {k: rep[name][k] for k in ("n_input", "n_retained", "n_excluded", "positive_source", "negative_source")},
          "excluded:", rep[name]["excluded"], "positive_not_best:", [x["pair_id"] for x in rep[name]["positive_not_best_answer"]])
# Pass/fail (review round 13: a check that only writes its findings into a JSON and exits 0 cannot be trusted by a runner).
failures = []
if not rep["tune_ids_equal_e23_draw"]:
    failures.append("tune ids differ from the E23 draw")
for model in ("llama3.1-8b", "qwen3-8b"):
    m = rep[model]
    expected = 2 * (rep["train"]["n_retained"] + rep["tune"]["n_retained"])
    if m["errors"]:
        failures.append("%s: encode_answer errors %s" % (model, m["errors"]))
    if m["n_encoded"] != expected:
        failures.append("%s: encoded %d answers, expected %d" % (model, m["n_encoded"], expected))
    if m["n_geometry_checked"] != m["n_encoded"] or not m["mask_geometry_ok"]:
        failures.append("%s: mask geometry failed (%d of %d checked)" % (model, m["n_geometry_checked"], m["n_encoded"]))
    if m["roundtrip_mismatch"] or m["special_in_answer"]:
        failures.append("%s: %d round-trip mismatches, %d answers containing special tokens" % (model, m["roundtrip_mismatch"], m["special_in_answer"]))
rep["failures"] = failures
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(rep, f, indent=1, ensure_ascii=False)
if failures:
    print("FAILED:", failures)
    sys.exit(1)
print("PASSED: all E25 data / tokenizer checks")
