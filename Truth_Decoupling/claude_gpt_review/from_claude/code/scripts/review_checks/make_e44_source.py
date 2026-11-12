#!/usr/bin/env python3
"""Source directory for the free-form E44 pipeline on a model that has no E25 run (Claude, 2026-11-12).

E43 / E44 take the model and the question splits from an E25 "source run" (--source-run: bank.json + questions.json). Of
bank.json they read only `model`, `model_config` (load_model uses it), `config` (qa_system, qa_user and the run settings) and
`num_layers`; questions.json holds the TruthfulQA question splits and is byte-identical for every old model (sha256
b61184c3...). This writes a new source directory for --model:
  questions.json  copied byte for byte from --like (checked)
  bank.json       {protocol, config} copied from --like's bank.json; model / model_config / num_layers / hidden_size from the
                  model's own config (periphery); a `derived_from` record. No DAS bank: E44 never reads one.
usage: make_e44_source.py --model qwen3.5-9b --like $ROOT/results_qa_das_v1u/llama3.1-8b/full --out $ROOT/results_newmodels_ff/source/qwen3.5-9b
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from core.config import load_model_config        # noqa: E402

KEEP_FROM_LIKE = ("protocol", "config")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--like", required=True, help="an existing E25 v1u source run directory (bank.json, questions.json)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.path.exists(args.out) and os.listdir(args.out):
        raise SystemExit("refusing to write into non-empty %s" % args.out)
    like_bank = json.load(open(os.path.join(args.like, "bank.json")))
    for k in KEEP_FROM_LIKE:
        if k not in like_bank:
            raise SystemExit("--like bank.json lacks %r" % k)
    if "qa_system" not in like_bank["config"] or "qa_user" not in like_bank["config"]:
        raise SystemExit("--like bank.json config lacks the QA prompt")
    cfg = load_model_config(args.model)
    for k in ("name", "path", "hidden_size", "num_layers"):
        if k not in cfg:
            raise SystemExit("model config lacks %r" % k)
    os.makedirs(args.out, exist_ok=True)
    q_src, q_dst = os.path.join(args.like, "questions.json"), os.path.join(args.out, "questions.json")
    shutil.copyfile(q_src, q_dst)
    if sha(q_src) != sha(q_dst):
        raise SystemExit("questions.json copy is not byte-identical")
    bank = {k: like_bank[k] for k in KEEP_FROM_LIKE}
    bank.update({"model": cfg["name"], "model_config": cfg, "hidden_size": int(cfg["hidden_size"]), "num_layers": int(cfg["num_layers"]),
                 "derived_from": {"dir": os.path.abspath(args.like), "bank_sha256": sha(os.path.join(args.like, "bank.json")),
                                  "questions_sha256": sha(q_src), "bank_model": like_bank.get("model"),
                                  "note": "no E25 DAS bank for this model; only the fields E43/E44 read (model, model_config, config, num_layers)"}})
    tmp = os.path.join(args.out, ".bank.json.tmp")            # atomic: the runner's guard only checks that bank.json exists
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(bank, f, indent=1, ensure_ascii=False)
    os.replace(tmp, os.path.join(args.out, "bank.json"))
    print("wrote %s (model %s, %d layers; questions.json sha256 %s)" % (args.out, cfg["name"], cfg["num_layers"], sha(q_dst)[:16]))


if __name__ == "__main__":
    main()
