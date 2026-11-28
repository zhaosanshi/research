"""Exploration (executor R-yi, Claude Opus 5.5, 2026-11-28): question lists and source directories for learning E and C
anew on PopQA and on FalseQA (work queue R, discussion/work_queue/2026-11-28_queue_r.md, R7 / R8, section 3.2). Not
reviewed by GPT. No existing file is changed; this writes new files only.

The direction program (scripts/e44_factorial_das.py, unchanged) takes its questions from a "source run" directory:
bank.json (model, model_config, config.qa_system / qa_user, num_layers, hidden_size) and questions.json (it reads only
splits.train and splits.tune; per question pair_id and question; e44_export_judge.py also reads gold / incorrect /
best_answer). This script writes such a directory for another question set:
  bank.json       byte copy of the model's TruthfulQA source bank.json (the same model loading and prompts)
  questions.json  {"relearn_1128": provenance, "splits": {"train": 96 questions, "tune": 192 questions}}
  relearn_1128_provenance.json   what was copied from where (sha256 of every input)

Phase ids (CPU, writes the two id files; ids and counts only, no question text):
  PopQA    train = random.Random(20261128).sample(sorted(train ids of popqa_splits_20261119.json), 96), in draw order;
           tune = the 192 tune ids of that file, in file order (the order of the C2 judging of 11-19).
  FalseQA  pool = the 1,187 pairs of the official train.csv as the gate step paired them (diag/gate_capture_1126.py:
           pair i = data row i of the label-1 block with data row i + 1187 of the label-0 block; question ids
           fqtr<iiii>f (false premise) / fqtr<iiii>v (valid premise)), less the pairs whose two questions have the same
           normalised text (the rule of falseqa_prep.load_pairs for every FalseQA bank of this project; 4 pairs here;
           a first draw over all 1,187 put one of them, fqtr0887, into the tune pairs: one question twice with two
           contradictory references, so the pool follows the bank rule); one rng = random.Random(20261128):
           training pairs = rng.sample(sorted pair ids, 48), then tune pairs = rng.sample(the other pair ids, sorted, 96).
           Pairs stay together. Of the 48 training pairs the first 40 drawn are gradient pairs and the last 8 drawn are
           held-out pairs: the 16 held-out positions that e44_factorial_das.split_train_val picks (torch.randperm(96)
           with seed 42 * 7919 + 13, the default --split-seed 42; computed here with the same torch) are filled with the
           16 questions of the 8 held-out pairs (f then v, pair by pair, positions ascending), every other position with
           the 80 questions of the gradient pairs (f then v, draw order). Tune: pairs in draw order, f then v.
           Checks (stop on any failure): train.csv is the 2,374-row three-column file with 1,187 label-1 rows followed by
           1,187 label-0 rows; no drawn question has the normalised text (falseqa_prep.norm) of any question of the
           held-out 600 bank or the confirmation 772 bank; the dev 100 bank is reported, not a stop.
           gold = falseqa_prep.gold_of(answer, label, "train") (one string; a list literal would stop).
  The PopQA id file also records how many drawn training questions share a normalised first answer with the test 500
  (the 11-19 record: 34 of the 300; information only, training never sees answers).

Phase source (CPU): one source directory for one model and one set.
  PopQA question records follow newsets_gen.py's export: pair_id "popqa:<id>", gold = the first 10 bank references
  (GOLD_CAP of the 11-19 judging), incorrect = [], best_answer = gold[0].
  FalseQA: pair_id "falseqa:<question id>", gold = gold_of(answer), incorrect = [], best_answer = gold[0].

usage (container, python3 -I, from /workspace/Truth_Decoupling):
  relearn_1128_source.py ids POPQA_SPLITS POPQA_BANK FQA_TRAIN_CSV FQA_HELD_BANK FQA_CONFIRM_BANK FQA_DEV_BANK OUT_POPQA_IDS OUT_FQA_IDS
  relearn_1128_source.py source popqa|falseqa IDS_JSON DATA(POPQA_BANK|FQA_TRAIN_CSV) LIKE_SOURCE_DIR OUT_DIR
"""
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import falseqa_prep as P  # noqa: E402

SEED = 20261128
N_TRAIN, N_TUNE = 96, 192
N_FQA_PAIRS = 1187
N_HELD = 16
SPLIT_SEED = 42            # e44_factorial_das.py --split-seed default
GOLD_CAP = 10              # newsets_gen.py GOLD_CAP


def sha_file(p):
    return P.sha_file(p)


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump(path, obj):
    if os.path.exists(path):
        raise SystemExit("%s exists" % path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def held_positions(n, seed=SPLIT_SEED):
    """The positions e44_factorial_das.split_train_val holds out (same expression, same torch)."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(int(seed) * 7919 + 13)
    order = torch.randperm(n, generator=g).tolist()
    return sorted(order[:N_HELD]), torch.__version__


def fqa_train_pairs(path):
    rows = list(csv.reader(open(path, encoding="utf-8", newline="")))
    if rows[0] != ["question", "answer", "label"] or len(rows) != 2 * N_FQA_PAIRS + 1 or any(len(r) != 3 for r in rows):
        raise SystemExit("FalseQA train.csv is not the 2,374-row three-column file")
    rows = rows[1:]
    if [r[2] for r in rows] != ["1"] * N_FQA_PAIRS + ["0"] * N_FQA_PAIRS:
        raise SystemExit("FalseQA train.csv: not 1,187 label-1 rows followed by 1,187 label-0 rows")
    pairs = {}
    for i in range(N_FQA_PAIRS):
        a, b = rows[i], rows[i + N_FQA_PAIRS]
        if not a[0].strip() or not b[0].strip():
            raise SystemExit("a train.csv row without question")
        pid = "fqtr%04d" % i
        pairs[pid] = [{"id": pid + "f", "question": a[0].strip(), "gold": P.gold_of(a[1], "1", "train"), "label": 1, "source_row": i},
                      {"id": pid + "v", "question": b[0].strip(), "gold": P.gold_of(b[1], "0", "train"), "label": 0,
                       "source_row": i + N_FQA_PAIRS}]
    return pairs


def phase_ids(a):
    if len(a) != 8:
        raise SystemExit(__doc__)
    popqa_splits, popqa_bank, fqa_csv, fqa_held, fqa_conf, fqa_dev, out_p, out_f = a
    for p in (out_p, out_f):
        if os.path.exists(p):
            raise SystemExit("%s exists" % p)
    # ---- PopQA
    sp = json.load(open(popqa_splits, encoding="utf-8"))
    if sp.get("dataset") != "popqa" or len(sp["train"]) != 300 or len(sp["tune"]) != N_TUNE or len(sp["test"]) != 500:
        raise SystemExit("unexpected PopQA split file")
    if set(sp["train"]) & set(sp["tune"]) or set(sp["train"]) & set(sp["test"]) or set(sp["tune"]) & set(sp["test"]):
        raise SystemExit("PopQA splits overlap")
    rng = random.Random(SEED)
    ptrain = rng.sample(sorted(sp["train"]), N_TRAIN)
    bank = {r["id"]: r for r in jl(popqa_bank)}
    for i in ptrain:
        if bank[i]["split"] != "train":
            raise SystemExit("drawn PopQA id %s is not a train question in the bank" % i)
    for i in sp["tune"]:
        if bank[i]["split"] != "tune":
            raise SystemExit("PopQA tune id %s is not a tune question in the bank" % i)
    norm_ans = lambda r: P.norm(r["gold"][0])  # noqa: E731
    test_ans = {norm_ans(bank[i]) for i in sp["test"]}
    test_q = {P.norm(bank[i]["question"]) for i in sp["test"]}
    tune_q = {P.norm(bank[i]["question"]) for i in sp["tune"]}
    shared_ans = sorted(i for i in ptrain if norm_ans(bank[i]) in test_ans)
    shared_q = [i for i in ptrain if P.norm(bank[i]["question"]) in test_q | tune_q]
    if shared_q:
        raise SystemExit("drawn PopQA training questions share normalised text with tune / test: %s" % shared_q)
    held_pos, tv = held_positions(N_TRAIN)
    pobj = {"dataset": "popqa", "purpose": "work queue R7: learn E and C anew on PopQA (relearn_1128)",
            "seed": SEED, "draw": "random.Random(seed).sample(sorted(train ids of the split file), 96), draw order kept",
            "split_file": os.path.basename(popqa_splits), "split_file_sha256": sha_file(popqa_splits),
            "bank_sha256": {os.path.basename(popqa_bank): sha_file(popqa_bank)},
            "train": ptrain, "tune": list(sp["tune"]),
            "held_out_positions": held_pos, "held_out_ids_expected": [ptrain[k] for k in held_pos],
            "held_out_rule": "e44_factorial_das.split_train_val (torch.randperm(96), seed 42*7919+13), torch %s" % tv,
            "checks": {"train_shares_question_text_with_tune_or_test": 0,
                       "train_shares_normalised_first_answer_with_test": len(shared_ans),
                       "train_ids_sharing_first_answer_with_test": shared_ans},
            "counts": {"train": len(ptrain), "tune": len(sp["tune"])}}
    # ---- FalseQA
    pairs = fqa_train_pairs(fqa_csv)
    same_text_pool = sorted(p for p, (f, v) in pairs.items() if P.norm(f["question"]) == P.norm(v["question"]))
    rng = random.Random(SEED)
    ids = sorted(set(pairs) - set(same_text_pool))
    tr = rng.sample(ids, 48)
    rest = sorted(set(ids) - set(tr))
    tu = rng.sample(rest, 96)
    if set(tr) & set(tu):
        raise SystemExit("FalseQA train / tune pairs overlap")
    gpairs, hpairs = tr[:40], tr[40:]
    gq = [q["id"] for p in gpairs for q in pairs[p]]
    hq = [q["id"] for p in hpairs for q in pairs[p]]
    held_pos, tv = held_positions(N_TRAIN)
    order, gi, hi = [], iter(gq), iter(hq)
    for k in range(N_TRAIN):
        order.append(next(hi) if k in held_pos else next(gi))
    if sorted(order) != sorted(gq + hq) or [order[k] for k in held_pos] != hq:
        raise SystemExit("FalseQA held-out placement failed")
    tune_q = [q["id"] for p in tu for q in pairs[p]]
    text = {q["id"]: q["question"] for p in pairs.values() for q in p}
    banks = {"heldout_600": jl(fqa_held), "confirm_772": jl(fqa_conf), "dev_100": jl(fqa_dev)}
    if len(banks["heldout_600"]) != 600 or len(banks["confirm_772"]) != 772 or len(banks["dev_100"]) != 100:
        raise SystemExit("FalseQA banks do not have 600 / 772 / 100 rows")
    hits = {}
    for name, rows in banks.items():
        bt = {P.norm(r["question"]) for r in rows}
        hits[name] = sorted(i for i in order + tune_q if P.norm(text[i]) in bt)
    if hits["heldout_600"] or hits["confirm_772"]:
        raise SystemExit("drawn FalseQA questions share normalised text with held-out / confirm: %s" % hits)
    same_text_pairs = sorted(p for p in tr + tu if P.norm(pairs[p][0]["question"]) == P.norm(pairs[p][1]["question"]))
    drawn_texts = [P.norm(text[i]) for i in order + tune_q]
    fobj = {"dataset": "falseqa", "purpose": "work queue R8: learn E and C anew on FalseQA (relearn_1128)",
            "source_file": "train.csv (FalseQA official train split)", "source_file_sha256": sha_file(fqa_csv),
            "pairing": "pair fqtr<iiii> = data row i of the label-1 block (id ...f, false premise) with data row i + 1187 of the "
                       "label-0 block (id ...v, valid premise), as diag/gate_capture_1126.py",
            "pool": "the 1,187 pairs less the %d pairs whose two questions have the same normalised text "
                    "(falseqa_prep.load_pairs rule): %s" % (len(same_text_pool), same_text_pool),
            "pool_pairs": len(ids),
            "seed": SEED, "draw": "one random.Random(seed): sample(sorted pool pair ids, 48) training pairs, then "
                                  "sample(sorted other pool pair ids, 96) tune pairs",
            "train_pairs_gradient": gpairs, "train_pairs_held_out": hpairs, "tune_pairs": tu,
            "train": order, "tune": tune_q,
            "held_out_positions": held_pos, "held_out_ids_expected": hq,
            "held_out_rule": "the 16 positions of e44_factorial_das.split_train_val (torch.randperm(96), seed 42*7919+13), "
                             "torch %s, filled with the 8 held-out pairs" % tv,
            "checks": {"normalised_text_hits": {k: len(v) for k, v in hits.items()}, "dev_100_hit_ids": hits["dev_100"],
                       "same_text_pairs_drawn": same_text_pairs,
                       "repeated_normalised_text_among_drawn": len(drawn_texts) - len(set(drawn_texts)),
                       "normaliser": "falseqa_prep.norm"},
            "bank_sha256": {os.path.basename(p): sha_file(p) for p in (fqa_held, fqa_conf, fqa_dev)},
            "counts": {"train": len(order), "tune": len(tune_q), "train_pairs": len(tr), "tune_pairs": len(tu),
                       "train_false_premise": sum(i.endswith("f") for i in order),
                       "tune_false_premise": sum(i.endswith("f") for i in tune_q)}}
    dump(out_p, pobj)
    dump(out_f, fobj)
    print("PopQA: train %d (held-out positions %s), tune %d; train ids sharing a first answer with test: %d" % (
        len(ptrain), held_pos, len(sp["tune"]), len(shared_ans)))
    print("FalseQA: train %d (%d pairs; held-out pairs %s), tune %d (%d pairs); text hits %s; same-text pairs drawn %s" % (
        len(order), len(tr), hpairs, len(tune_q), len(tu), {k: len(v) for k, v in hits.items()}, same_text_pairs))


def phase_source(a):
    if len(a) != 5:
        raise SystemExit(__doc__)
    which, ids_json, data, like, out = a
    if which not in ("popqa", "falseqa"):
        raise SystemExit("set must be popqa or falseqa")
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit("refusing to write into non-empty %s" % out)
    idf = json.load(open(ids_json, encoding="utf-8"))
    if idf.get("dataset") != which or len(idf["train"]) != N_TRAIN or len(idf["tune"]) != N_TUNE:
        raise SystemExit("id file is not a %s id file of 96 + 192" % which)
    if len(set(idf["train"] + idf["tune"])) != N_TRAIN + N_TUNE:
        raise SystemExit("duplicate ids in the id file")
    if which == "popqa":
        bank = {r["id"]: r for r in jl(data)}
        if sha_file(data) not in idf["bank_sha256"].values():
            raise SystemExit("PopQA bank differs from the one the id file records")

        def rec(i, split):
            b = bank[i]
            if b["split"] != split:
                raise SystemExit("%s is not a %s question" % (i, split))
            g = [x for x in b["gold"]][:GOLD_CAP]
            if not g or not all(isinstance(x, str) and x.strip() for x in g):
                raise SystemExit("bad gold of %s" % i)
            return {"pair_id": "popqa:%s" % i, "question": b["question"], "gold": g, "incorrect": [], "best_answer": g[0]}
    else:
        if sha_file(data) != idf["source_file_sha256"]:
            raise SystemExit("train.csv differs from the one the id file records")
        qs = {q["id"]: q for p in fqa_train_pairs(data).values() for q in p}

        def rec(i, split):
            q = qs[i]
            return {"pair_id": "falseqa:%s" % i, "question": q["question"], "gold": list(q["gold"]), "incorrect": [],
                    "best_answer": q["gold"][0]}
    train = [rec(i, "train") for i in idf["train"]]
    tune = [rec(i, "tune") for i in idf["tune"]]
    for part in (train, tune):
        if any(not q["question"].strip() for q in part):
            raise SystemExit("empty question text")
    held = [train[k]["pair_id"] for k in idf["held_out_positions"]]
    os.makedirs(out, exist_ok=True)
    like_bank = os.path.join(like, "bank.json")
    lb = json.load(open(like_bank, encoding="utf-8"))
    for k in ("model", "model_config", "config", "num_layers", "hidden_size"):
        if k not in lb:
            raise SystemExit("--like bank.json lacks %s" % k)
    shutil.copyfile(like_bank, os.path.join(out, "bank.json"))
    if sha_file(like_bank) != sha_file(os.path.join(out, "bank.json")):
        raise SystemExit("bank.json copy is not byte-identical")
    prov = {"set": which, "ids_file": os.path.basename(ids_json), "ids_file_sha256": sha_file(ids_json),
            "data_file": os.path.basename(data), "data_file_sha256": sha_file(data),
            "bank_json_copied_from": os.path.abspath(like_bank), "bank_json_sha256": sha_file(like_bank),
            "model": lb["model"], "held_out_ids_expected": held,
            "note": "written by scripts/newsets/relearn_1128_source.py (work queue R7 / R8); bank.json is the model's TruthfulQA "
                    "source bank.json byte for byte; questions.json holds the new set's train (96) and tune (192) splits"}
    dump(os.path.join(out, "questions.json"), {"relearn_1128": prov, "splits": {"train": train, "tune": tune}})
    prov["questions_json_sha256"] = sha_file(os.path.join(out, "questions.json"))
    prov["builder_sha256"] = sha_file(os.path.abspath(__file__))
    dump(os.path.join(out, "relearn_1128_provenance.json"), prov)
    print("wrote %s: model %s, %s train %d tune %d, questions.json sha256 %s; held-out ids expected %s" % (
        out, lb["model"], which, len(train), len(tune), prov["questions_json_sha256"][:16], held))


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("ids", "source"):
        raise SystemExit(__doc__)
    {"ids": phase_ids, "source": phase_source}[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
