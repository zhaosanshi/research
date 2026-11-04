"""Unified record schema + download / sampling / split helpers for dataset adapters.

Every adapter turns one public dataset into a list of records:

  {
    "id": str,                 # dataset-unique id
    "task": "tf" | "qa" | "mc",
    "dataset": str,
    "text": str,               # statement (tf) or question (qa / mc)
    "label": bool | None,      # tf: truth value
    "gold": [str],             # qa: acceptable answers
    "incorrect": [str],        # qa: known wrong answers (TruthfulQA)
    "choices": [str],          # mc
    "answer_idx": int | None,  # mc
    "pair_id": str | None,     # tf pairs (same subject / relation, opposite truth)
    "meta": {...}              # anything else (relation, category, subject, ...)
  }

Adapters are periphery: they know URLs and column names. Core never imports them.
"""
import csv
import hashlib
import io
import json
import os
import random
import sys
import urllib.request


def record(id_, task, dataset, text, label=None, gold=None, incorrect=None, choices=None, answer_idx=None,
           pair_id=None, meta=None):
    return {"id": str(id_), "task": task, "dataset": dataset, "text": text, "label": label,
            "gold": list(gold or []), "incorrect": list(incorrect or []), "choices": list(choices or []),
            "answer_idx": answer_idx, "pair_id": pair_id, "meta": dict(meta or {})}


def sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest, overwrite=False, timeout=120):
    """Download `url` to `dest` (atomic). Fails loudly on HTTP errors; returns dest."""
    if os.path.exists(dest) and not overwrite:
        return dest
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "Truth_Decoupling/0.1 (research)"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)
    print("  downloaded %s (%.1f MB)" % (dest, os.path.getsize(dest) / 1e6), file=sys.stderr)
    return dest


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# sampling and splitting (seeded, deterministic)
# --------------------------------------------------------------------------- #

def balanced_sample_tf(records, n_total, seed, by_pair=True):
    """Balanced true/false sample. With by_pair, whole pairs are kept together
    (so the same pair never straddles a split); n_total counts records."""
    rng = random.Random(seed)
    if by_pair and any(r.get("pair_id") for r in records):
        pairs = {}
        for r in records:
            pairs.setdefault(r["pair_id"], []).append(r)
        keys = sorted(k for k, v in pairs.items() if len(v) == 2 and {v[0]["label"], v[1]["label"]} == {True, False})
        rng.shuffle(keys)
        out = []
        for k in keys:
            if len(out) + 2 > n_total:
                break
            out.extend(sorted(pairs[k], key=lambda r: not r["label"]))
        return out
    pos = sorted([r for r in records if r["label"] is True], key=lambda r: r["id"])
    neg = sorted([r for r in records if r["label"] is False], key=lambda r: r["id"])
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n_total // 2, len(pos), len(neg))
    out = pos[:k] + neg[:k]
    rng.shuffle(out)
    return out


def split_records(records, calib, dev, seed, unit="pair"):
    """Deterministic calib / dev / test split. `calib`/`dev` are counts of *units*
    (pairs when pair_id present and unit == "pair", else records). Test = the rest."""
    rng = random.Random(seed)
    if unit == "pair" and any(r.get("pair_id") for r in records):
        groups = {}
        for r in records:
            groups.setdefault(r["pair_id"] or r["id"], []).append(r)
        keys = sorted(groups)
        rng.shuffle(keys)
        parts = {"calib": keys[:calib], "dev": keys[calib:calib + dev], "test": keys[calib + dev:]}
        return {name: [r for k in ks for r in groups[k]] for name, ks in parts.items()}
    recs = sorted(records, key=lambda r: r["id"])
    rng.shuffle(recs)
    return {"calib": recs[:calib], "dev": recs[calib:calib + dev], "test": recs[calib + dev:]}


def stratified_sample(records, n_total, key_fn, seed):
    """Roughly equal counts per stratum (key_fn(record)), deterministic."""
    rng = random.Random(seed)
    strata = {}
    for r in records:
        strata.setdefault(key_fn(r), []).append(r)
    keys = sorted(strata)
    for k in keys:
        strata[k] = sorted(strata[k], key=lambda r: r["id"])
        rng.shuffle(strata[k])
    out, i = [], 0
    while len(out) < n_total and any(strata[k] for k in keys):
        k = keys[i % len(keys)]
        if strata[k]:
            out.append(strata[k].pop())
        i += 1
    rng.shuffle(out)
    return out
