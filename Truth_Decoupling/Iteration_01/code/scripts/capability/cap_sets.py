"""Question lists for the capability check (work queue F0 / F1): MMLU stratified 1,000 and GSM8K test 1,319.

Writes only IDs and counts, never question text, so the output can go into the public repo.
  MMLU  (cais/mmlu, config "all", split "test", 14,042 rows, 57 subjects): per-subject quotas by largest remainder,
        then a fixed-seed draw inside each subject. ID = "mmlu:<subject>:<row index inside that subject, in file order>".
  GSM8K (openai/gsm8k, config "main", split "test", 1,319 rows): all rows. ID = "gsm8k:<row index>".

Usage: python3 -I cap_sets.py mmlu RAW_PARQUET OUT_JSON [--seed 20261119] [--n 1000]
       python3 -I cap_sets.py gsm8k RAW_PARQUET OUT_JSON
       python3 -I cap_sets.py --selftest
"""
import argparse
import collections
import hashlib
import json
import random

SEED = 20261119


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def largest_remainder(shares, total):
    s = float(sum(shares.values()))
    raw = {k: total * v / s for k, v in shares.items()}
    q = {k: int(raw[k]) for k in raw}
    for k in sorted(raw, key=lambda k: (-(raw[k] - q[k]), k))[:total - sum(q.values())]:
        q[k] += 1
    assert sum(q.values()) == total
    return q


def mmlu_ids(subjects, n, seed):
    """subjects: list of subject names in file order (one per row)."""
    per = collections.defaultdict(list)
    for s in subjects:
        per[s].append("mmlu:%s:%d" % (s, len(per[s])))
    quotas = largest_remainder({s: len(v) for s, v in per.items()}, n)
    rng = random.Random(seed)
    out = []
    for s in sorted(per):
        out.extend(sorted(rng.sample(per[s], quotas[s]), key=lambda x: int(x.rsplit(":", 1)[1])))
    return out, quotas


def selftest():
    subs = ["a"] * 50 + ["b"] * 30 + ["c"] * 20
    ids, q = mmlu_ids(subs, 10, 1)
    assert q == {"a": 5, "b": 3, "c": 2}, q
    assert len(ids) == len(set(ids)) == 10
    assert collections.Counter(x.split(":")[1] for x in ids) == collections.Counter({"a": 5, "b": 3, "c": 2})
    assert all(0 <= int(x.rsplit(":", 1)[1]) < {"a": 50, "b": 30, "c": 20}[x.split(":")[1]] for x in ids)
    assert mmlu_ids(subs, 10, 1)[0] == ids, "not deterministic"
    print("selftest PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set", nargs="?", choices=["mmlu", "gsm8k"])
    ap.add_argument("raw", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not (a.set and a.raw and a.out):
        ap.error("set, raw and out are required")
    import pyarrow.parquet as pq
    t = pq.read_table(a.raw)
    res = {"set": a.set, "source_file_sha256": sha256(a.raw), "rows": t.num_rows}
    if a.set == "mmlu":
        ids, quotas = mmlu_ids(t.column("subject").to_pylist(), a.n, a.seed)
        res.update({"seed": a.seed, "n": a.n, "subjects": len(quotas), "quotas": quotas, "ids": ids})
    else:
        res.update({"ids": ["gsm8k:%d" % i for i in range(t.num_rows)]})
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in res.items() if k not in ("ids", "quotas")}, sort_keys=True))


if __name__ == "__main__":
    main()
