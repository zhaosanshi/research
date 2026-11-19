"""Fixed-seed splits for the TriviaQA and PopQA pilot (work queue C0 / C1).

Writes only question IDs and counts, never question text, so the output can go into the public repo.

TriviaQA (rc.nocontext, validation; the test answers are not public):
  1. keep the first row of each question_id (17,944 rows -> 9,960 questions, the count used by neighbouring papers);
  2. drop question_ids whose normalised question text is shared with another question_id;
  3. shuffle with the seed and fill test, then tune, then train; a question whose normalised answer value already
     belongs to another split is skipped, so no normalised question text and no normalised answer value is shared
     between two splits.
PopQA (the single official test file, 14,267 rows):
  1. rows whose question text is shared by rows with different subjects are ambiguous and dropped (whole group);
  2. rows whose question text is shared by rows with the same subject are merged into one item (multi-valued
     relations); the item keeps the smallest row id and the union of the gold answer lists;
  3. per-split relation quotas by largest remainder (16 relations), then shuffle with the seed and fill test, tune,
     train; a subject that already belongs to another split is skipped, so subjects never cross splits.
The Pro subset is the first PRO_N test IDs of a second fixed-seed shuffle of the test IDs.

Usage: python3 -I newsets_splits.py {triviaqa|popqa} RAW_PATH OUT_JSON [--seed 20261119]
       python3 -I newsets_splits.py extend {triviaqa|popqa} RAW_PATH BASE_SPLITS_JSON OUT_JSON [--name test2] [--n 500]
                                   [--pro-n 100] [--seed 20261120]
         a fresh split from the items no existing split uses, with the same rules: TriviaQA items whose normalised
         answer value is in an existing split are skipped, PopQA items whose subject is in an existing split are
         skipped and the relation quotas follow the cleaned pool; the Pro subset comes from a second shuffle (seed + 1)
       python3 -I newsets_splits.py --selftest
"""
import argparse
import collections
import csv
import hashlib
import json
import random
import re
import string
import sys

SIZES = (("test", 500), ("tune", 192), ("train", 300))
PRO_N = 100
SEED = 20261119

_PUNC = set(string.punctuation + "\u2018\u2019\u00b4`")


def tqa_norm(s):
    """TriviaQA's official answer normalisation (triviaqa utils/dataset_utils.normalize_answer)."""
    s = s.replace("_", " ").lower()
    s = "".join(" " if ch in _PUNC else ch for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split()).strip()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def largest_remainder(shares, total):
    """Integer quotas summing to total, proportional to shares (dict key -> count); ties broken by key order."""
    s = float(sum(shares.values()))
    raw = {k: total * v / s for k, v in shares.items()}
    q = {k: int(raw[k]) for k in raw}
    left = total - sum(q.values())
    order = sorted(raw, key=lambda k: (-(raw[k] - q[k]), k))
    for k in order[:left]:
        q[k] += 1
    assert sum(q.values()) == total
    return q


def fill(items, rng, group_of, quota_key=None, quotas=None):
    """items: list of dicts with 'id'. group_of(item) -> key that must not cross splits.
    quotas: {split: {quota_key(item): n}} or None (then plain sizes). Fills splits in SIZES order."""
    order = list(items)
    rng.shuffle(order)
    need = {name: n for name, n in SIZES}
    left = {name: dict(quotas[name]) for name, _ in SIZES} if quotas else None
    owner = {}
    out = {name: [] for name, _ in SIZES}
    skipped = 0
    for it in order:
        g = group_of(it)
        cands = [owner[g]] if g in owner else [name for name, _ in SIZES]
        placed = False
        for name in cands:
            if len(out[name]) >= need[name]:
                continue
            if left is not None and left[name].get(quota_key(it), 0) <= 0:
                continue
            out[name].append(it["id"])
            owner[g] = name
            if left is not None:
                left[name][quota_key(it)] -= 1
            placed = True
            break
        if not placed and g in owner:
            skipped += 1  # its group already lives in a split that is full (or out of quota) for it
        if all(len(out[n]) >= need[n] for n, _ in SIZES):
            break
    for name, n in SIZES:
        if len(out[name]) != n:
            raise SystemExit("could not fill split %s: %d of %d" % (name, len(out[name]), n))
    return out, skipped


def load_triviaqa(path):
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["question_id", "question", "answer"])
    seen, items = set(), []
    for qid, q, a in zip(t.column("question_id").to_pylist(), t.column("question").to_pylist(), t.column("answer").to_pylist()):
        if qid in seen:
            continue
        seen.add(qid)
        items.append({"id": qid, "nq": tqa_norm(q), "na": a["normalized_value"]})
    return items


def triviaqa_items(items):
    by_text = collections.Counter(it["nq"] for it in items)
    kept = [it for it in items if by_text[it["nq"]] == 1]
    return kept, {"rows_first_per_question_id": len(items), "dropped_shared_question_text": len(items) - len(kept)}


def load_popqa(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [{"id": r["id"], "q": r["question"], "subj": r["subj_id"], "prop": r["prop"],
             "gold": json.loads(r["possible_answers"])} for r in rows]


def popqa_items(rows):
    by_q = collections.defaultdict(list)
    for r in rows:
        by_q[r["q"]].append(r)
    items, merged_groups, dropped_rows = [], 0, 0
    for q, g in by_q.items():
        if len(g) == 1:
            r = g[0]
            items.append({"id": r["id"], "subj": r["subj"], "prop": r["prop"], "merged_ids": [r["id"]]})
            continue
        if len(set(r["subj"] for r in g)) > 1:
            dropped_rows += len(g)
            continue
        ids = sorted((r["id"] for r in g), key=lambda x: int(x))
        props = sorted(set(r["prop"] for r in g))
        if len(props) != 1:
            raise SystemExit("same subject and text but different relations: %r" % ids)
        merged_groups += 1
        items.append({"id": ids[0], "subj": g[0]["subj"], "prop": props[0], "merged_ids": ids})
    info = {"rows": len(rows), "dropped_ambiguous_rows": dropped_rows, "merged_same_subject_groups": merged_groups,
            "items": len(items)}
    return items, info


def split_triviaqa(path, seed):
    items, info = triviaqa_items(load_triviaqa(path))
    out, skipped = fill(items, random.Random(seed), group_of=lambda it: it["na"])
    info["skipped_answer_in_other_split"] = skipped
    return out, info, {}


def split_popqa(path, seed):
    items, info = popqa_items(load_popqa(path))
    shares = collections.Counter(it["prop"] for it in items)
    quotas = {name: largest_remainder(shares, n) for name, n in SIZES}
    out, skipped = fill(items, random.Random(seed), group_of=lambda it: it["subj"], quota_key=lambda it: it["prop"],
                        quotas=quotas)
    info["skipped_subject_in_other_split"] = skipped
    merged = {it["id"]: it["merged_ids"] for it in items if len(it["merged_ids"]) > 1}
    used = set(out["test"]) | set(out["tune"]) | set(out["train"])
    return out, info, {k: v for k, v in merged.items() if k in used}


def finish(name, path, seed, out, info, merged):
    pro = list(out["test"])
    random.Random(seed + 1).shuffle(pro)
    res = {"dataset": name, "source_file_sha256": sha256(path), "seed": seed, "sizes": dict(SIZES),
           "info": info, "train": out["train"], "tune": out["tune"], "test": out["test"],
           "test_pro100": sorted(pro[:PRO_N], key=str)}
    if merged:
        res["merged_row_ids"] = merged
    return res


def extend(dataset, raw, base, name, n, pro_n, seed):
    """A fresh split disjoint from every split of base (IDs, and TriviaQA answers / PopQA subjects)."""
    used = set(base["train"]) | set(base["tune"]) | set(base["test"])
    if dataset == "triviaqa":
        items, info = triviaqa_items(load_triviaqa(raw))
        group = lambda it: it["na"]  # noqa: E731
    else:
        items, info = popqa_items(load_popqa(raw))
        group = lambda it: it["subj"]  # noqa: E731
    taken = {group(it) for it in items if it["id"] in used}
    if len(taken) == 0 or not used <= {it["id"] for it in items}:
        raise SystemExit("the base split file does not match this raw file")
    pool = [it for it in items if it["id"] not in used and group(it) not in taken]
    rng = random.Random(seed)
    order = list(pool)
    rng.shuffle(order)
    if dataset == "popqa":
        shares = collections.Counter(it["prop"] for it in items)
        left = largest_remainder(shares, n)
        out = []
        for it in order:
            if left.get(it["prop"], 0) > 0:
                out.append(it["id"])
                left[it["prop"]] -= 1
            if len(out) == n:
                break
    else:
        out = [it["id"] for it in order[:n]]
    if len(out) != n:
        raise SystemExit("could not fill %d items" % n)
    pro = list(out)
    random.Random(seed + 1).shuffle(pro)
    res = {"dataset": dataset, "source_file_sha256": sha256(raw), "seed": seed, "name": name,
           "base_splits_seed": base.get("seed"), "info": dict(info, pool_after_exclusion=len(pool), excluded_groups=len(taken)),
           name: out, name + "_pro100": sorted(pro[:pro_n], key=str)}
    if dataset == "popqa":
        merged = {it["id"]: it["merged_ids"] for it in items if len(it["merged_ids"]) > 1 and it["id"] in set(out)}
        if merged:
            res["merged_row_ids"] = merged
    return res


def selftest():
    assert tqa_norm("The  Beatles!") == "beatles"
    assert tqa_norm("Rock_and_roll") == "rock and roll"
    assert tqa_norm("O\u2019Neill") == "o neill"
    q = largest_remainder({"a": 5, "b": 3, "c": 2}, 7)
    assert q == {"a": 4, "b": 2, "c": 1}, q
    # fill: groups never cross splits, sizes exact
    global SIZES
    keep = SIZES
    SIZES = (("test", 5), ("tune", 3), ("train", 2))
    try:
        items = [{"id": str(i), "g": i // 2} for i in range(40)]
        out, _ = fill(items, random.Random(1), group_of=lambda it: it["g"])
        owner = {}
        for name, ids in out.items():
            for x in ids:
                g = int(x) // 2
                assert owner.setdefault(g, name) == name, "group crossed splits"
        assert [len(out[n]) for n, _ in SIZES] == [5, 3, 2]
        # quotas respected
        items = [{"id": str(i), "g": i, "p": "x" if i % 4 else "y"} for i in range(80)]
        shares = collections.Counter(it["p"] for it in items)
        quotas = {name: largest_remainder(shares, n) for name, n in SIZES}
        out, _ = fill(items, random.Random(2), group_of=lambda it: it["g"], quota_key=lambda it: it["p"], quotas=quotas)
        for name, ids in out.items():
            c = collections.Counter("x" if int(x) % 4 else "y" for x in ids)
            assert dict(c) == {k: v for k, v in quotas[name].items() if v}, (name, c, quotas[name])
        # popqa grouping: ambiguous dropped, same-subject merged
        rows = [{"id": "1", "q": "Q1", "subj": "s1", "prop": "r", "gold": ["a"]},
                {"id": "2", "q": "Q1", "subj": "s2", "prop": "r", "gold": ["b"]},
                {"id": "3", "q": "Q2", "subj": "s3", "prop": "r", "gold": ["c"]},
                {"id": "10", "q": "Q2", "subj": "s3", "prop": "r", "gold": ["d"]},
                {"id": "4", "q": "Q3", "subj": "s4", "prop": "r", "gold": ["e"]}]
        items, info = popqa_items(rows)
        assert sorted(it["id"] for it in items) == ["3", "4"], items
        assert info["dropped_ambiguous_rows"] == 2 and info["merged_same_subject_groups"] == 1
        assert [it for it in items if it["id"] == "3"][0]["merged_ids"] == ["3", "10"]
        # triviaqa: shared normalised text dropped
        items, info = triviaqa_items([{"id": "a", "nq": "x", "na": "1"}, {"id": "b", "nq": "x", "na": "2"},
                                      {"id": "c", "nq": "y", "na": "3"}])
        assert [it["id"] for it in items] == ["c"] and info["dropped_shared_question_text"] == 2
    finally:
        SIZES = keep
    print("selftest PASS")


def main_extend(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["triviaqa", "popqa"])
    ap.add_argument("raw")
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--name", default="test2")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--pro-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261120)
    a = ap.parse_args(argv)
    with open(a.base, encoding="utf-8") as f:
        base = json.load(f)
    if base.get("dataset") != a.dataset:
        raise SystemExit("base split file is for %r" % base.get("dataset"))
    res = extend(a.dataset, a.raw, base, a.name, a.n, a.pro_n, a.seed)
    new = res[a.name]
    assert len(set(new)) == len(new) and not set(new) & (set(base["train"]) | set(base["tune"]) | set(base["test"]))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print(json.dumps({k: res[k] for k in ("dataset", "seed", "name", "info")}, sort_keys=True))


def main():
    if sys.argv[1:2] == ["extend"]:
        main_extend(sys.argv[2:])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", choices=["triviaqa", "popqa"])
    ap.add_argument("raw", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not (a.dataset and a.raw and a.out):
        ap.error("dataset, raw and out are required")
    fn = split_triviaqa if a.dataset == "triviaqa" else split_popqa
    out, info, merged = fn(a.raw, a.seed)
    res = finish(a.dataset, a.raw, a.seed, out, info, merged)
    all_ids = res["train"] + res["tune"] + res["test"]
    assert len(set(all_ids)) == len(all_ids), "an ID appears in two splits"
    assert set(res["test_pro100"]) <= set(res["test"]) and len(res["test_pro100"]) == PRO_N
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print(json.dumps({k: res[k] for k in ("dataset", "seed", "sizes", "info")}, sort_keys=True))


if __name__ == "__main__":
    main()
