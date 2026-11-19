# Iteration_01 analysis (Claude Opus 5.5, 2026-11-20), read-only, no model, no judge call.
# TriviaQA, single E minus no write (alias judge), on the first test split and on the fresh split test2, divided by
# whether the question's normalised answer is shared with another question of the deduplicated pool. Question: does
# the swing between the two test sets (+43 vs -135, five-model totals on 500) come from test2 excluding every answer
# an earlier split uses (novelty/ours/2026-11-20_rechosen_caps_validation.md, section 5.3)?
# usage (needs pyarrow, so inside the container):
#   docker exec -i -u 1000:1001 -w /workspace/Truth_Decoupling truth_dec python3 -I Iteration_01/diag/newsets_tqa_shared_answers.py
import collections
import glob
import json
import sys

sys.path.insert(0, "Iteration_01/diag")
sys.path.insert(0, "Iteration_01/code/scripts/newsets")
import newsets_score as S     # noqa: E402
import alias_match as AM      # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

RAW = "data_external/triviaqa_20261119/rc.nocontext/validation-00000-of-00001.parquet"
SC = {"R": 1, "W": -1, "X": 0}


def main():
    t = pq.read_table(RAW, columns=["question_id", "answer"])
    na = {}
    for qid, a in zip(t.column("question_id").to_pylist(), t.column("answer").to_pylist()):
        na.setdefault(qid, a["normalized_value"])     # first row per question_id, as newsets_splits.py
    freq = collections.Counter(na.values())
    bank = {}
    for p in sorted(glob.glob(S.NS + "/bank/triviaqa_bank*.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                bank.setdefault(r["split"], {})[r["id"]] = r
    for sp in ("test", "test2"):
        ids = list(bank[sp])
        c = sum(freq[na[i]] >= 2 for i in ids)
        print("%s: %d questions, answer shared with another pool question: %d (%.0f%%)" % (sp, len(ids), c, 100.0 * c / len(ids)))
    M = S.models()
    tot = collections.defaultdict(int)
    for name in S.ORDER:
        m = M[name]
        out = []
        for sp in ("test", "test2"):
            b = bank[sp]
            sc = {}
            for key, arm in (("x", "x"), ("E", "x:E%d.w.p2" % m["el"])):
                with open("%s/gen/triviaqa/%s/%s/%s.json" % (S.NS, name, sp, S.tok(arm)), encoding="utf-8") as f:
                    g = json.load(f)
                if not g.get("completed") or sorted(r["id"] for r in g["rows"]) != sorted(b):
                    raise SystemExit("%s %s %s is not a complete pass" % (name, sp, arm))
                sc[key] = {r["id"]: SC[AM.verdict(r["text"], b[r["id"]]["gold"],
                                                  not (r["rule_abstain"] or not r["text"].strip()))] for r in g["rows"]}
            for grp, keep in (("shared", lambda i: freq[na[i]] >= 2), ("unique", lambda i: freq[na[i]] == 1),
                              ("all", lambda i: True)):
                q = [i for i in b if keep(i)]
                d = sum(sc["E"][i] - sc["x"][i] for i in q)
                base = sum(sc["x"][i] for i in q)
                tot[(sp, grp)] += d
                tot[(sp, grp, "n")] = len(q)
                out.append("%s %s n=%d no-write net %+d, E minus no write %+d" % (sp, grp, len(q), base, d))
        print("%-12s " % name + " | ".join(out))
    for sp in ("test", "test2"):
        print("five-model total %s: " % sp + ", ".join(
            "%s (n=%d) %+d" % (g, tot[(sp, g, "n")], tot[(sp, g)]) for g in ("shared", "unique", "all")))


if __name__ == "__main__":
    main()
