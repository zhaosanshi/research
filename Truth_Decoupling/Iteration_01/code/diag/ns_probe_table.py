# Iteration_01 analysis (Claude Fable 5.2, 2026-11-21), read-only, no model, no judge call. EXPLORATION, not a result.
# TriviaQA / PopQA: a write (or another prompt) used as a PROBE of the model's own no-write answer.
#   stable(P): the no-write answer A0 is kept if the pass P answers and its answer is A0 (compared as
#              release_origin.norm, the comparison of N1's own rule); otherwise the question is left blank.
#   stable(P1 and P2): kept only if both passes return A0.
# So every system returns the model's own no-write answer or a blank; the passes differ only in what probes it.
# Probe families (every member a full pass over the split; a family enters only with the files that exist on tune and on
# the reading split):
#   E          x:E<l>.w.p<s>                         the E write alone
#   C caution  x:C<l>.w.p<k>                         the C write toward caution alone
#   C answer   x:C<l>.w.m<k>                         the C write toward answering alone
#   E+C caut.  x:E<l>.w.p<s>+C<l>.w.p<k>             both writes in one pass
#   E+C answ.  x:E<l>.w.p<s>+C<l>.w.m<k>             both writes in one pass
#   E and Cc   stable(E at s) and stable(C caution at k), two passes
#   prompt     conf                                  the must-answer prompt, nothing written (a control: is any change
#                                                    of the input as good a probe as the E write?)
# Per family the member is chosen on the task's tune split by net score (ties: the earlier member = lower strengths);
# its numbers are read on the other split. Reference systems: no write; single E answering by itself at its own best
# tune strength (its answers as they are); and, where the confidence file exists, the usual confidence filter at the
# same answer count as the E probe (the must-answer pass, its most confident answers by mean log-probability).
# Verdicts by alias match; paired intervals with questions resampled.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/ns_probe_table.py [--split2 test2] [TASK ...]
import json
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
sys.path.insert(0, "Iteration_01/code/scripts/newsets")
import newsets_score as S     # noqa: E402
import alias_match as AM      # noqa: E402
import release_origin as ro   # noqa: E402

NS = S.NS
ES = ("0.25", "0.5", "0.75", "1", "1.5", "2", "3", "4")
KS = ("0.25", "0.5", "1", "2")


class Passes:
    def __init__(self, task, model, split):
        self.task, self.model, self.split = task, model, split
        self.bank_path = "%s/bank/%s_bank_%s.jsonl" % (NS, task, "test2_20261120" if split == "test2" else "20261119")
        with open(self.bank_path, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f]
        self.b = {r["id"]: r for r in rows if r["split"] == split}
        self.ids = list(self.b)
        self.sha = S.sha_file(self.bank_path)
        self.cache = {}

    def get(self, arm):
        """(states, answered, normalized texts) of a complete full pass, or None if the file does not exist."""
        if arm in self.cache:
            return self.cache[arm]
        f = "%s/gen/%s/%s/%s/%s.json" % (NS, self.task, self.model, self.split, S.tok(arm))
        out = None
        if os.path.exists(f):
            with open(f, encoding="utf-8") as h:
                g = json.load(h)
            k = g.get("contract") or {}
            if not (g.get("completed") and k.get("full_pass") and k.get("split") == self.split and k.get("model") == self.model
                    and k.get("items_file_sha256") == self.sha and [r["id"] for r in g["rows"]] == self.ids):
                raise SystemExit("%s is not a complete %s pass of %s" % (f, self.split, self.model))
            st, ans, tx = [], [], []
            for r in g["rows"]:
                blank = bool(r["rule_abstain"]) or not r["text"].strip()
                st.append(AM.verdict(r["text"], self.b[r["id"]]["gold"], not blank) if not blank else "X")
                ans.append(not blank)
                tx.append(ro.norm(r["text"]))
            out = (st, ans, tx)
        self.cache[arm] = out
        return out


def stable(base, probes):
    """States of the filter: the no-write state where every probe answers with the no-write answer, else blank."""
    st, ans, tx = base
    return [st[i] if ans[i] and all(p[1][i] and p[2][i] == tx[i] for p in probes) else "X" for i in range(len(st))]


def net(v):
    return sum((x == "R") - (x == "W") for x in v)


def cnt(v):
    return "%+4d (ans %3d, R %3d W %3d)" % (net(v), sum(x != "X" for x in v), v.count("R"), v.count("W"))


def diff(a, b):
    return S.ps.ci([((x == "R") - (x == "W")) - ((y == "R") - (y == "W")) for x, y in zip(a, b)])


def conf_top(task, name, split, X, n):
    """The must-answer pass with only its n most confident answers kept (mean log-probability of the answer tokens,
    files of newsets_gen.py conf, bound to the generation by newsets_score.conf_rows); None if it cannot be built."""
    try:
        g, _ = S.gen_file(task, name, split, "conf")
        rows = S.conf_rows(task, name, split, "conf", {"%s:%s" % (task, r["id"]): r["text"] for r in g["rows"]})
    except (SystemExit, OSError):
        return None
    cf = X.get("conf")
    if cf is None:
        return None
    ids = ["%s:%s" % (task, i) for i in X.ids]
    cand = [i for i in range(len(ids)) if cf[1][i] and rows[ids[i]].get("mean_logp") is not None]
    if len(cand) < n:
        return None
    kept = set(sorted(cand, key=lambda i: (-rows[ids[i]]["mean_logp"], i))[:n])
    return [cf[0][i] if i in kept else "X" for i in range(len(ids))]


def families(m):
    e, c = "x:E%d.w.p" % m["el"], "x:C%d.w." % m["cl"]
    return [
        ("E", [((e + s,), "E %s" % s) for s in ES]),
        ("C caution", [((c + "p" + k,), "C caution %s" % k) for k in KS]),
        ("C answer", [((c + "m" + k,), "C answer %s" % k) for k in KS]),
        ("E+C caution, one pass", [(("%s%s+%sp%s" % (e, s, c[2:], k),), "E %s + C caution %s" % (s, k)) for s in ES for k in KS]),
        ("E+C answer, one pass", [(("%s%s+%sm%s" % (e, s, c[2:], k),), "E %s + C answer %s" % (s, k)) for s in ES for k in KS]),
        ("E and C caution, two passes", [((e + s, c + "p" + k), "E %s and C caution %s" % (s, k)) for s in ES for k in KS]),
        ("prompt (control)", [(("conf",), "must-answer prompt")]),
    ]


def main():
    args = sys.argv[1:]
    second = "test"
    if args[:1] == ["--split2"]:
        second, args = args[1], args[2:]
    M = S.models()
    for task in args or list(S.TASKS):
        print("\n# %s (alias match): members chosen on tune (192), numbers on %s" % (task, second))
        tot = {}
        for name in S.ORDER:
            m = M[name]
            T, X = Passes(task, name, "tune"), Passes(task, name, second)
            bt, bx = T.get("x"), X.get("x")
            if bt is None or bx is None:
                print("\n## %s: no no-write pass" % name)
                continue
            print("\n## %s: no write: tune %s | %s %s" % (name, cnt(bt[0]), second, cnt(bx[0])))
            eb = [(s, T.get("x:E%d.w.p%s" % (m["el"], s)), X.get("x:E%d.w.p%s" % (m["el"], s))) for s in ES]
            eb = [(s, a, b) for s, a, b in eb if a and b]
            ref = {"no write": bx[0]}
            if eb:
                s, a, b = max(eb, key=lambda r: (net(r[1][0]), -eb.index(r)))
                ref["single E answering (E %s)" % s] = b[0]
                print("    single E answering by itself, best on tune E %s: tune %s | %s %s" % (s, cnt(a[0]), second, cnt(b[0])))
            got = {}
            for fam, members in families(m):
                rows = []
                for arms, label in members:
                    pt, px = [T.get(a) for a in arms], [X.get(a) for a in arms]
                    if all(pt) and all(px):
                        rows.append((label, stable(bt, pt), stable(bx, px)))
                    elif all(pt):
                        rows.append((label, stable(bt, pt), None))
                if not rows:
                    continue
                both = [r for r in rows if r[2] is not None]
                top = max(rows, key=lambda r: (net(r[1]), -rows.index(r)))
                line = "    %-28s %2d members on tune (%2d also on %s); best on tune: %s, tune %s" % (fam, len(rows), len(both), second, top[0], cnt(top[1]))
                if both:
                    pick = max(both, key=lambda r: (net(r[1]), -both.index(r)))
                    got[fam] = pick
                    if pick[0] != top[0]:
                        line += " | best with a %s file: %s, tune %s" % (second, pick[0], cnt(pick[1]))
                    line += " || %s %s | minus %s" % (second, cnt(pick[2]), " | ".join("%s %s" % (k, diff(pick[2], v)) for k, v in ref.items()))
                print(line)
            if "E" in got:
                for fam in got:
                    if fam != "E":
                        print("        %-28s minus the E probe: %s" % (fam, diff(got[fam][2], got["E"][2])))
                        tot.setdefault("%s minus the E probe" % fam, []).append(dict(zip(X.ids, [((a == "R") - (a == "W")) - ((b == "R") - (b == "W")) for a, b in zip(got[fam][2], got["E"][2])])))
                n = sum(v != "X" for v in got["E"][2])
                top = conf_top(task, name, second, X, n)
                if top is not None:
                    print("        confidence filter (must-answer prompt, its %d most confident answers) %s | the E probe minus it: %s" % (
                        n, cnt(top), diff(got["E"][2], top)))
                    tot.setdefault("E probe minus the confidence filter, same answer count", []).append(
                        dict(zip(X.ids, [((a == "R") - (a == "W")) - ((b == "R") - (b == "W")) for a, b in zip(got["E"][2], top)])))
                for k, v in ref.items():
                    key = "E probe minus " + ("single E answering" if k.startswith("single E") else k)
                    tot.setdefault(key, []).append(dict(zip(X.ids, [((a == "R") - (a == "W")) - ((b == "R") - (b == "W")) for a, b in zip(got["E"][2], v)])))
        print("\n## %s, models together (a line only if all %d models have it; questions resampled)" % (task, len(S.ORDER)))
        for key, per in tot.items():
            if len(per) == len(S.ORDER):
                common = sorted(set.intersection(*[set(p) for p in per]))
                print("    %-52s %s (%d questions)" % (key, S.ps.ci([sum(p[q] for p in per) for q in common]), len(common)))


if __name__ == "__main__":
    main()
