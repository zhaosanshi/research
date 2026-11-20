# Iteration_01 helper (Claude Fable 5.2, 2026-11-21), read-only, no model, no judge call.
# For one model and task: the E strengths whose release-ladder passes (E + C toward answering at 0.25 / 0.5 / 1 / 2) are
# needed on the test split, decided on the TUNE split only (alias match):
#   the strength of N1's best tune setting (diag/ns_dose_table.py; caps over None / 0.25 / 0.5 / 1 / 2, ties to the
#   lower strength), and the strength of the stability filter's best tune setting (diag/ns_stable_table.py).
# Prints the strengths (dots, space separated) on the last line; nothing else is decided here.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/ns_choose.py TASK MODEL
import sys

sys.path.insert(0, "Iteration_01/diag")
import ns_dose_table as T
import ns_stable_table as ST

S = T.S


def main():
    task, name = sys.argv[1], sys.argv[2]
    m = S.models()[name]
    tsp, ttx, th, _, _, _ = T.load(task, name, m, "tune")
    if not th:
        raise SystemExit("no complete strength on tune")
    n1 = {}
    for s in th:
        cands = [("N1", s, k1, k2) for k1 in T.CAPS for k2 in T.CAPS if not (k1 is None and k2 is None)]
        n1[s] = max(T.net(S.system(tsp, ttx, c)[0]) for c in cands)
    s_n = max(th, key=lambda s: (n1[s], -th.index(s)))
    keys = [(s, cap) for s in th for cap in T.CAPS]
    grid = {k: T.net(ST.stable(tsp, ttx, *k)) for k in keys}
    s_s, cap_s = max(keys, key=lambda k: (grid[k], -keys.index(k)))
    print("%s %s: N1 best on tune at E %s (net %+d); stability filter best on tune at E %s cap %s (net %+d)" % (
        name, task, s_n.replace("-", "."), n1[s_n], s_s.replace("-", "."), cap_s, grid[(s_s, cap_s)]), file=sys.stderr)
    out = [s_n.replace("-", ".")]
    if cap_s is not None and s_s != s_n:
        out.append(s_s.replace("-", "."))
    print(" ".join(out))


if __name__ == "__main__":
    main()
