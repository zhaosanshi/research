# Iteration_01 (main session, Claude Fable 5.2, 2026-11-24; plan discussion/work_queue/2026-11-24_queue_n.md, progress
# entry of 20:06: selection rule, reading and predictions written before any test verdict): the late single-E setting
# picked on the tune split, read on the TEST split against N1 (archived test verdicts; the two caps chosen on tune),
# against single E at the main-table setting and against no write. Natural answer counts. Read-only, no model, no judge
# call. Loaders, the N1 chooser and the bootstrap are those of depth_scan.py (main_table_layers.n1_pick on tune,
# newsets_score.system, e44_round1_tables.boot); the test split is main_table_same_n.Data(name, 1).
# Checks before anything is printed: N1 on tune equals the archived tune score; N1 and single E on test equal the
# archived main table (results/write_positions/main_table_prompt_budget1_20261121.txt).
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/depth_test.py KEY LAYER STRENGTH
#   e.g. python3 -B Iteration_01/diag/depth_test.py llama 17 8
import sys

sys.path.insert(0, "Iteration_01/diag")
import depth_scan as DS  # noqa: E402

REF_TEST = {"llama": dict(n1=(476, 62), e=(319, 40), n=523),   # (answers, net) of the archived main table, test split
            "q9": dict(n1=(444, 186), e=(408, 168), n=523)}


def work(k, split, name):
    return "%s/eval_extra/judge_%s_%s" % (k["run"], split, name)


def usual(k, i):
    import main_table_same_n as MT
    import final_records as fr
    D = MT.Data(k["name"], i)
    utx = fr.raw_texts(D.m["ec"][i])
    st, bl, tx = {}, {}, {}
    for q in D.sp.qs:
        kk = (q, "baseline_x")
        st[kk], bl[kk], tx[kk] = D.sp.st[kk], D.sp.blank[kk], utx[kk]
    return st, bl, tx, list(D.sp.qs)


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in REF_TEST:
        raise SystemExit("usage: depth_test.py llama|q9 LAYER STRENGTH")
    import main_table_layers as ML
    import newsets_score as S
    import final_records as fr
    import policy_select2 as ps
    import release_origin as ro
    key, layer, strength = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    k = DS.KEYS[key]
    s = DS.hy(k["es"])
    sh = S.shell(k["el"], k["cl"])
    need = ["baseline_x", sh.cE(s)] + [sh.cEC(s, kk) for kk in ps.LAD]
    # tune: the pick of N1 (the archived main-table choice)
    st, bl, tx, qs0 = usual(k, 0)
    owner = {}
    DS.add_dir(work(k, "tune", "posprompt%s_tune" % k["tag"]), st, bl, tx, False, owner)
    DS.add_dir(work(k, "tune", "posprompt2%s_tune" % k["tag"]), st, bl, tx, True, owner)
    sp0 = S.NewSplit(k["el"], k["cl"], st, bl, need, qs=qs0)
    pick = ML.n1_pick(sp0, fr.Texts(tx, tx), s)
    n0 = ps.score(S.system(sp0, fr.Texts(tx, tx), pick)[0], 1)
    if n0 != k["ref_n1_tune"]:
        DS.die("N1 on tune scores %+d, the archived main table says %+d" % (n0, k["ref_n1_tune"]))
    # test
    st, bl, tx, qs0 = usual(k, 1)
    owner = {}
    DS.add_dir(work(k, "test", "posprompt%s_test" % k["tag"]), st, bl, tx, False, owner)
    DS.add_dir(work(k, "test", "posprompt2%s_test" % k["tag"]), st, bl, tx, True, owner)
    DS.add_dir(work(k, "test", "depthE_test"), st, bl, tx, False, owner)
    cond = DS.cond_e(layer, DS.hy(strength))
    txs = fr.Texts(tx, tx)
    # the archived systems on every test question they have (must equal the archived main table)
    qsf = DS.common_qs(st, need, qs0)
    spf = S.NewSplit(k["el"], k["cl"], st, bl, need, qs=qsf)
    ref = REF_TEST[key]
    cn = DS.counts(*S.system(spf, txs, pick)[:2])
    ce = DS.counts(*S.system(spf, txs, ("E", s))[:2])
    if len(qsf) != ref["n"] or (cn[0], cn[3]) != ref["n1"] or (ce[0], ce[3]) != ref["e"]:
        DS.die("test check: %d questions, N1 %s, single E %s; the archived main table has %d questions, N1 %s, single E %s" % (
            len(qsf), (cn[0], cn[3]), (ce[0], ce[3]), ref["n"], ref["n1"], ref["e"]))
    # the comparison: questions where the new arm has a verdict too (a cell the judge declined has none)
    qs = DS.common_qs(st, need + [cond], qs0)
    dropped = [q for q in qsf if q not in set(qs)]
    sp = S.NewSplit(k["el"], k["cl"], st, bl, need, qs=qs)
    n1s, n1a, recs = S.system(sp, txs, pick)
    es, ea, _ = S.system(sp, txs, ("E", s))
    ls, la = DS.single_e(st, bl, tx, layer, DS.hy(strength), qs)
    base = ([st[(q, "baseline_x")] for q in qs], [not bl[(q, "baseline_x")] for q in qs])
    print("# Plan N, %s: the late single E (layer %d, strength %s) on the test split against N1 (depth_test.py)" % (k["name"], layer, strength))
    print("N1 = E%d at %s, blanks re-asked with E + C%d, caps chosen on tune (%s). Checks passed: N1 on tune %+d; on the %d test "
          "questions N1 (%d answers, net %+d) and single E (%d answers, net %+d) equal the archived main table." % (
              k["el"], k["es"], k["cl"], ro.name_of(pick), n0, len(qsf), cn[0], cn[3], ce[0], ce[3]))
    print("Read on %d questions: %d left out because the judge declined to grade the new arm's answer (%s); each could move "
          "the new arm's net by at most 1, so every difference below by at most %d." % (
              len(qs), len(dropped), ", ".join(dropped) or "none", len(dropped)))
    print("Score right +1 / no answer 0 / wrong -1; paired bootstrap 10,000 resamples, seed 20261111; natural answer counts.\n")
    print("| system | answers | right | wrong | net |")
    print("|---|---|---|---|---|")
    rows = (("no write", base), ("single E, layer %d at %s (main table)" % (k["el"], k["es"]), (es, ea)), ("N1", (n1s, n1a)),
            ("single E, layer %d at %s" % (layer, strength), (ls, la)))
    for name, (x, a) in rows:
        print("| %s | %d | %d | %d | %+d |" % ((name,) + DS.counts(x, a)))
    d = lambda p, q: DS.boot([DS.val(a) - DS.val(b) for a, b in zip(p, q)])
    print("\n| difference | net [95%] |")
    print("|---|---|")
    print("| late single E minus N1 | %s |" % d(ls, n1s))
    print("| late single E minus single E at the main-table setting | %s |" % d(ls, es))
    print("| late single E minus no write | %s |" % d(ls, base[0]))
    print("| N1 minus single E at the main-table setting | %s |" % d(n1s, es))
    print("| N1 minus no write | %s |" % d(n1s, base[0]))
    # where the two systems differ
    both = sum(1 for a, b in zip(la, n1a) if a and b)
    only_l = [(x, y) for x, y, a, b in zip(ls, n1s, la, n1a) if a and not b]
    only_n = [(x, y) for x, y, a, b in zip(ls, n1s, la, n1a) if b and not a]
    print("\nanswered by both %d; only by the late single E %d (right %d, wrong %d); only by N1 %d (right %d, wrong %d)" % (
        both, len(only_l), sum(x == "R" for x, _ in only_l), sum(x == "W" for x, _ in only_l),
        len(only_n), sum(y == "R" for _, y in only_n), sum(y == "W" for _, y in only_n)))
    bb = [(x, y) for x, y, a, b in zip(ls, n1s, la, n1a) if a and b]
    print("on the %d questions both answer: late single E right %d wrong %d; N1 right %d wrong %d" % (
        len(bb), sum(x == "R" for x, _ in bb), sum(x == "W" for x, _ in bb), sum(y == "R" for _, y in bb), sum(y == "W" for _, y in bb)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
