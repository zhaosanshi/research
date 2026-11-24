# Iteration_01 (main session, Claude Fable 5.2, 2026-11-24; plan discussion/work_queue/2026-11-24_queue_n.md, progress
# entry of 20 h): single E at each layer of the depth scan against N1 (E at the main-table layer and strength, blanks
# re-asked with E + C at the current C layer, two caps chosen on tune) and against single E at the main-table setting,
# on the same tune questions. Read-only; every loader, the N1 chooser and the bootstrap are those of depth_scan.py
# (main_table_layers.n1_pick, newsets_score.system, e44_round1_tables.boot). No model, no judge call.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/depth_late_vs_n1.py KEY   (llama | q9)
import sys

sys.path.insert(0, "Iteration_01/diag")
import depth_scan as DS  # noqa: E402


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in DS.KEYS:
        raise SystemExit("usage: depth_late_vs_n1.py llama|q9")
    import main_table_layers as ML
    import newsets_score as S
    import final_records as fr
    import policy_select2 as ps
    import release_origin as ro
    key = sys.argv[1]
    k = DS.KEYS[key]
    choice = DS.load_choice(key)
    st, bl, tx, qs0 = DS.usual(k)
    owner = {}
    DS.add_dir(DS.work(k, "posprompt%s_tune" % k["tag"]), st, bl, tx, False, owner)
    DS.add_dir(DS.work(k, "posprompt2%s_tune" % k["tag"]), st, bl, tx, True, owner)
    # the depth-scan dir may repeat a condition of the main-table dir (the rule can choose the main-table arm itself):
    # read it apart and merge; a repeated cell must carry the same verdict, blank flag and text
    st2 = {kk: v for kk, v in st.items() if kk[1] == "baseline_x"}
    bl2 = {kk: v for kk, v in bl.items() if kk[1] == "baseline_x"}
    tx2 = {kk: v for kk, v in tx.items() if kk[1] == "baseline_x"}
    DS.add_dir(DS.work(k, DS.NAME_E), st2, bl2, tx2, False, {})
    rep = 0
    for kk in st2:
        if kk in st:
            if (st[kk], bl[kk], tx[kk]) != (st2[kk], bl2[kk], tx2[kk]):
                DS.die("cell %s differs between the main-table dir and the depth-scan dir" % (kk,))
            rep += kk[1] != "baseline_x"
        else:
            st[kk], bl[kk], tx[kk] = st2[kk], bl2[kk], tx2[kk]
    s = DS.hy(k["es"])
    txs = fr.Texts(tx, tx)
    sh = S.shell(k["el"], k["cl"])
    need = ["baseline_x", sh.cE(s)] + [sh.cEC(s, kk) for kk in ps.LAD]
    rows0, _ = DS.stage1_arms(choice)
    conds = [DS.cond_e(l, x) for l, _, x in rows0 if x is not None]
    qs = DS.common_qs(st, sorted(set(need + conds)), qs0)
    sp = S.NewSplit(k["el"], k["cl"], st, bl, need, qs=qs)
    pick = ML.n1_pick(sp, txs, s)
    n1s, n1a, _ = S.system(sp, txs, pick)
    es, ea, _ = S.system(sp, txs, ("E", s))
    base = ([st[(q, "baseline_x")] for q in qs], [not bl[(q, "baseline_x")] for q in qs])
    print("# Plan N, %s: single E per layer against N1 and against single E at the main-table setting, tune split (depth_late_vs_n1.py)" % k["name"])
    print("%d tune questions with every verdict. N1 = E%d at %s, blanks re-asked with E + C%d, caps chosen on these questions (%s)."
          % (len(qs), k["el"], k["es"], k["cl"], ro.name_of(pick)))
    print("Score right +1 / no answer 0 / wrong -1; paired bootstrap 10,000 resamples, seed 20261111; natural answer counts.\n")
    if rep:
        print("(%d cells of the depth-scan dir repeat a main-table condition; identical in verdict, blank flag and text)\n" % rep)
    print("| system | answers | right | wrong | net |")
    print("|---|---|---|---|---|")
    for name, (x, a) in (("no write", base), ("single E, layer %d at %s (main table)" % (k["el"], k["es"]), (es, ea)), ("N1", (n1s, n1a))):
        print("| %s | %d | %d | %d | %+d |" % ((name,) + DS.counts(x, a)))
    print("\n| layer | position | line | strength | answers | right | wrong | net | minus N1 [95%] | minus single E at the main-table setting [95%] |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    seen = set()
    for l, line, x in rows0:
        if x is None or (l, x) in seen:
            continue
        seen.add((l, x))
        ls, la = DS.single_e(st, bl, tx, l, DS.hy(x), qs)
        a, r, w, net = DS.counts(ls, la)
        print("| %d | %.3f | %.1f | %s | %d | %d | %d | %+d | %s | %s |" % (
            l, (l + 1.0) / k["nl"], line, x, a, r, w, net,
            DS.boot([DS.val(p) - DS.val(q) for p, q in zip(ls, n1s)]),
            DS.boot([DS.val(p) - DS.val(q) for p, q in zip(ls, es)])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
