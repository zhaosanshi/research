# Iteration_01 helper (Claude Fable 5.2, 2026-11-20), read-only on verdicts; writes one small JSON next to the work dir.
# The mean-difference baseline gets its own best setting: among the judged tune conditions E_L<layer>_x-md-p<strength>
# of a work dir (layers and strengths of logs/newsets_20261119/md_sweep.sh), pick the one with the highest tune score
# (right - wrong, DeepSeek Pro; questions without a verdict in any of these conditions are left out of all of them).
# Ties go to the lower strength, then to the layer nearest to the layer of the learned direction.
# Prints "LAYER STRENGTH" on the last line (the launcher reads it) and writes <work dir>/../mdsweep_choice.json.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/md_choose.py WORK_DIR DAS_LAYER
import json
import re
import sys

sys.path.insert(0, "Iteration_01/diag")
import policy_select2 as ps


def main():
    work, das_layer = sys.argv[1], int(sys.argv[2])
    st, _ = ps.load(work, "pro")
    pat = re.compile(r"^E_L(\d+)_x-md-p([0-9-]+)$")
    conds = sorted({c for _, c in st if pat.match(c)})
    if not conds:
        raise SystemExit("no mean-difference conditions in %s" % work)
    qs = sorted({q for q, c in st if c in conds})
    kept = [q for q in qs if all(st.get((q, c), "?") != "?" for c in conds)]
    rows = []
    for c in conds:
        m = pat.match(c)
        layer, strength = int(m.group(1)), float(m.group(2).replace("-", "."))
        s = [st[(q, c)] for q in kept]
        rows.append({"condition": c, "layer": layer, "strength": strength, "right": s.count("R"), "wrong": s.count("W"),
                     "net": s.count("R") - s.count("W")})
    base = None
    if all((q, "baseline_x") in st for q in kept):
        b = [st[(q, "baseline_x")] for q in kept]
        base = b.count("R") - b.count("W")
    best = max(rows, key=lambda r: (r["net"], -r["strength"], -abs(r["layer"] - das_layer)))
    out = {"work": work, "questions": len(kept), "dropped": len(qs) - len(kept), "no_write_net": base, "das_layer": das_layer,
           "rows": rows, "choice": best}
    with open(work.rstrip("/") + "/../mdsweep_choice.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    for r in rows:
        print("L%-2d strength %-4g net %+4d (right %3d wrong %3d)%s" % (r["layer"], r["strength"], r["net"], r["right"], r["wrong"],
                                                                 "  <- chosen" if r is best else ""))
    print("questions %d (dropped %d); no write net %s" % (len(kept), len(qs) - len(kept), base))
    print("%d %g" % (best["layer"], best["strength"]))


if __name__ == "__main__":
    main()
