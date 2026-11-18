# Iteration_01 analysis (Claude Opus 5.5, 2026-11-19), read-only, no model, no judge call.
# One per-question record of what a scheme finally hands in, shared by scoring and by printing. GPT 2026-11-18
# (discussion/novelty_review/2026-11-18_gpt_results_recheck.md section 4): the case printer of b_vs_n1.py showed answers the
# scheme never handed in (a vetoed C answer printed as N1's). Work queue A1 (discussion/work_queue/2026-11-19_queue.md).
# For every question of a split and a setting cfg, records() returns a dict:
#   p      question id
#   cond   the condition whose answer is handed in
#   src    "ec" (the E+C grid dir) or "d" (the must-answer dir): where that answer and its verdict live
#   text   the raw answer handed in (judge2/cells.jsonl "prediction")
#   state  R right / W wrong / H hedge / X blank / ? no verdict, exactly as the existing scorers count it
#   cand   None (not blank under E at s, or the setting has no E), "own" (also blank with no write) or "E's" (answered with no
#          write, blank only under E)
#   rung   the C strength of the first non-blank answer on the ladder (None if there is none)
#   note   what the scheme did with the question (see NOTES)
# Settings: ("base",) ("E", s) ("C", k) ("ALL", s, k) ("EC", s, cap) ("BO", s, cap) ("T2", s, k1, k2) ("N1", s, k1, k2)
# ("B", s) ("DE", s) ("Dnone",) ("DA",); k1 is the cap of E's blanks and k2 the cap of own blanks, as in release_origin.py.
# The states equal those of release_origin.states / two_caps.states / c_first.states / policy_select2.Split.states:
#   --selftest  synthetic split with the three cases GPT asked for (vetoed, above the cap, blank on one side only), checked
#               against the existing scorers and by question id, final condition and raw text
#   --check     every setting of every family on the real runs of the five models (tune, test, and the full test split of
#               the two Gemma-3), records against the existing scorers
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/final_records.py --selftest | --check
import json
import sys

sys.path.insert(0, "Iteration_01/diag")
import policy_select2 as ps
import c_first as cf
import two_caps as tc
import release_origin as ro

LAD = ps.LAD
CAPS = ro.CAPS
NOTES = ("no write", "E alone", "C alone", "E and C on every question", "released", "vetoed: the original answer came back",
         "above the cap", "blank at every dose", "not re-asked", "re-asked with the must-answer prompt",
         "must-answer prompt", "not a candidate")


def raw_texts(work):
    out = {}
    for line in open(work + "/judge2/cells.jsonl", encoding="utf-8"):
        r = json.loads(line)
        out[(r["pair_id"], r["condition"])] = r["prediction"]
    return out


class Texts:
    """Raw answers of one split: the E+C grid dir, and the must-answer dir when the split has one."""

    def __init__(self, ec, d=None):
        self.ec = ec
        self.d = d or {}

    @classmethod
    def of(cls, m, i):
        return cls(raw_texts(m["ec"][i]), raw_texts(m["d"][i]) if m.get("d") else None)


def rec(p, cond, src, sp, tx, cand=None, rung=None, note=""):
    st = sp.st if src == "ec" else sp.dst
    text = (tx.ec if src == "ec" else tx.d).get((p, cond))
    return dict(p=p, cond=cond, src=src, text=text, state=st[(p, cond)], cand=cand, rung=rung, note=note)


def first_answer(sp, p, s, ladder):
    for k in ladder:
        if not sp.blank[(p, sp.cEC(s, k))]:
            return k
    return None


def cand_of(sp, p, s):
    if not sp.blank[(p, sp.cE(s))]:
        return None
    return "own" if sp.blank[(p, "baseline_x")] else "E's"


def ladder_record(sp, tx, p, s, cap, veto):
    """A question blank under E at s, re-asked with C up the ladder to cap; veto: an E's blank whose first answer is its
    original (no-write) answer, compared as release_origin.norm, stays blank and the ladder stops there."""
    e = sp.cE(s)
    cand = cand_of(sp, p, s)
    if cap is None:
        return rec(p, e, "ec", sp, tx, cand, None, "not re-asked")
    k = first_answer(sp, p, s, tc.lad(cap))
    if k is None:
        later = first_answer(sp, p, s, LAD)
        return rec(p, e, "ec", sp, tx, cand, later, "above the cap" if later is not None else "blank at every dose")
    c = sp.cEC(s, k)
    if veto and cand == "E's" and ro.norm(tx.ec[(p, "baseline_x")]) == ro.norm(tx.ec[(p, c)]):
        return rec(p, e, "ec", sp, tx, cand, k, "vetoed: the original answer came back")
    return rec(p, c, "ec", sp, tx, cand, k, "released")


def records(sp, cfg, tx):
    kind = cfg[0]
    out = []
    for p in sp.qs:
        if kind == "base":
            out.append(rec(p, "baseline_x", "ec", sp, tx, None, None, "no write"))
        elif kind == "C":
            out.append(rec(p, sp.cC(cfg[1]), "ec", sp, tx, None, None, "C alone"))
        elif kind == "Dnone":
            out.append(rec(p, "baseline_conf", "d", sp, tx, None, None, "must-answer prompt"))
        elif kind == "DA":
            out.append(rec(p, "baseline_conf_E2", "d", sp, tx, None, None, "must-answer prompt"))
        elif kind == "DE":
            out.append(rec(p, sp.dE(cfg[1]), "d", sp, tx, None, None, "must-answer prompt"))
        else:
            s = cfg[1]
            e = sp.cE(s)
            cand = cand_of(sp, p, s)
            if kind == "ALL":
                out.append(rec(p, sp.cEC(s, cfg[2]), "ec", sp, tx, cand, None, "E and C on every question"))
            elif cand is None:
                out.append(rec(p, e, "ec", sp, tx, None, None, "E alone" if kind == "E" else "not a candidate"))
            elif kind == "E":
                out.append(rec(p, e, "ec", sp, tx, cand, first_answer(sp, p, s, LAD), "not re-asked"))
            elif kind == "B":
                out.append(rec(p, sp.dE(s), "d", sp, tx, cand, None, "re-asked with the must-answer prompt"))
            elif kind == "EC":
                out.append(ladder_record(sp, tx, p, s, cfg[2], veto=False))
            elif kind == "BO":
                out.append(ladder_record(sp, tx, p, s, cfg[2] if cand == "own" else None, veto=False))
            elif kind == "T2":
                out.append(ladder_record(sp, tx, p, s, cfg[3] if cand == "own" else cfg[2], veto=False))
            elif kind == "N1":
                out.append(ladder_record(sp, tx, p, s, cfg[3] if cand == "own" else cfg[2], veto=True))
            else:
                raise ValueError(cfg)
    return out


def states(recs):
    return [r["state"] for r in recs]


def answered(recs, sp, dblank=None):
    """True where the answer handed in is not blank for the rule detector (nor empty). dblank: the blank flags of the
    must-answer dir (policy_select2.load(...)[1]); needed only when a record comes from it."""
    db = dblank if dblank is not None else getattr(sp, "dblank", None)
    out = []
    for r in recs:
        if r["src"] == "ec":
            out.append(not sp.blank[(r["p"], r["cond"])])
        elif db is None:
            raise ValueError("answered(): a must-answer record needs the blank flags of the must-answer dir")
        else:
            out.append(not db[(r["p"], r["cond"])])
    return out


def existing(sp, cfg, normtx):
    """The scorers the analyses so far used, for comparison."""
    if cfg[0] in ("N1", "N2"):
        return ro.states(sp, normtx, cfg)
    if cfg[0] == "T2":
        return tc.states(sp, cfg)
    if cfg[0] in ("ALL", "BO", "CF"):
        return cf.states(sp, cfg)
    return sp.states(cfg)


def settings(sp):
    out = [("base",)] + [("C", k) for k in sp.cks] + [("E", s) for s in "12"]
    for s in "12":
        out += [("ALL", s, k) for k in LAD] + [("EC", s, k) for k in LAD] + [("BO", s, k) for k in LAD]
        out += [(kind, s, k1, k2) for kind in ("T2", "N1") for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]
    if sp.dst is not None:
        out += [("B", s) for s in "12"] + [("DE", s) for s in "12"] + [("Dnone",), ("DA",)]
    return out


def compare(sp, tx, label):
    normtx = {k: ro.norm(v) for k, v in tx.ec.items()}
    bad = 0
    cfgs = settings(sp)
    for cfg in cfgs:
        r = records(sp, cfg, tx)
        if [x["p"] for x in r] != sp.qs:
            raise SystemExit("%s %s: question order differs" % (label, cfg))
        miss = [x["p"] for x in r if x["text"] is None]
        if miss:
            raise SystemExit("%s %s: no raw answer for %d questions, e.g. %s" % (label, cfg, len(miss), miss[:3]))
        if states(r) != existing(sp, cfg, normtx):
            bad += 1
            print("  MISMATCH %s %s" % (label, cfg))
    print("  %-44s %3d settings, %3d questions: %s" % (label, len(cfgs), len(sp.qs), "all equal" if not bad else "%d MISMATCHES" % bad))
    return bad


def check():
    import confirm_fresh as cfr
    bad = 0
    for name in ro.FIVE:
        m = cf.MODELS[name]
        for i, split in ((0, "tune"), (1, "test")):
            sp = ps.Split(m, i)
            bad += compare(sp, Texts.of(m, i), "%s %s (%s)" % (name, split, m["judge"][i]))
        if name in cfr.FULL:
            ec_full, d_full = cfr.FULL[name]
            mf = dict(m, ec=(m["ec"][0], ec_full), d=(m["d"][0], d_full), judge=(m["judge"][0], "pro"))
            bad += compare(ps.Split(mf, 1), Texts.of(mf, 1), "%s full test (pro)" % name)
    print("check: %s" % ("PASS" if not bad else "FAIL (%d)" % bad))
    return bad


# ---------------------------------------------------------------- synthetic split
class Fake(ps.Split):
    """A Split built in memory: E layer 11, C layer 19 (order EC), strengths 1 and 2, the four C rungs."""

    def __init__(self, rows):
        self.m = dict(e=11, c=19, order="EC")
        self.st, self.blank, self.dst, self.dblank = {}, {}, {}, {}
        self.ec_text, self.d_text = {}, {}
        self.qs = sorted(rows)
        self.cks = list(LAD)
        for p, cells in rows.items():
            for cond, (text, state, src) in cells.items():
                blank = ro.norm(text) in ("i don t know", "")
                if src == "ec":
                    self.st[(p, cond)], self.blank[(p, cond)], self.ec_text[(p, cond)] = state, blank, text
                else:
                    self.dst[(p, cond)], self.dblank[(p, cond)], self.d_text[(p, cond)] = state, blank, text


def synthetic():
    """q1 vetoed (E's blank, the original 'Paris' comes back at C 0.5); q2 own blank first answered at C 1 (above a cap of 0.5);
    q3 blank under E at strength 1 only (blank on one side when the two schemes use different strengths); q4 not a
    candidate; q5 own blank at every dose; q6 E's blank released with a new answer at C 0.25."""
    IDK = "I don't know."
    e = lambda s: "E_L11_x-w-p%s" % s
    ec = lambda s, k: "X_x--E11-w-p%s-and-C19-w-m%s" % (s, k)
    c = lambda k: "C_L19_x-w-m%s" % k
    d = lambda s: "E_L11_conf-w-p%s" % s
    base = {}

    def q(no_write, e1, e2, ladder2, ladder1=None, calone=None, must=("Answer.", "R")):
        cells = {"baseline_x": no_write + ("ec",), e("1"): e1 + ("ec",), e("2"): e2 + ("ec",)}
        for k, v in zip(LAD, ladder2):
            cells[ec("2", k)] = v + ("ec",)
        for k, v in zip(LAD, ladder1 or ladder2):
            cells[ec("1", k)] = v + ("ec",)
        for k, v in zip(LAD, calone or [no_write] * 4):
            cells[c(k)] = v + ("ec",)
        for s in "12":
            cells[d(s)] = must + ("d",)
        cells["baseline_conf"] = must + ("d",)
        cells["baseline_conf_E2"] = must + ("d",)
        return cells

    X = (IDK, "X")
    base["q1"] = q(("Paris", "W"), X, X, [X, ("Paris.", "W"), ("Lyon.", "R"), ("Lyon.", "R")], must=("Lyon.", "R"))
    base["q2"] = q(X, X, X, [X, X, ("Rome.", "R"), ("Rome.", "R")], must=("Milan.", "W"))
    base["q3"] = q(("Oslo.", "R"), X, ("Oslo.", "R"), [("Oslo.", "R")] * 4, ladder1=[X, ("Bergen.", "W"), ("Bergen.", "W"), ("Bergen.", "W")],
                   must=("Oslo.", "R"))
    base["q4"] = q(("Red.", "R"), ("Red.", "R"), ("Red.", "R"), [("Red.", "R")] * 4)
    base["q5"] = q(X, X, X, [X, X, X, X], must=("Blue.", "W"))
    base["q6"] = q(("Ten.", "W"), X, X, [("Nine.", "R")] * 4, must=("Eight.", "W"))
    return Fake(base)


def selftest():
    sp = synthetic()
    tx = Texts(sp.ec_text, sp.d_text)
    normtx = {k: ro.norm(v) for k, v in tx.ec.items()}
    fails = []

    def expect(cfg, p, cond, text, state, note, rung="-"):
        r = {x["p"]: x for x in records(sp, cfg, tx)}[p]
        got = (r["cond"], r["text"], r["state"], r["note"]) + ((r["rung"],) if rung != "-" else ())
        want = (cond, text, state, note) + ((rung,) if rung != "-" else ())
        if got != want:
            fails.append("%s %s: got %s, want %s" % (cfg, p, got, want))

    n1 = ("N1", "2", "2", "2")
    expect(n1, "q1", "E_L11_x-w-p2", "I don't know.", "X", "vetoed: the original answer came back", "0-5")
    expect(n1, "q2", "X_x--E11-w-p2-and-C19-w-m1", "Rome.", "R", "released", "1")
    expect(("N1", "2", "2", "0-5"), "q2", "E_L11_x-w-p2", "I don't know.", "X", "above the cap", "1")
    expect(n1, "q3", "E_L11_x-w-p2", "Oslo.", "R", "not a candidate")
    expect(("N1", "1", "2", "2"), "q3", "X_x--E11-w-p1-and-C19-w-m0-5", "Bergen.", "W", "released", "0-5")
    expect(("EC", "1", "2"), "q3", "X_x--E11-w-p1-and-C19-w-m0-5", "Bergen.", "W", "released", "0-5")
    expect(n1, "q4", "E_L11_x-w-p2", "Red.", "R", "not a candidate")
    expect(n1, "q5", "E_L11_x-w-p2", "I don't know.", "X", "blank at every dose", None)
    expect(n1, "q6", "X_x--E11-w-p2-and-C19-w-m0-25", "Nine.", "R", "released", "0-25")
    expect(("N1", "2", None, "2"), "q6", "E_L11_x-w-p2", "I don't know.", "X", "not re-asked")
    expect(("EC", "2", "2"), "q1", "X_x--E11-w-p2-and-C19-w-m0-5", "Paris.", "W", "released", "0-5")
    expect(("B", "2"), "q1", "E_L11_conf-w-p2", "Lyon.", "R", "re-asked with the must-answer prompt")
    expect(("B", "1"), "q3", "E_L11_conf-w-p1", "Oslo.", "R", "re-asked with the must-answer prompt")
    expect(("B", "2"), "q3", "E_L11_x-w-p2", "Oslo.", "R", "not a candidate")
    expect(("BO", "2", "2"), "q1", "E_L11_x-w-p2", "I don't know.", "X", "not re-asked")
    expect(("BO", "2", "2"), "q2", "X_x--E11-w-p2-and-C19-w-m1", "Rome.", "R", "released", "1")
    expect(("T2", "2", "0-5", "0-5"), "q1", "X_x--E11-w-p2-and-C19-w-m0-5", "Paris.", "W", "released", "0-5")
    # N1 at strength 2 against B at strength 1: q3 is a candidate of B only; on it N1 hands in E2 and B the must-answer answer
    a, b = records(sp, ("N1", "2", "2", "2"), tx), records(sp, ("B", "1"), tx)
    only_b = [x["p"] for x, y in zip(a, b) if x["cand"] is None and y["cand"] is not None]
    if only_b != ["q3"]:
        fails.append("candidates of B only: %s, want ['q3']" % only_b)
    # every setting: the records give the states of the existing scorers
    for cfg in settings(sp):
        if states(records(sp, cfg, tx)) != existing(sp, cfg, normtx):
            fails.append("states differ from the existing scorer for %s" % (cfg,))
    for f in fails:
        print("  FAIL " + f)
    print("selftest: %d settings compared, %d named checks; %s" % (len(settings(sp)), 18, "PASS" if not fails else "FAIL (%d)" % len(fails)))
    return len(fails)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(1 if selftest() else 0)
    if "--check" in sys.argv:
        sys.exit(1 if check() else 0)
    print(__doc__ or "usage: final_records.py --selftest | --check")
