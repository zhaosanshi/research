#!/usr/bin/env python3
# Iteration_01 exploration (Claude Opus 5.5, 2026-11-22), work queue discussion/work_queue/2026-11-22_queue_g.md,
# "1 supplement: G4, let the model arbitrate between two answers" (written 19:19, before any reading of this item).
# Offline: no model, no judge call, no generation. Nothing is chosen (no threshold, no strength, no cap).
# Rule: when one question has two different answers, both are turned into self-judge statements (selfjudge_probe.py,
# the E45 code path) and the one with the higher margin = logp(True) - logp(False) is taken; a tie keeps E's answer.
# "Different" = different statements (selfjudge_probe.statement); "substantive" = answered (the answer counts as
# answered under the existing scorer of that pass).
#   G4a  questions E answered: the original model's own answer (plain prompt, no write; sources tq_orig / qq_orig) vs
#        E's answer, where both are answered and differ; every other question keeps E's answer (same answer count).
#   G4b  questions blank under E that N1 re-asked: C's released answer vs the must-answer + E answer (GPT round four,
#        discussion/validation_dataset/2026-11-22d_candidate_pair.py, group "C", cutoff None, redone as it is: pairs =
#        questions where both are answered; equal answer text -> one common verdict; a switch to the must-answer + E
#        answer only when the statements differ and its margin is strictly higher; a tie keeps C's answer).
#   G4c  G4a, then rule Z on the chosen answer: margin < 0 -> no answer (negated-premise openings, NEG_RE, not judged).
# Conditions: no write ("baseline") and true/false E at strength 3 ("E_a3", models with a true/false E only); the pick,
# and in G4c rule Z, use the margins of the same condition.
# Readings (TruthfulQA test split; (QA)^2 held-out 470, all 570 alongside; per model and models together):
#   1  pairs with exactly one right and one wrong answer: how many each condition picks right; "always E" alongside.
#   2  G4a - single E (same answers handed in); G4a written - G4a no write; G4c - (single E + rule Z) (= S1 of
#      selfjudge_stage1.computeZ, threshold 0, same condition).
#   3  N1 with its first pass replaced by G4a: minus strong single E at N1's answer count, minus two-pass single E whose
#      first pass is G4a with the NO-WRITE self-judge (the version that needs no direction); the plain references
#      N1 - strong and N1 - two-pass single E alongside.
#   4  G4b: GPT's table (direct C / no-write pick / written pick, written - no write, written - direct C, the
#      one-right-one-wrong pairs), first exactly as GPT computed it (two models, the common 523 / 470 questions, its
#      bootstrap: random.Random(20261116), 4000 draws, order statistics 100 and 3899); must reproduce 40 / 30 / 43,
#      +13 [+3, +25] and 3 / -1 / 7, +8 [+2, +16]; then per model on each model's own list and (QA)^2 570, and N1 with
#      the picked answer in place of C's (minus N1, minus strong single E at N1's answer count).
# Inputs: selfjudge_stage1.load_all (every source checked there against the native files and scorers), the probes of
# selfjudge_probe.read_probe (score files bound to texts, statements and metadata), and tq_orig / qq_orig items.
# Intervals elsewhere: TruthfulQA policy_select2.ci (seed 20261116), (QA)^2 newsets_score.joint_ci (seed 20261119),
# models together on the sorted common TruthfulQA questions / the (QA)^2 bank order, as selfjudge_stage1.tableZ.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/selfjudge_arbitrate.py MODEL [MODEL ...]
import os
import random
import sys

import numpy as np

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets"]
import selfjudge_probe as SP     # noqa: E402
import selfjudge_stage1 as ST    # noqa: E402

GPT_EXPECT = {"TQA": ([40, 30, 43], "+13 [+3,+25]", [33, 20, 26]), "QA2": ([3, -1, 7], "+8 [+2,+16]", [16, 7, 11])}


def probe_rows(model, source, key):
    """{key(item): (answer_sha1, statement_sha1, {cond: margin}, neg, value, answered)} for the answered items."""
    doc, _, z = SP.read_probe(model, source=source) if source else SP.read_probe(model)
    conds = SP.conds_of(model, source)
    ans = [x for x in doc["items"] if x["answered"]]
    out = {}
    for j, x in enumerate(ans):
        if z["ids"][j] != x["id"] or z["answer_sha1"][j] != x["answer_sha1"]:
            raise ValueError(f"{model} {source}: score row {j} is not item {x['id']}")
        out[key(x)] = (x["answer_sha1"], str(z["statement_sha1"][j]), {c: float(z[c + "__margin"][j]) for c in conds},
                       x["neg"], x.get("value", SP.V.get(x.get("state"))), True)
    return doc, out


def load(models):
    A = ST.load_all(models)
    D = {}
    for m in models:
        conds = SP.conds_of(m, "source")
        ids, tq, info = A[m]["tq"]
        qids, qq = A[m]["qq"]
        # statements of every scored answer
        _, sE = probe_rows(m, "", lambda x: (x["split"], x["id"]))
        _, sO = probe_rows(m, "tq_orig", lambda x: x["id"])
        _, sR = probe_rows(m, "tq_release", lambda x: x["id"])
        _, sF = probe_rows(m, "tq_forced", lambda x: x["id"])
        odoc = SP.read_json(os.path.join(SP.OUT, m, "tq_orig_items.json"))
        oit = {x["id"]: x for x in odoc["items"]}
        rit = {x["id"]: x for x in SP.read_json(os.path.join(SP.OUT, m, "tq_release_items.json"))["items"]}
        fit = {x["id"]: x for x in SP.read_json(os.path.join(SP.OUT, m, "tq_forced_items.json"))["items"]}
        T = {}
        for q in ids:
            o = oit["test|" + q]
            r = {c: tq[c][q] for c in conds}
            b = r["baseline"]
            if b["eans"] and (("test", q) not in sE or sE[("test", q)][3] != b["negE"]):
                raise ValueError(f"{m}: E statement of {q}")
            if any(abs(sE[("test", q)][2][c] - r[c]["mE"]) > 0 for c in conds if b["eans"]):
                raise ValueError(f"{m}: E margin of {q} differs between loaders")
            T[q] = dict(e=b["e"], eans=b["eans"], n1=b["n1"], issued=b["issued"], f=b["f"], fans=b["fans"], conf=b["conf"],
                        negE=b["negE"], mE={c: r[c]["mE"] for c in conds}, stE=sE[("test", q)][1] if b["eans"] else None,
                        o=o["value"], oans=o["answered"], stO=sO["test|" + q][1] if o["answered"] else None,
                        mO=sO["test|" + q][2] if o["answered"] else None, negO=o["neg"],
                        rel=None if b["rel"] is None else dict(v=b["rel"][0], neg=b["rel"][2], m={c: r[c]["rel"][1] for c in conds},
                                                               st=sR["test|" + q][1], sha=rit["test|" + q]["answer_sha1"]),
                        F=None if not b["fans"] else dict(v=b["f"], neg=b["negF"], m={c: r[c]["mF"] for c in conds},
                                                          st=sF["test|" + q][1], sha=fit["test|" + q]["answer_sha1"]))
            if o["answered"] and sO["test|" + q][4] != o["value"]:
                raise ValueError(f"{m}: original value of {q}")
        # (QA)^2
        _, qE = probe_rows(m, "qaqa", lambda x: x["id"])
        _, qO = probe_rows(m, "qq_orig", lambda x: x["id"])
        _, qR = probe_rows(m, "qq_release", lambda x: x["id"])
        _, qF = probe_rows(m, "qq_forced", lambda x: x["id"])
        qoit = {x["id"]: x for x in SP.read_json(os.path.join(SP.OUT, m, "qq_orig_items.json"))["items"]}
        qrit = {x["id"]: x for x in SP.read_json(os.path.join(SP.OUT, m, "qq_release_items.json"))["items"]}
        qfit = {x["id"]: x for x in SP.read_json(os.path.join(SP.OUT, m, "qq_forced_items.json"))["items"]}
        if list(qoit) != list(qids):
            raise ValueError(f"{m}: (QA)^2 original roster differs")
        Q = {}
        for q in qids:
            o = qoit[q]
            r = {c: qq[c][q] for c in conds}
            b = r["baseline"]
            if b["eans"] and any(abs(qE[q][2][c] - r[c]["mE"]) > 0 for c in conds):
                raise ValueError(f"{m}: (QA)^2 E margin of {q} differs between loaders")
            Q[q] = dict(e=b["e"], eans=b["eans"], n1=b["n1"], issued=b["issued"], f=b["f"], fans=b["fans"], conf=b["conf"],
                        negE=b["negE"], mE={c: r[c]["mE"] for c in conds}, stE=qE[q][1] if b["eans"] else None,
                        o=o["value"], oans=o["answered"], stO=qO[q][1] if o["answered"] else None,
                        mO=qO[q][2] if o["answered"] else None, negO=o["neg"], pilot=b["pilot"],
                        rel=None if b["rel"] is None else dict(v=b["rel"][0], neg=b["rel"][2], m={c: r[c]["rel"][1] for c in conds},
                                                               st=qR[q][1], sha=qrit[q]["answer_sha1"]),
                        F=None if not b["fans"] else dict(v=b["f"], neg=b["negF"], m={c: r[c]["mF"] for c in conds},
                                                          st=qF[q][1], sha=qfit[q]["answer_sha1"]))
        D[m] = dict(conds=conds, tq=(ids, T, info), qq=(qids, Q), info=odoc["splits"])
    return D


# ------------------------------------------------------------------------------------------------------------------ #
def g4(ids, d, conds):
    """Per-question values of every system; summary counts."""
    pos = {q: i for i, q in enumerate(ids)}
    ranked = sorted([q for q in ids if d[q]["fans"] and d[q]["conf"] is not None], key=lambda q: (-d[q]["conf"], pos[q]))
    nN1 = sum(d[q]["issued"] for q in ids)
    nrel = sum(d[q]["rel"] is not None for q in ids)
    if nN1 > len(ranked):
        raise SystemExit("strong single E cannot reach N1's answer count")
    top = set(ranked[:nN1])
    eb = [q for q in ranked if not d[q]["eans"]]
    if len(eb) < nrel:
        raise SystemExit("two-pass single E cannot fill N1's releases")
    fill = set(eb[:nrel])
    V = {"E": {q: d[q]["e"] for q in ids}, "N1": {q: d[q]["n1"] for q in ids},
         "B(N1)": {q: d[q]["f"] if q in top else 0 for q in ids},
         "TP": {q: d[q]["f"] if q in fill else d[q]["e"] for q in ids}}
    S = {}
    for c in conds:
        g, gc, s1, sw, ngc, ns1 = {}, {}, {}, 0, 0, 0
        disc = dict(n=0, pick=0, alwaysE=0)
        for q in ids:
            x = d[q]
            g[q], gc[q], s1[q] = x["e"], x["e"], x["e"]
            if not x["eans"]:
                continue
            take_o = bool(x["oans"] and x["stO"] != x["stE"] and x["mO"][c] > x["mE"][c])
            pair = bool(x["oans"] and x["stO"] != x["stE"])
            v, m_, neg = (x["o"], x["mO"][c], x["negO"]) if take_o else (x["e"], x["mE"][c], x["negE"])
            g[q] = v
            sw += take_o
            gc[q] = 0 if (not neg and m_ < 0) else v
            ngc += not (not neg and m_ < 0)
            s1[q] = 0 if (not x["negE"] and x["mE"][c] < 0) else x["e"]
            ns1 += not (not x["negE"] and x["mE"][c] < 0)
            if pair and x["o"] * x["e"] == -1:
                disc["n"] += 1
                disc["pick"] += v == 1
                disc["alwaysE"] += x["e"] == 1
        V["G4a_" + c], V["G4c_" + c], V["S1_" + c] = g, gc, s1
        V["N1G4a_" + c] = {q: g[q] if d[q]["eans"] else d[q]["n1"] for q in ids}
        S[c] = dict(switch=sw, disc=disc, pairs=sum(1 for q in ids if d[q]["eans"] and d[q]["oans"] and d[q]["stO"] != d[q]["stE"]),
                    nG4c=ngc, nS1=ns1)
    V["TPG4a0"] = {q: d[q]["f"] if q in fill else V["G4a_baseline"][q] for q in ids}
    # G4b on this list
    for c in conds:
        V["N1G4b_" + c] = {}
    gb = dict(pairs=0, usable=0, disc=0, pick={c: 0 for c in conds}, inc=0, val={c: 0 for c in conds})
    for q in ids:
        x = d[q]
        for c in conds:
            V["N1G4b_" + c][q] = x["n1"]
        if x["rel"] is None or x["F"] is None:
            continue
        a, f = x["rel"]["v"], x["F"]["v"]
        if x["rel"]["sha"] == x["F"]["sha"]:
            f = a
        usable = x["rel"]["st"] != x["F"]["st"]
        gb["pairs"] += 1
        gb["usable"] += usable
        gb["inc"] += a
        disc = a * f == -1 and usable
        gb["disc"] += disc
        for c in conds:
            take = usable and (x["F"]["m"][c] - x["rel"]["m"][c]) > 0
            v = f if take else a
            V["N1G4b_" + c][q] = v
            gb["val"][c] += v
            gb["pick"][c] += disc and v == 1
    nets = {k: sum(v.values()) for k, v in V.items()}
    return V, S, gb, nets


def diffs(conds):
    out = [("G4a-E[%s]" % c, "G4a_" + c, "E") for c in conds]
    if "E_a3" in conds:
        out.append(("G4a written-no write", "G4a_E_a3", "G4a_baseline"))
    out += [("G4c-S1[%s]" % c, "G4c_" + c, "S1_" + c) for c in conds]
    out += [("ref N1-B(N1)", "N1", "B(N1)"), ("ref N1-TP", "N1", "TP")]
    out += [("N1G4a-B(N1)[%s]" % c, "N1G4a_" + c, "B(N1)") for c in conds]
    out += [("N1G4a-TPG4a0[%s]" % c, "N1G4a_" + c, "TPG4a0") for c in conds]
    out += [("G4b N1G4b-N1[%s]" % c, "N1G4b_" + c, "N1") for c in conds]
    if "E_a3" in conds:
        out.append(("G4b written-no write", "N1G4b_E_a3", "N1G4b_baseline"))
    out += [("G4b N1G4b-B(N1)[%s]" % c, "N1G4b_" + c, "B(N1)") for c in conds]
    return out


def report(label, ids, d, conds, ci):
    V, S, gb, nets = g4(ids, d, conds)
    print("  %s: %d questions | answers: E %d, N1 %d | nets: E %+d, N1 %+d, strong at N1's count %+d, two-pass single E %+d" % (
        label, len(ids), sum(d[q]["eans"] for q in ids), sum(d[q]["issued"] for q in ids), nets["E"], nets["N1"],
        nets["B(N1)"], nets["TP"]))
    for c in conds:
        s = S[c]
        print("    [%s] G4a: pairs (both answered, statements differ) %d, switched to the original %d | one right one wrong %d: "
              "picked right %d, always-E right %d | nets G4a %+d, G4c %+d (answers %d), S1 %+d (answers %d), N1+G4a %+d" % (
                  c, s["pairs"], s["switch"], s["disc"]["n"], s["disc"]["pick"], s["disc"]["alwaysE"], nets["G4a_" + c],
                  nets["G4c_" + c], s["nG4c"], nets["S1_" + c], s["nS1"], nets["N1G4a_" + c]))
    print("    G4b: pairs %d (statements differ %d), one right one wrong %d: picked right %s | direct C %+d, %s" % (
        gb["pairs"], gb["usable"], gb["disc"], ", ".join("%s %d" % (c, gb["pick"][c]) for c in conds), gb["inc"],
        ", ".join("%s pick %+d" % (c, gb["val"][c]) for c in conds)))
    line = []
    for key, a, b in diffs(conds):
        line.append("%s %s" % (key, ci([V[a][q] - V[b][q] for q in ids])))
    print("    " + " | ".join(line))
    return V


# ------------------------------------------------------------------------------------------------------------------ #
def gpt_replay(D, models):
    """GPT round four, group C, cutoff None, two models, common lists, its bootstrap."""
    def interval(col):
        n = len(col)
        rng = random.Random(20261116)
        dr = np.array([np.bincount([rng.randrange(n) for _ in range(n)], minlength=n) for _ in range(4000)])
        dist = np.sort(dr @ np.asarray(col))
        return "%+d [%+d,%+d]" % (sum(col), dist[100], dist[3899])

    ok = True
    for task in ("TQA", "QA2"):
        if task == "TQA":
            ids = sorted(set.intersection(*(set(D[m]["tq"][0]) for m in models)))
            per = {m: D[m]["tq"][1] for m in models}
        else:
            ids = sorted(q for q in D[models[0]]["qq"][0] if not D[models[0]]["qq"][1][q]["pilot"])
            per = {m: D[m]["qq"][1] for m in models}
        tab = np.zeros((len(ids), 6), dtype=np.int64)
        pairs = 0
        for m in models:
            for i, q in enumerate(ids):
                x = per[m][q]
                if x["rel"] is None or x["F"] is None:
                    continue
                pairs += 1
                a, f = x["rel"]["v"], x["F"]["v"]
                if x["rel"]["sha"] == x["F"]["sha"]:
                    f = a
                usable = x["rel"]["st"] != x["F"]["st"]
                m0 = x["F"]["m"]["baseline"] - x["rel"]["m"]["baseline"]
                m3 = x["F"]["m"]["E_a3"] - x["rel"]["m"]["E_a3"]
                c0 = f if (usable and m0 > 0) else a
                c3 = f if (usable and m3 > 0) else a
                disc = a * f == -1 and usable
                tab[i] += (a, c0, c3, int(disc), int(disc and c0 == 1), int(disc and c3 == 1))
        got = (tab[:, :3].sum(0).tolist(), interval(tab[:, 2] - tab[:, 1]), tab[:, 3:].sum(0).tolist())
        print("# GPT replay %s (%d questions, %d model x question pairs): direct C / no-write pick / written pick %s | written - no write %s | "
              "written - direct C %s | one right one wrong / no-write picks right / written picks right %s (GPT: %s, %s, %s)" % (
                  task, len(ids), pairs, got[0], got[1], interval(tab[:, 2] - tab[:, 0]), got[2], *GPT_EXPECT[task]))
        ok = ok and got == GPT_EXPECT[task]
    return ok


def main():
    import tq_same_n_decomp as TQ
    import newsets_score as NS
    ci_tq = TQ.ps.ci

    def ci_qq(v):
        tot, lo, hi = NS.joint_ci([list(v)])
        return "%+d [%+d, %+d]" % (tot, lo, hi)

    models = tuple(sys.argv[1:])
    if not models or any(m not in SP.MODELS for m in models):
        raise SystemExit("usage: selfjudge_arbitrate.py MODEL [MODEL ...]")
    D = load(models)
    print("# selfjudge G4 (work queue 2026-11-22_queue_g, '1 supplement', 19:19): models %s; conditions per model %s" % (
        ", ".join(models), {m: D[m]["conds"] for m in models}))
    for m in models:
        print("# %s original-answer source: %s" % (m, {k: {kk: v[kk] for kk in ("n", "answered")} for k, v in D[m]["info"].items()}))
    if set(models) == {"llama3.1-8b", "qwen3.5-9b"}:
        if not gpt_replay(D, ("llama3.1-8b", "qwen3.5-9b")):
            raise SystemExit("GPT REPLAY MISMATCH: stopping before the G4 readings")
        print("# GPT round-four G4b numbers reproduced")
    Vs = {}
    for m in models:
        conds = D[m]["conds"]
        print("=" * 100)
        ids, T, (layer, s, n1cfg) = D[m]["tq"]
        print("%s | E L%d strength %s, N1 %s" % (m, layer, s, n1cfg))
        Vs[(m, "tq")] = report("TruthfulQA test", ids, T, conds, ci_tq)
        qids, Q = D[m]["qq"]
        held = [q for q in qids if not Q[q]["pilot"]]
        Vs[(m, "q470")] = report("(QA)^2 470 held out", held, Q, conds, ci_qq)
        Vs[(m, "q570")] = report("(QA)^2 570", qids, Q, conds, ci_qq)
    if len(models) < 2:
        return
    conds = tuple(c for c in D[models[0]]["conds"] if all(c in D[m]["conds"] for m in models))
    print("=" * 100)
    print("%d models together (TruthfulQA: sorted common questions, every quantity recomputed there; (QA)^2: bank order)" % len(models))
    common = sorted(set.intersection(*(set(D[m]["tq"][0]) for m in models)))
    for label, sets in (("TruthfulQA test, %d common questions" % len(common), {m: (common, D[m]["tq"][1]) for m in models}),
                        ("(QA)^2 470 held out", {m: ([q for q in D[m]["qq"][0] if not D[m]["qq"][1][q]["pilot"]], D[m]["qq"][1]) for m in models}),
                        ("(QA)^2 570", {m: (D[m]["qq"][0], D[m]["qq"][1]) for m in models})):
        res = {m: g4(sets[m][0], sets[m][1], conds) for m in models}
        ids = sets[models[0]][0]
        if any(sets[m][0] != ids for m in models):
            raise SystemExit("question lists differ")
        disc = {c: [sum(res[m][1][c]["disc"][k] for m in models) for k in ("n", "pick", "alwaysE")] for c in conds}
        gbp = {c: sum(res[m][2]["pick"][c] for m in models) for c in conds}
        print("  %s: G4a one right one wrong %s | G4b one right one wrong %d, picked right %s" % (
            label, " ; ".join("[%s] %d pairs, picked right %d, always-E right %d" % (c, *disc[c]) for c in conds),
            sum(res[m][2]["disc"] for m in models), ", ".join("%s %d" % (c, gbp[c]) for c in conds)))
        line = []
        for key, a, b in diffs(conds):
            if label.startswith("Truthful"):
                line.append("%s %s" % (key, ci_tq([sum(res[m][0][a][q] - res[m][0][b][q] for m in models) for q in ids])))
            else:
                tot, lo, hi = NS.joint_ci([[res[m][0][a][q] - res[m][0][b][q] for q in ids] for m in models])
                line.append("%s %+d [%+d, %+d]" % (key, tot, lo, hi))
        print("    " + " | ".join(line))


if __name__ == "__main__":
    main()
