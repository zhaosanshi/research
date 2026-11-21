# Iteration_01 analysis (Claude Fable 5.2, 2026-11-21), read-only, no model, no judge call.
# The comparator the main table did not have: a PROMPT-ONLY system that also selects by the model's own confidence
# (nothing written into the model). TruthfulQA free-form, DeepSeek Pro, score = right - wrong.
# Inputs: N1 at each model's main-table setting (diag/main_table_layers.py, same rule and budget; its per-question
# states are taken from that program, not recomputed); the no-write answers of the run's 11 prompts with their verdicts
# (work dirs judge_<split>_promptsweep_<split>, diag/prompt_sweep_table.py); the confidence of the model in those
# answers = mean log-probability of the answer's own tokens under its own prompt (eval_extra/
# confidence_prompts_<split>_<prompt>[_b1].json of scripts/review_checks/ff_prompt_confidence.py, launcher
# explore_20261120/prompt_conf.sh). A confidence row is used only if its text hash equals the judged answer's.
#
# Rules, written here before any number of this table was read:
#   P1  same number of answers: the must-answer prompt with the honest sentence (conf_E2, the "prompt only" row of the
#       main table) keeps its N most confident answers and says nothing elsewhere, N = the number of answers N1 gives on
#       test (ties by question order). This is the strong comparator of the main table with the sentence in the prompt
#       in place of the E write. Not comparable when conf_E2 has fewer than N answers with a confidence.
#   P2  own operating point, chosen on tune only: over every prompt that has a confidence file on both splits and every
#       cut "keep the n most confident answered answers" (n = 0 .. all), the pair with the highest TUNE net. Ties: the
#       smaller n within a prompt (as diag/main_table_same_n.best_cut), then conf_E2 first, then the prompt's name.
#       On test the same prompt keeps its answers whose confidence is at least the confidence of the last answer kept
#       on tune (all answers when the tune cut keeps all, none when it keeps none). A cut never falls between two
#       answers of equal confidence, so that the tune net is the net of that threshold rule (added after the review of
#       2026-11-21; it changes nothing unless two answers of a prompt have exactly the same confidence).
# Two stops added after that review: the prompts with verdicts must be exactly the 11 named below (the main table's
# "best prompt on tune" chooses among whatever was judged), and the conf_E2 states read here must equal, question by
# question, the "prompt only" states of the main table (they come from two judge work dirs).
#   For reference only (it uses the test labels, so it is an upper bound and not a system): the best cut of P2's
#   prompt on test.
# Printed per model: the rows, N1 minus each with a paired interval; then the five models together on the questions
# all five have (per question the differences are summed, questions resampled).
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/prompt_conf_table.py [BUDGET ...]
import contextlib
import glob
import hashlib
import io
import json
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import main_table_layers as ML

DM = ML.DM
ps, ro = ML.ps, ML.ro
PROMPTS = ("x", "x_E1", "x_F", "conf", "conf_E1", "conf_E2", "conf_M", "caut", "caut_E1", "caut_E2", "caut_M")


def val(x):
    return (x == "R") - (x == "W")


def net(st):
    return sum(map(val, st))


def judged(run, split, qs):
    """{prompt: (states, answered, {question: judged answer text})} for the prompts with complete Pro verdicts."""
    w = "%s/eval_extra/judge_%s_promptsweep_%s" % (run, split, split)
    if not os.path.exists(w + "/judge2/single_pro.jsonl"):
        return {}
    st, bl = ps.load(w, "pro")
    found = {c[len("baseline_"):] for _, c in st if c.startswith("baseline_")}
    if found != set(PROMPTS):
        raise SystemExit("%s holds the prompts %s, this table chooses among %s" % (w, sorted(found), sorted(PROMPTS)))
    text = {}
    with open(w + "/judge2/cells.jsonl", encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            text[(r["pair_id"], r["condition"])] = r["prediction"]
    out = {}
    for p in PROMPTS:
        c = "baseline_" + p
        if all(st.get((q, c), "?") != "?" for q in qs):
            out[p] = ([st[(q, c)] for q in qs], [not bl[(q, c)] for q in qs], {q: text[(q, c)] for q in qs})
    return out


def confidence(run, split, prompt, qs, texts):
    """({question: mean log-prob or None}, note) of a prompt's answers, or (None, why)."""
    files = [f for f in ("%s/eval_extra/confidence_prompts_%s_%s.json" % (run, split, prompt),
                         "%s/eval_extra/confidence_prompts_%s_%s_b1.json" % (run, split, prompt)) if os.path.exists(f)]
    if not files:
        return None, "no confidence file"
    with open(files[0], encoding="utf-8") as f:
        d = json.load(f)
    t = d.get("targets", {}).get("prompt:" + prompt)
    if d.get("split") != split or d.get("partial") or t is None or t.get("writes") or t.get("base_prompt") != prompt:
        return None, "%s is not a complete no-write %s file of this prompt" % (os.path.basename(files[0]), split)
    rows = {r["pair_id"]: r for r in t["rows"]}
    bad = [q for q in qs if q not in rows or hashlib.sha1(texts[q].encode("utf-8")).hexdigest() != rows[q]["text_sha1"]]
    if bad:
        return None, "%d confidence rows do not belong to the judged answers (e.g. %s)" % (len(bad), bad[:2])
    return {q: rows[q]["mean_logp"] for q in qs}, "scoring batch %s, greedy reproduction %.4f" % (
        d.get("scoring_batch_size"), t["greedy_reproduction"]["rate"])


def ranked(answered, conf, qs):
    cand = [i for i, a in enumerate(answered) if a and conf[qs[i]] is not None]
    return sorted(cand, key=lambda i: (-conf[qs[i]], i))


def keep(states, kept):
    """(states, answered) of a system that hands in only the answers in kept."""
    kept = set(kept)
    return ["X" if i not in kept else s for i, s in enumerate(states)], [i in kept for i in range(len(states))]


def best_cut(states, order, confs=None):
    """(n, net) of the best prefix of the confidence order; ties go to fewer answers. With confs (the confidence of
    each position) a prefix may not end between two answers of equal confidence."""
    best, cur = (0, 0), 0
    for n, i in enumerate(order, 1):
        cur += val(states[i])
        if confs is not None and n < len(order) and confs[order[n]] == confs[i]:
            continue
        if cur > best[1]:
            best = (n, cur)
    return best


def model_block(model, B, tot):
    mt = {}
    with contextlib.redirect_stdout(io.StringIO()):
        ML.model_block(model, [B], mt)
    got = mt.get((B, "n1 states"), {}).get(model)
    if got is None:
        print("\n## %s: no main-table setting within budget %g" % (model, B))
        return
    qs, n1, n1_ans, layer, s, main_prompt_only = got
    Dt, tsp, ttx, _, _, _ = DM.merged_all(model, 0)
    Dx, xsp, xtx, _, _, _ = DM.merged_all(model, 1)
    if list(xsp.qs) != list(qs):
        raise SystemExit("%s: the main table's test questions differ from the run's" % model)
    run = Dx.m["ec"][1].split("/eval_extra/")[0]
    tune, test = judged(run, "tune", tsp.qs), judged(run, "test", qs)
    N = sum(n1_ans)
    print("\n## %s: N1 at layer %d, E strength %s (budget %g): answers %d, net %+d | test %d questions, tune %d" % (
        model, layer, DM.show(s), B, N, net(n1), len(qs), len(tsp.qs)))
    conf, notes = {}, []
    for p in PROMPTS:
        if p not in tune or p not in test:
            notes.append("%s: no complete verdicts" % p)
            continue
        ct, wt = confidence(run, "tune", p, tsp.qs, tune[p][2])
        cx, wx = confidence(run, "test", p, qs, test[p][2])
        if ct is None or cx is None:
            notes.append("%s: tune %s; test %s" % (p, wt, wx))
            continue
        conf[p] = (ct, cx)
        notes.append("%s: tune %s; test %s" % (p, wt, wx))
    diffs = []

    def row(label, st, ans):
        print("    %-86s answers %3d | right %3d wrong %3d | net %+4d" % (
            label, sum(ans), sum(x == "R" for x in st), sum(x == "W" for x in st), net(st)))

    if "conf_E2" in test:
        st, ans, _ = test["conf_E2"]
        differ = [q for q, x, y in zip(qs, st, main_prompt_only) if val(x) != val(y)]
        if differ:
            raise SystemExit("%s: conf_E2 as judged in the prompt-sweep work dir differs from the main table's prompt-only row on %d questions (e.g. %s)" % (
                model, len(differ), differ[:3]))
        row("prompt only, every answer (conf_E2)", st, ans)
        diffs.append(("prompt only", st))
    if "conf_E2" in conf:
        st, ans, _ = test["conf_E2"]
        order = ranked(ans, conf["conf_E2"][1], qs)
        if len(order) >= N:
            p1, a1 = keep(st, order[:N])
            row("P1 conf_E2, its %d most confident answers" % N, p1, a1)
            diffs.append(("P1 (same answer count)", p1))
        else:
            print("    P1 not comparable: conf_E2 has %d answers with a confidence, N1 gives %d" % (len(order), N))
    else:
        print("    P1 not available: conf_E2 has no usable confidence (see the notes below)")
    cands = []
    for p in conf:
        st, ans, _ = tune[p]
        order = ranked(ans, conf[p][0], tsp.qs)
        n, sc = best_cut(st, order, [conf[p][0][q] for q in tsp.qs])
        cands.append((p, n, sc, order))
    if cands:
        p, n, sc, order = sorted(cands, key=lambda c: (-c[2], c[0] != "conf_E2", c[0]))[0]
        st, ans, _ = test[p]
        xorder = ranked(ans, conf[p][1], qs)
        if n == 0:
            kept, how = [], "keeps nothing"
        elif n == len(order):
            kept, how = xorder, "keeps every answer (%d of %d on tune)" % (n, len(order))
        else:
            tau = conf[p][0][tsp.qs[order[n - 1]]]
            kept = [i for i in xorder if conf[p][1][qs[i]] >= tau]
            how = "keeps the %d most confident of %d on tune (confidence at least %.3f)" % (n, len(order), tau)
        p2, a2 = keep(st, kept)
        row("P2 chosen on tune: %s, %s; tune net %+d" % (p, how, sc), p2, a2)
        diffs.append(("P2 (prompt and cut chosen on tune)", p2))
        un, usc = best_cut(st, xorder, [conf[p][1][q] for q in qs])
        print("    %-86s answers %3d | net %+4d   (N1 minus it %+d)" % (
            "for reference, chosen with the test labels: best cut of %s on test" % p, un, usc, net(n1) - usc))
        print("    tune, best cut per prompt: " + " | ".join("%s %+d (keep %d of %d)" % (c[0], c[2], c[1], len(c[3])) for c in sorted(cands, key=lambda c: -c[2])))
    print("    N1 minus: " + " | ".join("%s %s" % (name, ps.ci([val(x) - val(y) for x, y in zip(n1, other)])) for name, other in diffs))
    print("    confidence files: " + "; ".join(notes))
    for name, other in diffs:
        t = tot.setdefault((B, name), [0, {}])
        t[0] += 1
        for q, x, y in zip(qs, n1, other):
            t[1][q] = t[1].get(q, 0) + val(x) - val(y)
    tot.setdefault((B, "questions"), []).append(set(qs))


def main():
    budgets = [float(a) for a in sys.argv[1:]] or [1.0]
    for B in budgets:
        tot = {}
        print("# budget %g: N1 against prompt-only systems that select by the model's own confidence (nothing written)" % B)
        for model in ro.FIVE:
            model_block(model, B, tot)
        sets = tot.get((B, "questions"), [])
        if len(sets) == len(ro.FIVE):
            common = sorted(set.intersection(*sets))
            print("\n## the five models together, on the %d questions all five have (per question the differences are summed; questions resampled)" % len(common))
            for (b, name), v in tot.items():
                if name == "questions":
                    continue
                if v[0] == len(ro.FIVE):
                    print("    N1 minus %-36s %s" % (name, ps.ci([v[1][q] for q in common])))
                else:
                    print("    N1 minus %-36s only %d of %d models have it" % (name, v[0], len(ro.FIVE)))
        print()


if __name__ == "__main__":
    main()
