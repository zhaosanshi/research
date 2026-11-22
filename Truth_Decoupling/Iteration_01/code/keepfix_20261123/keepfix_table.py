# -*- coding: utf-8 -*-
"""Queue K (Claude Opus 5.5, 2026-11-23; discussion/work_queue/2026-11-23_queue_k.md): checks and the tune-split
readings of the E direction trained with the keep term (E_k), Llama-3.1-8B. Read-only, no model, no judge call.
Exploration.

Systems (P = x_E2, the plain prompt + the honest sentence; every write at "prompt"; E at layer 16, C at layer 19):
  S0 = P, no write; SE(E, s) = P + E at strength s; SN1(E, s) = N1 on P (each question SE's answer; blanks re-asked with
  P + E + C at 0.25 / 0.5 / 1 / 2 up to two caps; a released answer of a question S0 answers must differ from S0's);
  only E = the plain prompt x + E (no sentence); no write = x, no write.
  E = the main table's E (results_e44_factorial_das/llama3.1-8b_side_E2) at 3.5, or E_k at 3.5 (nominal) and at s'
  (aligned: s' * gap_k = the main E's amount at 3.5; the main reading).
Modes (scoring container, from /workspace/Truth_Decoupling):
  check-tq   --run R --main MAINRUN --arms ARM...   the TruthfulQA tune files of E_k: base P answers = the main run's;
             every E_k part: direction file / run of R, amount = beta * gap_k, positions prompt (CPU, python -I is fine)
  check-fact --run R [--gen-name G] --arms ARM...   the same for the PopQA / TriviaQA tune files of E_k
             (newsets/gen_prompt/G/<task>/<model>/tune, G = keepfix by default)
  train      the training records (main E, lambda 0 and 1), the lambda 0 comparison, the unit test
  lam TAG... queue K section 5 (TAG in l0p03 l0p1 l0p3, the usable lambdas): the training records of the three lambdas,
             then for the usable ones TruthfulQA tune (192, DeepSeek Pro: SE / SN1, caps n1_pick on tune; only E), (QA)^2 100
             tune (pilot) questions (DeepSeek Pro, caps over every pair on the 100, ties to the earlier pair; by assumption
             type), PopQA / TriviaQA tune (192 each, alias match; right -> wrong answers and those opening with a negation,
             diag/selfjudge_probe.NEG_RE), each at the aligned strength (@a, the main reading) and at nominal 3.5 (@3.5),
             next to the main E at 3.5 and at half strength 1.75 and P alone (S0); then the section 5 sum and the choice.
  Every reading also splits the questions the plain model answers right without a write into "right under P too" and
  "right under x only" (P = x_E2 alone): per system right / wrong / neither in each class (GPT review 2026-11-23, 2.3).
Intervals: TruthfulQA policy_select2.ci, the other sets newsets_score.joint_ci (paired per question).
"""
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = "/workspace/Truth_Decoupling"
MODEL = "llama3.1-8b"
DISPLAY = "Llama-3.1-8B"
EL, CL, S_MAIN = 16, 19, "3.5"
MAINR = "results_e44_factorial_das/llama3.1-8b_side_E2"
R1 = "Iteration_01/results/keepfix/llama3.1-8b/e44/llama3.1-8b_keepfix"
R0 = "Iteration_01/results/keepfix/llama3.1-8b/e44/llama3.1-8b_keepfix_l0"
NSK = "Iteration_01/newsets/keepfix"
NSD = "Iteration_01/newsets"
LADK = ("0.25", "0.5", "1", "2")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tok(arm):
    return arm.replace(":", "--").replace("+", "-and-").replace(".", "-")


def dash(s):
    return s.replace(".", "-")


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def aligned():
    return Path(NSK, "aligned_s_%s.txt" % MODEL).read_text().strip()


# ----------------------------------------------------------------------------- checks (CPU)
def ek_parts_ok(parts, run, gap, dsha, what):
    bad = []
    e = [p for p in parts if p["family"] == "E"]
    if len(e) != 1:
        return ["%s: %d E parts" % (what, len(e))]
    p = e[0]
    if p["layer"] != EL or p["kind"] != "w" or p.get("direction_run") != Path(run).name or p.get("direction_file") != "E_L%d.pt" % EL:
        bad.append("%s: E part layer / kind / run" % what)
    if p.get("direction_file_sha256") != dsha:
        bad.append("%s: E part direction file" % what)
    if not math.isclose(float(p["amount"]), float(p["beta"]) * gap, rel_tol=1e-9):
        bad.append("%s: E amount %r is not %r * gap_k" % (what, p["amount"], p["beta"]))
    return bad


def ek_of(run):
    lg = jload(Path(run) / "train_logs" / ("E_L%d.json" % EL))
    dsha = sha(Path(run) / "dirs" / ("E_L%d.pt" % EL))
    if lg.get("direction_file_sha256") != dsha or not lg.get("keep"):
        raise SystemExit("%s: the train log does not describe a keep-term direction file" % run)
    return float(lg["gap"]), dsha


def check_tq(argv):
    run, main, arms = argv[argv.index("--run") + 1], argv[argv.index("--main") + 1], argv[argv.index("--arms") + 1:]
    gap, dsha = ek_of(run)
    bad = []
    b1, b0 = jload(Path(run) / "eval_extra" / "base_x_E2_tune.json"), jload(Path(main) / "eval_extra" / "base_x_E2_tune.json")
    keys = ("pair_id", "text", "token_ids", "finished", "stop_id")
    same = sum(all(x[k] == y[k] for k in keys) for x, y in zip(b1["rows"], b0["rows"]))
    if b1["system"] != b0["system"] or len(b1["rows"]) != len(b0["rows"]) or same != len(b0["rows"]):
        bad.append("P answers differ from the main run's (%d of %d identical)" % (same, len(b0["rows"])))
    for a in arms:
        g = jload(Path(run) / "eval_extra" / "gen" / ("%s-at-prompt_tune.json" % tok(a)))
        if g.get("positions") != "prompt" or g.get("arm") != a or g.get("completed") is not True:
            bad.append("%s: positions / arm / completed" % a)
        if g["base_prompt"] == "x" and g.get("baseline_reproduction_verified") is not True:
            bad.append("%s: the no-write x answers were not reproduced" % a)
        bad += ek_parts_ok(g["parts"], run, gap, dsha, a)
    print("check-tq: P answers identical %d / %d; %d arms; E_k gap %.6f, direction %s; %s" % (
        same, len(b0["rows"]), len(arms), gap, dsha[:16], "PASS" if not bad else "FAIL %s" % bad), flush=True)
    if bad:
        raise SystemExit(1)


def check_fact(argv):
    run, arms = argv[argv.index("--run") + 1], argv[argv.index("--arms") + 1:]
    gname = argv[argv.index("--gen-name") + 1] if "--gen-name" in argv else "keepfix"
    if "--gen-name" in argv and argv.index("--gen-name") > argv.index("--arms"):
        raise SystemExit("give --gen-name before --arms")
    gap, dsha = ek_of(run)
    bad, n = [], 0
    for task in ("popqa", "triviaqa"):
        for a in arms:
            g = jload(Path(NSD, "gen_prompt", gname, task, MODEL, "tune", tok(a) + ".json"))
            c = g["contract"]
            if c.get("positions") != "prompt" or c.get("arm") != a or g.get("completed") is not True or c.get("full_pass") is not True:
                bad.append("%s %s: positions / arm / completed / full pass" % (task, a))
            bad += ek_parts_ok(c["parts"], run, gap, dsha, "%s %s" % (task, a))
            n += 1
    print("check-fact: %d files; E_k gap %.6f, direction %s; %s" % (n, gap, dsha[:16], "PASS" if not bad else "FAIL %s" % bad), flush=True)
    if bad:
        raise SystemExit(1)


# ----------------------------------------------------------------------------- readings
def imports():
    sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks"]
    import n1_on_sentence as N
    import n1_on_sentence3 as N3
    import qaqa_pilot_table as Q
    import ns_prompt_table as NP
    import selfjudge_probe as SPB
    return N, N3, Q, NP, SPB


def neg(SPB, t):
    return bool(SPB.NEG_RE.match(t or ""))


def show_train():
    print("\n# training records (Llama-3.1-8B, E at layer %d)" % EL)
    for name, run in (("main E", MAINR), ("lambda 0 (check)", R0), ("lambda 1 = E_k", R1)):
        p = Path(run) / "train_logs" / ("E_L%d.json" % EL)
        if not p.exists():
            print("    %-18s no record" % name)
            continue
        lg = jload(p)
        tr, va = lg["losses"]["train"], lg["losses"].get("val") or {}
        k = lg.get("keep") or {}
        print("    %-18s gap %.6f | md norm %.4f | cos(w, md) %.4f | cause loss trained %.4f (no write %.4f) | isolate %.5f | held-out cause %.4f isolate %.5f | %.0f s" % (
            name, lg["gap"], lg["md_norm"], lg["cosine_w_md"], tr["trained"]["cause"], tr["no_write"]["cause"], tr["trained"]["isolate"],
            va.get("trained", {}).get("cause", float("nan")), va.get("trained", {}).get("isolate", float("nan")), lg["seconds"]))
        if k:
            dg = k.get("diagnostics") or {}
            print("        keep: lambda %g, A %.6f, keep questions %s %s; keep loss (mean over answer tokens, by class) with the saved direction at +A %s, at -A %s, no write %s" % (
                k["lambda"], k["amount"], k["keep_set"]["n_used"], k["keep_set"].get("n_used_by_class"),
                dg.get("keep_loss_trained_at_A"), dg.get("keep_loss_opposite_side"), dg.get("keep_loss_no_write")))
            print("        sign check %s | draws %s | memory %s | cap %s" % (k.get("sign_check"), lg["training"].get("keep_draws_sha256", "")[:16],
                                                                         k.get("memory"), k.get("memory_cap")))
            print("        per epoch: " + " | ".join("%d: cause %.4f keep %.4f (negative-sign steps %d)" % (
                x["epoch"], x["train_cause"], x["keep_train_mean"], x["keep_sign_negative_steps"]) for x in lg["training"]["history"]))
    for tag in ("l0", "l1"):
        p = Path(NSK, "compare_%s_%s.json" % (tag, MODEL))
        if p.exists():
            d = jload(p)
            print("    compare %s with the main E: elementwise equal %s | cosine %.8f | max abs diff %.3g | gap %s | init equal %s | contract equal without keep %s" % (
                tag, d["elementwise_equal"], d["cosine"], d["max_abs_diff"], d["gap"], d["init_sha256_equal"], d["contract_equal_without_keep"]))
    for f, label in (("signcheck_%s.json" % MODEL, "sign check"), ("aligned_s_%s.txt" % MODEL, "aligned strength s'")):
        p = Path(NSK, f)
        if p.exists():
            print("    %s: %s" % (label, p.read_text().strip()))
    p = Path(NSK, "unittest_%s.json" % MODEL)
    if p.exists():
        u = jload(p)
        print("    unit test %s on %s" % ("PASS" if u["passed"] else "FAIL", u.get("four")))
        for r in u["rows"]:
            print("        %s: train code at A %.6f, hand %.6f (diff %.2e) | at 0 %.6f, no hook %.6f (diff %.2e)" % (
                r["key"], r["train_code_at_A"], r["hand_at_A"], r["diff_at_A"], r["train_code_at_0"], r["hand_no_hook"], r["diff_at_0"]))
        gch = u.get("gradient_check") or {}
        print("        keep loss alone: loss %.5f, gradient norm %.4g (tangent %.4g, radial %.4g, finite %s); steps %s" % (
            gch.get("loss", float("nan")), gch.get("grad_norm", float("nan")), gch.get("grad_tangent_norm", float("nan")),
            gch.get("grad_radial", float("nan")), gch.get("grad_finite"),
            " ; ".join("t %.2f: down %.5f up %.5f" % (x["t"], x["down"], x["up"]) for x in gch.get("steps", []))))
        for x in gch.get("positions", []):
            print("        positions %s: written %s, changed = prompt rule %s, rest bit-identical %s, added along u %.4f..%.4f, off-axis %.2e" % (
                x["key"], x["expected_first_last"], x["changed_equals_prompt_rule"], x["rest_bit_identical"],
                x["added_along_u_min"], x["added_along_u_max"], x["off_axis_rel_max"]))
        print("        batch of 4 %.6f | all keep questions: main E at A %s, no write %s | step %s | memory %s | cap %s" % (
            u["batch_of_4"], u["all_keep_questions"]["main_E_at_A"], u["all_keep_questions"]["no_write"],
            {k: v for k, v in u["step"].items() if k != "error"}, u["memory"], u.get("memory_cap")))
    p = Path(NSK, "keep_ids_%s.json" % MODEL)
    if p.exists():
        d = jload(p)
        print("    keep questions: %d (%s)" % (d["n_keep"], " | ".join(
            "%s train %d: x right %d / wrong %d / blank %d, x_E2 right %d / wrong %d / blank %d; kept %d = right under both %d + x only %d" % (
                t, v["n_train"], v["right"], v["wrong"], v["blank"], v["x_E2_right"], v["x_E2_wrong"], v["x_E2_blank"], len(v["ids"]),
                v["keep_both"], v["keep_x_only"]) for t, v in d["tasks"].items())))


def line(names, sysd, qs, diffs, ci, title, extra=None):
    print("\n## %s (%d questions)" % (title, len(qs)))
    print("    " + " | ".join("%s %+d (%d)" % (n, sum(sysd[n][q][0] for q in qs), sum(sysd[n][q][1] for q in qs)) for n in names))
    print("    " + " | ".join("%s - %s %s" % (a, b, ci([sysd[a][q][0] - sysd[b][q][0] for q in qs])) for a, b in diffs))
    if extra:
        print("    " + extra)


def classes(sysd, qs, names, title):
    """The questions the plain model answers right without a write, split by P (x_E2, no write): right under both /
    right under x only; per system the answers right, wrong and neither (blank or no information) in each class."""
    both = [q for q in qs if sysd["no write"][q][0] == 1 and sysd["S0"][q][0] == 1]
    xonly = [q for q in qs if sysd["no write"][q][0] == 1 and sysd["S0"][q][0] != 1]
    print("    %s, right without a write: %d = right under P too %d + right under x only %d" % (title, len(both) + len(xonly), len(both), len(xonly)))
    for label, cq in (("right under both", both), ("right under x only", xonly)):
        print("      %s (%d): " % (label, len(cq)) + " | ".join("%s right %d wrong %d neither %d" % (
            n, sum(sysd[n][q][0] == 1 for q in cq), sum(sysd[n][q][0] == -1 for q in cq), sum(sysd[n][q][0] == 0 for q in cq)) for n in names))


def tq_k_build(N, s_dash, qs, kdir):
    """N.build on the prefix x_E2 with this queue's work dir in place of the n1s dir (same renaming, same checks)."""
    MP, ps, S, fr = N.MP, N.ps, N.S, N.MP.fr
    display, layer, _ = N.A.KEY[MODEL]
    D = MP.MT.Data(display, 0)
    run = D.m["ec"][0].split("/eval_extra/")[0]
    cl = D.m["c"]
    st, bl, tx = {}, {}, {}

    def put(k, v, b, t):
        if k in st and st[k] != v:
            raise SystemExit("condition %s %s appears twice with different verdicts" % (k[0], k[1]))
        st[k], bl[k], tx[k] = v, b, t

    def rename(c):
        if c == "baseline_x_E2":
            return "baseline_x"
        if c.startswith("X_x_E2--") and c.endswith(N.SUF):
            return "X_x--" + c[len("X_x_E2--"):-len(N.SUF)]
        if c.startswith("E_L") and "_x_E2-" in c:
            return c.replace("_x_E2-", "_x-", 1)
        return None
    for w in (MP.work(run, "tune", "h22c4"), kdir):
        if not MP.has(w):
            raise FileNotFoundError("no Pro verdicts in %s" % w)
        st1, bl1 = ps.load(w, "pro")
        t1 = fr.raw_texts(w)
        for (q, c), v in st1.items():
            if c != "baseline_x_E2" and not (c.startswith("X_x_E2--") or "_x_E2-" in c):
                continue
            c2 = rename(c)
            if c2 is None:
                continue
            put((q, c2), v, bl1[(q, c)], t1[(q, c)])
    sh = S.shell(layer, cl)
    need = ["baseline_x", sh.cE(s_dash)] + [sh.cEC(s_dash, k) for k in ps.LAD]
    sp = S.NewSplit(layer, cl, st, bl, need, qs=qs)
    return sp, fr.Texts(tx, tx)


def single_cells(N, w, cond, qs):
    st, bl = N.ps.load(w, "pro")
    t = N.MP.fr.raw_texts(w)
    return ({q: (N.V[st[(q, cond)]], not bl[(q, cond)]) for q in qs if st.get((q, cond), "?") != "?"},
            {q: t[(q, cond)] for q in qs if (q, cond) in t})


def texts_of(sp, tx, cfg, N):
    recs = N.S.system(sp, tx, cfg)[2]
    return {r["p"]: r["text"] for r in recs}


KR = "Iteration_01/results/keepfix/llama3.1-8b/e44"
LAM_OF = {"l0p03": "0.03", "l0p1": "0.1", "l0p3": "0.3"}


def tag_info(tag):
    """Layout of one lambda of queue K section 5 (keepfix_llama_lam.sh)."""
    return {"tag": tag, "label": "k%s" % LAM_OF[tag], "run": "%s/llama3.1-8b_keepfix_%s" % (KR, tag),
            "s": Path(NSK, "aligned_s_%s_%s.txt" % (MODEL, tag)).read_text().strip(),
            "gen": "keepfix_%s" % tag, "qwork": "keepfix_%s_pilot100" % tag}


def show_lam(tags):
    print("\n# queue K section 5: lambda 0.03 / 0.1 / 0.3 (usable: signcheck --min-gap 0.718 and cosine with the main E >= 0.5)")
    for tag in ("l0p03", "l0p1", "l0p3"):
        run = "%s/llama3.1-8b_keepfix_%s" % (KR, tag)
        p = Path(run) / "train_logs" / ("E_L%d.json" % EL)
        if not p.exists():
            print("    lambda %-5s no record" % LAM_OF[tag])
            continue
        lg = jload(p)
        tr, va = lg["losses"]["train"], lg["losses"].get("val") or {}
        k = lg["keep"]
        dg = k.get("diagnostics") or {}
        sc = Path(NSK, "signcheck_%s_%s.json" % (MODEL, tag))
        cp = Path(NSK, "compare_%s_%s.json" % (tag, MODEL))
        scd = jload(sc) if sc.exists() else {}
        cpd = jload(cp) if cp.exists() else {}
        al = Path(NSK, "aligned_s_%s_%s.txt" % (MODEL, tag))
        print("    lambda %-5s gap %.6f | cosine with the main E %s | cos(w, md) %.4f | cause loss trained %.4f (no write %.4f, main E 0.0182) | held-out cause %.4f | %s | aligned s' %s" % (
            LAM_OF[tag], lg["gap"], ("%.4f" % cpd["cosine"]) if cpd else "-", lg["cosine_w_md"], tr["trained"]["cause"], tr["no_write"]["cause"],
            va.get("trained", {}).get("cause", float("nan")), "usable" if scd.get("pass") and cpd and cpd["cosine"] >= 0.5 else "NOT usable",
            al.read_text().strip() if al.exists() else "-"))
        print("        keep loss per answer token (all / right under both / right under x only): saved direction at +A %s | -A %s | no write %s" % (
            dg.get("keep_loss_trained_at_A"), dg.get("keep_loss_opposite_side"), dg.get("keep_loss_no_write")))
        print("        sign check %s | draws %s | per epoch: %s" % (
            {kk: scd.get(kk) for kk in ("pass", "sign_check", "lambda0_vs_lambda1_same")}, lg["training"].get("keep_draws_sha256", "")[:16],
            " | ".join("%d: cause %.4f keep %.4f neg %d" % (x["epoch"], x["train_cause"], x["keep_train_mean"], x["keep_sign_negative_steps"])
                       for x in lg["training"]["history"])))


def tq(tags):
    N, N3, Q, NP, SPB = imports()
    MP, ML, ps = N.MP, N.ML, N.ps
    ct = N.restricted(MODEL, 0)
    sysd, txt, cfgs, qsets = {}, {}, {}, []
    for s in (S_MAIN, "1.75"):
        sp, tx = N.build(MODEL, 0, "x_E2", dash(s), ct)
        cfg = ML.n1_pick(sp, tx, dash(s))
        cfgs["main %s" % s] = cfg
        s0 = N.run_sys(sp, tx, ("base",))
        if "S0" in sysd and any(s0[q] != sysd["S0"][q] for q in sp.qs if q in sysd["S0"]):
            raise SystemExit("S0 differs between builds")
        sysd["S0"] = s0
        sysd["SE main %s" % s], sysd["SN1 main %s" % s] = N.run_sys(sp, tx, ("E", dash(s))), N.run_sys(sp, tx, cfg)
        txt["SE main %s" % s], txt["SN1 main %s" % s] = texts_of(sp, tx, ("E", dash(s)), N), texts_of(sp, tx, cfg, N)
        qsets.append(set(sp.qs))
    display, layer, _ = N.A.KEY[MODEL]
    D = MP.MT.Data(display, 0)
    run = D.m["ec"][0].split("/eval_extra/")[0]
    sysd["no write"] = {q: (N.V[D.sp.st[(q, "baseline_x")]], not D.sp.blank[(q, "baseline_x")]) for q in D.sp.qs}
    sysd["only E main 3.5"], txt["only E main 3.5"] = single_cells(N, MP.work(run, "tune", "pospromptl16"), "E_L16_x-w-p3-5", ct)
    knames, onames = [], []
    for t in tags:
        kdir = "%s/eval_extra/judge_tune_kf_tune" % t["run"]
        xdir = "%s/eval_extra/judge_tune_kfx_tune" % t["run"]
        for s, at in ((t["s"], "a"), (S_MAIN, S_MAIN)):
            ksp, ktx = tq_k_build(N, dash(s), ct, kdir)
            kcfg = ML.n1_pick(ksp, ktx, dash(s))
            nm = "%s@%s" % (t["label"], at)
            cfgs[nm] = kcfg
            s0k = N.run_sys(ksp, ktx, ("base",))
            if any(s0k[q] != sysd["S0"][q] for q in ksp.qs if q in sysd["S0"]):
                raise SystemExit("S0 differs between the builds")
            sysd["SE " + nm], sysd["SN1 " + nm] = N.run_sys(ksp, ktx, ("E", dash(s))), N.run_sys(ksp, ktx, kcfg)
            txt["SE " + nm], txt["SN1 " + nm] = texts_of(ksp, ktx, ("E", dash(s)), N), texts_of(ksp, ktx, kcfg, N)
            qsets.append(set(ksp.qs))
            knames += ["SE " + nm, "SN1 " + nm]
            sysd["only E " + nm], txt["only E " + nm] = single_cells(N, xdir, "E_L16_x-w-p%s" % dash(s), ct)
            onames.append("only E " + nm)
    qs = [q for q in ct if all(q in x for x in qsets) and all(q in sysd[n] for n in sysd)]
    print("\n# TruthfulQA tune, DeepSeek Pro, %d of the main table's %d tune questions; P = x_E2 (S0 = P alone, no write); <k lambda>@a = E_k at the aligned strength (write = the main E's at 3.5), @3.5 = nominal; N1 caps (n1_pick on tune): %s" % (
        len(qs), len(ct), " | ".join("%s %s" % (k, v) for k, v in cfgs.items())))
    print("    aligned strengths: " + " | ".join("%s s' = %s" % (t["label"], t["s"]) for t in tags))
    names = ["S0", "SE main 3.5", "SN1 main 3.5", "SE main 1.75", "SN1 main 1.75"] + knames
    diffs = []
    for t in tags:
        nm = "%s@a" % t["label"]
        diffs += [("SE " + nm, "SE main 3.5"), ("SN1 " + nm, "SN1 main 3.5"), ("SE " + nm, "SE main 1.75"), ("SN1 " + nm, "SN1 main 1.75"),
                  ("SE " + nm, "S0"), ("SN1 " + nm, "S0")]
    diffs += [("SN1 main 3.5", "S0"), ("SN1 main 1.75", "S0")]
    negs = " | ".join("%s: opening with a negation %d of %d answered" % (n, sum(1 for q in qs if sysd[n][q][1] and neg(SPB, txt[n].get(q))),
                                                                    sum(sysd[n][q][1] for q in qs)) for n in names[1:])
    line(names, sysd, qs, diffs, ps.ci, "TruthfulQA tune on P (x_E2)", negs)
    classes(sysd, qs, names, "TruthfulQA tune")
    names2 = ["no write", "only E main 3.5"] + onames
    diffs2 = [(n, "only E main 3.5") for n in onames] + [(n, "no write") for n in onames] + [("only E main 3.5", "no write")]
    line(names2, sysd, qs, diffs2, ps.ci, "TruthfulQA tune: plain prompt x, single E (no sentence)")
    return {n: sum(sysd[n][q][0] for q in qs) for n in names}, len(qs)


def qaqa(tags):
    N, N3, Q, NP, SPB = imports()
    G = Q.G

    class PR(Q.Raw):
        """Passes over the 570-question bank with verdicts on the 100 pilot questions only; S0 (P) from the main passes."""

        def __init__(self, main_R, gdir, works):
            Q.Raw.__init__(self, MODEL, "qaqafull", N3.BANKQ, works)
            self.full_ids = list(self.ids)
            self.ids = [q for q in self.full_ids if q in N3.PILOT]
            self.gdir = Path(gdir)
            self.rows["x"], self.cond["x"] = main_R.rows["x"], main_R.cond["x"]
            for q in self.full_ids:
                k = (q, main_R.cond["x"])
                if k in main_R.val:
                    self.val[k], self.pred[k] = main_R.val[k], main_R.pred[k]

        def has(self, arm):
            if arm in self.rows:
                return True
            p = self.gdir / (G.arm_token(arm) + ".json")
            if not p.is_file():
                return False
            g = G.load_generation(p, self.args, self.full_ids)
            k = g["contract"]
            if k["arm"] != arm or k.get("positions") != ("prompt" if ":" in arm else None):
                raise SystemExit("%s is not the pass %s at prompt" % (p.name, arm))
            rows = {r["id"]: r for r in g["rows"]}
            if any((q, k["condition"]) not in self.val and rows[q]["text"].strip() for q in self.ids):
                return False
            for q in self.ids:
                key = (q, k["condition"])
                if key in self.pred and self.pred[key].strip() != rows[q]["text"].strip():
                    raise SystemExit("the judged answer of %s under %s is not the text now in %s" % (q, arm, p.name))
            self.rows[arm], self.cond[arm] = rows, k["condition"]
            return True

    def take(R, s):
        arms = ["x_E2:E%d.w.p%s" % (EL, s)] + ["x_E2:E%d.w.p%s+C%d.w.m%s" % (EL, s, CL, k) for k in LADK]
        for a in arms:
            R.need(a)
            alias = a.replace("x_E2:", "x:")
            R.rows[alias], R.cond[alias] = R.rows[a], R.cond[a]
        return N3.qaqa_sn1(R, s)
    main_R = N3.qaqa_raw(MODEL, S_MAIN)
    caps_m, sn_m = N3.qaqa_sn1(main_R, S_MAIN)
    pilot = [q for q in main_R.ids if q in N3.PILOT]
    base = Q.Raw(MODEL, "qaqafull", N3.BANKQ, ("test_full570", "test_full570_tuned", "test_full570_strong"))
    sysd = {"no write": base.arm_system("x"), "S0": main_R.arm_system("x"), "SE main 3.5": main_R.arm_system(main_R.e(S_MAIN)),
            "SN1 main 3.5": (sn_m[0], sn_m[1])}
    txt = {"SE main 3.5": {q: main_R.text(main_R.e(S_MAIN), q) for q in pilot}}
    caps = {"main 3.5": caps_m}
    hR = PR(main_R, Path(NSD) / "gen_prompt" / "keepfix_main" / "qaqafull" / MODEL / "test", ["keepfix_main_pilot100"])
    c, sn = take(hR, "1.75")
    caps["main 1.75"] = c
    sysd["SE main 1.75"], sysd["SN1 main 1.75"] = hR.arm_system(hR.e("1.75")), (sn[0], sn[1])
    txt["SE main 1.75"] = {q: hR.text(hR.e("1.75"), q) for q in pilot}
    knames = []
    for t in tags:
        kR = PR(main_R, Path(NSD) / "gen_prompt" / t["gen"] / "qaqafull" / MODEL / "test", [t["qwork"]])
        for s, at in ((t["s"], "a"), (S_MAIN, S_MAIN)):
            nm = "%s@%s" % (t["label"], at)
            c, sn = take(kR, s)
            caps[nm] = c
            sysd["SE " + nm], sysd["SN1 " + nm] = kR.arm_system(kR.e(s)), (sn[0], sn[1])
            txt["SE " + nm] = {q: kR.text(kR.e(s), q) for q in pilot}
            knames += ["SE " + nm, "SN1 " + nm]
    per = {n: {q: (v[0][q], bool(v[1][q])) for q in pilot} for n, v in sysd.items()}
    jci = lambda v: "%+d [%+d, %+d]" % tuple(N3.Q.S.joint_ci([v]))  # noqa: E731
    print("\n# (QA)^2 100 tune (pilot) questions, DeepSeek Pro; P = x_E2 (S0 = P alone, no write); SN1 caps chosen on these 100 (E's blanks / own blanks): %s" % (
        " | ".join("%s %s / %s" % (k, v[0], v[1]) for k, v in caps.items())))
    names = ["S0", "SE main 3.5", "SN1 main 3.5", "SE main 1.75", "SN1 main 1.75"] + knames
    diffs = []
    for t in tags:
        nm = "%s@a" % t["label"]
        diffs += [("SE " + nm, "SE main 3.5"), ("SN1 " + nm, "SN1 main 3.5"), ("SE " + nm, "SE main 1.75"), ("SN1 " + nm, "SN1 main 1.75"),
                  ("SE " + nm, "S0"), ("SN1 " + nm, "S0")]
    diffs += [("SN1 main 3.5", "S0"), ("SN1 main 1.75", "S0")]
    for t, label in (("all", "all types"),) + Q.TYPES:
        qs = [q for q in pilot if t == "all" or main_R.qtype[q] == t]
        se = [n for n in names if n.startswith("SE")]
        negs = " | ".join("%s: opening with a negation %d of %d answered" % (
            n, sum(1 for q in qs if per[n][q][1] and neg(SPB, txt[n][q])), sum(per[n][q][1] for q in qs)) for n in se)
        line(["no write"] + names, per, qs, diffs, jci, "(QA)^2 tune 100, %s" % label, negs)
        classes(per, qs, names, "(QA)^2 tune 100, %s" % label)
    return {n: sum(per[n][q][0] for q in pilot) for n in names}, len(pilot)


def fact(tags):
    N, N3, Q, NP, SPB = imports()
    G, AM = NP.G, NP.AM
    jci = lambda v: "%+d [%+d, %+d]" % tuple(NP.S.joint_ci([v]))  # noqa: E731

    def load(task, path, arm, bank, bp):
        g = G.load_generation(path, SimpleNamespace(task=task, split="tune", model=MODEL, items=bp), list(bank))
        if g["contract"]["arm"] != arm or g["contract"].get("positions") != ("prompt" if ":" in arm else None):
            raise SystemExit("%s is not the pass %s" % (path, arm))
        out, txt, ids = {}, {}, {}
        for r in g["rows"]:
            blank = bool(r["rule_abstain"]) or not r["text"].strip()
            v = AM.verdict(r["text"], bank[r["id"]]["gold"], not blank)
            q = "%s:%s" % (task, r["id"])
            out[q], txt[q], ids[q] = ((v == "R") - (v == "W"), not blank), r["text"], r["token_ids"]
        return out, txt, ids
    nets, sizes = {}, {}
    for task in ("triviaqa", "popqa"):
        bank, bp = NP.bank(task, "tune")
        base = Path(NSD, "gen_prompt")
        arm = lambda s: "x_E2:E%d.w.p%s" % (EL, s)  # noqa: E731
        files = {"no write": (base / task / MODEL / "tune" / "x.json", "x"),
                 "S0": (base / "keepfix_main" / task / MODEL / "tune" / "x_E2.json", "x_E2"),
                 "SE main 3.5": (base / "keepfix_main" / task / MODEL / "tune" / (tok(arm(S_MAIN)) + ".json"), arm(S_MAIN)),
                 "SE main 1.75": (base / "keepfix_main" / task / MODEL / "tune" / (tok(arm("1.75")) + ".json"), arm("1.75"))}
        knames = []
        for t in tags:
            for s, at in ((t["s"], "a"), (S_MAIN, S_MAIN)):
                nm = "SE %s@%s" % (t["label"], at)
                files[nm] = (base / t["gen"] / task / MODEL / "tune" / (tok(arm(s)) + ".json"), arm(s))
                knames.append(nm)
        sysd, txt, tid = {}, {}, {}
        for n, (p, a) in files.items():
            sysd[n], txt[n], tid[n] = load(task, p, a, bank, bp)
        qs = list(sysd["S0"])
        same = []
        for n, a in (("S0", "x_E2"), ("SE main 3.5", arm(S_MAIN)), ("SE main 1.75", arm("1.75"))):
            other = base / "h22" / task / MODEL / "tune" / (tok(a) + ".json")
            if other.exists():
                _, _, oid = load(task, other, a, bank, bp)
                same.append("%s = queue I section 8's file: %d / %d identical" % (n, sum(oid[q] == tid[n][q] for q in qs), len(qs)))
            else:
                same.append("%s: queue I section 8's file not there" % n)
        names = ["no write", "S0", "SE main 3.5", "SE main 1.75"] + knames
        diffs = []
        for t in tags:
            nm = "SE %s@a" % t["label"]
            diffs += [(nm, "SE main 3.5"), (nm, "SE main 1.75"), (nm, "S0"), (nm, "no write")]
        diffs += [("SE main 3.5", "S0"), ("SE main 1.75", "S0"), ("SE main 3.5", "no write"), ("SE main 1.75", "no write")]
        flips = []
        for n in names[2:]:
            for ref in ("S0", "no write"):
                rw = [q for q in qs if sysd[ref][q][0] == 1 and sysd[n][q][0] == -1]
                flips.append("%s: right under %s -> wrong %d (opening with a negation %d)" % (n, ref, len(rw), sum(neg(SPB, txt[n][q]) for q in rw)))
        negs = " | ".join("%s: answered %d, opening with a negation %d" % (n, sum(sysd[n][q][1] for q in qs),
                                                                         sum(1 for q in qs if sysd[n][q][1] and neg(SPB, txt[n][q]))) for n in names)
        line(names, sysd, qs, diffs, jci, "%s tune, alias match; P = x_E2 (S0 = P alone, no write)" % task, negs)
        print("    " + " | ".join(flips))
        classes(sysd, qs, names[1:], "%s tune" % task)
        print("    " + " | ".join(same))
        nets[task] = {n: sum(sysd[n][q][0] for q in qs) for n in names}
        sizes[task] = len(qs)
    return nets, sizes


def choose(tags, tqn, qan, fan):
    """Queue K section 5: sum = TruthfulQA tune SN1 + (QA)^2 100 tune SN1 + TriviaQA tune SE + PopQA tune SE; the largest
    sum among the usable lambdas, ties to the larger lambda; at the aligned strength (main reading) and at nominal 3.5."""
    (tq, n_tq), (qa, n_qa), (fa, n_fa) = tqn, qan, fan

    def parts(sn1, se):
        return [tq[sn1], qa[sn1], fa["triviaqa"][se], fa["popqa"][se]]
    rows = [("old E (main 3.5)", parts("SN1 main 3.5", "SE main 3.5")), ("old E half (1.75)", parts("SN1 main 1.75", "SE main 1.75"))]
    for t in tags:
        for at in ("a", S_MAIN):
            nm = "%s@%s" % (t["label"], at)
            rows.append((nm, parts("SN1 " + nm, "SE " + nm)))
    sizes = [n_tq, n_qa, n_fa["triviaqa"], n_fa["popqa"]]
    print("\n# queue K section 5: the sum (TruthfulQA tune SN1 + (QA)^2 100 tune SN1 + TriviaQA tune SE + PopQA tune SE); sizes %s" % sizes)
    old = rows[0][1]
    for name, v in rows:
        print("    %-18s %+4d %+4d %+4d %+4d | sum %+4d | per 100 questions %s | change against the old E per 100 %s" % (
            (name,) + tuple(v) + (sum(v), " ".join("%+.1f" % (100.0 * x / n) for x, n in zip(v, sizes)),
                                  " ".join("%+.1f" % (100.0 * (x - o) / n) for x, o, n in zip(v, old, sizes)))))
    for at, label in (("a", "aligned strength (main reading)"), (S_MAIN, "nominal 3.5")):
        cand = [(sum(v), float(LAM_OF[t["tag"]]), t["label"]) for t in tags for name, v in rows if name == "%s@%s" % (t["label"], at)]
        if cand:
            best = max(cand)
            print("    chosen at the %s: %s (sum %+d; old E %+d) -> %s than the old E" % (
                label, best[2], best[0], sum(old), "higher" if best[0] > sum(old) else "not higher"))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "check-tq":
        check_tq(sys.argv[2:])
    elif mode == "check-fact":
        check_fact(sys.argv[2:])
    elif mode == "train":
        show_train()
    elif mode == "lam":
        tags = [tag_info(t) for t in sys.argv[2:]]
        show_train()
        show_lam(tags)
        if tags:
            tqn, qan, fan = tq(tags), qaqa(tags), fact(tags)
            choose(tags, tqn, qan, fan)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
