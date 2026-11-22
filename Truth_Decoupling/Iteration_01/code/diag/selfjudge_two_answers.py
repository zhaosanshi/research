#!/usr/bin/env python3
# Iteration_01 exploration (Claude Opus 5.5, 2026-11-22), work queue discussion/work_queue/2026-11-22_queue_j.md
# ("two answers, pick one"; rules written 22:16, committed b7f1861, before any self-judge score or reading of this item).
# No judge call, no key, no generation. Nothing is chosen (no threshold, no strength, no cap).
# Candidates per model and question (TruthfulQA, the main table's tune / test lists, as diag/plain_honesty_fill.py):
#   a_S = S0 = the plain prompt with the honest sentence, no write (condition baseline_x_E2, work dir
#         eval_extra/judge_<split>_h22c4_<split>; text checked against eval_extra/base_x_E2_<split>.json);
#   a_N = N1 of the main table (S.system at ML.n1_pick on the tune split, as plain_honesty_fill.py); its answer is single
#         E's answer (selfjudge_probe items.json / probe.npz) or the released answer (tq_release_items.json / probe);
#   a_T = must-answer + honest sentence, no write (baseline_conf_E2 of the must-answer work dir D.m["d"][split]; text
#         checked against answers["conf_E2"] of frozen_<split>.json of that run).
#   Verdicts: the existing DeepSeek Pro verdicts (policy_select2.load(..., "pro")); answered = not blank there.
# Self-judge score: selfjudge_probe.py's score path unchanged (E45 code path), margin = logp(True) - logp(False) of
# 'The answer to the question "{q}" is: {a}' (selfjudge_probe.statement); conditions selfjudge_probe.conds_of(model,
# source): baseline for all five models, E_a3 in addition for the three with a true/false E. New answer sources
# (registered here into selfjudge_probe.SOURCES at run time; selfjudge_probe.py itself is not changed):
#   tj_s0    the answered S0 answers, tune + test       tj_conf  the answered conf_E2 answers, tune + test
#   tj_n1chk 16 N1 test answers (seed 20261122) already scored in probe.npz / tq_release_probe.npz (queue section 3.4)
#   tj_n1ctx diagnosis only (check 4 missed on Llama-3.1-8B): the complete original scoring batches (8 statements, the
#            order of probe.npz / tq_release_probe.npz) that hold the 16 check answers, to see whether the differences
#            come from the batch neighbours (export-ctx MODEL; check4 then also compares these)
# Systems ("different" = different statements; ties keep S0):
#   A1   one side answered -> that one; both answered, same statement -> S0's; both answered, different -> higher margin.
#   A1z  A1, then rule Z on the handed-in answer (margin < 0 -> no answer; NEG_RE openings exempt).
#   S0, N1, F1 (= S0, its blanks from N1), T2 (= S0, its blanks from conf_E2; reference only); S0z, N1z (= every N1
#   answer judged, selfjudge_stage1.computeZ's S3), F1z; TA1 / TA1z = A1 / A1z with a_N replaced by a_T.
# Readings (queue section 2): 1 one-right-one-wrong pairs picked right; 2 A1 - S0 / F1 / N1 / TA1; 3 A1z - S0z / F1z /
# N1z / TA1z; 4 written minus not written (A1, A1z); 5 the same at equal answer counts: the side with more answers drops
# its handed-in answers from the lowest margin up (ties by roster order) until the counts are equal, per model; the
# margins are those of the condition of the comparison (reading 4: each side its own condition, see the progress note).
# Totals: five models (no write) and three models (both conditions) on the sorted common questions; intervals
# policy_select2.ci (4000, seed 20261116) on per-question sums over the models.
# usage (host, from /data/sanshizhao/Truth_Decoupling):
#   python3 -B Iteration_01/diag/selfjudge_two_answers.py export MODEL      (writes the three *_items.json)
#   python3 -B Iteration_01/diag/selfjudge_two_answers.py check4 MODEL      (after scoring tj_n1chk)
#   python3 -B Iteration_01/diag/selfjudge_two_answers.py check3            (queue 3.3 from the verdicts alone)
#   section 5 (batch size 1): export1 MODEL (host); launcher with SOURCES tj1_chk,tj1_n1,tj1_s0,tj1_conf; checkbc MODEL
#   python3 -B Iteration_01/diag/selfjudge_two_answers.py read              (checks 3 and 4, then all readings)
#   score (container, GPU): launchers/explore_20261120/selfjudge_two_answers_1122.sh MODEL GPU SOURCES
#     = python3 -B Iteration_01/diag/selfjudge_two_answers.py score MODEL SRC[,SRC]
import os
import sys

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets"]
import selfjudge_probe as SP     # noqa: E402

NEW = ("tj_s0", "tj_conf", "tj_n1chk", "tj_n1ctx")
SP.SOURCES.update(NEW)
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
WRITE = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b")
V = SP.V
SPLITS = ((1, "test"), (0, "tune"))
# queue section 3.3 (test): (net, answers) of S0 and N1 per model; F1 and T2 on the five models' common questions
REF_S0 = {"llama3.1-8b": (33, 454), "qwen3.5-9b": (168, 464), "qwen3.8-27b": (162, 373), "gemma3-12b": (156, 501),
          "gemma3-27b": (186, 488)}
REF_N1 = {"llama3.1-8b": (62, 476), "qwen3.5-9b": (186, 444), "qwen3.8-27b": (175, 490), "gemma3-12b": (158, 489),
          "gemma3-27b": (144, 510)}
REF5 = {"n": 520, "F1": (781, 2518), "T2": (689, 2598)}
NCHK = 16


def native(model, i):
    """The main table's list of split i and the verdicts / texts of S0, conf_E2 and N1, as plain_honesty_fill.py."""
    import iti_table as A
    MP, ML, S, ps = A.MP, A.ML, A.S, A.ps
    name = ("tune", "test")[i]
    display, layer, s = A.KEY[model]
    T, err = MP.prompt_split(display, layer, i, s)
    if T is None:
        raise SystemExit("%s %s: %s" % (model, name, err))
    src = [x for x in ML.sources(display)[2] if x[0] == layer]
    want = set(src[0][3 if i else 1].qs)
    ids = [q for q in T[3] if q in want]
    if ids != T[3]:
        T, err = MP.prompt_split(display, layer, i, s, qs=ids)
    sp, tx, _, ids = T
    tT, _ = MP.prompt_split(display, layer, 0, s)
    n1cfg = ML.n1_pick(tT[0], tT[1], s)
    n1 = S.system(sp, tx, n1cfg)
    D = MP.MT.Data(display, i)
    run = D.m["ec"][i].split("/eval_extra/")[0]
    w = MP.work(run, name, "h22c4")
    st, bl = ps.load(w, "pro")
    st1, bl1 = ps.load(D.m["d"][i], "pro")
    return dict(name=name, ids=list(ids), n1cfg=list(n1cfg), n1=n1, run=run, w=w, d=D.m["d"][i], st=st, bl=bl, st1=st1,
                bl1=bl1, fr=MP.fr)


# ------------------------------------------------------------------------------------------------------------------ #
def export(model):
    step1 = SP.read_json(os.path.join(SP.OUT, model, "items.json"))
    qtext = {(x["split"], x["id"]): x["question"] for x in step1["items"]}
    s0, cf = [], []
    info = {}
    for i, name in SPLITS:
        N = native(model, i)
        ids = N["ids"]
        if ids != [x["id"] for x in step1["items"] if x["split"] == name]:
            raise SystemExit("%s %s: the main list differs from selfjudge items.json" % (model, name))
        # S0 texts: judged cells vs the generation file
        gpath = os.path.join(N["run"], "eval_extra", "base_x_E2_%s.json" % name)
        g = SP.read_json(gpath)
        if g.get("completed") is not True or g.get("partial") or g.get("split") != name or g.get("prompt") != "x_E2":
            raise SystemExit("%s: not the complete x_E2 no-write pass of %s" % (gpath, name))
        gt = {r["pair_id"]: r["text"] for r in g["rows"]}
        # conf_E2 texts: judged cells vs the frozen pass of the must-answer work dir's run
        fpath = os.path.join(os.path.dirname(N["d"]), "frozen_%s.json" % name)
        fz = SP.read_json(fpath)
        if fz.get("split") != name or fz.get("completed") is not True or fz.get("model") not in (model, model + "-dm"):
            raise SystemExit("%s: not the complete frozen pass of %s" % (fpath, name))
        ft = {r["pair_id"]: r["answers"]["conf_E2"]["text"] for r in fz["questions"]}
        if len(gt) != len(g["rows"]) or len(ft) != len(fz["questions"]):
            raise SystemExit("%s %s: duplicate question in a generation file" % (model, name))
        tw, td = N["fr"].raw_texts(N["w"]), N["fr"].raw_texts(N["d"])
        qw = cell_questions(N["w"], "baseline_x_E2")
        qd = cell_questions(N["d"], "baseline_conf_E2")
        cnt = dict(s0_verdict=0, s0_answered=0, conf_verdict=0, conf_answered=0, s0_checked=0, conf_checked=0)
        for q in ids:
            qt = qtext[(name, q)]
            for cond, st, bl, tx, gen, qq, out, tag in (
                    ("baseline_x_E2", N["st"], N["bl"], tw, gt, qw, s0, "s0"),
                    ("baseline_conf_E2", N["st1"], N["bl1"], td, ft, qd, cf, "conf")):
                if st.get((q, cond), "?") == "?":
                    continue
                t = tx[(q, cond)]
                if SP.sha1(t) != SP.sha1(gen[q]):
                    raise SystemExit("%s %s: the judged %s text of %s is not the generation file's" % (model, name, cond, q))
                if qq[q].strip().rstrip("?").strip() != qt.strip().rstrip("?").strip():
                    raise SystemExit("%s %s: question text of %s differs from items.json" % (model, name, q))
                cnt[tag + "_checked"] += 1
                cnt[tag + "_verdict"] += 1
                a = not bl[(q, cond)]
                cnt[tag + "_answered"] += a
                out.append(dict(id=name + "|" + q, split=name, question=qt, answer=t, answer_sha1=SP.sha1(t),
                                value=V[st[(q, cond)]], state=st[(q, cond)], answered=bool(a), neg=bool(SP.NEG_RE.match(t))))
        info[name] = dict(n=len(ids), s0_work=N["w"], s0_generation=gpath, s0_generation_sha256=SP.sha256_file(gpath),
                          conf_work=N["d"], conf_generation=fpath, conf_generation_sha256=SP.sha256_file(fpath), **cnt)
        print("%s %s: %d questions | S0 verdicts %d, answered %d, texts = generation (sha1) %d | conf_E2 verdicts %d, "
              "answered %d, texts = generation (sha1) %d" % (model, name, len(ids), cnt["s0_verdict"], cnt["s0_answered"],
                                                              cnt["s0_checked"], cnt["conf_verdict"], cnt["conf_answered"],
                                                              cnt["conf_checked"]))
    for src, items, cond in (("tj_s0", s0, "baseline_x_E2"), ("tj_conf", cf, "baseline_conf_E2")):
        SP.save_items(os.path.join(SP.OUT, model, src + "_items.json"),
                      dict(model=model, source=src, condition=cond, splits=info, neg_re=SP.NEG_RE.pattern, items=items))
        print("  %s: %d items (%d answered) -> %s_items.json" % (src, len(items), sum(x["answered"] for x in items), src))
    # check 4: 16 N1 test answers already scored
    import numpy as np
    n1 = n1_answers(model)
    pool = sorted(k for k, x in n1.items() if k[0] == "test")
    rng = np.random.default_rng(20261122)
    pick = sorted(rng.choice(len(pool), NCHK, replace=False).tolist())
    items = []
    for j in pick:
        k = pool[j]
        x = n1[k]
        items.append(dict(id=k[0] + "|" + k[1], split=k[0], question=x["question"], answer=x["answer"], answer_sha1=x["sha"],
                          value=x["v"], answered=True, neg=x["neg"], n1_from=x["src"]))
    SP.save_items(os.path.join(SP.OUT, model, "tj_n1chk_items.json"),
                  dict(model=model, source="tj_n1chk", seed=20261122, items=items))
    print("  tj_n1chk: %d N1 test answers (E %d, released %d) -> tj_n1chk_items.json" % (
        len(items), sum(x["n1_from"] == "E" for x in items), sum(x["n1_from"] == "rel" for x in items)))


def export_ctx(model):
    """Diagnosis: the original batches (size 8, answered-item order of each source) holding the tj_n1chk answers."""
    chk = SP.read_json(os.path.join(SP.OUT, model, "tj_n1chk_items.json"))["items"]
    pd = [x for x in SP.read_probe(model)[0]["items"] if x["answered"]]
    rd = [x for x in SP.read_probe(model, source="tq_release")[0]["items"] if x["answered"]]
    pos = {"E": {x["split"] + "|" + x["id"]: j for j, x in enumerate(pd)}, "rel": {x["id"]: j for j, x in enumerate(rd)}}
    lists = {"E": [dict(id=x["split"] + "|" + x["id"], split=x["split"], question=x["question"], answer=x["answer"],
                        answer_sha1=x["answer_sha1"], value=V[x["n1_state"]], answered=True, neg=x["neg"], n1_from="E")
                   for x in pd],
             "rel": [dict(id=x["id"], split=x["split"], question=None, answer=x["answer"], answer_sha1=x["answer_sha1"],
                          value=x["value"], answered=True, neg=x["neg"], n1_from="rel") for x in rd]}
    q = {x["split"] + "|" + x["id"]: x["question"] for x in SP.read_probe(model)[0]["items"]}
    for x in lists["rel"]:
        x["question"] = q[x["id"]]
    items, seen, tail = [], set(), []
    for x in chk:
        b = pos[x["n1_from"]][x["id"]] // 8
        if (x["n1_from"], b) in seen:
            continue
        seen.add((x["n1_from"], b))
        blk = lists[x["n1_from"]][8 * b:8 * b + 8]
        if len(blk) != 8:
            # the last (short) batch of a source: kept short by putting it at the very end
            if 8 * b + len(blk) != len(lists[x["n1_from"]]) or tail:
                raise SystemExit("%s: batch %d of %s is short but not the last one" % (model, b, x["n1_from"]))
            tail = blk
            continue
        items += blk
    items += tail
    SP.save_items(os.path.join(SP.OUT, model, "tj_n1ctx_items.json"), dict(model=model, source="tj_n1ctx", items=items))
    print("tj_n1ctx: %d batches, %d statements" % (len(seen), len(items)))


def cell_questions(work, cond):
    import json
    out = {}
    for line in open(work + "/judge2/cells.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["condition"] == cond:
            out[r["pair_id"]] = r["question"]
    return out


def n1_answers(model):
    """{(split, id): N1's answered text with its existing margins}: single E's answer (probe) or the released one."""
    doc, _, z = SP.read_probe(model)
    conds = [c for c in ("baseline", "E_a3") if c in SP.conds_of(model, "")]
    out = {}
    ans = [x for x in doc["items"] if x["answered"]]
    for j, x in enumerate(ans):
        if z["ids"][j] != x["id"] or z["answer_sha1"][j] != x["answer_sha1"]:
            raise ValueError("%s: probe row %d is not item %s" % (model, j, x["id"]))
        if not x["n1_keeps_e"] or not x["n1_answered"]:
            raise SystemExit("%s: N1 does not keep E's answer of %s" % (model, x["id"]))
        out[(x["split"], x["id"])] = dict(question=x["question"], answer=x["answer"], sha=x["answer_sha1"], neg=x["neg"],
                                          v=V[x["n1_state"]], st=str(z["statement_sha1"][j]), src="E",
                                          m={c: float(z[c + "__margin"][j]) for c in conds})
    rdoc, _, rz = SP.read_probe(model, source="tq_release")
    rc = [c for c in ("baseline", "E_a3") if c in SP.conds_of(model, "tq_release")]
    if rc != conds:
        raise SystemExit("%s: release conditions %s differ from %s" % (model, rc, conds))
    ra = [x for x in rdoc["items"] if x["answered"]]
    for j, x in enumerate(ra):
        if rz["ids"][j] != x["id"] or rz["answer_sha1"][j] != x["answer_sha1"]:
            raise ValueError("%s: release row %d is not item %s" % (model, j, x["id"]))
        k = (x["split"], x["id"].split("|", 1)[1])
        if k in out:
            raise SystemExit("%s: %s is both an E answer and a release" % (model, k))
        out[k] = dict(question=None, answer=x["answer"], sha=x["answer_sha1"], neg=x["neg"], v=x["value"],
                      st=str(rz["statement_sha1"][j]), src="rel", m={c: float(rz[c + "__margin"][j]) for c in conds})
    q = {(x["split"], x["id"]): x["question"] for x in doc["items"]}
    for k, x in out.items():
        x["question"] = q[k]
        if SP.sha1(SP.statement(x["question"], x["answer"])) != x["st"]:
            raise SystemExit("%s: statement of N1 answer %s" % (model, k))
    return out


# ------------------------------------------------------------------------------------------------------------------ #
def score(model, sources):
    if sources and all(s in B1 for s in sources):
        return score1(model, sources)
    for s in sources:
        if s not in NEW:
            raise SystemExit("unknown source %s" % s)
    SP.score(model, None, False, 5, sources=list(sources))


def check4(model, quiet=False):
    import numpy as np
    doc, meta, z = SP.read_probe(model, source="tj_n1chk")
    n1 = n1_answers(model)
    conds = list(SP.conds_of(model, "tj_n1chk"))
    ans = [x for x in doc["items"] if x["answered"]]
    worst = {}
    for c in conds:
        d = []
        for j, x in enumerate(ans):
            k = (x["split"], x["id"].split("|", 1)[1])
            if n1[k]["sha"] != x["answer_sha1"] or n1[k]["st"] != str(z["statement_sha1"][j]):
                raise SystemExit("%s: check item %s is not N1's answer" % (model, x["id"]))
            d.append(abs(float(z[c + "__margin"][j]) - n1[k]["m"][c]))
        worst[c] = (len(d), float(np.max(d)), float(np.mean(d)))
    line = "check 4 %s: %s" % (model, " | ".join("%s %d answers max |diff| %.5f mean %.6f" % ((c,) + worst[c]) for c in conds))
    ok = len(ans) == NCHK and all(v[1] <= SP.TOL for v in worst.values())
    if os.path.exists(os.path.join(SP.OUT, model, "tj_n1ctx_probe.json")):
        cd, _, cz = SP.read_probe(model, source="tj_n1ctx")
        ca = [x for x in cd["items"] if x["answered"]]
        dd = {c: [] for c in conds}
        for j, x in enumerate(ca):
            k = (x["split"], x["id"].split("|", 1)[1])
            if n1[k]["sha"] != x["answer_sha1"]:
                raise SystemExit("%s: context item %s is not N1's answer" % (model, x["id"]))
            for c in conds:
                dd[c].append(abs(float(cz[c + "__margin"][j]) - n1[k]["m"][c]))
        line += " || diagnosis, the original batches rescored: " + " | ".join(
            "%s %d statements max |diff| %.5f" % (c, len(dd[c]), max(dd[c])) for c in conds)
    if not quiet:
        print(line + (" -> OK" if ok else " -> FAILED"))
    return ok, line


# ------------------------------------------------------------------------------------------------------------------ #
# Queue J section 5 (22:45, commit 1ef3c38): every self-judge score is recomputed one statement at a time (batch size 1,
# set in memory only), so that a margin no longer depends on its batch neighbours. Sources (export1 MODEL, host):
#   tj1_s0 / tj1_conf  the same items as tj_s0 / tj_conf;  tj1_n1  every answered N1 answer, tune + test (E's answer or
#   the released one);  tj1_chk  the 16 tj_n1chk answers, scored twice in order and once shuffled (check (b)).
# score1 (container, GPU) is selfjudge_probe.score_model's scoring path rebuilt from the same imports of the same code
# copy (copied: the direction / centre set-up and the forward loop, nothing else); before any batch-1 pass it must
# reproduce the stored batch-8 scores of the first 64 tj_s0 statements exactly (identity of the path), and the label
# map / centre shift must equal the validated full pass. Readings use only the tj1_* scores; the batch scores of the
# 22:2x run (tj_s0 / tj_conf / probe / tq_release) are read once more for the sensitivity lines.
B1 = ("tj1_s0", "tj1_conf", "tj1_n1", "tj1_chk")
SP.SOURCES.update(B1)
EQ_N = 64


def export1(model):
    for a, b in (("tj_s0", "tj1_s0"), ("tj_conf", "tj1_conf"), ("tj_n1chk", "tj1_chk")):
        doc = SP.read_json(os.path.join(SP.OUT, model, a + "_items.json"))
        doc = dict(doc, source=b, copied_from=a + "_items.json")
        SP.save_items(os.path.join(SP.OUT, model, b + "_items.json"), doc)
    n1 = n1_answers(model)
    items = [dict(id=k[0] + "|" + k[1], split=k[0], question=x["question"], answer=x["answer"], answer_sha1=x["sha"],
                  value=x["v"], answered=True, neg=x["neg"], n1_from=x["src"]) for k, x in sorted(n1.items())]
    SP.save_items(os.path.join(SP.OUT, model, "tj1_n1_items.json"), dict(model=model, source="tj1_n1", items=items))
    print("%s: tj1_s0 / tj1_conf / tj1_chk copied; tj1_n1 %d N1 answers (tune %d, test %d)" % (
        model, len(items), sum(x["split"] == "tune" for x in items), sum(x["split"] == "test" for x in items)))


def score1(model, sources):
    import fcntl
    import numpy as np
    from pathlib import Path
    cfg = SP.MODELS[model]
    code = os.path.abspath(cfg["code"])
    sys.path[:0] = [os.path.join(code, "scripts"), code]
    from common import load_split, tf_prompt
    from core.config import load_experiment, load_model_config
    from core.directions import load_das_direction, mean_diff_direction
    from core.intervene import SteerSpec, next_token_logits
    from core.io_utils import set_seed
    from core.model_io import capture_last_token, load_model
    from core.readout import build_label_token_map, label_logprobs_from_logits
    write = cfg["das"] is not None
    ref = SP.read_json((cfg["ref"] or cfg["code_ref"]) + ".json")
    for f, h in ref["code_sha256"].items():
        if SP.sha256_file(os.path.join(code, "scripts", f)) != h:
            raise SystemExit("%s: %s differs from the file E45 ran with" % (code, f))
    _, accepted, _ = SP.read_probe(model)
    exp = load_experiment("E2b_tqa")
    bs8 = exp.get("batch_size", 8)
    if bs8 != 8:
        raise SystemExit("the experiment's batch size is %s, expected 8" % bs8)
    mcfg = load_model_config(cfg.get("config", model))
    if mcfg.get("name") != model:
        raise SystemExit("model config is for %s" % mcfg.get("name"))
    prompts = exp["prompts"]
    st = prompts["statement"]
    set_seed(exp.get("seed", 42))
    recs, est = [], []
    for ds in exp["eval_datasets"]:
        recs += load_split(ds, exp.get("eval_split", "test"))
        est += load_split(ds, "calib") + load_split(ds, "dev")
    y_est = np.array([bool(r["label"]) for r in est])
    bundle = load_model(mcfg)
    EL = cfg["tf_layer"]
    mI = build_label_token_map(bundle.tokenizer, tf_prompt(bundle, prompts, recs[0]["text"], allow_idk=True),
                               st["label_variants"])
    if accepted["label_map"] != {k: list(map(int, v)) for k, v in mI.items()}:
        raise SystemExit("label tokens differ from the validated full pass")
    specs = {"baseline": None}
    if write:
        eI = [tf_prompt(bundle, prompts, r["text"], allow_idk=True) for r in est]
        eB = [tf_prompt(bundle, prompts, r["text"], allow_idk=False) for r in est]
        aB, _ = capture_last_token(bundle, eB, [EL], components=("resid",), batch_size=bs8)
        aI, _ = capture_last_token(bundle, eI, [EL], components=("resid",), batch_size=bs8)
        HB = aB[("resid", EL)].numpy().astype(np.float64)
        HI = aI[("resid", EL)].numpy().astype(np.float64)
        wMD, _ = mean_diff_direction(HB[y_est], HB[~y_est])
        wE, _ = load_das_direction(cfg["das"], "E", EL, variant="tf-ce-bidir", align_to=wMD)
        cI = (HI[y_est].mean(0) + HI[~y_est].mean(0)) / 2
        shift = float(np.dot(cI - (HB[y_est].mean(0) + HB[~y_est].mean(0)) / 2, wE) / ((HI @ wE).std() + 1e-12))
        if not np.isclose(shift, accepted["centre_shift_along_E_in_sd"], rtol=0, atol=1e-6):
            raise SystemExit("intervention centre differs from the validated full pass")
        specs["E_a3"] = [SteerSpec(wE, EL, mode="amplify", alpha=3.0, center=cI)]
    else:
        shift = None

    def run(texts, sp, b):
        out = []
        for i in range(0, len(texts), b):
            out.append(next_token_logits(bundle, bundle.encode(texts[i:i + b]), sp).numpy())
        lg = np.concatenate(out)
        lp = [label_logprobs_from_logits(x, mI, use_abstain=True) for x in lg]
        lpT = np.array([d["True"] for d in lp], dtype=np.float64)
        lpF = np.array([d["False"] for d in lp], dtype=np.float64)
        lpU = np.array([d["Abstain"] for d in lp], dtype=np.float64)
        if not np.isfinite([lpT, lpF, lpU]).all():
            raise ValueError("non-finite label score")
        pred = np.array(["True", "False", "Abstain"])[np.argmax([lpT, lpF, lpU], axis=0)]
        return dict(margin=lpT - lpF, gate=np.maximum(lpT, lpF) - lpU, choice=pred, lpT=lpT, lpF=lpF, lpU=lpU)

    def texts_of(src):
        doc = SP.read_json(os.path.join(SP.OUT, model, src + "_items.json"))
        items = [x for x in doc["items"] if x["answered"]]
        if any(SP.sha1(x["answer"]) != x["answer_sha1"] for x in items):
            raise SystemExit("%s: answer text changed" % src)
        stmts = [SP.statement(x["question"], x["answer"]) for x in items]
        texts = [tf_prompt(bundle, prompts, t, allow_idk=True) for t in stmts]
        if build_label_token_map(bundle.tokenizer, texts[0], st["label_variants"]) != mI:
            raise SystemExit("label map differs on %s" % src)
        return items, stmts, texts
    conds = SP.conds_of(model, "x")
    # identity of the path: the stored batch-8 tj_s0 scores of the first 64 statements, exactly
    sdoc, _, sz = SP.read_probe(model, source="tj_s0")
    items, stmts, texts = texts_of("tj_s0")
    for c in conds:
        got = run(texts[:EQ_N], specs[c], bs8)["margin"]
        d = float(np.max(np.abs(got - sz[c + "__margin"][:EQ_N])))
        print("path identity %s %s: batch 8, first %d tj_s0 statements, max |diff| vs stored %.6f" % (model, c, EQ_N, d), flush=True)
        if d != 0.0:
            raise SystemExit("PATH IDENTITY FAILED")
    root = Path(SP.OUT) / model
    with open(root / "probe.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for src in sources:
            if (root / (src + "_probe.json")).exists():
                SP.read_probe(model, source=src)
                print("%s: validated batch-1 scores already exist" % src)
                continue
            items, stmts, texts = texts_of(src)
            res = {c: run(texts, specs[c], 1) for c in conds}
            extra = {}
            if src == "tj1_chk":
                rng = np.random.default_rng(20261122)
                perm = rng.permutation(len(texts))
                for c in conds:
                    again = run(texts, specs[c], 1)["margin"]
                    sh = np.empty(len(texts))
                    sh[perm] = run([texts[k] for k in perm], specs[c], 1)["margin"]
                    d2 = float(np.max(np.abs(again - res[c]["margin"])))
                    d3 = float(np.max(np.abs(sh - res[c]["margin"])))
                    extra[c] = dict(second_pass_max_abs_diff=d2, shuffled_pass_max_abs_diff=d3)
                    print("check (b) %s %s: %d statements, batch 1: second pass max |diff| %.6f, shuffled pass max |diff| %.6f" % (
                        model, c, len(texts), d2, d3), flush=True)
            values = dict(ids=np.array([x["id"] for x in items]), answer_sha1=np.array([x["answer_sha1"] for x in items]),
                          statement_sha1=np.array([SP.sha1(t) for t in stmts]),
                          **{"%s__%s" % (c, k): v for c, r in res.items() for k, v in r.items()})
            meta = dict(model=model, code=cfg["code"], das=cfg["das"], tf_layer=cfg["tf_layer"], conditions=list(conds),
                        batch_size=1, centre_shift_along_E_in_sd=shift,
                        label_map={k: list(map(int, v)) for k, v in mI.items()},
                        items_sha256=SP.sha256_file(str(root / (src + "_items.json"))), n_items=len(items),
                        path_identity_batch8_first=EQ_N, check_b=extra or None)
            SP.save_probe(str(root / (src + "_probe")), values, meta)
            print("%s probe (batch 1): %d statements x %d conditions" % (src, len(items), len(conds)), flush=True)


def check_bc(model):
    """Section 5 item 2 (b) and (c) on the stored files."""
    import numpy as np
    _, meta, _ = SP.read_probe(model, source="tj1_chk")
    b = meta["check_b"]
    okb = all(v["second_pass_max_abs_diff"] == 0 and v["shuffled_pass_max_abs_diff"] == 0 for v in b.values())
    n1 = n1_answers(model)
    doc, _, z = SP.read_probe(model, source="tj1_n1")
    ans = [x for x in doc["items"] if x["answered"]]
    lines = []
    for c in [c for c in ("baseline", "E_a3") if c in SP.conds_of(model, "x")]:
        d = []
        for j, x in enumerate(ans):
            k = (x["split"], x["id"].split("|", 1)[1])
            if n1[k]["sha"] != x["answer_sha1"]:
                raise SystemExit("tj1_n1 item %s" % x["id"])
            d.append(abs(float(z[c + "__margin"][j]) - n1[k]["m"][c]))
        d = np.array(d)
        lines.append("%s n=%d |batch 1 - batch 8|: median %.4f, 95%% %.4f, max %.4f, share exactly 0 %.3f" % (
            c, len(d), np.median(d), np.percentile(d, 95), d.max(), float((d == 0).mean())))
    return okb, "check (b) %s: %s -> %s | check (c) %s" % (model, " ; ".join(
        "%s second %.6f shuffled %.6f" % (c, v["second_pass_max_abs_diff"], v["shuffled_pass_max_abs_diff"]) for c, v in b.items()),
        "OK" if okb else "FAILED", " ; ".join(lines))


# ------------------------------------------------------------------------------------------------------------------ #
def probe_map(model, src):
    doc, _, z = SP.read_probe(model, source=src)
    conds = [c for c in ("baseline", "E_a3") if c in SP.conds_of(model, src)]
    ans = [x for x in doc["items"] if x["answered"]]
    out = {}
    for j, x in enumerate(ans):
        if z["ids"][j] != x["id"]:
            raise ValueError("%s %s: row %d" % (model, src, j))
        out[x["id"]] = dict(st=str(z["statement_sha1"][j]), m={c: float(z[c + "__margin"][j]) for c in conds},
                            neg=x["neg"], sha=x["answer_sha1"], v=x["value"])
    return {x["id"]: x for x in doc["items"]}, out


def load(model, batch1=True):
    """{split: (ids, rows)}; rows[q] = {"S"|"N"|"T": dict(v, a, st, m, neg) }, only the questions with both the S0 and
    the conf_E2 verdict (as plain_honesty_fill.py); also N1 on every question of the list (for the S3 cross-check)."""
    n1 = n1_answers(model)
    if batch1:
        si, sm = probe_map(model, "tj1_s0")
        ci_, cm = probe_map(model, "tj1_conf")
        _, nm = probe_map(model, "tj1_n1")
        if set(nm) != {k[0] + "|" + k[1] for k in n1}:
            raise SystemExit("%s: tj1_n1 does not cover N1's answers" % model)
        for k, x in n1.items():
            y = nm[k[0] + "|" + k[1]]
            if y["sha"] != x["sha"] or y["st"] != x["st"]:
                raise SystemExit("%s: tj1_n1 item %s" % (model, k))
            x["m"] = y["m"]
    else:
        si, sm = probe_map(model, "tj_s0")
        ci_, cm = probe_map(model, "tj_conf")
    out = {}
    for i, name in SPLITS:
        N = native(model, i)
        ids = N["ids"]
        rows, alln = {}, {}
        for k, q in enumerate(ids):
            nv, na = V[N["n1"][0][k]], bool(N["n1"][1][k])
            if N["n1"][2][k]["p"] != q:
                raise SystemExit("%s %s: N1 record order" % (model, name))
            x = n1.get((name, q))
            if na != (x is not None) or (x is not None and x["v"] != nv):
                raise SystemExit("%s %s: N1 of %s differs between S.system and the self-judge items" % (model, name, q))
            Nn = dict(v=nv, a=na, st=x["st"] if x else None, m=x["m"] if x else None, neg=x["neg"] if x else False)
            alln[q] = Nn
            if N["st"].get((q, "baseline_x_E2"), "?") == "?" or N["st1"].get((q, "baseline_conf_E2"), "?") == "?":
                continue
            r = {"N": Nn}
            for tag, items, mp, st, bl, cond in (("S", si, sm, N["st"], N["bl"], "baseline_x_E2"),
                                                 ("T", ci_, cm, N["st1"], N["bl1"], "baseline_conf_E2")):
                it = items[name + "|" + q]
                v, a = V[st[(q, cond)]], not bl[(q, cond)]
                if it["value"] != v or it["answered"] != a:
                    raise SystemExit("%s %s: %s item of %s differs from the verdicts" % (model, name, tag, q))
                p = mp.get(name + "|" + q)
                if a != (p is not None):
                    raise SystemExit("%s %s: %s score of %s missing or extra" % (model, name, tag, q))
                r[tag] = dict(v=v, a=a, st=p["st"] if p else None, m=p["m"] if p else None, neg=it["neg"])
            rows[q] = r
        out[name] = (ids, rows, alln)
    return out


def check3_native():
    """Check 3 from the verdicts alone (no self-judge score needed): S0 / N1 per model, F1 / T2 on the common list."""
    per = {}
    for m in ORDER:
        N = native(m, 1)
        row = {}
        for k, q in enumerate(N["ids"]):
            if N["st"].get((q, "baseline_x_E2"), "?") == "?" or N["st1"].get((q, "baseline_conf_E2"), "?") == "?":
                continue
            s0 = (V[N["st"][(q, "baseline_x_E2")]], not N["bl"][(q, "baseline_x_E2")])
            t = (V[N["st1"][(q, "baseline_conf_E2")]], not N["bl1"][(q, "baseline_conf_E2")])
            n1 = (V[N["n1"][0][k]], bool(N["n1"][1][k]))
            row[q] = dict(S0=s0, N1=n1, F1=s0 if s0[1] else n1, T2=s0 if s0[1] else t)
        per[m] = row
        got = [(sum(r[x][0] for r in row.values()), sum(r[x][1] for r in row.values())) for x in ("S0", "N1")]
        print("check 3 %-12s %d q: S0 %+d (%d) [ref %+d (%d)] | N1 %+d (%d) [ref %+d (%d)] -> %s" % (
            m, len(row), *got[0], *REF_S0[m], *got[1], *REF_N1[m],
            "OK" if got[0] == REF_S0[m] and got[1] == REF_N1[m] else "MISMATCH"))
    common = sorted(set.intersection(*(set(v) for v in per.values())))
    g = {x: (sum(per[m][q][x][0] for m in ORDER for q in common), sum(per[m][q][x][1] for m in ORDER for q in common))
         for x in ("F1", "T2")}
    print("check 3 five models, %d common q: F1 %+d (%d) [ref %+d (%d)] | T2 %+d (%d) [ref %+d (%d)] -> %s" % (
        len(common), *g["F1"], *REF5["F1"], *g["T2"], *REF5["T2"],
        "OK" if len(common) == REF5["n"] and g["F1"] == REF5["F1"] and g["T2"] == REF5["T2"] else "MISMATCH"))


def zdrop(x, c):
    return x["a"] and not x["neg"] and x["m"][c] < 0


def systems(r, c, nv=False):
    """{system: (value, answered, tag of the handed-in answer)} of one question under condition c. nv: on the questions
    both S0 and N1 answer with the same statement, A1 hands in N1's answer (its verdict) instead of S0's (section 5,
    item 4: the range of influence of those questions; TA1 is not touched)."""
    S, N, T = r["S"], r["N"], r["T"]

    def pick(other, ot):
        if S["a"] and not other["a"]:
            return "S"
        if other["a"] and not S["a"]:
            return ot
        if not S["a"]:
            return ot          # neither answered: as F1 / T2 (the second candidate's cell)
        if S["st"] == other["st"]:
            return ot if (nv and ot == "N") else "S"
        return ot if other["m"][c] > S["m"][c] else "S"

    def val(tag):
        # the cell's verdict value is kept even when the cell is blank (plain_honesty_fill.py's convention, which the
        # section 3.3 references follow); only rule Z and the equal-count cut turn a handed-in answer into 0
        return (r[tag]["v"], r[tag]["a"], tag if r[tag]["a"] else None)

    def z(t):
        v, a, tag = t
        return (0, False, None) if tag is not None and zdrop(r[tag], c) else t

    out = {"S0": val("S"), "N1": val("N"), "F1": val("S" if S["a"] else "N"), "T2": val("S" if S["a"] else "T"),
           "A1": val(pick(N, "N")), "TA1": val(pick(T, "T"))}
    for k in ("S0", "N1", "F1", "A1", "TA1"):
        out[k + "z"] = z(out[k])
    return out


def margin(r, tag, c):
    return r[tag]["m"][c]


def equal_n(qs, rows, a, ca, b, cb, nv=False):
    """Per-question values of a (cond ca) and b (cond cb) at equal answer counts: the side with more answers drops its
    handed-in answers from the lowest margin up (of its own condition; ties by position in qs)."""
    va = {q: systems(rows[q], ca, nv)[a] for q in qs}
    vb = {q: systems(rows[q], cb, nv)[b] for q in qs}
    na, nb = sum(v[1] for v in va.values()), sum(v[1] for v in vb.values())

    def cut(vals, cond, k):
        pos = {q: i for i, q in enumerate(qs)}
        cand = sorted([q for q in qs if vals[q][1]], key=lambda q: (margin(rows[q], vals[q][2], cond), pos[q]))
        drop = set(cand[:k])
        return {q: ((0, False, None) if q in drop else vals[q]) for q in qs}
    if na > nb:
        va = cut(va, ca, na - nb)
    elif nb > na:
        vb = cut(vb, cb, nb - na)
    return va, vb, (na, nb)


def comparisons(conds):
    out = []
    for c in conds:
        for b in ("S0", "F1", "N1", "TA1"):
            out.append(("A1[%s]-%s" % (c, b), "A1", c, b, c))
        for b in ("S0z", "F1z", "N1z", "TA1z"):
            out.append(("A1z[%s]-%s[%s]" % (c, b, c), "A1z", c, b, c))
    if "E_a3" in conds:
        out.append(("A1[E_a3]-A1[baseline]", "A1", "E_a3", "A1", "baseline"))
        out.append(("A1z[E_a3]-A1z[baseline]", "A1z", "E_a3", "A1z", "baseline"))
    return out


def block(label, models, data, name, qs_of, conds, ci, keys=None, nv=False, brief=False):
    """Print readings 1-5 for one model (len(models) == 1) or the models together on one common list. keys: only these
    comparisons; brief: no nets / reading 1 (the sensitivity and same-statement lines)."""
    print("  %s" % label)
    if brief:
        return diffs(models, data, name, qs_of, conds, ci, keys, nv, "    ")
    # nets
    for c in conds:
        parts = []
        for k in ("S0", "N1", "F1", "T2", "A1", "TA1", "S0z", "N1z", "F1z", "A1z", "TA1z"):
            tv = sum(systems(data[m][name][1][q], c)[k][0] for m in models for q in qs_of(m))
            ta = sum(systems(data[m][name][1][q], c)[k][1] for m in models for q in qs_of(m))
            parts.append("%s %+d (%d)" % (k, tv, ta))
        print("    [%s] nets (answers): %s" % (c, " | ".join(parts)))
    # reading 1
    for c in conds:
        for other, lab in (("N", "S0 vs N1 (A1)"), ("T", "S0 vs conf_E2 (TA1, side)")):
            n = pr = aS = aO = same = sv = 0
            for m in models:
                for q in qs_of(m):
                    r = data[m][name][1][q]
                    S, O = r["S"], r[other]
                    if S["a"] and O["a"]:
                        if S["st"] == O["st"]:
                            same += 1
                            sv += S["v"] != O["v"]
                            continue
                        if S["v"] * O["v"] == -1:
                            n += 1
                            t = other if O["m"][c] > S["m"][c] else "S"
                            pr += r[t]["v"] == 1
                            aS += S["v"] == 1
                            aO += O["v"] == 1
            print("    reading 1 [%s] %s: one right one wrong %d: self-judge picks right %d (%s) | always S0 right %d | always "
                  "other right %d | (both answered, same statement %d, of which verdicts differ %d)" % (
                      c, lab, n, pr, "%.3f" % (pr / n) if n else "n/a", aS, aO, same, sv))
    diffs(models, data, name, qs_of, conds, ci, keys, nv, "    ")


def diffs(models, data, name, qs_of, conds, ci, keys, nv, ind):
    rows_nat, rows_eq = [], []
    for key, a, ca, b, cb in comparisons(conds):
        if keys is not None and key not in keys:
            continue
        dn, de = {}, {}
        counts = []
        for m in models:
            qs = qs_of(m)
            R = data[m][name][1]
            for q in qs:
                sa, sb = systems(R[q], ca, nv)[a], systems(R[q], cb, nv)[b]
                dn[q] = dn.get(q, 0) + sa[0] - sb[0]
            va, vb, nn = equal_n(qs, R, a, ca, b, cb, nv)
            counts.append(nn)
            for q in qs:
                de[q] = de.get(q, 0) + va[q][0] - vb[q][0]
        qall = qs_of(models[0])
        na, nb = sum(x[0] for x in counts), sum(x[1] for x in counts)
        rows_nat.append("%s %s (answers %d vs %d)" % (key, ci([dn[q] for q in qall]), na, nb))
        rows_eq.append("%s %s (at %d)" % (key, ci([de[q] for q in qall]), sum(min(x) for x in counts)))
    print(ind + "natural answer counts:")
    for x in rows_nat:
        print(ind + "  " + x)
    print(ind + "reading 5, equal answer counts (per model, the side with more answers drops its lowest-margin answers):")
    for x in rows_eq:
        print(ind + "  " + x)


def read():
    import iti_table as A
    import selfjudge_stage1 as ST
    ci = A.ps.ci
    print("# selfjudge two answers (work queue 2026-11-22_queue_j, rules 22:16 b7f1861, section 5 22:45 1ef3c38); TruthfulQA, "
          "DeepSeek Pro verdicts; MAIN scores = batch size 1 (tj1_*)")
    print("# check 4 (section 5 item 2):")
    ok = True
    for m in ORDER:
        _, line = check4(m, quiet=True)
        print("#   (a) / old rule: " + line)
        o, line = check_bc(m)
        print("#   " + line)
        ok &= o
    if not ok:
        raise SystemExit("CHECK (b) FAILED: stopping before any reading")
    data = {m: load(m) for m in ORDER}
    data8 = {m: load(m, batch1=False) for m in ORDER}
    print("# check 3 (test): references")
    for m in ORDER:
        ids, rows, _ = data[m]["test"]
        got = {k: (sum(systems(rows[q], "baseline")[k][0] for q in rows), sum(systems(rows[q], "baseline")[k][1] for q in rows))
               for k in ("S0", "N1")}
        hit = got["S0"] == REF_S0[m] and got["N1"] == REF_N1[m]
        print("#   %-12s %d q: S0 %+d (%d) [ref %+d (%d)] | N1 %+d (%d) [ref %+d (%d)] -> %s" % (
            m, len(rows), *got["S0"], *REF_S0[m], *got["N1"], *REF_N1[m], "OK" if hit else "MISMATCH"))
        ok &= hit
    common = sorted(set.intersection(*(set(data[m]["test"][1]) for m in ORDER)))
    g5 = {k: (sum(systems(data[m]["test"][1][q], "baseline")[k][0] for m in ORDER for q in common),
              sum(systems(data[m]["test"][1][q], "baseline")[k][1] for m in ORDER for q in common)) for k in ("F1", "T2")}
    hit = len(common) == REF5["n"] and g5["F1"] == REF5["F1"] and g5["T2"] == REF5["T2"]
    print("#   five models, %d common q: F1 %+d (%d) [ref %+d (%d)] | T2 %+d (%d) [ref %+d (%d)] -> %s" % (
        len(common), *g5["F1"], *REF5["F1"], *g5["T2"], *REF5["T2"], "OK" if hit else "MISMATCH"))
    ok &= hit
    if not ok:
        raise SystemExit("CHECK 3 MISMATCH: stopping before any reading")
    # cross-check on the batch scores: N1z on every test question = selfjudge_stage1.computeZ's S3 (rule Z, threshold 0)
    Z = ST.load_all(ORDER)
    for m in ORDER:
        ids, rows, alln = data8[m]["test"]
        for c in [c for c in ("baseline", "E_a3") if c in SP.conds_of(m, "x")]:
            mine = sum(0 if (x["a"] and zdrop(x, c)) else x["v"] for x in alln.values())
            zi, ztq, _ = Z[m]["tq"]
            if list(zi) != list(ids):
                raise SystemExit("%s: computeZ list differs" % m)
            _, sm = ST.computeZ(zi, ztq[c], 0.0, None, True)
            print("#   N1z cross-check (batch scores) %s [%s]: here %+d, selfjudge_stage1 S3 %+d -> %s" % (
                m, c, mine, sm["nets"]["S3"], "OK" if mine == sm["nets"]["S3"] else "MISMATCH"))
            if mine != sm["nets"]["S3"]:
                raise SystemExit("N1z differs from S3")
    print("# all checks passed")

    def sens_keys(conds):
        return ["A1[%s]-F1" % c for c in conds] + ["A1z[%s]-S0z[%s]" % (c, c) for c in conds]

    def all_blocks(label, models, name, qs_of, conds):
        block(label, models, data, name, qs_of, conds, ci)
        print("    SENSITIVITY, the existing batch scores (tj_s0 / tj_conf / probe / tq_release):")
        block("", models, data8, name, qs_of, conds, ci, keys=sens_keys(conds), brief=True)
        print("    SAME STATEMENT, DIFFERENT VERDICTS: A1 / A1z hand in N1's answer (and verdict) on the same-statement questions:")
        keys = [k for k, a, _, b, _ in comparisons(conds) if a in ("A1", "A1z")]
        block("", models, data, name, qs_of, conds, ci, keys=keys, nv=True, brief=True)

    for _, name in SPLITS:
        print("=" * 110)
        print("TruthfulQA %s" % name)
        for m in ORDER:
            conds = [c for c in ("baseline", "E_a3") if c in SP.conds_of(m, "x")]
            qs = [q for q in data[m][name][0] if q in data[m][name][1]]
            all_blocks("%s: %d questions" % (m, len(qs)), (m,), name, lambda _m, qs=qs: qs, conds)
        c5 = sorted(set.intersection(*(set(data[m][name][1]) for m in ORDER)))
        all_blocks("five models, %d common questions (no write)" % len(c5), ORDER, name, lambda _m: c5, ["baseline"])
        c3 = sorted(set.intersection(*(set(data[m][name][1]) for m in WRITE)))
        all_blocks("three models (%s), %d common questions" % (", ".join(WRITE), len(c3)), WRITE, name, lambda _m: c3,
                   ["baseline", "E_a3"])


def main():
    a = sys.argv[1:]
    if len(a) == 2 and a[0] == "export" and a[1] in ORDER:
        export(a[1])
    elif len(a) == 3 and a[0] == "score" and a[1] in ORDER:
        score(a[1], [x for x in a[2].split(",") if x])
    elif len(a) == 2 and a[0] == "export1" and a[1] in ORDER:
        export1(a[1])
    elif len(a) == 2 and a[0] == "checkbc" and a[1] in ORDER:
        ok, line = check_bc(a[1])
        print(line)
        if not ok:
            raise SystemExit(1)
    elif len(a) == 2 and a[0] == "export-ctx" and a[1] in ORDER:
        export_ctx(a[1])
    elif len(a) == 2 and a[0] == "check4" and a[1] in ORDER:
        ok, _ = check4(a[1])
        if not ok:
            raise SystemExit(1)
    elif a == ["check3"]:
        check3_native()
    elif a == ["read"]:
        read()
    else:
        raise SystemExit("usage: selfjudge_two_answers.py export MODEL | score MODEL SRC[,SRC] | check4 MODEL | read")


if __name__ == "__main__":
    main()
