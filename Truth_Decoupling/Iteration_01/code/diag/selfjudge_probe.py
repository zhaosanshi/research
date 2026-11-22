#!/usr/bin/env python3
# Iteration_01 exploration (Claude Opus 5.5, 2026-11-22), work queue discussion/work_queue/2026-11-22_queue_d.md,
# section 1. Not reviewed by GPT; the readings are exploration readings only. No judge call, no key, no generation.
#
# Question: on the TruthfulQA questions that single E (main-table setting, plain prompt, write rule "prompt";
# Llama-3.1-8B layer 16 strength 3.5, Qwen3.5-9B layer 15 strength 2) actually answers, does the model's own
# true/false judgement of its answer, with the true/false E of E45 written into that judging pass, separate the wrong
# answers from the right ones better than the same judgement without the write, and well enough that "judged false ->
# no answer" adds net score (right +1, no answer 0, wrong -1)?
#
# Steps (subcommands):
#   export MODEL   (host, no model) the main table's tune / test question lists (exactly as diag/tq_same_n_decomp.py:
#                  MP.prompt_split, restricted to the main table's source lists), single E's judged answer text and its
#                  DeepSeek Pro state per question (S.system(sp, tx, ("E", s))), N1's state at the main table's N1
#                  choice (ML.n1_pick on tune) and whether N1 keeps E's answer; every E text is checked against the
#                  generation file of that pass (sha1 of the stripped text). Writes
#                  Iteration_01/newsets/selfjudge/<MODEL>/items.json (question texts live there only, never in the repo).
#   score MODEL [--regress-limit N] [--regress-only]
#                  (container, GPU) E45's code path, imported unchanged from the code copy E45 ran with (its file hashes
#                  must equal the ones the E45 json records): the Unknown-allowed statement prompt
#                  (common.tf_prompt(..., allow_idk=True)), the label map of E45's first test statement, the true/false E
#                  = DAS tf-ce-bidir at E45's E layer (load_das_direction, sign aligned to the forced-prompt mean
#                  difference), "amplify" h <- h + a w w^T (h - c_I), c_I = (mu_true + mu_false) / 2 of the 192
#                  estimation statements under the Unknown-allowed prompt at that layer. Four conditions: baseline (no
#                  write), E_a1, E_a2, E_a3 (E45's three strengths). Regression first: the four conditions on E45's own
#                  1,442 test statements (same order, same batch size) against E45's npz; any |margin difference| > 0.02
#                  stops the run. Then the probe: statement = 'The answer to the question "{q}" is: {a}' built exactly
#                  as data_adapters/truthfulqa_tf.py builds it from (question, best answer): q = question stripped, "?"
#                  stripped and re-added; a = answer stripped, trailing "." stripped and one "." added (checked to
#                  reproduce all 1,634 E45 statements from their meta question / answer). Writes probe.npz / probe.json.
#   table          (host, no model) R1-R4 below for the models that have both files.
# Readings per answered statement and condition: margin = logp(True) - logp(False) (label log-prob masses, E45's
# label_logprobs_from_logits), gate = max(logp True, logp False) - logp Unknown, choice = three-way arg-max.
#   R1  AUROC of margin for "E's answer right (Pro R) vs wrong (Pro W)" (H = Pro "said nothing" and X = Pro abstain are
#       left out), per condition, tune and test separately, 95% interval by resampling questions (4000, seed
#       20261122); "written minus not written" paired on the same resamples.
#   R2  per condition one threshold chosen on TUNE only (function choose_thresholds): an answer with margin < t becomes
#       no answer; t = the tune-net maximiser among -inf, midpoints of consecutive distinct tune margins, +inf; ties go
#       to the smaller t (fewer answers dropped; "drop nothing" allowed). Then read once on TEST (function read_test):
#       dropped answers (right / wrong / other), net against single E and against N1 (the same threshold on the E
#       answers N1 keeps), paired over the test questions of the main list (policy_select2.ci, 4000 resamples).
#   R3  at each written condition's test drop count k: drop the k lowest NO-WRITE margins instead (ties by question
#       order) and compare; the mean answer log-prob of single E's pass is not stored (generation rows carry no
#       log-probs), so that comparator is not available.
#   R4  the same split by whether the answer opens with a negated premise, regex NEG_RE below (case-insensitive,
#       matched at the start after quotes / asterisks / spaces):
#         ^[\s"'*]*(no|none|nothing|nobody|never|neither|not|there(\s+(is|are|was|were)\s+no|['\u2019]s\s+no|
#         \s+(isn't|aren't|wasn't|weren't)))\b
# usage (host, from /data/sanshizhao/Truth_Decoupling):
#   python3 -B Iteration_01/diag/selfjudge_probe.py export llama3.1-8b
#   python3 -B Iteration_01/diag/selfjudge_probe.py table
#   score: launchers/explore_20261120/selfjudge_1122.sh (container, GPU 1)
# step 2 (work queue section 4; see the block above export_qaqa):
#   docker exec -u 1000:1001 -w /workspace/Truth_Decoupling truth_dec python3 -B Iteration_01/diag/selfjudge_probe.py export-qaqa MODEL
#   launchers/explore_20261120/selfjudge_1122.sh MODEL --qaqa
# queue e stage one: launchers/explore_20261120/selfjudge_1122.sh MODEL --sources SRC[,SRC] (SRC_items.json written by
#   diag/selfjudge_stage1.py; --qaqa is the same as --sources qaqa)
#   python3 -B Iteration_01/diag/selfjudge_probe.py step2
import argparse
import hashlib
import json
import os
import re
import sys

OUT = "Iteration_01/newsets/selfjudge"
NEG_RE = re.compile(r"^[\s\"'*]*(no|none|nothing|nobody|never|neither|not|there(\s+(is|are|was|were)\s+no|['\u2019]s\s+no|"
                    r"\s+(isn't|aren't|wasn't|weren't)))\b", re.IGNORECASE)
TEMPLATE = 'The answer to the question "{q}" is: {a}'
CONDS = ("baseline", "E_a1", "E_a2", "E_a3")
CONDS2 = ("baseline", "E_a3")          # step 2 (work queue section 4): no write and true/false E at strength 3 only
QAQA_BANK = "Iteration_01/newsets/bank/qaqa_full_bank_20261122.jsonl"
QAQA_PILOT = "Iteration_01/newsets/splits/qaqa_pilot_20261122.json"
QAQA_WORKS = ["test_full570"]
QTYPES = ("no_failure", "failure", "unverifiable")
ALPHAS = (1.0, 2.0, 3.0)
TOL = 0.02
V = {"R": 1, "W": -1, "X": 0, "H": 0}
MODELS = {
    "llama3.1-8b": dict(display="Llama-3.1-8B", layer=16, s="3-5", code="code_e45_20261112",
                        das="results_das_tfce/E1/llama3.1-8b", tf_layer=11, ref="results_e45_20261112/E45/llama3.1-8b/e45_v1"),
    "qwen3.5-9b": dict(display="Qwen3.5-9B", layer=15, s="2", code="code_newmodels_20261112",
                       das="results_newmodels_20261112/E1/qwen3.5-9b", tf_layer=31,
                       ref="results_e45_newmodels_20261119/E45/qwen3.5-9b/e45_nm_Cp"),
}
ORDER = ("llama3.1-8b", "qwen3.5-9b")


def sha1(text):
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def statement(question, answer):
    """data_adapters/truthfulqa_tf.py: q = text.strip().rstrip('?').strip() + '?'; a = best.strip(), .rstrip('.') + '.'."""
    q = question.strip().rstrip("?").strip() + "?"
    return TEMPLATE.format(q=q, a=answer.strip().rstrip(".") + ".")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------------------------------------------------------ #
# export (host)
# ------------------------------------------------------------------------------------------------------------------ #
def main_lists(A, model):
    """The main table's tune and test splits of single E at "prompt", as diag/tq_same_n_decomp.one builds them."""
    MP, ML = A.MP, A.ML
    display, layer, s = A.KEY[model]
    cfg = MODELS[model]
    if (display, layer, s) != (cfg["display"], cfg["layer"], cfg["s"]):
        raise SystemExit("%s: main-table setting %s differs from the work queue's %s" % (model, (display, layer, s), cfg))
    tune, err = MP.prompt_split(display, layer, 0, s)
    test, err2 = MP.prompt_split(display, layer, 1, s)
    if tune is None or test is None:
        raise SystemExit("%s: %s %s" % (model, err, err2))
    src = [x for x in ML.sources(display)[2] if x[0] == layer]
    if len(src) != 1:
        raise SystemExit("%s: layer %s is not one of the main table's sources" % (model, layer))
    ct = [q for q in tune[3] if q in set(src[0][1].qs)]
    cx = [q for q in test[3] if q in set(src[0][3].qs)]
    if ct != tune[3]:
        tune, err = MP.prompt_split(display, layer, 0, s, qs=ct)
    if cx != test[3]:
        test, err = MP.prompt_split(display, layer, 1, s, qs=cx)
    if tune is None or test is None or tune[3] != ct or test[3] != cx:
        raise SystemExit("%s: the restricted splits are not the main table's" % model)
    return display, layer, s, tune, test


def export(model):
    sys.path.insert(0, "Iteration_01/diag")
    import iti_table as A
    MP, ML, S = A.MP, A.ML, A.S
    display, layer, s, tune, test = main_lists(A, model)
    n1cfg = ML.n1_pick(tune[0], tune[1], s)
    tf = {}
    for split in ("calib", "dev", "test"):
        with open("code/data/processed/truthfulqa_tf/%s.jsonl" % split, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                tf.setdefault(r["pair_id"], (split, r["meta"]["question"]))
    items, info = [], {}
    for i, (split, (sp, tx, _, ids)) in enumerate((("tune", tune), ("test", test))):
        es, ea, er = S.system(sp, tx, ("E", s))
        ns, na, nr = S.system(sp, tx, n1cfg)
        econd = S.shell(layer, 0).cE(s)
        _, run, _ = MP.usual(display, i)
        w1 = MP.work(run, split, "posprompt" + "l%d" % layer)
        qtext = {}
        for line in open(w1 + "/judge2/cells.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if qtext.setdefault(r["pair_id"], r["question"]) != r["question"]:
                raise SystemExit("%s: two question texts for %s" % (w1, r["pair_id"]))
        gpath = "%s/eval_extra/gen/x--E%d-w-p%s-at-prompt_%s.json" % (run, layer, s, split)
        g = read_json(gpath)
        if g.get("completed") is not True or g.get("partial") or g.get("split") != split or g.get("positions") != "prompt":
            raise SystemExit("%s is not the complete prompt-rule generation of this split" % gpath)
        gtext = {r["pair_id"]: r["text"] for r in g["rows"]}
        n_ans = 0
        for k, q in enumerate(ids):
            if er[k]["p"] != q or nr[k]["p"] != q or er[k]["cond"] != econd:
                raise SystemExit("%s %s: record order / condition differs at %s" % (model, split, q))
            text = er[k]["text"]
            if text is None or sha1(text) != sha1(gtext[q]):
                raise SystemExit("%s %s: the judged E text of %s is not the generation file's" % (model, split, q))
            keeps = bool(ea[k]) and nr[k]["cond"] == econd
            if ea[k] and (not keeps or ns[k] != es[k] or not na[k]):
                raise SystemExit("%s %s: N1 does not keep E's answer of %s unchanged" % (model, split, q))
            tsplit, tq = tf.get(q, (None, None))
            if tq is not None and tq.strip().rstrip("?").strip() != qtext[q].strip().rstrip("?").strip():
                raise SystemExit("%s: question text of %s differs from the true/false data" % (model, q))
            n_ans += bool(ea[k])
            items.append(dict(id=q, split=split, question=qtext[q], answer=text, answer_sha1=sha1(text), state=es[k],
                              answered=bool(ea[k]), n1_state=ns[k], n1_answered=bool(na[k]), n1_keeps_e=keeps,
                              neg=bool(NEG_RE.match(text)), tf_split=tsplit))
        info[split] = dict(n=len(ids), e_answered=n_ans, work_dir=w1, generation=gpath, generation_sha256=sha256_file(gpath))
    tune_ids = {x["id"] for x in items if x["split"] == "tune"}
    if tune_ids & {x["id"] for x in items if x["split"] == "test"}:
        raise SystemExit("tune and test share questions")
    out_dir = os.path.join(OUT, model)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "items.json")
    if os.path.exists(path):
        raise SystemExit("%s exists" % path)
    doc = dict(model=model, display=display, e_layer=layer, e_strength=s, e_condition=S.shell(layer, 0).cE(s),
               n1_choice=list(n1cfg), splits=info, neg_re=NEG_RE.pattern, items=items)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=0)
    for split in ("tune", "test"):
        xs = [x for x in items if x["split"] == split]
        a = [x for x in xs if x["answered"]]
        print("%s %s: %d questions, E answered %d (R %d, W %d, H %d, X %d), negated-premise opening %d; in the true/false "
              "estimation set (calib+dev) %d, in true/false test %d, not in the true/false data %d" % (
                  model, split, len(xs), len(a), *[sum(x["state"] == c for x in a) for c in "RWHX"], sum(x["neg"] for x in a),
                  sum(x["tf_split"] in ("calib", "dev") for x in a), sum(x["tf_split"] == "test" for x in a),
                  sum(x["tf_split"] is None for x in a)))
    print("N1 choice %s; written %s" % (n1cfg, path))


# ------------------------------------------------------------------------------------------------------------------ #
# score (container, GPU)
# ------------------------------------------------------------------------------------------------------------------ #
def score(model, regress_limit, regress_only, n_show, sources=()):
    import numpy as np
    cfg = MODELS[model]
    code = os.path.abspath(cfg["code"])
    sys.path[:0] = [os.path.join(code, "scripts"), code]
    import e45_tf_idk_ec as E45                                                       # noqa: F401  (constants only)
    from common import load_split, tf_prompt
    from core.config import load_experiment, load_model_config
    from core.directions import load_das_direction, mean_diff_direction
    from core.intervene import SteerSpec, next_token_logits
    from core.io_utils import set_seed
    from core.model_io import capture_last_token, load_model
    from core.readout import build_label_token_map, decide_label, label_logprobs_from_logits

    ref = read_json(cfg["ref"] + ".json")
    refz = np.load(cfg["ref"] + ".npz")
    if not (ref.get("completed") and ref["model"] == model and ref["e_layer"] == cfg["tf_layer"] and ref["wording"] == "pik"
            and ref["das"]["variant"] == "tf-ce-bidir" and ref["das"]["dir"].rstrip("/").endswith(cfg["das"])
            and list(ref["alphas_E"]) == list(ALPHAS) == list(E45.ALPHAS_E) and ref["limit"] is None):
        raise SystemExit("%s: the E45 reference run is not the expected one" % cfg["ref"])
    for f, h in ref["code_sha256"].items():
        if sha256_file(os.path.join(code, "scripts", f)) != h:
            raise SystemExit("%s: %s differs from the file E45 ran with" % (code, f))
    exp = load_experiment("E2b_tqa")
    mcfg = load_model_config(model)
    prompts = exp["prompts"]
    st = prompts["statement"]
    set_seed(exp.get("seed", 42))
    bs = exp.get("batch_size", 8)
    recs = []
    for ds in exp["eval_datasets"]:
        recs += load_split(ds, exp.get("eval_split", "test"))
    est = []
    for ds in exp["eval_datasets"]:
        est += load_split(ds, "calib") + load_split(ds, "dev")
    if [r["id"] for r in recs] != [str(x) for x in refz["ids"]]:
        raise SystemExit("the E45 statements are not in the order of the reference npz")
    bad = [r["id"] for r in recs + est if statement(r["meta"]["question"], r["meta"]["answer"]) != r["text"]]
    if bad:
        raise SystemExit("statement() does not reproduce %d E45 statements, e.g. %s" % (len(bad), bad[:3]))
    print("statement(): reproduces all %d E45 statements (test %d + estimation %d)" % (len(recs) + len(est), len(recs), len(est)), flush=True)
    y_est = np.array([bool(r["label"]) for r in est])
    bundle = load_model(mcfg)
    EL = cfg["tf_layer"]
    tI = [tf_prompt(bundle, prompts, r["text"], allow_idk=True) for r in recs]
    mI = build_label_token_map(bundle.tokenizer, tI[0], st["label_variants"])
    if set(mI) != {"True", "False", "Abstain"}:
        raise SystemExit("label map %s" % sorted(mI))
    eB = [tf_prompt(bundle, prompts, r["text"], allow_idk=False) for r in est]
    eI = [tf_prompt(bundle, prompts, r["text"], allow_idk=True) for r in est]
    aB, _ = capture_last_token(bundle, eB, [EL], components=("resid",), batch_size=bs)
    aI, _ = capture_last_token(bundle, eI, [EL], components=("resid",), batch_size=bs)
    HB = aB[("resid", EL)].numpy().astype(np.float64)
    HI = aI[("resid", EL)].numpy().astype(np.float64)
    wMD, _ = mean_diff_direction(HB[y_est], HB[~y_est])
    wE, das_meta = load_das_direction(cfg["das"], "E", EL, variant="tf-ce-bidir", align_to=wMD)
    if wE is None:
        raise SystemExit("no DAS direction at layer %d in %s" % (EL, cfg["das"]))
    cI = (HI[y_est].mean(0) + HI[~y_est].mean(0)) / 2
    proj = HI @ wE
    shift = float(np.dot(cI - (HB[y_est].mean(0) + HB[~y_est].mean(0)) / 2, wE) / (proj.std() + 1e-12))
    print("true/false E: layer %d, centre shift along E %+.6f sd (E45 json %+.6f)" % (
        EL, shift, ref["E_diagnostics"]["centre_shift_along_E_in_sd"]), flush=True)
    specs = {"baseline": None}
    for a in ALPHAS:
        specs["E_a%g" % a] = [SteerSpec(wE, EL, mode="amplify", alpha=a, center=cI)]

    def run(texts, sp):
        out = []
        for i in range(0, len(texts), bs):
            out.append(next_token_logits(bundle, bundle.encode(texts[i:i + bs]), sp).numpy())
        lg = np.concatenate(out)
        lp = [label_logprobs_from_logits(x, mI, use_abstain=True) for x in lg]
        pred = np.array([decide_label(d) for d in lp])
        lpT = np.array([d["True"] for d in lp], dtype=np.float64)
        lpF = np.array([d["False"] for d in lp], dtype=np.float64)
        lpU = np.array([d["Abstain"] for d in lp], dtype=np.float64)
        gate = np.maximum(lpT, lpF) - lpU
        if not np.array_equal(pred != "Abstain", gate >= 0):
            raise RuntimeError("three-way arg-max and the gate disagree")
        return dict(margin=lpT - lpF, gate=gate, choice=pred, lpT=lpT, lpF=lpF, lpU=lpU)

    for src in sources:
        # ---- step 2 (work queue d section 4, step 3) and queue e stage one: answers of another source (src_items.json:
        # qaqa = (QA)^2 single E; tq_forced / tq_release / qq_forced / qq_release from diag/selfjudge_stage1.py), the
        # same code path, no regression (step 1's passed), conditions CONDS2 ------------------------------------------ #
        doc = read_json(os.path.join(OUT, model, src + "_items.json"))
        items = [x for x in doc["items"] if x["answered"]]
        for x in items:
            if sha1(x["answer"]) != x["answer_sha1"]:
                raise SystemExit("%s_items.json: answer text of %s changed" % (src, x["id"]))
        stmts = [statement(x["question"], x["answer"]) for x in items]
        texts = [tf_prompt(bundle, prompts, t, allow_idk=True) for t in stmts]
        if build_label_token_map(bundle.tokenizer, texts[0], st["label_variants"]) != mI:
            raise SystemExit("the label map differs on the %s prompts" % src)
        res = {c: run(texts, specs[c]) for c in CONDS2}
        stem = os.path.join(OUT, model, src + "_probe")
        if os.path.exists(stem + ".npz"):
            raise SystemExit("%s.npz exists" % stem)
        np.savez(stem + ".npz", ids=np.array([x["id"] for x in items]), answer_sha1=np.array([x["answer_sha1"] for x in items]),
                 statement_sha1=np.array([sha1(t) for t in stmts]),
                 **{"%s__%s" % (c, k): v for c, r in res.items() for k, v in r.items()})
        rng = np.random.default_rng(20261122)
        show = sorted(rng.choice(len(items), min(n_show, len(items)), replace=False).tolist())
        meta = dict(model=model, code=cfg["code"], das=cfg["das"], tf_layer=EL, conditions=list(CONDS2),
                    centre_shift_along_E_in_sd=shift, label_map={k: list(map(int, v)) for k, v in mI.items()},
                    items_sha256=sha256_file(os.path.join(OUT, model, src + "_items.json")), n_items=len(items),
                    n_answers_with_newline=sum("\n" in x["answer"] for x in items),
                    shown_probe_prompts=[dict(id=items[k]["id"], value=items[k]["value"], prompt=texts[k]) for k in show])
        with open(stem + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        print("%s probe: %d answered statements x %d conditions -> %s.npz" % (src, len(items), len(CONDS2), stem), flush=True)
    if sources:
        return

    # ---- regression against E45's npz ----------------------------------------------------------------------------- #
    n = regress_limit or len(recs)
    half_a = np.asarray(refz["half_a"])[:n]
    reg = {}
    for c in CONDS:
        r = run(tI[:n], specs[c])
        d = np.abs(r["margin"] - refz[c + "__margin"][:n])
        dg = np.abs(r["gate"] - refz[c + "__gate"][:n])
        flips = int((((r["choice"] != "Abstain")) != np.asarray(refz[c + "__answer"][:n]).astype(bool)).sum())
        reg[c] = dict(n=int(n), n_test=int((~half_a).sum()), max_abs_margin_diff=float(d.max()),
                      max_abs_margin_diff_test=float(d[~half_a].max()) if (~half_a).any() else None,
                      mean_abs_margin_diff=float(d.mean()), max_abs_gate_diff=float(dg.max()), answer_flips=flips)
        print("regression %-9s %4d statements (test %4d): max |margin diff| %.5f (test %.5f), mean %.6f, max |gate diff| %.5f, "
              "answer flips %d" % (c, n, reg[c]["n_test"], reg[c]["max_abs_margin_diff"], reg[c]["max_abs_margin_diff_test"] or 0.0,
                                   reg[c]["mean_abs_margin_diff"], reg[c]["max_abs_gate_diff"], flips), flush=True)
    if any(v["max_abs_margin_diff"] > TOL for v in reg.values()):
        raise SystemExit("REGRESSION FAILED: some margin differs from E45 by more than %.2f" % TOL)
    print("REGRESSION OK (%d statements, tolerance %.2f)" % (n, TOL), flush=True)
    if regress_only:
        return

    # ---- the probe -------------------------------------------------------------------------------------------------- #
    doc = read_json(os.path.join(OUT, model, "items.json"))
    items = [x for x in doc["items"] if x["answered"]]
    for x in items:
        if sha1(x["answer"]) != x["answer_sha1"]:
            raise SystemExit("items.json: answer text of %s changed" % x["id"])
    stmts = [statement(x["question"], x["answer"]) for x in items]
    texts = [tf_prompt(bundle, prompts, t, allow_idk=True) for t in stmts]
    if build_label_token_map(bundle.tokenizer, texts[0], st["label_variants"]) != mI:
        raise SystemExit("the label map differs on the probe prompts")
    res = {c: run(texts, specs[c]) for c in CONDS}
    out_dir = os.path.join(OUT, model)
    stem = os.path.join(out_dir, "probe")
    if os.path.exists(stem + ".npz"):
        raise SystemExit("%s.npz exists" % stem)
    np.savez(stem + ".npz", ids=np.array([x["id"] for x in items]), split=np.array([x["split"] for x in items]),
             answer_sha1=np.array([x["answer_sha1"] for x in items]),
             statement_sha1=np.array([sha1(t) for t in stmts]),
             **{"%s__%s" % (c, k): v for c, r in res.items() for k, v in r.items()})
    rng = np.random.default_rng(20261122)
    test_idx = [i for i, x in enumerate(items) if x["split"] == "test"]
    show = sorted(rng.choice(test_idx, min(n_show, len(test_idx)), replace=False).tolist())
    ref_show = [i for i, r in enumerate(recs) if r["pair_id"] in {items[k]["id"] for k in show}]
    meta = dict(model=model, code=cfg["code"], das=cfg["das"], tf_layer=EL, conditions=list(CONDS), alphas=list(ALPHAS),
                das_meta={k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in (das_meta or {}).items()},
                centre_shift_along_E_in_sd=shift, label_map={k: list(map(int, v)) for k, v in mI.items()},
                regression=reg, items_sha256=sha256_file(os.path.join(OUT, model, "items.json")), n_items=len(items),
                n_answers_with_newline=sum("\n" in x["answer"] for x in items),
                shown_probe_prompts=[dict(id=items[k]["id"], state=items[k]["state"], prompt=texts[k]) for k in show],
                shown_reference_prompts=[dict(id=recs[k]["id"], prompt=tI[k]) for k in ref_show[:4]])
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print("probe: %d answered statements x %d conditions -> %s.npz" % (len(items), len(CONDS), stem), flush=True)


# ------------------------------------------------------------------------------------------------------------------ #
# table (host)
# ------------------------------------------------------------------------------------------------------------------ #
def auroc(score_, y):
    import numpy as np
    s = np.asarray(score_, dtype=np.float64)
    y = np.asarray(y, dtype=bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def choose_thresholds(rows, conds=CONDS):
    """TUNE ONLY. rows: [(margin by condition, value)] of the answered tune statements. Returns {cond: (t, drops, gain)}."""
    out = {}
    for c in conds:
        m = [r[0][c] for r in rows]
        v = [r[1] for r in rows]
        u = sorted(set(m))
        cands = [float("-inf")] + [(a + b) / 2.0 for a, b in zip(u[:-1], u[1:])] + [float("inf")]
        best = None
        for t in cands:                      # ascending: on ties the first (fewest dropped) wins
            gain = -sum(vi for mi, vi in zip(m, v) if mi < t)
            if best is None or gain > best[2]:
                best = (t, sum(mi < t for mi in m), gain)
        out[c] = best
    return out


def read_test(items, mg, thr, ci):
    """TEST, read once at the tune thresholds. items: every test question of the main list (answered or not)."""
    out = {}
    for c in CONDS:
        t = thr[c][0]
        drop = [x["answered"] and mg[x["id"]][c] < t for x in items]
        d_e = [-V[x["state"]] if dr else 0 for x, dr in zip(items, drop)]
        d_n1 = [-V[x["n1_state"]] if dr and x["n1_keeps_e"] else 0 for x, dr in zip(items, drop)]
        out[c] = dict(t=t, drop=drop, k=sum(drop), R=sum(dr and x["state"] == "R" for x, dr in zip(items, drop)),
                      W=sum(dr and x["state"] == "W" for x, dr in zip(items, drop)),
                      O=sum(dr and x["state"] in "HX" for x, dr in zip(items, drop)), d_e=d_e, d_n1=d_n1,
                      ci_e=ci(d_e), ci_n1=ci(d_n1))
    return out


def lowest_k(items, mg, cond, k):
    """Drop the k answered test statements with the lowest margin under cond (ties by question order)."""
    cand = sorted([i for i, x in enumerate(items) if x["answered"]], key=lambda i: (mg[items[i]["id"]][cond], i))
    keep = set(cand[:k])
    return [-V[x["state"]] if i in keep else 0 for i, x in enumerate(items)]


def table():
    import numpy as np
    sys.path.insert(0, "Iteration_01/diag")
    import policy_select2 as ps
    ci = ps.ci
    for model in ORDER:
        d = os.path.join(OUT, model)
        if not (os.path.exists(d + "/items.json") and os.path.exists(d + "/probe.npz")):
            print("%s: no probe yet" % model)
            continue
        doc = read_json(d + "/items.json")
        meta = read_json(d + "/probe.json")
        if meta["items_sha256"] != sha256_file(d + "/items.json"):
            raise SystemExit("%s: probe.json was made from another items.json" % model)
        z = np.load(d + "/probe.npz")
        items = doc["items"]
        ans = [x for x in items if x["answered"]]
        if [str(i) for i in z["ids"]] != [x["id"] for x in ans] or [str(a) for a in z["answer_sha1"]] != [x["answer_sha1"] for x in ans] \
                or [str(s) for s in z["split"]] != [x["split"] for x in ans]:
            raise SystemExit("%s: probe rows are not the answered items" % model)
        if any(sha1(statement(x["question"], x["answer"])) != str(h) for x, h in zip(ans, z["statement_sha1"])):
            raise SystemExit("%s: statement texts differ" % model)
        key = [(x["split"], x["id"]) for x in ans]
        if len(set(key)) != len(key):
            raise SystemExit("%s: duplicate rows" % model)
        mg = {}
        for j, x in enumerate(ans):
            mg[x["split"] + "|" + x["id"]] = {c: float(z[c + "__margin"][j]) for c in CONDS}
        print("=" * 100)
        print("%s | single E %s (%s) | N1 choice %s | true/false E layer %d, strengths %s | regression: %s" % (
            model, doc["e_condition"], doc["e_strength"], doc["n1_choice"], meta["tf_layer"], meta["alphas"],
            ", ".join("%s max %.4f" % (c, meta["regression"][c]["max_abs_margin_diff"]) for c in CONDS)))
        sp = {s: [x for x in items if x["split"] == s] for s in ("tune", "test")}
        for s in ("tune", "test"):
            a = [x for x in sp[s] if x["answered"]]
            print("  %s: %d questions; E answered %d: right %d, wrong %d, said nothing (H) %d, Pro abstain (X) %d; "
                  "negated-premise opening %d; answers with a newline %d" % (
                      s, len(sp[s]), len(a), *[sum(x["state"] == c for x in a) for c in "RWHX"], sum(x["neg"] for x in a),
                      sum("\n" in x["answer"] for x in a)))
        # choice distribution
        for s in ("tune", "test"):
            idx = [j for j, x in enumerate(ans) if x["split"] == s]
            print("  %s self-judgement choice (True / False / Unknown): %s" % (s, " | ".join(
                "%s %d/%d/%d" % (c, *[int((z[c + "__choice"][idx] == lab).sum()) for lab in ("True", "False", "Abstain")]) for c in CONDS)))
        # R1
        print("  R1 AUROC of margin, right vs wrong (H and X left out); [95% by questions, 4000, seed 20261122]")
        rng = np.random.default_rng(20261122)
        for s in ("tune", "test"):
            rw = [x for x in sp[s] if x["answered"] and x["state"] in "RW"]
            y = np.array([x["state"] == "R" for x in rw])
            M = {c: np.array([mg[s + "|" + x["id"]][c] for x in rw]) for c in CONDS}
            point = {c: auroc(M[c], y) for c in CONDS}
            boots = {c: [] for c in CONDS}
            for _ in range(4000):
                ix = rng.integers(0, len(rw), len(rw))
                for c in CONDS:
                    boots[c].append(auroc(M[c][ix], y[ix]))
            B = {c: np.array(boots[c]) for c in CONDS}
            line = "    %-5s n=%d (right %d, wrong %d): " % (s, len(rw), int(y.sum()), int((~y).sum()))
            line += " | ".join("%s %.3f [%.3f, %.3f]" % (c, point[c], np.nanpercentile(B[c], 2.5), np.nanpercentile(B[c], 97.5)) for c in CONDS)
            print(line)
            print("          written minus not written: " + " | ".join(
                "%s %+.3f [%+.3f, %+.3f]" % (c, point[c] - point["baseline"], np.nanpercentile(B[c] - B["baseline"], 2.5),
                                              np.nanpercentile(B[c] - B["baseline"], 97.5)) for c in CONDS[1:]))
        # R2 (thresholds on tune only, then one test read)
        mtu = {x["id"]: mg["tune|" + x["id"]] for x in sp["tune"] if x["answered"]}
        thr = choose_thresholds([(mtu[x["id"]], V[x["state"]]) for x in sp["tune"] if x["answered"]])
        mte = {x["id"]: mg["test|" + x["id"]] for x in sp["test"] if x["answered"]}
        te = read_test(sp["test"], mte, thr, ci)
        net_e = sum(V[x["state"]] for x in sp["test"])
        net_n1 = sum(V[x["n1_state"]] for x in sp["test"])
        print("  R2 judged false -> no answer; threshold chosen on tune, read once on test. Test nets: single E %+d, N1 %+d "
              "(%d test questions)" % (net_e, net_n1, len(sp["test"])))
        for c in CONDS:
            t, kt, gt = thr[c]
            r = te[c]
            print("    %-9s tune: t=%s drops %d, tune gain %+d | test: drops %d (right %d, wrong %d, other %d) | "
                  "vs single E %s | vs N1 %s" % (c, "%+.4f" % t if abs(t) != float("inf") else ("-inf" if t < 0 else "+inf"),
                                                  kt, gt, r["k"], r["R"], r["W"], r["O"], r["ci_e"], r["ci_n1"]))
        # R3
        print("  R3 same number dropped on test, the k lowest NO-WRITE margins instead (mean answer log-prob of single E: "
              "not stored)")
        for c in CONDS[1:]:
            k = te[c]["k"]
            nw = lowest_k(sp["test"], mte, "baseline", k)
            diff = [a - b for a, b in zip(te[c]["d_e"], nw)]
            print("    %-9s k=%d: written vs single E %s | k lowest no-write vs single E %s (right %d, wrong %d) | written "
                  "minus no-write-k %s" % (c, k, te[c]["ci_e"], ci(nw), sum(v == -1 for v in nw), sum(v == 1 for v in nw), ci(diff)))
        # R4
        print("  R4 negated-premise opening vs other (regex in the file head)")
        for s in ("tune", "test"):
            for g, lab in ((True, "neg"), (False, "other")):
                rw = [x for x in sp[s] if x["answered"] and x["state"] in "RW" and x["neg"] == g]
                a = [x for x in sp[s] if x["answered"] and x["neg"] == g]
                y = np.array([x["state"] == "R" for x in rw])
                print("    %-5s %-5s answered %d (right %d, wrong %d, other %d): AUROC %s" % (
                    s, lab, len(a), sum(x["state"] == "R" for x in a), sum(x["state"] == "W" for x in a),
                    sum(x["state"] in "HX" for x in a),
                    " | ".join("%s %.3f" % (c, auroc([mg[s + "|" + x["id"]][c] for x in rw], y)) for c in CONDS)))
        for c in CONDS:
            r = te[c]
            for g, lab in ((True, "neg"), (False, "other")):
                dr = [x for x, d_ in zip(sp["test"], r["drop"]) if d_ and x["neg"] == g]
                print("    test drops %-9s %-5s %d (right %d, wrong %d, other %d)" % (
                    c, lab, len(dr), sum(x["state"] == "R" for x in dr), sum(x["state"] == "W" for x in dr),
                    sum(x["state"] in "HX" for x in dr)))


# ------------------------------------------------------------------------------------------------------------------ #
# step 2 (work queue section 4): rules R5a / R5b, TruthfulQA (offline) and (QA)^2 (validation)
#   R5a  two thresholds: the answers are split by NEG_RE into "negated-premise opening" and "other"; each class gets its
#        own threshold chosen on the TruthfulQA TUNE split (choose_thresholds on that class only; "drop nothing"
#        allowed); an answer of a class whose margin is below its class threshold becomes no answer.
#   R5b  judge "other" only: negated-premise openings are always kept; "other" uses R5a's "other" threshold.
#   Conditions: no write and true/false E at strength 3 (CONDS2). The thresholds are chosen once (step2_thresholds),
#   printed at the head of the output, and then read on the TruthfulQA test split and on (QA)^2 (570 questions, and
#   the 470 that are not in the 100-question pilot) without any further choice.
# export-qaqa MODEL (container, no GPU): single E's pass on the 570-question (QA)^2 bank at the main-table layer and
#   strength (qaqa_pilot_table.SET), write rule "prompt", judge work dir test_full570, read with qaqa_pilot_table.Raw
#   (value A +1, B -1, C / H / empty 0; answered = not blank for the rule detector and not empty); every answered text
#   must be the text DeepSeek Pro judged. Writes Iteration_01/newsets/selfjudge/<MODEL>/qaqa_items.json.
# score MODEL --qaqa (container, GPU): the same code path as "score" on those answers, conditions CONDS2, no
#   regression (step 1's passed). Writes qaqa_probe.{npz,json}.
# step2 (host): the readings.
# ------------------------------------------------------------------------------------------------------------------ #
def export_qaqa(model):
    sys.path.insert(0, "Iteration_01/diag")
    import qaqa_pilot_table as qt
    cfg = MODELS[model]
    el, s = qt.SET[model][:2]
    if el != cfg["layer"] or s.replace(".", "-") != cfg["s"]:
        raise SystemExit("%s: (QA)^2 setting %s %s differs from the TruthfulQA main-table setting" % (model, el, s))
    R = qt.Raw(model, "qaqafull", QAQA_BANK, QAQA_WORKS)
    arm = R.e(s)
    R.need(arm)
    with open(QAQA_BANK, encoding="utf-8") as f:
        bank = {r["id"]: r for r in (json.loads(l) for l in f)}
    pilot = read_json(QAQA_PILOT)
    if len(bank) != 570 or len(pilot["ids"]) != 100 or any(q not in bank or bank[q]["qtype"] != pilot["types"][q] for q in pilot["ids"]):
        raise SystemExit("the pilot ids are not 100 questions of the 570-question bank")
    pset = set(pilot["ids"])
    cond = R.cond[arm]
    items = []
    for q in R.ids:
        text = R.text(arm, q)
        answered = not R.blank(arm, q)
        if answered and ((q, cond) not in R.pred or R.pred[(q, cond)].strip() != text.strip()):
            raise SystemExit("%s: the answer of %s is not the judged text" % (model, q))
        items.append(dict(id=q, question=bank[q]["question"], answer=text, answer_sha1=sha1(text), value=R.cell(arm, q),
                          answered=answered, qtype=bank[q]["qtype"], pilot=q in pset, neg=bool(NEG_RE.match(text))))
    path = os.path.join(OUT, model, "qaqa_items.json")
    if os.path.exists(path):
        raise SystemExit("%s exists" % path)
    gpath = R.gdir / (qt.G.arm_token(arm) + ".json")
    doc = dict(model=model, arm=arm, condition=cond, works=QAQA_WORKS, bank=QAQA_BANK, bank_sha256=sha256_file(QAQA_BANK),
               generation=str(gpath), generation_sha256=sha256_file(str(gpath)), neg_re=NEG_RE.pattern, items=items)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=0)
    a = [x for x in items if x["answered"]]
    print("%s (QA)^2 %s: %d questions, E answered %d (right %d, wrong %d, other %d), negated-premise opening %d; pilot %d -> %s" % (
        model, arm, len(items), len(a), sum(x["value"] == 1 for x in a), sum(x["value"] == -1 for x in a),
        sum(x["value"] == 0 for x in a), sum(x["neg"] for x in a), len(pset), path))


def step2_thresholds(tq_rows):
    """TUNE ONLY (TruthfulQA). tq_rows: [(margin by condition, value, neg)] of the answered tune statements.
    Returns {cond: {"neg": (t, drops, gain), "other": (t, drops, gain)}}."""
    by = {g: choose_thresholds([(m, v) for m, v, n in tq_rows if n == (g == "neg")], CONDS2) for g in ("neg", "other")}
    return {c: {g: by[g][c] for g in ("neg", "other")} for c in CONDS2}


def step2_drops(rows, mg, thr, rule, c):
    """rows: [dict(id, answered, neg, v)]; mg: {id: {cond: margin}}. The drop flags under rule a (R5a) or b (R5b)."""
    out = []
    for x in rows:
        if not x["answered"] or (rule == "b" and x["neg"]):
            out.append(False)
        else:
            out.append(mg[x["id"]][c] < thr[c]["neg" if x["neg"] else "other"][0])
    return out


def step2_read(label, rows, mg, thr, ci, qtype=False):
    def cnt(xs):
        return "%d (right %d, wrong %d, other %d)" % (len(xs), sum(x["v"] == 1 for x in xs), sum(x["v"] == -1 for x in xs),
                                                       sum(x["v"] == 0 for x in xs))
    print("  %s: %d questions; single E net %+d" % (label, len(rows), sum(x["v"] for x in rows)))
    for rule in ("a", "b"):
        dl = {}
        for c in CONDS2:
            dr = step2_drops(rows, mg, thr, rule, c)
            dl[c] = [-x["v"] if d else 0 for x, d in zip(rows, dr)]
            print("    R5%s %-9s drops neg %s | other %s | vs single E %s" % (
                rule, c, cnt([x for x, d in zip(rows, dr) if d and x["neg"]]), cnt([x for x, d in zip(rows, dr) if d and not x["neg"]]),
                ci(dl[c])))
            if qtype:
                print("                       by type: " + " | ".join("%s %s" % (t, cnt([x for x, d in zip(rows, dr) if d and x["qtype"] == t]))
                                                          for t in QTYPES))
        print("    R5%s written minus not written %s" % (rule, ci([a - b for a, b in zip(dl["E_a3"], dl["baseline"])])))


def step2_auroc(label, rows, mg):
    import numpy as np
    rng = np.random.default_rng(20261122)
    for g, sel in (("neg", lambda x: x["neg"]), ("other", lambda x: not x["neg"]), ("all", lambda x: True)):
        rw = [x for x in rows if x["answered"] and x["v"] != 0 and sel(x)]
        y = np.array([x["v"] == 1 for x in rw])
        M = {c: np.array([mg[x["id"]][c] for x in rw]) for c in CONDS2}
        pt = {c: auroc(M[c], y) for c in CONDS2}
        B = {c: [] for c in CONDS2}
        for _ in range(4000):
            ix = rng.integers(0, len(rw), len(rw)) if rw else np.array([], dtype=int)
            for c in CONDS2:
                B[c].append(auroc(M[c][ix], y[ix]))
        B = {c: np.array(B[c]) for c in CONDS2}
        with np.errstate(all="ignore"):
            q = lambda a: (np.nanpercentile(a, 2.5), np.nanpercentile(a, 97.5)) if np.isfinite(a).any() else (float("nan"), float("nan"))
            print("    R1 %-5s %-5s n=%d (right %d, wrong %d): %s | written minus not written %+.3f [%+.3f, %+.3f]" % (
                label, g, len(rw), int(y.sum()), int((~y).sum()),
                " | ".join("%s %.3f [%.3f, %.3f]" % (c, pt[c], *q(B[c])) for c in CONDS2),
                pt["E_a3"] - pt["baseline"], *q(B["E_a3"] - B["baseline"])))


def step2():
    import numpy as np
    sys.path.insert(0, "Iteration_01/diag")
    import policy_select2 as ps
    ci = ps.ci
    data = {}
    print("# selfjudge step 2 (work queue 2026-11-22_queue_d, section 4); conditions %s" % (CONDS2,))
    print("# THRESHOLDS, chosen on the TruthfulQA tune split only (frozen for everything below): t, tune drops, tune net gain")
    for model in ORDER:
        d = os.path.join(OUT, model)
        doc = read_json(d + "/items.json")
        meta = read_json(d + "/probe.json")
        if meta["items_sha256"] != sha256_file(d + "/items.json"):
            raise SystemExit("%s: probe.json was made from another items.json" % model)
        z = np.load(d + "/probe.npz")
        ans = [x for x in doc["items"] if x["answered"]]
        if [str(i) for i in z["ids"]] != [x["id"] for x in ans] or [str(a) for a in z["answer_sha1"]] != [x["answer_sha1"] for x in ans]:
            raise SystemExit("%s: probe rows are not the answered items" % model)
        mg = {x["split"] + "|" + x["id"]: {c: float(z[c + "__margin"][j]) for c in CONDS2} for j, x in enumerate(ans)}
        rows = {s: [dict(id=s + "|" + x["id"], answered=x["answered"], neg=x["neg"], v=V[x["state"]])
                    for x in doc["items"] if x["split"] == s] for s in ("tune", "test")}
        thr = step2_thresholds([(mg[x["id"]], x["v"], x["neg"]) for x in rows["tune"] if x["answered"]])
        data[model] = (mg, rows, thr)
        for c in CONDS2:
            print("  %-12s %-9s " % (model, c) + " | ".join("%s t=%s drops %d gain %+d" % (
                g, "%+.4f" % thr[c][g][0] if abs(thr[c][g][0]) != float("inf") else ("-inf" if thr[c][g][0] < 0 else "+inf"),
                thr[c][g][1], thr[c][g][2]) for g in ("neg", "other")))
    print("# NEG_RE: %s" % NEG_RE.pattern)
    for model in ORDER:
        mg, rows, thr = data[model]
        print("=" * 100)
        print("%s | TruthfulQA (exploration: rules fixed after step 1 was seen)" % model)
        step2_read("TruthfulQA test", rows["test"], mg, thr, ci)
        d = os.path.join(OUT, model)
        if not os.path.exists(d + "/qaqa_probe.npz"):
            print("  (QA)^2: no probe yet")
            continue
        qd = read_json(d + "/qaqa_items.json")
        qm = read_json(d + "/qaqa_probe.json")
        if qm["items_sha256"] != sha256_file(d + "/qaqa_items.json"):
            raise SystemExit("%s: qaqa_probe.json was made from another qaqa_items.json" % model)
        z = np.load(d + "/qaqa_probe.npz")
        qa = [x for x in qd["items"] if x["answered"]]
        if [str(i) for i in z["ids"]] != [x["id"] for x in qa] or [str(a) for a in z["answer_sha1"]] != [x["answer_sha1"] for x in qa]:
            raise SystemExit("%s: (QA)^2 probe rows are not the answered items" % model)
        if any(sha1(statement(x["question"], x["answer"])) != str(h) for x, h in zip(qa, z["statement_sha1"])):
            raise SystemExit("%s: (QA)^2 statement texts differ" % model)
        qmg = {x["id"]: {c: float(z[c + "__margin"][j]) for c in CONDS2} for j, x in enumerate(qa)}
        qrows = [dict(id=x["id"], answered=x["answered"], neg=x["neg"], v=x["value"], qtype=x["qtype"], pilot=x["pilot"])
                 for x in qd["items"]]
        print("%s | (QA)^2 validation, %s, judge dirs %s; thresholds as above" % (model, qd["arm"], qd["works"]))
        for lab, rs in (("570", qrows), ("470 (pilot 100 left out)", [x for x in qrows if not x["pilot"]])):
            a = [x for x in rs if x["answered"]]
            print("  (QA)^2 %s: E answered %d (right %d, wrong %d, other %d); neg %d (right %d, wrong %d), other %d (right %d, wrong %d)" % (
                lab, len(a), sum(x["v"] == 1 for x in a), sum(x["v"] == -1 for x in a), sum(x["v"] == 0 for x in a),
                sum(x["neg"] for x in a), sum(x["neg"] and x["v"] == 1 for x in a), sum(x["neg"] and x["v"] == -1 for x in a),
                sum(not x["neg"] for x in a), sum(not x["neg"] and x["v"] == 1 for x in a), sum(not x["neg"] and x["v"] == -1 for x in a)))
            print("    answered by type: " + " | ".join("%s %d (right %d, wrong %d)" % (
                t, sum(x["qtype"] == t for x in a), sum(x["qtype"] == t and x["v"] == 1 for x in a),
                sum(x["qtype"] == t and x["v"] == -1 for x in a)) for t in QTYPES))
            step2_read("(QA)^2 " + lab, rs, qmg, thr, ci, qtype=True)
            step2_auroc(lab[:3], rs, qmg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("export", "score", "table", "export-qaqa", "step2"))
    ap.add_argument("model", nargs="?")
    ap.add_argument("--regress-limit", type=int, default=None)
    ap.add_argument("--regress-only", action="store_true")
    ap.add_argument("--qaqa", action="store_true", help="score: the (QA)^2 answers of qaqa_items.json (step 2)")
    ap.add_argument("--sources", default="", help="score: comma list of answer sources SRC (SRC_items.json), e.g. tq_forced,tq_release")
    ap.add_argument("--show", type=int, default=5)
    a = ap.parse_args()
    if a.cmd not in ("table", "step2") and a.model not in MODELS:
        raise SystemExit("model must be one of %s" % sorted(MODELS))
    if a.cmd == "export":
        export(a.model)
    elif a.cmd == "export-qaqa":
        export_qaqa(a.model)
    elif a.cmd == "score":
        srcs = (["qaqa"] if a.qaqa else []) + [x for x in a.sources.split(",") if x]
        score(a.model, a.regress_limit, a.regress_only, a.show, sources=srcs)
    elif a.cmd == "step2":
        step2()
    else:
        table()


if __name__ == "__main__":
    main()
