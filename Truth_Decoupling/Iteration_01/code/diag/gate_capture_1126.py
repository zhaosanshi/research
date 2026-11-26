#!/usr/bin/env python3
# Iteration_01 exploration (executor shen, Claude Opus 5.5, 2026-11-26), work queue
# discussion/work_queue/2026-11-26_queue_p.md, sections 2 and 4: capture for the per-question switch ("gate").
# Not reviewed by GPT. GPU forward passes only: no generation, no write, no judge call.
#
# What is read: for every question, the output of EVERY decoder layer (block output, index 0 .. L-1) at the LAST prompt
# token of the plain prompt, no write, float32; one forward per question (batch size 1, no padding), through the
# engine's own forward helper core.model_io.forward_logits (explicit position ids, as every engine forward) inside the
# engine's float32 output projection (X.E.fp32_output_projection), so that the next-token logits are the ones the
# generation used.
# Model loading: exactly as scripts/capability/cap_mcq_sweep.py and newsets_gen.py gen do it: newsets_gen.load_models on
# code/configs/newsets/frozen_models_20261119.json, newsets_gen.engine (the model's own code copy, e44_eval_extra.py),
# newsets_gen.context (the run's engine Context; this is what produced every no-write answer of the new sets).
# Prompt rendering: X.pd.render_prompt(ctx.bundle, system, template, text) (chat template, generation header, Qwen
# thinking stub as configured), the renderer of every generation:
#   free-answer sets (FalseQA, (QA)^2, TruthfulQA, PopQA, TriviaQA): system = the run's prompt "x", template = the run's
#     user template ctx.user_template ("Question: {question}\nAnswer:");
#   MMLU: system "x", template "{question}", user text cap_prompts.mmlu_user (newsets_gen.load_items, as cap_mcq_sweep);
#   GSM8K: system "neutral" ("You are a helpful assistant."), template "{question}", user text cap_prompts.gsm8k_user:
#     this is the system the GSM8K no-write generations used (capability/gen_main/gsm8k/<model>/b8/neutral.json; the
#     docstring of cap_prompts.py says so too). The task brief said "the same plain system prompt"; the plan says "the
#     prompt of the capability evaluation", which is "neutral" for GSM8K. The plan is followed.
# Sets (name: source; label 1 = trap where the set has labels, else -1):
#   fqa_train      FalseQA official TRAIN split, newsets/raw_falseqa_20261123/train.csv (2,374 rows = 1,187 pairs; label 1
#                  = false premise; pair = row i of the label-1 block with row i of the label-0 block; ids fqtr<iiii>f /
#                  fqtr<iiii>v, group fqtr<iiii>); every pair kept (4 pairs have the same normalised text twice)
#   fqa_held       FalseQA held-out 600 (rows of falseqa_full_bank_20261123.jsonl with set heldout; label = premise false)
#   qaqa           (QA)^2 full bank 570 (bank order; label = qtype failure / unverifiable); the 100 pilot ids are
#                  training-only, the other 470 are reading-only
#   tq_train, tq_tune, tq_test   the run's frozen_train / frozen_tune / frozen_test questions (96 / 192 / 529)
#   popqa_train, popqa_tune, popqa_test, popqa_test2, triviaqa_* likewise (bank field split; test2 from the test2 bank)
#   mmlu_test      the 1,000-item list capability/sets/mmlu_1000_20261119.json
#   mmlu_extra     500 MMLU test items drawn outside that list (draw file, see "draw" below)
#   gsm8k_test     capability/sets/gsm8k_first304_20261124.json (first 304 test items)
#   gsm8k_train    300 GSM8K TRAIN items drawn (draw file). The train split was not on the server; it was fetched from
#                  huggingface openai/gsm8k at revision 740312add88f781978c0658806c59bc2815b9866 (the revision whose
#                  test parquet is byte-identical to the server's data_external/gsm8k_20261119 copy) into
#                  data_external/gsm8k_train_gate_1126/gsm8k_main_train_gate_1126.parquet (sha256 checked below).
# Draw (phase "draw", CPU, once for all models): random.Random(20261126).sample over the candidates, then sorted in file
#   order. MMLU candidates: every test-parquet item (id mmlu:<subject>:<index within subject>, newsets_gen.load_items'
#   ids) not in the 1,000 list and whose normalised user text is not the text of a list item. GSM8K candidates: every
#   train row (id gsm8ktrain:<row>) whose normalised user text is not the text of one of the 304 test items.
# Checks (printed; a failure stops the run):
#   (1) the rendered prompt: tokenizer(prefix_text) == render_prompt's prefix_ids == bundle.encode([prefix_text])
#       (the generation's own encoder, one prompt, so no padding); these are the ids fed to the forward;
#   (2) generation records of the no-write answers (the records hold the answer token ids, system text and template,
#       not the prompt token ids): contract system text and template equal ours, identity_sha256 equals the loaded
#       Context's, the record's item ids contain ours; TruthfulQA test: frozen_test.json user template, prompt "x" and
#       content_start["x"] of every question equal ours;
#   (3) arg-max of our float32 next-token logits equals the first token of the record's no-write answer: agreement and
#       the largest gap (top logit - logit of that token) printed per set; a set with records and agreement below 90%
#       stops the run (the generation ran left-padded batches of 8, so near ties may flip);
#       MMLU: our letter log-probabilities (log-sum-exp over every single-token spelling, as cap_mcq_sweep) against the
#       no-write arm "x" of capability/ecmmlu_1126/<model>.json: pick agreement (below 90% stops) and largest difference;
#   (4) the first 16 prompts of tq_test are captured a second time at the end; the two captures must be bit-identical;
#   (5) every layer hook fires once per forward on the full prompt width.
#   The planned minimum (32 TruthfulQA test questions, 16 of each other set with records) is covered: checks (1)-(3)
#   run on every question of every set that has records.
# Output (server only): Iteration_01/newsets/gate_1126/<model>/<set>_gate_1126.npz with keys (ids), text_sha1 (sha1 of
#   the normalised question / user text; no text), labels (int8), groups, feats (float32 [N, L, d]); and
#   <model>/capture_gate_1126.json (counts, checks, hashes). Existing files are never overwritten.
# usage (model interpreter, from /workspace/Truth_Decoupling):
#   python3 -I Iteration_01/diag/gate_capture_1126.py draw
#   <model python> -I Iteration_01/diag/gate_capture_1126.py capture MODEL
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

PROJ = Path("/workspace/Truth_Decoupling")
IT = PROJ / "Iteration_01"
NEWSETS = IT / "code" / "scripts" / "newsets"
CAPDIR = IT / "code" / "scripts" / "capability"
FROZEN_MODELS = IT / "code" / "configs" / "newsets" / "frozen_models_20261119.json"
OUT = IT / "newsets" / "gate_1126"
DRAWS = OUT / "draws_gate_1126.json"
FQA_TRAIN = IT / "newsets" / "raw_falseqa_20261123" / "train.csv"
FQA_BANK = IT / "newsets" / "falseqa" / "falseqa_full_bank_20261123.jsonl"
FQA_HELD = IT / "newsets" / "falseqa" / "falseqa_heldout_bank_20261123.jsonl"
QAQA_BANK = IT / "newsets" / "bank" / "qaqa_full_bank_20261122.jsonl"
QAQA_PILOT = IT / "newsets" / "bank" / "qaqa_pilot_bank_20261122.jsonl"
BANKS = {"popqa": (IT / "newsets/bank/popqa_bank_20261119.jsonl", IT / "newsets/bank/popqa_bank_test2_20261120.jsonl"),
         "triviaqa": (IT / "newsets/bank/triviaqa_bank_20261119.jsonl", IT / "newsets/bank/triviaqa_bank_test2_20261120.jsonl")}
MMLU_PARQUET = PROJ / "data_external/mmlu_20261119/all/test-00000-of-00001.parquet"
MMLU_IDS = IT / "capability/sets/mmlu_1000_20261119.json"
GSM_TEST = PROJ / "data_external/gsm8k_20261119/main/test-00000-of-00001.parquet"
GSM_TEST_SHA = "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"
GSM_TRAIN = PROJ / "data_external/gsm8k_train_gate_1126/gsm8k_main_train_gate_1126.parquet"
GSM_TRAIN_SHA = "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d"
GSM_IDS = IT / "capability/sets/gsm8k_first304_20261124.json"
ECMMLU = IT / "capability/ecmmlu_1126"
SEED = 20261126
N_MMLU_EXTRA, N_GSM_TRAIN = 500, 300
N_FQA_PAIRS = 1187
MIN_AGREE = 0.90
N_REPEAT = 16
SETS = ("fqa_train", "fqa_held", "qaqa", "tq_train", "tq_tune", "tq_test",
        "popqa_train", "popqa_tune", "popqa_test", "popqa_test2",
        "triviaqa_train", "triviaqa_tune", "triviaqa_test", "triviaqa_test2",
        "mmlu_test", "mmlu_extra", "gsm8k_test", "gsm8k_train")
MODELS = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")

sys.path[:0] = [str(NEWSETS), str(CAPDIR)]
import newsets_gen as NG   # noqa: E402
import cap_prompts as CP   # noqa: E402


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def tsha(s):
    return hashlib.sha1(norm(s).encode("utf-8")).hexdigest()


def sha_file(p):
    return NG.sha_file(p)


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def rj(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ---- MMLU / GSM8K items --------------------------------------------------------------------------------------------- #
def mmlu_all():
    """[(id, user text)] of every MMLU test item, ids as newsets_gen.load_items reads them (mmlu:<subject>:<k>)."""
    import pyarrow.parquet as pq
    out, per = [], {}
    for r in pq.read_table(MMLU_PARQUET).to_pylist():
        k = per.get(r["subject"], 0)
        per[r["subject"]] = k + 1
        out.append(("mmlu:%s:%d" % (r["subject"], k), CP.mmlu_user(r["subject"], r["question"], r["choices"])))
    return out


def gsm_train_all():
    import pyarrow.parquet as pq
    return [("gsm8ktrain:%d" % i, CP.gsm8k_user(r["question"])) for i, r in enumerate(pq.read_table(GSM_TRAIN).to_pylist())]


def listed(task, items, ids):
    a = NG.argparse.Namespace(task=task, items=items, ids=ids, split="all", only_ids=None)
    got, template = NG.load_items(a)
    if template != "{question}":
        raise SystemExit("%s: unexpected template" % task)
    return got


def make_draws():
    if sha_file(GSM_TRAIN) != GSM_TRAIN_SHA or sha_file(GSM_TEST) != GSM_TEST_SHA:
        raise SystemExit("GSM8K parquet hashes differ from the recorded ones")
    m_list = listed("mmlu", MMLU_PARQUET, MMLU_IDS)
    lid = {i for i, _ in m_list}
    ltx = {norm(t) for _, t in m_list}
    m_all = mmlu_all()
    if not lid <= {i for i, _ in m_all}:
        raise SystemExit("MMLU list ids are not test-parquet ids")
    m_out = [x for x in m_all if x[0] not in lid]
    m_cand = [x for x in m_out if norm(x[1]) not in ltx]
    m_pick = sorted(random.Random(SEED).sample(range(len(m_cand)), N_MMLU_EXTRA))
    g_list = listed("gsm8k", GSM_TEST, GSM_IDS)
    gtx = {norm(t) for _, t in g_list}
    g_all = gsm_train_all()
    g_cand = [x for x in g_all if norm(x[1]) not in gtx]
    g_pick = sorted(random.Random(SEED).sample(range(len(g_cand)), N_GSM_TRAIN))
    doc = {"seed": SEED, "rule": "random.Random(seed).sample(range(len(candidates)), n), sorted (file order)",
           "mmlu": {"parquet_sha256": sha_file(MMLU_PARQUET), "list_sha256": sha_file(MMLU_IDS), "items": len(m_all),
                    "outside_list": len(m_out), "left_out_same_text_as_a_list_item": len(m_out) - len(m_cand),
                    "candidates": len(m_cand), "ids": [m_cand[k][0] for k in m_pick]},
           "gsm8k_train": {"parquet_sha256": GSM_TRAIN_SHA, "source": "huggingface openai/gsm8k main train, revision 740312add88f781978c0658806c59bc2815b9866",
                           "rows": len(g_all), "left_out_same_text_as_a_test304_item": len(g_all) - len(g_cand),
                           "candidates": len(g_cand), "ids": [g_cand[k][0] for k in g_pick]}}
    return doc


def run_draw():
    doc = make_draws()
    OUT.mkdir(parents=True, exist_ok=True)
    if DRAWS.exists():
        if rj(DRAWS) != doc:
            raise SystemExit("%s exists and differs from a fresh draw" % DRAWS)
        print("draw file exists and equals a fresh draw")
    else:
        with open(DRAWS, "x", encoding="utf-8") as f:
            json.dump(doc, f, indent=0)
    for k in ("mmlu", "gsm8k_train"):
        d = doc[k]
        print("draw %s: %d candidates (%s), %d drawn" % (k, d["candidates"], ", ".join("%s %s" % (a, b) for a, b in d.items()
                                                                                    if a.startswith(("left_out", "outside", "items", "rows"))), len(d["ids"])))


# ---- the sets of one model ------------------------------------------------------------------------------------------ #
def items_of(model, m):
    """{set: dict(system=name of the system prompt, template=None (the run's) or "{question}", items=[(id, text, label,
    group)], record=None or dict(kind, path))}."""
    S = {}
    rows = list(csv.reader(open(FQA_TRAIN, encoding="utf-8", newline="")))
    if rows[0] != ["question", "answer", "label"] or len(rows) != 2 * N_FQA_PAIRS + 1:
        raise SystemExit("FalseQA train.csv is not the 2,374-row three-column file")
    rows = rows[1:]
    if [r[2] for r in rows] != ["1"] * N_FQA_PAIRS + ["0"] * N_FQA_PAIRS:
        raise SystemExit("FalseQA train.csv: not 1,187 label-1 rows followed by 1,187 label-0 rows")
    it = []
    for i in range(N_FQA_PAIRS):
        it.append(("fqtr%04df" % i, rows[i][0], 1, "fqtr%04d" % i))
    for i in range(N_FQA_PAIRS):
        it.append(("fqtr%04dv" % i, rows[i + N_FQA_PAIRS][0], 0, "fqtr%04d" % i))
    S["fqa_train"] = dict(system="x", template=None, items=it, record=None)
    fb = jl(FQA_BANK)
    held = [r for r in fb if r["set"] == "heldout"]
    hb = jl(FQA_HELD)
    if [(r["id"], r["question"]) for r in held] != [(r["id"], r["question"]) for r in hb] or len(held) != 600:
        raise SystemExit("FalseQA held-out rows of the full bank differ from the held-out bank")
    S["fqa_held"] = dict(system="x", template=None, record=dict(kind="gen", path=IT / "newsets/gen_prompt/falseqa" / model / "test/x.json"),
                         items=[(r["id"], r["question"], int(r["premise"] == "false"), r["pair_id"]) for r in held])
    qb = jl(QAQA_BANK)
    pil = {r["id"]: r for r in jl(QAQA_PILOT)}
    if len(qb) != 570 or not set(pil) <= {r["id"] for r in qb} or len(pil) != 100:
        raise SystemExit("(QA)^2 banks")
    for r in qb:
        if r["id"] in pil and (pil[r["id"]]["question"] != r["question"] or pil[r["id"]]["qtype"] != r["qtype"]):
            raise SystemExit("(QA)^2 pilot row differs from the full bank row %s" % r["id"])
    S["qaqa"] = dict(system="x", template=None, record=dict(kind="gen", path=IT / "newsets/gen_prompt/qaqafull" / model / "test/x.json"),
                     items=[(r["id"], r["question"], int(r["qtype"] != "no_failure"), r["id"]) for r in qb])
    for sp in ("train", "tune", "test"):
        p = PROJ / m["run_dir"] / ("frozen_%s.json" % sp)
        fr = rj(p)
        if fr["split"] != sp or not fr.get("completed"):
            raise SystemExit("%s: not a completed frozen %s file" % (p, sp))
        S["tq_" + sp] = dict(system="x", template=None, record=dict(kind="frozen", path=p) if sp == "test" else None,
                             items=[(q["pair_id"], q["question"], 1, q["pair_id"]) for q in fr["questions"]])
    for task, (b1, b2) in BANKS.items():
        r1, r2 = jl(b1), jl(b2)
        for sp in ("train", "tune", "test", "test2"):
            rr = [r for r in (r2 if sp == "test2" else r1) if r["split"] == sp]
            rec = None
            gp = IT / "newsets/gen_prompt" / task / model / sp / "x.json"
            if sp != "train":
                rec = dict(kind="gen", path=gp)
            S["%s_%s" % (task, sp)] = dict(system="x", template=None, record=rec,
                                          items=[(r["id"], r["question"], 0, r["id"]) for r in rr])
    S["mmlu_test"] = dict(system="x", template="{question}", record=dict(kind="mmlu", path=ECMMLU / ("%s.json" % model)),
                          items=[(i, t, 0, i) for i, t in listed("mmlu", MMLU_PARQUET, MMLU_IDS)])
    d = rj(DRAWS)
    fresh = make_draws()
    if d != fresh:
        raise SystemExit("the draw file differs from a fresh draw")
    allm = dict(mmlu_all())
    S["mmlu_extra"] = dict(system="x", template="{question}", record=None, items=[(i, allm[i], 0, i) for i in d["mmlu"]["ids"]])
    gdir = sorted((IT / "capability/gen_main/gsm8k" / model).glob("*/neutral.json"))
    if len(gdir) != 1:
        raise SystemExit("%s: expected one GSM8K no-write neutral.json, found %d" % (model, len(gdir)))
    S["gsm8k_test"] = dict(system="neutral", template="{question}", record=dict(kind="gsm", path=gdir[0]),
                           items=[(i, t, 0, i) for i, t in listed("gsm8k", GSM_TEST, GSM_IDS)])
    allg = dict(gsm_train_all())
    S["gsm8k_train"] = dict(system="neutral", template="{question}", record=None, items=[(i, allg[i], 0, i) for i in d["gsm8k_train"]["ids"]])
    if tuple(S) != SETS:
        raise SystemExit("set order")
    for name, s in S.items():
        ids = [x[0] for x in s["items"]]
        if len(set(ids)) != len(ids) or any(not isinstance(x[1], str) or not x[1].strip() for x in s["items"]):
            raise SystemExit("%s: repeated id or empty text" % name)
    return S


def record_rows(rec, model, ids, system_text, template, ident):
    """{id: first answer token id or -1} from a no-write generation record, after the contract checks; TruthfulQA also
    returns content starts. MMLU: {id: {letter: logp}}."""
    p = rec["path"]
    g = rj(p)
    if rec["kind"] == "frozen":
        if g["user_template"] != template or g["prompt_systems"]["x"] != system_text or g["identity_sha256"] != ident:
            raise SystemExit("%s: frozen_test template / prompt x / identity differ from the loaded run" % p)
        out = {}
        for q in g["questions"]:
            t = q["answers"]["x"]["token_ids"]
            out[q["pair_id"]] = (t[0] if t else -1, int(q["content_start"]["x"]))
        if set(ids) - set(out):
            raise SystemExit("%s: questions missing from the frozen record" % p)
        return out
    if rec["kind"] == "mmlu":
        if g["model"] != model or g.get("identity_sha256") != ident or "x" not in g["arms"]:
            raise SystemExit("%s: not this run's MMLU file" % p)
        return {r["id"]: r["logp"] for r in g["arms"]["x"]["rows"]}
    c = g["contract"]
    if (c["system_text"] != system_text or c["user_template"] != template or c["identity_sha256"] != ident or c["parts"]
            or not g.get("completed") or c["model"] != model):
        raise SystemExit("%s: contract (system / template / identity / no write / model) differs" % p)
    out = {r["id"]: (r["token_ids"][0] if r["token_ids"] else -1) for r in g["rows"]}
    if set(ids) - set(out):
        raise SystemExit("%s: items missing from the record" % p)
    return out


def run_capture(model):
    import numpy as np
    import torch
    m = NG.load_models(FROZEN_MODELS, model)
    sets = items_of(model, m)
    out_dir = OUT / model
    for name in SETS:
        if (out_dir / ("%s_gate_1126.npz" % name)).exists():
            raise SystemExit("%s exists" % (out_dir / ("%s_gate_1126.npz" % name)))
    if (out_dir / "capture_gate_1126.json").exists():
        raise SystemExit("capture_gate_1126.json exists")
    print("%s: %s" % (model, ", ".join("%s %d" % (k, len(v["items"])) for k, v in sets.items())), flush=True)
    if not torch.cuda.is_available():
        # a model config with a device map would otherwise load on the processor and run there
        raise SystemExit("RuntimeError: No CUDA GPUs are available (refusing to load %s without a card)" % model)
    X = NG.engine(m)
    import core.model_io as MIO
    if Path(MIO.__file__).resolve().parent.parent != (PROJ / m["code"]).resolve():
        raise SystemExit("core.model_io imported from %s, not from the model's code copy" % MIO.__file__)
    t0 = time.time()
    ctx = NG.context(X, m, 0)
    systems = NG.systems_of(ctx)
    bundle = ctx.bundle
    L = bundle.num_layers
    tok = bundle.tokenizer
    print("%s: code %s, identity %s, layers %d, hidden %d, loaded in %.0fs; system x %r; neutral %r; template %r" % (
        model, m["code"], ctx.ident_sha[:16], L, bundle.hidden_size, time.time() - t0, systems["x"], systems["neutral"],
        ctx.user_template), flush=True)
    letter_ids = {}
    for Lt in "ABCD":                    # as cap_mcq_sweep: every single-token spelling of the letter
        cands = []
        for spelling in (Lt, " " + Lt):
            ii = tok.encode(spelling, add_special_tokens=False)
            if len(ii) == 1 and ii[0] not in cands:
                cands.append(ii[0])
        letter_ids[Lt] = cands
    store = {}

    def hook(l):
        def f(mod, inp, out):
            h = MIO._as_hidden(out)
            store.setdefault(l, []).append((h.shape[1], h[0, -1, :].detach().float().to("cpu")))
        return f
    hs = [bundle.layer(l).register_forward_hook(hook(l)) for l in range(L)]
    meta = {"model": model, "code": m["code"], "identity_sha256": ctx.ident_sha, "layers": L, "hidden": bundle.hidden_size,
            "systems": {"x": systems["x"], "neutral": systems["neutral"]}, "user_template": ctx.user_template,
            "draws_sha256": sha_file(DRAWS), "program_sha256": sha_file(__file__), "sets": {}}

    def forward(ids):
        store.clear()
        inputs = {"input_ids": torch.tensor([ids], device=bundle.device), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=bundle.device)}
        lg = MIO.forward_logits(bundle, inputs, grad=False, logits_to_keep=1)
        if any(len(store.get(l, [])) != 1 or store[l][0][0] != len(ids) for l in range(L)):
            raise SystemExit("a layer hook did not fire exactly once on the full prompt")
        return torch.stack([store[l][0][1] for l in range(L)], 0).numpy(), lg[0, -1].float()

    first_repeat = None
    try:
        with X.E.fp32_output_projection(bundle, enabled=True):
            for name in SETS:
                s = sets[name]
                system = systems[s["system"]]
                template = s["template"] or ctx.user_template
                items = s["items"]
                ids = [x[0] for x in items]
                rec = record_rows(s["record"], model, ids, system, template, ctx.ident_sha) if s["record"] else None
                N = len(items)
                feats = np.zeros((N, L, bundle.hidden_size), dtype=np.float32)
                agree = n_cmp = 0
                gaps, cs_bad, mm = [], 0, []
                t1 = time.time()
                for k, (iid, text, _, _) in enumerate(items):
                    p = X.pd.render_prompt(bundle, system, template, text)
                    pid = p["prefix_ids"]
                    e1 = tok(p["prefix_text"], add_special_tokens=False)["input_ids"]
                    e2 = bundle.encode([p["prefix_text"]])["input_ids"][0].tolist()
                    if e1 != pid or e2 != pid:
                        raise SystemExit("%s %s %s: encoded prompt differs from render_prompt's ids" % (model, name, iid))
                    H, lg = forward(pid)
                    feats[k] = H
                    if rec is not None:
                        if s["record"]["kind"] == "mmlu":
                            lp = torch.log_softmax(lg.double(), -1)
                            mine = {Lt: float(torch.logsumexp(lp[torch.tensor(v, device=lp.device)], 0)) for Lt, v in letter_ids.items()}
                            old = rec[iid]
                            pk = max("ABCD", key=lambda a: (mine[a], -"ABCD".index(a)))
                            po = max("ABCD", key=lambda a: (old[a], -"ABCD".index(a)))
                            n_cmp += 1
                            agree += pk == po
                            mm.append(max(abs(mine[a] - old[a]) for a in "ABCD" if old[a] > -5) if any(old[a] > -5 for a in "ABCD") else 0.0)
                        else:
                            r = rec[iid]
                            first = r[0] if s["record"]["kind"] == "frozen" else r
                            if s["record"]["kind"] == "frozen" and r[1] != p["content_start"]:
                                cs_bad += 1
                            if first >= 0:
                                n_cmp += 1
                                am = int(lg.argmax())
                                agree += am == first
                                if am != first:
                                    gaps.append(float(lg.max() - lg[first]))
                if cs_bad:
                    raise SystemExit("%s TruthfulQA test: content start differs from frozen_test.json at %d questions" % (model, cs_bad))
                info = dict(n=N, layers=L, seconds=round(time.time() - t1, 1), system=s["system"], record=str(s["record"]["path"]) if s["record"] else None,
                            compared=n_cmp, agree=agree)
                if rec is not None and s["record"]["kind"] == "mmlu":
                    info["largest_letter_logp_difference"] = max(mm) if mm else None
                    print("%s %-15s %4d prompts | MMLU letter pick = ecmmlu x pick %d / %d, largest letter log-prob difference %.4f | %.0fs" % (
                        model, name, N, agree, n_cmp, max(mm) if mm else float("nan"), info["seconds"]), flush=True)
                elif rec is not None:
                    info.update(disagree=len(gaps), largest_gap=max(gaps) if gaps else 0.0, gaps_over_1=sum(g > 1.0 for g in gaps))
                    print("%s %-15s %4d prompts | arg-max = first token of the no-write answer %d / %d; disagreements %d, largest gap %.4f, gaps > 1 logit %d%s | %.0fs" % (
                        model, name, N, agree, n_cmp, len(gaps), info["largest_gap"], info["gaps_over_1"],
                        "; content starts equal frozen_test.json at every question" if s["record"]["kind"] == "frozen" else "", info["seconds"]), flush=True)
                else:
                    print("%s %-15s %4d prompts | no generation record (training only): rendered with the same function, not compared | %.0fs" % (
                        model, name, N, info["seconds"]), flush=True)
                if rec is not None and (n_cmp == 0 or agree < MIN_AGREE * n_cmp):
                    raise SystemExit("%s %s: agreement with the generation record below %.0f%%" % (model, name, 100 * MIN_AGREE))
                if name == "tq_test":
                    first_repeat = (items[:N_REPEAT], system, template, feats[:N_REPEAT].copy())
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / ("%s_gate_1126.npz" % name)
                np.savez(path, keys=np.array(ids), text_sha1=np.array([tsha(x[1]) for x in items]),
                         labels=np.array([x[2] for x in items], dtype=np.int8), groups=np.array([x[3] for x in items]), feats=feats)
                info["npz_sha256"] = sha_file(path)
                meta["sets"][name] = info
                del feats
            its, system, template, ref = first_repeat
            rep = np.stack([forward(X.pd.render_prompt(bundle, system, template, x[1])["prefix_ids"])[0] for x in its], 0)
            same = bool(np.array_equal(rep.view(np.uint32), ref.view(np.uint32)))
            meta["repeat_check"] = dict(set="tq_test", n=len(its), bit_identical=same)
            print("%s: repeat check, first %d TruthfulQA test prompts captured twice: bit-identical %s" % (model, len(its), same), flush=True)
            if not same:
                raise SystemExit("%s: the repeated capture is not bit-identical" % model)
    finally:
        for h in hs:
            h.remove()
    ctx.still_current()
    with open(out_dir / "capture_gate_1126.json", "x", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print("%s: capture done in %.0fs" % (model, time.time() - t0), flush=True)


def main():
    if sys.argv[1:] == ["draw"]:
        run_draw()
    elif len(sys.argv) == 3 and sys.argv[1] == "capture" and sys.argv[2] in MODELS:
        run_capture(sys.argv[2])
    else:
        raise SystemExit("usage: gate_capture_1126.py draw | capture MODEL")


if __name__ == "__main__":
    main()
