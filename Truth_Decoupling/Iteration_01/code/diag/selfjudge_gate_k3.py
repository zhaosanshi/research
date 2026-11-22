#!/usr/bin/env python3
# Iteration_01 exploration (Claude Opus 5.5, 2026-11-22), work queue discussion/work_queue/2026-11-22_queue_e.md, item
# "1g supplement: third switch K3" (written 17:55, before any K3 reading). Llama-3.1-8B and Qwen3.5-9B. Not reviewed by
# GPT; exploration readings only. Forward passes only; no generation, no judge call; existing answers and verdicts.
#
# K3 = a linear probe on the model's residual stream that says whether a question is a "trap" question.
#   Prompt: the free-answer line's plain prompt x, no write: system = the run's prompt system "x", user = the run's user
#     template "Question: {question}\nAnswer:", rendered by core.qa_prompt_das.render_prompt (chat template, generation
#     header; Qwen thinking off) -- the renderer the free-answer generations used (newsets_gen.py / the e44 engine).
#     The system text and template are read from the generation records themselves (newsets x.json contracts and the
#     TruthfulQA run's frozen_test.json) and must be identical across all six sets. Checks (capture step): the loaded
#     code copy's qa_prompt_das.py equals the generation copy's byte for byte, ModelBundle.build_chat source and the
#     model config "chat" / "path" entries are identical, the encoded prompt equals render_prompt's prefix_ids, on
#     TruthfulQA the user-content start equals the frozen record's, and the arg-max next token at the last prompt
#     position (FP32 output projection as the engine generates, computed here in vocabulary chunks from the head's
#     input) is compared with the first generated token of the
#     no-write x answer of every question (printed; the run stops when a set agrees on fewer than 90% or when any
#     disagreement is more than a near tie, top logit - logit of the x token > 1.0).
#   Features: the output of every decoder layer at the last prompt token (end of the assistant header), float32, taken
#     from the generation's own prefill: core.model_io.greedy_generate (model.generate, left-padded batches of 8 in item
#     order, exactly as qa_prompt_das.generate_prompts runs the no-write x pass) with one new token. (A plain forward
#     with explicit position ids, core.model_io.capture_last_token, gave the same arg-max on every Llama prompt but
#     differed on 3-5% of the left-padded Qwen3.5-9B prompts, none of the unpadded (QA)^2 ones; the prefill of
#     generate is the state that produced the answers.)
#   Sets: (QA)^2 570 (bank order), TruthfulQA test (the 529 frozen questions; readings use the main list),
#     PopQA test / test2, TriviaQA test / test2 (500 each). Files newsets/selfjudge/<MODEL>/k3_<set>.npz.
# Training (fit step, CPU, sklearn): only the 100 (QA)^2 tune ("pilot") questions with their official type; positive =
#   failure + unverifiable, negative = no_failure. Features standardised (mean / variance from these 100 only); logistic
#   regression, L2, C = 1, solver liblinear, random_state 0; nothing tuned.
#   Layer choice: 5-fold stratified cross-validation inside the 100 (shuffle, random_state 0); in each fold the
#   standardisation is fit on the four training folds only (pipeline); the layer score is the mean of the five fold
#   AUROCs (the pooled out-of-fold AUROC is printed too); the highest mean wins, ties to the lower layer. The E layer
#   (Llama 16, Qwen3.5-9B 15) is read as well. The 470 held-out questions take part in no choice. The chosen layer and
#   the E layer are then refit on all 100 and applied to every set.
# Decision: trap = predicted probability > 0.5 (no threshold chosen).
# Readings (table step, host): as selfjudge_gate.py table, with K3 next to K0 / K1, using that file's gate() (G, B, Bg
#   built at the same answer count) unchanged; the K0 reference (Llama G-Bg +10, Qwen3.5-9B -2, five-model G-B / G-Bg)
#   is checked first and a mismatch stops before any reading.
# usage (from /data/sanshizhao/Truth_Decoupling):
#   (GPU)  Iteration_01/logs/newsets_20261119/selfjudge_gate_k3_1122.sh MODEL   (capture + fit, container, GPU 1)
#   (host) python3 -B Iteration_01/diag/selfjudge_gate_k3.py table
import hashlib
import json
import os
import re
import sys

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets"]
import selfjudge_probe as SP     # noqa: E402
import selfjudge_gate as SG      # noqa: E402

FROZEN_MODELS = "Iteration_01/code/configs/newsets/frozen_models_20261119.json"
BANKS = {("popqa", "test"): "Iteration_01/newsets/bank/popqa_bank_20261119.jsonl",
         ("popqa", "test2"): "Iteration_01/newsets/bank/popqa_bank_test2_20261120.jsonl",
         ("triviaqa", "test"): "Iteration_01/newsets/bank/triviaqa_bank_20261119.jsonl",
         ("triviaqa", "test2"): "Iteration_01/newsets/bank/triviaqa_bank_test2_20261120.jsonl"}
SETS = ("qaqa", "tq_test", "popqa_test", "popqa_test2", "triviaqa_test", "triviaqa_test2")
MIN_AGREE = 0.90     # arg-max agreement with the first x token, per set
MAX_GAP = 1.0        # every disagreement must be a near tie: top logit - logit of the x token <= 1.0 (FP32)
CV_SEED = 0
NFOLD = 5

# These originals predate the fit checksum and were independently rebuilt from all
# twelve captured arrays. Keep their validated identifiers for read-only migration.
LEGACY = {
    "llama3.1-8b": ("cfbdd66588ce92393d0821c1308a4a76712b73e9f7cb5aa688efcf56c0c4f7c0",
                    "7368b52f302236408a178098b6db2dcb356360b93a90dde67cf54af5fe4bacc1"),
    "qwen3.5-9b": ("c9d7f1a5526cd6576e3feccfe8de8495885b57768328fab8d52cd0af34408a28",
                   "19de2ad8cd4b4adbf168b77e2178041174bda5c857af8ff33d7cabd0f9df8850"),
}
LEGACY_INPUTS = ("4458128f48a846bc1a099e75d148ca5d55ef01fc715bf9d3c61f3f34db787841",
                 "a3f6342aa9feca2e24f88003f2a05f1ba4f0f32cff9da5caaa62eaa9ac7782fa")


def payload_digest(doc):
    fields = {k: v for k, v in doc.items() if k != "payload_sha256"}
    raw = json.dumps(fields, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def training_inputs():
    return (SP.sha256_file(SP.QAQA_BANK), SP.sha256_file(SP.QAQA_PILOT))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def build_chat_source(path):
    src = open(path, encoding="utf-8").read()
    m = re.search(r"\n    def build_chat\(.*?(?=\n    def )", src, re.S)
    if m is None:
        raise SystemExit("no build_chat in %s" % path)
    return m.group(0)


def set_items(model, verbose=True):
    """[(set name, system, template, [dict(key, question, first_token, content_start)])] from the generation records."""
    fm = SP.read_json(FROZEN_MODELS)["models"][model]
    out = []
    bank = read_jsonl(SP.QAQA_BANK)
    g = SP.read_json("Iteration_01/newsets/gen_prompt/qaqafull/%s/test/x.json" % model)
    k = g["contract"]
    if (k["items_file_sha256"] != SP.sha256_file(SP.QAQA_BANK) or k["base_prompt"] != "x" or k["parts"]
            or (k["model"], k["task"], k["split"], k["arm"]) != (model, "qaqafull", "test", "x") or not g.get("completed")):
        raise SystemExit("%s (QA)^2 x.json is not the no-write x pass of the bank" % model)
    if [r["id"] for r in g["rows"]] != [b["id"] for b in bank] or any(b["split"] != "test" for b in bank):
        raise SystemExit("%s (QA)^2 x.json rows are not the bank" % model)
    out.append(("qaqa", k["system_text"], k["user_template"],
                [dict(key="qaqa|" + b["id"], question=b["question"], first=(r["token_ids"][0] if r["token_ids"] else -1), cs=None)
                 for b, r in zip(bank, g["rows"])]))
    fr = SP.read_json(os.path.join(fm["run_dir"], "frozen_test.json"))
    if fr["model"] != model or fr["split"] != "test" or not fr.get("completed"):
        raise SystemExit("%s frozen_test.json" % model)
    out.append(("tq_test", fr["prompt_systems"]["x"], fr["user_template"],
                [dict(key="tq_test|" + q["pair_id"], question=q["question"],
                      first=(q["answers"]["x"]["token_ids"][0] if q["answers"]["x"]["token_ids"] else -1), cs=q["content_start"]["x"])
                 for q in fr["questions"]]))
    for (task, split), path in BANKS.items():
        rows = [r for r in read_jsonl(path) if r["split"] == split]
        g = SP.read_json("Iteration_01/newsets/gen_prompt/%s/%s/%s/x.json" % (task, model, split))
        k = g["contract"]
        if (k["items_file_sha256"] != SP.sha256_file(path) or k["base_prompt"] != "x" or k["parts"]
                or (k["model"], k["task"], k["split"], k["arm"]) != (model, task, split, "x") or not g.get("completed")):
            raise SystemExit("%s %s %s x.json is not the no-write x pass of the bank" % (model, task, split))
        if [r["id"] for r in g["rows"]] != [r["id"] for r in rows]:
            raise SystemExit("%s %s %s x.json rows are not the bank split" % (model, task, split))
        out.append(("%s_%s" % (task, split), k["system_text"], k["user_template"],
                    [dict(key="%s_%s|%s:%s" % (task, split, task, b["id"]), question=b["question"],
                          first=(r["token_ids"][0] if r["token_ids"] else -1), cs=None) for b, r in zip(rows, g["rows"])]))
    if len({(s, t) for _, s, t, _ in out}) != 1:
        raise SystemExit("%s: the plain prompt x differs across the generation records" % model)
    # the same question lists (and texts) as K1's items, which the 1g readings are keyed on
    k1 = SP.read_json(os.path.join(SP.OUT, model, "k1_items.json"))["items"]
    for name, _, _, its in out:
        pre = name + "|"
        ref = [x for x in k1 if x["key"].startswith(pre)]
        if name == "tq_test":            # the frozen order differs from the main list's; K3 is keyed, so compare as sets
            byk = {x["key"]: x for x in its}
            if any(r["key"] not in byk for r in ref):
                raise SystemExit("%s TruthfulQA: a main-list question is not in the frozen record" % model)
            mine = [byk[r["key"]] for r in ref]
        else:
            mine = its
        if [x["key"] for x in mine] != [x["key"] for x in ref]:
            raise SystemExit("%s %s: question keys differ from k1_items.json" % (model, name))
        bad = sum(a["question"].strip() != b["question"].strip() for a, b in zip(mine, ref))
        if verbose:
            print("%s %s: %d questions (K1 list %d), question texts differing from K1's: %d" % (model, name, len(its), len(ref), bad), flush=True)
        if bad or len({x["key"] for x in its}) != len(its):
            raise SystemExit("%s %s: mismatched text or duplicate question" % (model, name))
    return fm, out


def capture(model):
    import numpy as np
    import torch
    cfg = SP.MODELS[model]
    code = os.path.abspath(cfg["code"])
    sys.path[:0] = [os.path.join(code, "scripts"), code]
    from core import qa_prompt_das as pd
    from core.config import load_model_config
    from core.io_utils import set_seed
    from core.model_io import _as_hidden as _hid, greedy_generate, load_model
    ref = SP.read_json(cfg["ref"] + ".json")
    for f, h in ref["code_sha256"].items():
        if SP.sha256_file(os.path.join(code, "scripts", f)) != h:
            raise SystemExit("%s: %s differs from the file E45 ran with" % (code, f))
    fm, sets = set_items(model)
    gen = fm["code"]
    checks = {}
    for rel in ("core/qa_prompt_das.py", "core/qa_precision.py"):
        a, b = SP.sha256_file(os.path.join(cfg["code"], rel)), SP.sha256_file(os.path.join(gen, rel))
        checks[rel] = a == b
        if a != b:
            raise SystemExit("%s differs between %s and the generation copy %s" % (rel, cfg["code"], gen))
    if build_chat_source(os.path.join(cfg["code"], "core/model_io.py")) != build_chat_source(os.path.join(gen, "core/model_io.py")):
        raise SystemExit("ModelBundle.build_chat differs between %s and %s" % (cfg["code"], gen))
    mc_a = SP.read_json(os.path.join(cfg["code"], "configs/models/%s.json" % model))
    mc_b = SP.read_json(os.path.join(gen, "configs/models/%s.json" % model))
    for key in ("chat", "path", "dtype"):
        if mc_a.get(key) != mc_b.get(key):
            raise SystemExit("model config %s differs between %s and %s" % (key, cfg["code"], gen))
    print("%s: loading code %s, generation code %s; qa_prompt_das / qa_precision / build_chat / chat config identical" % (
        model, cfg["code"], gen), flush=True)
    out_dir = os.path.join(SP.OUT, model)
    for name, _, _, _ in sets:
        if os.path.exists(os.path.join(out_dir, "k3_%s.npz" % name)):
            raise SystemExit("k3_%s.npz exists" % name)
    set_seed(42)
    bundle = load_model(load_model_config(model))
    L = bundle.num_layers
    head = bundle.model.get_output_embeddings()

    def fp32_argmax(x):
        """Arg-max of the output projection in FP32 (as core.qa_precision.fp32_output_projection computes it: input and
        weight cast to FP32, no TF32), in vocabulary chunks so that no full FP32 copy of the head sits on the GPU."""
        old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            xf = x.float()
            best_v, best_i = None, None
            W = head.weight
            for a in range(0, W.shape[0], 32768):
                lg = torch.nn.functional.linear(xf, W[a:a + 32768].float(),
                                                None if head.bias is None else head.bias[a:a + 32768].float())
                v, i = lg.max(-1)
                i = i + a
                if best_v is None:
                    best_v, best_i = v, i
                else:
                    up = v > best_v
                    best_v, best_i = torch.where(up, v, best_v), torch.where(up, i, best_i)
            return best_i.cpu().numpy(), best_v.cpu().numpy()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old
    meta = dict(model=model, code=cfg["code"], generation_code=gen, system=sets[0][1], user_template=sets[0][2], layers=L,
                sets={}, checks=checks)
    for name, system, template, its in sets:
        ps = [pd.render_prompt(bundle, system, template, x["question"]) for x in its]
        texts = [p["prefix_text"] for p in ps]
        for p in ps:
            if bundle.tokenizer(p["prefix_text"], add_special_tokens=False)["input_ids"] != p["prefix_ids"]:
                raise SystemExit("%s %s: encoding differs from render_prompt's ids" % (model, name))
        if name == "tq_test":
            bad = [x["key"] for x, p in zip(its, ps) if p["content_start"] != x["cs"]]
            if bad:
                raise SystemExit("%s TruthfulQA: content start differs from the frozen record at %d questions" % (model, len(bad)))
        # the generation's own prefill: core.model_io.greedy_generate (model.generate on the left-padded batch of 8, as
        # qa_prompt_das.generate_prompts calls it) with one new token, so the forward is exactly the one that produced
        # the first answer token; hooks keep the last position of every block output and of the head input.
        head_in, rs = [], []
        for i in range(0, len(texts), 8):
            inputs = bundle.encode(texts[i:i + 8])
            width = inputs["input_ids"].shape[1]
            calls = {l: [] for l in range(L)}
            hin = []
            hs = [bundle.layer(l).register_forward_hook(
                lambda mod, inp, out, l=l: calls[l].append((_hid(out).shape[1], _hid(out)[:, -1, :].detach().float().cpu())))
                for l in range(L)]
            hs.append(head.register_forward_pre_hook(lambda mod, args: hin.append((args[0].shape[1], args[0][:, -1, :].detach()))))
            try:
                greedy_generate(bundle, inputs, 1)
            finally:
                for h in hs:
                    h.remove()
            # the head may see only the last position (generate passes logits_to_keep), the blocks see the whole width
            if any(len(calls[l]) < 1 or calls[l][0][0] != width for l in range(L)) or not hin or hin[0][0] not in (width, 1):
                raise SystemExit("%s %s: the hooks did not see the prefill of batch %d (width %d; block calls %s, head %s)" % (
                    model, name, i, width, [(len(calls[l]), calls[l][0][0] if calls[l] else None) for l in (0, L - 1)],
                    [c[0] for c in hin]))
            rs.append(np.stack([calls[l][0][1].numpy() for l in range(L)], axis=1))
            head_in.append(hin[0][1])
        H = np.concatenate(rs, 0).astype(np.float32)
        X = torch.cat(head_in, 0)
        am, top = fp32_argmax(X)
        if len(am) != len(its):
            raise SystemExit("%s %s: %d head inputs for %d prompts" % (model, name, len(am), len(its)))
        first = np.array([x["first"] for x in its])
        has = first >= 0
        agree = int((am[has] == first[has]).sum())
        # gap = top logit - logit of the x answer's first token (FP32); 0 where they agree
        fi = torch.as_tensor(np.where(has, first, 0), device=X.device)
        lf = (X.float() * head.weight[fi].float()).sum(-1)
        if head.bias is not None:
            lf = lf + head.bias[fi].float()
        gap = np.where(has, top - lf.cpu().numpy(), 0.0)
        gap = np.where(am == first, 0.0, gap)
        meta["sets"][name] = dict(n=len(its), with_first_token=int(has.sum()), argmax_agrees=agree, max_gap=float(gap.max()),
                                  disagree_gaps=sorted(float(g) for g in gap[gap > 0]),
                                  read_token=ps[0]["read_token"], prompt_length_min=min(p["prompt_length"] for p in ps),
                                  prompt_length_max=max(p["prompt_length"] for p in ps), shown_prompt=texts[0])
        print("%s %s: %d prompts, read token %r, arg-max next token = first token of the x answer %d / %d; largest gap "
              "(top logit - logit of that token) where they differ %.4f" % (
                  model, name, len(its), ps[0]["read_token"], agree, int(has.sum()), float(gap.max())), flush=True)
        if not has.any() or agree < MIN_AGREE * has.sum() or gap.max() > MAX_GAP:
            raise SystemExit("%s %s: arg-max agreement below %.0f%% or a disagreement more than %.1f logit: the prompt may not "
                             "be the generation's" % (model, name, 100 * MIN_AGREE, MAX_GAP))
        np.savez(os.path.join(out_dir, "k3_%s.npz" % name), keys=np.array([x["key"] for x in its]),
                 question_sha1=np.array([SP.sha1(x["question"]) for x in its]), resid=H, argmax=am, first_token=first)
        meta["sets"][name]["npz_sha256"] = SP.sha256_file(os.path.join(out_dir, "k3_%s.npz" % name))
    with open(os.path.join(out_dir, "k3_capture.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print("%s: capture done" % model, flush=True)


# ------------------------------------------------------------------------------------------------------------------ #
def load_set(model, name):
    import numpy as np
    z = np.load(os.path.join(SP.OUT, model, "k3_%s.npz" % name))
    return [str(k) for k in z["keys"]], z["resid"]


def validate_captured_rows(model, inputs, cap):
    import numpy as np
    if cap["model"] != model or set(cap["sets"]) != set(SETS):
        raise SystemExit("foreign capture or incomplete sets")
    expected = set()
    legacy = {}
    if any("npz_sha256" not in s for s in cap["sets"].values()):
        old = os.path.join(SP.OUT, model, "k3_fit.json")
        if not os.path.isfile(old) or SP.sha256_file(old) != LEGACY[model][0]:
            raise SystemExit(model + ": legacy capture has no checked feature manifest")
        legacy = SP.read_json(old)["npz_sha256"]
    for name, _, _, items in inputs:
        path = os.path.join(SP.OUT, model, "k3_%s.npz" % name)
        h = cap["sets"][name].get("npz_sha256", legacy.get(name))
        if not h or SP.sha256_file(path) != h:
            raise SystemExit("%s: captured feature digest differs" % path)
        keys = [item["key"] for item in items]
        with np.load(path, allow_pickle=False) as z:
            if (len(set(keys)) != len(keys) or cap["sets"][name]["n"] != len(keys)
                    or z["keys"].tolist() != keys
                    or z["question_sha1"].tolist() != [SP.sha1(x["question"]) for x in items]):
                raise SystemExit("%s: captured rows differ from source questions" % path)
        expected.update(keys)
    return expected


def checked_fit(model):
    import math
    out_dir = os.path.join(SP.OUT, model)
    path = os.path.join(out_dir, "k3_fit.json")
    cap_path = os.path.join(out_dir, "k3_capture.json")
    doc = SP.read_json(path)
    cap = SP.read_json(cap_path)
    if "payload_sha256" in doc:
        intact = (doc["payload_sha256"] == payload_digest(doc)
                  and tuple(doc["training_inputs"]) == training_inputs()
                  and doc["capture_sha256"] == SP.sha256_file(cap_path))
    else:
        intact = ((SP.sha256_file(path), SP.sha256_file(cap_path)) == LEGACY.get(model)
                  and training_inputs() == LEGACY_INPUTS)
    if not intact or doc["model"] != model or doc["n_train"] != 100:
        raise SystemExit("foreign or incomplete K3 fit, capture, bank, or pilot: " + model)
    _, inputs = set_items(model, verbose=False)
    expected = validate_captured_rows(model, inputs, cap)
    for n in SETS:
        if doc["npz_sha256"][n] != SP.sha256_file(os.path.join(out_dir, "k3_%s.npz" % n)):
            raise SystemExit("%s: fit belongs to another %s capture" % (model, n))
    for l in (doc["selected_layer"], SP.MODELS[model]["layer"]):
        scores = doc["probs"][str(l)]
        if set(scores) != expected or not all(math.isfinite(v) and 0 <= v <= 1 for v in scores.values()):
            raise SystemExit("%s: invalid K3 probabilities at layer %d" % (model, l))
    return doc, cap


def fit(model):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out = os.path.join(SP.OUT, model, "k3_fit.json")
    if os.path.exists(out):
        raise SystemExit("%s exists" % out)
    _, inputs = set_items(model, verbose=False)
    cap_path = os.path.join(SP.OUT, model, "k3_capture.json")
    cap = SP.read_json(cap_path)
    validate_captured_rows(model, inputs, cap)
    bank = read_jsonl(SP.QAQA_BANK)
    pilot = SP.read_json(SP.QAQA_PILOT)
    pset = set(pilot["ids"])
    keys, H = load_set(model, "qaqa")
    if (keys != ["qaqa|" + r["id"] for r in bank] or len(pset) != 100
            or len(pilot["ids"]) != len(pset) or set(pilot["types"]) != pset):
        raise SystemExit("wrong bank or tune roster")
    tr = [i for i, row in enumerate(bank) if row["id"] in pset]
    if len(tr) != 100 or any(bank[i]["qtype"] != pilot["types"][bank[i]["id"]] for i in tr):
        raise SystemExit("the 100 tune questions")
    y = np.array([bank[i]["qtype"] != "no_failure" for i in tr])
    L = H.shape[1]
    folds = list(StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=CV_SEED).split(np.zeros(len(y)), y))

    def model_():
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="liblinear", random_state=0))
    cv = []
    for l in range(L):
        X = H[tr, l, :].astype(np.float64)
        fa, oof = [], np.zeros(len(y))
        for a, b in folds:
            clf = model_().fit(X[a], y[a])
            p = clf.predict_proba(X[b])[:, 1]
            oof[b] = p
            fa.append(SP.auroc(p, y[b]))
        cv.append(dict(layer=l, mean=float(np.mean(fa)), folds=[float(v) for v in fa], pooled=float(SP.auroc(oof, y))))
        print("%s layer %2d: 5-fold AUROC mean %.4f (folds %s), pooled out-of-fold %.4f" % (
            model, l, cv[-1]["mean"], " ".join("%.3f" % v for v in fa), cv[-1]["pooled"]), flush=True)
    best = max(c["mean"] for c in cv)
    sel = min(c["layer"] for c in cv if c["mean"] == best)
    el = SP.MODELS[model]["layer"]
    probs = {}
    info = {}
    for l in sorted({sel, el}):
        clf = model_().fit(H[tr, l, :].astype(np.float64), y)
        probs[str(l)] = {}
        for name in SETS:
            ks, Hs = load_set(model, name)
            p = clf.predict_proba(Hs[:, l, :].astype(np.float64))[:, 1]
            probs[str(l)].update({k: float(v) for k, v in zip(ks, p)})
        ptr = clf.predict_proba(H[tr, l, :].astype(np.float64))[:, 1]
        info[str(l)] = dict(train_accuracy=float(((ptr > 0.5) == y).mean()), train_auroc=float(SP.auroc(ptr, y)),
                            n_iter=[int(v) for v in np.ravel(clf[-1].n_iter_)], coef_norm=float(np.linalg.norm(clf[-1].coef_)))
    doc = dict(model=model, rule=dict(C=1.0, solver="liblinear", random_state=0, folds=NFOLD, cv_seed=CV_SEED, score="mean fold AUROC",
                                      ties="lower layer", trap="probability > 0.5"),
               n_train=len(tr), n_pos=int(y.sum()), cv=cv, selected_layer=sel, e_layer=el, fit=info, probs=probs,
               npz_sha256={n: SP.sha256_file(os.path.join(SP.OUT, model, "k3_%s.npz" % n)) for n in SETS})
    doc["training_inputs"] = training_inputs()
    doc["capture_sha256"] = SP.sha256_file(cap_path)
    doc["payload_sha256"] = payload_digest(doc)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=0)
    print("%s: selected layer %d (mean 5-fold AUROC %.4f); E layer %d (%.4f) -> %s" % (model, sel, best, el, cv[el]["mean"], out), flush=True)


# ------------------------------------------------------------------------------------------------------------------ #
def table():
    import newsets_score as NS
    import selfjudge_stage1 as S1
    import policy_select2 as ps
    ci_tq = ps.ci

    def ci_one(v):
        tot, lo, hi = NS.joint_ci([list(v)])
        return "%+d [%+d, %+d]" % (tot, lo, hi)

    ORDER, FIVE = SG.ORDER, SG.FIVE
    print("# selfjudge 1g supplement K3 (work queue 2026-11-22_queue_e): linear probe on the plain-prompt residual stream; "
          "trained on the 100 (QA)^2 tune questions only; trap = probability > 0.5")
    # ---- K0 reference (as selfjudge_gate.table) ----------------------------------------------------------------- #
    per = {}
    for m in FIVE:
        ids, d, rows = SG.load_qq(None, m)
        held = [q for q in ids if not rows[q]["pilot"]]
        trap = {q: rows[q]["qtype"] != "no_failure" for q in ids}
        per[m] = (held, SG.gate(held, d, trap, sj=False))
    held = per[FIVE[0]][0]
    if any(per[m][0] != held for m in FIVE) or len(held) != 470:
        raise SystemExit("(QA)^2 held-out lists")
    gb = ci_tq([sum(per[m][1]["G"][q] - per[m][1]["B"][q] for m in FIVE) for q in held])
    gbg = ci_tq([sum(per[m][1]["G"][q] - per[m][1]["Bg"][q] for m in FIVE) for q in held])
    okk = gb == SG.K0_EXPECT["gb"] and gbg == SG.K0_EXPECT["gbg"]
    print("# K0 reference, five models, (QA)^2 470: G-B %s (expected %s), G-Bg %s (expected %s)" % (gb, SG.K0_EXPECT["gb"], gbg, SG.K0_EXPECT["gbg"]))
    for m in FIVE:
        r = per[m][1]
        v = sum(r["G"][q] - r["Bg"][q] for q in held)
        print("#   %-12s G-Bg %+d, answers after gate %d" % (m, v, r["n"]["G"]))
        if m in SG.K0_EXPECT["bg"]:
            okk = okk and v == SG.K0_EXPECT["bg"][m] and r["n"]["G"] == SG.K0_EXPECT["n"][m]
    if not okk:
        raise SystemExit("K0 REFERENCE MISMATCH: stopping before any reading")
    print("# K0 reference reproduced")
    A1 = S1.load_all()
    K1 = {m: SG.k1_margins(m) for m in ORDER}
    TQ = SG.load_tq(A1)
    QQ = {m: SG.load_qq(A1, m) for m in ORDER}
    FIT, K3 = {}, {}
    for m in ORDER:
        doc, cap = checked_fit(m)
        FIT[m] = doc
        K3[m] = {"sel": doc["probs"][str(doc["selected_layer"])], "E": doc["probs"][str(doc["e_layer"])]}
        print("# %s capture: %s" % (m, " | ".join("%s n %d, arg-max = first x token %d / %d" % (
            n, v["n"], v["argmax_agrees"], v["with_first_token"]) for n, v in cap["sets"].items())))
    TYPES = ("no_failure", "failure", "unverifiable")
    print("=" * 100)
    print("CV: 5-fold stratified AUROC inside the 100 tune questions, per layer (mean of folds; pooled out-of-fold)")
    for m in ORDER:
        doc = FIT[m]
        print(" %s (train %d, positive %d): selected layer %d, E layer %d" % (m, doc["n_train"], doc["n_pos"], doc["selected_layer"], doc["e_layer"]))
        print("   " + " ".join("L%d %.3f" % (c["layer"], c["mean"]) for c in doc["cv"]))
        print("   pooled: " + " ".join("L%d %.3f" % (c["layer"], c["pooled"]) for c in doc["cv"]))
        for l, v in doc["fit"].items():
            print("   refit on 100 at layer %s: train accuracy %.2f, train AUROC %.3f, liblinear iterations %s" % (l, v["train_accuracy"], v["train_auroc"], v["n_iter"]))
    # ---- reading 1 ------------------------------------------------------------------------------------------------ #
    SW1 = (("K1", lambda m: K1[m], 0.0), ("K3 selected layer", lambda m: K3[m]["sel"], 0.5), ("K3 E layer", lambda m: K3[m]["E"], 0.5))
    print("=" * 100)
    print("READING 1: switch accuracy (K1 trap = margin > 0; K3 trap = probability > 0.5)")
    for m in ORDER:
        ids, d, rows = QQ[m]
        ids_t = TQ[m][0]
        print(" %s (K3 selected layer %d, E layer %d)" % (m, FIT[m]["selected_layer"], FIT[m]["e_layer"]))
        for swn, get, thr in SW1:
            k = get(m)
            for lab, sub in (("470 held out", [q for q in ids if not rows[q]["pilot"]]), ("570", ids), ("100 tune", [q for q in ids if rows[q]["pilot"]])):
                y = [rows[q]["qtype"] != "no_failure" for q in sub]
                sc = [k["qaqa|" + q] for q in sub]
                pt, lo, hi = SG.auroc_ci(sc, y)
                acc = sum((s > thr) == t for s, t in zip(sc, y))
                print("   %-17s (QA)^2 %-12s AUROC %.3f [%.3f, %.3f] | accuracy %d / %d (%.1f%%) | judged trap: %s" % (
                    swn, lab, pt, lo, hi, acc, len(sub), 100.0 * acc / len(sub), " | ".join(
                        "%s %d / %d" % (t, sum(k["qaqa|" + q] > thr for q in sub if rows[q]["qtype"] == t), sum(rows[q]["qtype"] == t for q in sub))
                        for t in TYPES)))
            ks = [k["tq_test|" + q] for q in ids_t]
            line = "   %-17s TruthfulQA test (main list %d) judged trap %d (%.1f%%)" % (swn, len(ids_t), sum(v > thr for v in ks), 100.0 * sum(v > thr for v in ks) / len(ks))
            for task, split in SG.FACT:
                fk = [v for kk, v in k.items() if kk.startswith("%s_%s|" % (task, split))]
                line += " | %s %s judged normal %d / %d (%.1f%%)" % (task, split, sum(v <= thr for v in fk), len(fk), 100.0 * sum(v <= thr for v in fk) / len(fk))
            print(line)
    # ---- reading 2 ------------------------------------------------------------------------------------------------ #
    print("=" * 100)
    print("READING 2: G against B and Bg (same answer count; selfjudge_gate.gate unchanged); G-N1, G-E for context; G+SJ side")
    for sw in ("K0", "K1", "K3"):
        print("-- switch %s%s" % (sw, " (selected layer)" if sw == "K3" else ""))
        tot = {}

        def trap_of(m, key, rows=None, q=None):
            if sw == "K0":
                return rows[q]["qtype"] != "no_failure"
            if sw == "K1":
                return K1[m][key] > 0
            return K3[m]["sel"][key] > 0.5
        for m in ORDER:
            ids, d, rows = QQ[m]
            trap = {q: trap_of(m, "qaqa|" + q, rows, q) for q in ids}
            print("  %s (QA)^2" % m)
            for lab, sub in (("470 held out", [q for q in ids if not rows[q]["pilot"]]), ("570", ids)):
                r = SG.gate(sub, d, trap)
                SG.show(lab, sub, r, ci_one)
                tot.setdefault(lab, {})[m] = (sub, SG.cells(r, sub))
            if sw == "K0":
                continue
            ids_t, d_t, cand = TQ[m]
            trap_t = {q: trap_of(m, "tq_test|" + q) for q in ids_t}
            print("  %s TruthfulQA test" % m)
            SG.show("TruthfulQA test", ids_t, SG.gate(ids_t, d_t, trap_t), ci_tq)
            for task, split in SG.FACT:
                for version in ("F", "D"):
                    fids, fd, fc, cfg = SG.load_fact(m, task, split, version)
                    trap_f = {q: trap_of(m, "%s_%s|%s" % (task, split, q)) for q in fids}
                    r = SG.gate(fids, fd, trap_f, strong_ok=False)
                    SG.show("%s %s N1 %s %s" % (task, split, version, cfg), fids, r, ci_one)
                    tot.setdefault("%s %s %s" % (task, split, version), {})[m] = (fids, SG.cells(r, fids))
        print("  two models together:")
        for lab, pm in tot.items():
            sub = pm[ORDER[0]][0]
            if pm[ORDER[1]][0] != sub:
                raise SystemExit("lists differ")
            line = []
            for k, _, _ in SG.DIFF:
                if any(pm[m][1][k] is None for m in ORDER):
                    line.append("%s not comparable" % k)
                else:
                    t, lo, hi = NS.joint_ci([pm[m][1][k] for m in ORDER])
                    line.append("%s %+d [%+d, %+d]" % (k, t, lo, hi))
            print("    %s, %d questions: %s" % (lab, len(sub), " | ".join(line)))
        if sw == "K0":
            continue
        common = sorted(set(TQ[ORDER[0]][0]) & set(TQ[ORDER[1]][0]))
        cs = {}
        for m in ORDER:
            ids_t, d_t, cand = TQ[m]
            trap_t = {q: trap_of(m, "tq_test|" + q) for q in common}
            cs[m] = SG.cells(SG.gate(common, d_t, trap_t), common)
        print("    TruthfulQA test, %d common questions: %s" % (len(common), " | ".join(
            "%s %s" % (k, "not comparable" if any(cs[m][k] is None for m in ORDER) else ci_tq([sum(cs[m][k][i] for m in ORDER) for i in range(len(common))]))
            for k, _, _ in SG.DIFF)))


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "capture" and sys.argv[2] in SG.ORDER:
        capture(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "fit" and sys.argv[2] in SG.ORDER:
        fit(sys.argv[2])
    elif sys.argv[1:] == ["table"]:
        table()
    else:
        raise SystemExit("usage: selfjudge_gate_k3.py capture MODEL | fit MODEL | table")


if __name__ == "__main__":
    main()
