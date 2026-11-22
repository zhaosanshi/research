# -*- coding: utf-8 -*-
"""Exploration (Claude Fable 5.2, 2026-11-22): Inference-Time Intervention (ITI; Li et al., NeurIPS 2023) as an
external baseline inside this project's own evaluation protocol: the same model loader and chat rendering (the
engine's Context, as scripts/newsets/newsets_gen.py builds it), the same greedy generation (48 new tokens, batches of
8, float32 output projection), the same judge inputs. What the authors did, item by item, with pointers into the
paper and the official code: novelty/prior_work/2026-11-22_iti_method_spec.md.

The method as implemented here (every step follows the official code unless marked "ours"):
  activations  for every (question, answer) pair of the fit questions (all reference answers: true = label 1,
               false = label 0), the text "Q: {question} A: {answer}" is run through the model and the input of the
               attention output projection (o_proj) at the LAST token is taken, per layer; split into heads it is the
               head-wise activation [layers, heads, head_dim].
  probes       the fit questions are split by question into a train part and a validation part (the official draw:
               a seeded random choice of 1 - val_ratio of the questions for training, val_ratio 0.2, seed 42); one
               logistic regression per head on the train part; heads are ranked by validation accuracy.
  direction    per head, the mass-mean shift: mean activation of the true pairs minus mean of the false pairs over
               all fit pairs, normalised.
  sigma        the standard deviation of activations projected on that direction. The paper (section 3.3) takes the
               training and validation activations; the official code takes a second activation bank, the last token
               of "Q: {question} A: {answer} Q: {a random other question}" (no label used), and takes it over ALL
               questions including its test fold (the maintainers acknowledged this in the repository's issue 46).
               Here both are computed on the fit questions only: --sigma endq (default, the code's bank) and
               --sigma pairs (the paper's wording).
  intervene    for the top K heads: alpha * sigma * direction is added to the head's activation at the last token of
               the forward ("lt" in the official code): with a key-value cache this is the last prompt token and then
               every generated token; the question tokens are untouched.
  ours         the fit questions are this project's training split (96 questions; optionally training + tune), K and
               alpha are chosen on the tune split under the same MMLU budget as the project's own direction, and the
               generation prompt is the project's plain chat prompt "x" (the official instruction prompt and few-shot
               primer belong to a completion-style evaluation that this project does not run).

  faithful (fit --faithful, added 2026-11-22 10 h after the read-only review; earlier fits are unchanged and keep
               their own rules, recorded in the fit file):
               - probe texts: the reference answer gets a full stop when it ends without . ! or ?, because the
                 official default probe bank (the multiple-choice targets of TruthfulQA) ends every choice with a
                 full stop, so the official "last token" is mostly the full stop (checked on rows of the public
                 dataset); the sigma bank keeps the answers as they are, as the official generation-split bank does;
               - sigma bank: the random other question is drawn once per question and shared by all its answers
                 (official tokenized_tqa_gen_end_q), from the fit questions only;
               - head order: numpy argsort of the flattened [layer, head] accuracies, reversed (official
                 get_top_heads), instead of "higher accuracy, then lower layer and head".

Phases (model interpreter, from /workspace/Truth_Decoupling):
  fit     --out FIT.pt [--fit-splits train[,tune]] [--val-ratio R] [--seed S] [--faithful]
  check   --fit FIT.pt --k K --alpha A        acceptance of the intervention code on the first two tune batches
  gen     --fit FIT.pt --split tune|test --k K --alpha A --out GEN.json [--first-batches N]
  mmlu    --fit FIT.pt --k K --alpha A --items PARQUET --ids IDS.json --out OUT.json [--first-batches N]
          (K = 0 scores the model without any intervention)
  gsm8k   --fit FIT.pt --k K --alpha A --items PARQUET --ids IDS.json --out OUT.json [--max-new-tokens N]
          the 1,319 GSM8K items under the intervention, neutral system prompt, as the project's own GSM8K passes
          (K = 0: no intervention, for the check against the usual no-write pass)
  export  --like JUDGE_INPUT.json --gen GEN.json [GEN.json ...] --out JUDGE_INPUT_OUT.json     (no model)
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
PROTOCOL = "iti_baseline_v1"
LETTERS = "ABCD"


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new(out, text):
    """Write a new file; never replace one (two lanes computing the same point used to race on the exists check).
    The text goes to a temporary file first and is then linked to the final name, which fails if the name exists."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(".%s.%d.tmp" % (out.name, os.getpid()))
    tmp.write_text(text, encoding="utf-8")
    try:
        os.link(tmp, out)
    except FileExistsError:
        raise SystemExit("%s exists" % out)
    finally:
        tmp.unlink()


def open_run(a):
    """(newsets_gen module, model config, engine module, context) with the model loaded."""
    import newsets_gen as NG
    m = NG.load_models(a.models, a.model)
    X = NG.engine(m)
    ctx = NG.context(X, m, 0)
    return NG, m, X, ctx


def head_shape(bundle):
    cfg = getattr(bundle.model.config, "text_config", None)
    if cfg is None or not hasattr(cfg, "num_attention_heads"):
        cfg = bundle.model.config
    heads = int(cfg.num_attention_heads)
    proj = bundle.attn(0).o_proj
    if proj.bias is not None or proj.in_features % heads:
        raise SystemExit("the attention output projection must be bias-free and split evenly into heads")
    return heads, proj.in_features // heads


def with_stop(answer):
    a = answer.rstrip()
    return a if a.endswith((".", "!", "?")) else a + "."


def pair_text(bundle, question, answer, next_question=None):
    """The official activation texts (format_truthfulqa / format_truthfulqa_end_q); the tokenizer's begin-of-sequence
    token is written out because bundle.encode adds no special tokens."""
    text = "Q: %s A: %s" % (question, answer)
    if next_question is not None:
        text += " Q: %s" % next_question
    return (bundle.tokenizer.bos_token or "") + text


def capture_last(X, bundle, texts, batch=8):
    """[N, layers, heads * head_dim] float32: the input of every layer's o_proj at the last token."""
    import torch
    L = bundle.num_layers
    store, handles = {}, []

    def make(l):
        def pre(module, args):
            store[l] = args[0][:, -1, :].detach().float().cpu()
        return pre

    for l in range(L):
        handles.append(bundle.attn(l).o_proj.register_forward_pre_hook(make(l)))
    out = []
    try:
        for i in range(0, len(texts), batch):
            inputs = bundle.encode(texts[i:i + batch])
            if not bool(inputs["attention_mask"][:, -1].all()):
                raise SystemExit("left padding expected: the last column must be every row's last token")
            store.clear()
            X.forward_logits(bundle, inputs, grad=False, logits_to_keep=1)
            if len(store) != L:
                raise SystemExit("an attention output projection did not run")
            out.append(torch.stack([store[l] for l in range(L)], dim=1))
    finally:
        for h in handles:
            h.remove()
    return torch.cat(out)


def run_fit(a):
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    NG, m, X, ctx = open_run(a)
    bundle = ctx.bundle
    H, d = head_shape(bundle)
    L = bundle.num_layers
    splits = [s for s in a.fit_splits.split(",") if s]
    if not splits or any(s not in ("train", "tune") for s in splits):
        raise SystemExit("--fit-splits takes train and / or tune; the test questions are never used for fitting")
    questions = [q for s in splits for q in ctx.prepared["splits"][s]]
    rows = []                                           # (question index, label, text, text of the sigma bank)
    other = np.random.RandomState(a.seed + 1)           # the official draw of the appended question has no seed
    for k, q in enumerate(questions):
        if a.faithful:                                  # one random question per question, shared by its answers
            nxt_q = questions[int(other.randint(len(questions)))]["question"]
        for label, key in ((1, "gold"), (0, "incorrect")):
            for ans in q[key]:
                nxt = nxt_q if a.faithful else questions[int(other.randint(len(questions)))]["question"]
                probe = with_stop(ans) if a.faithful else ans
                rows.append((k, label, pair_text(bundle, q["question"], probe), pair_text(bundle, q["question"], ans, nxt)))
    t0 = time.time()
    with torch.no_grad():
        acts = capture_last(X, bundle, [r[2] for r in rows]).numpy().reshape(len(rows), L, H, d)
        endq = capture_last(X, bundle, [r[3] for r in rows]).numpy().reshape(len(rows), L, H, d)
    if not (np.isfinite(acts).all() and np.isfinite(endq).all()):
        raise SystemExit("non-finite activations")
    qidx = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    n_q = len(questions)
    train_q = np.random.RandomState(a.seed).choice(n_q, size=int(n_q * (1 - a.val_ratio)), replace=False)
    val_q = set(range(n_q)) - set(train_q.tolist())
    is_val = np.array([k in val_q for k in qidx])
    tr, va = ~is_val, is_val
    if min(y[tr].sum(), (1 - y[tr]).sum(), y[va].sum(), (1 - y[va]).sum()) < 2:
        raise SystemExit("a part has fewer than two pairs of a class")
    acc = np.zeros((L, H))
    unit = np.zeros((L, H, d), dtype=np.float32)
    sigma = np.zeros((L, H))
    sigma_pairs = np.zeros((L, H))
    for l in range(L):
        for h in range(H):
            x = acts[:, l, h, :]
            clf = LogisticRegression(random_state=a.seed, max_iter=1000).fit(x[tr], y[tr])
            acc[l, h] = clf.score(x[va], y[va])
            com = x[y == 1].mean(0) - x[y == 0].mean(0)
            u = com / np.linalg.norm(com)
            unit[l, h] = u
            sigma_pairs[l, h] = float(np.std(x @ u))
            sigma[l, h] = float(np.std(endq[:, l, h, :] @ u))
        print("layer %2d: best validation accuracy %.3f" % (l, acc[l].max()), flush=True)
    rule = "numpy_argsort_desc" if a.faithful else "accuracy_then_layer_head"
    rank = ranked(acc, rule)
    out = Path(a.out)
    if out.exists():
        raise SystemExit("%s exists" % out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"protocol": PROTOCOL, "phase": "fit", "model": a.model, "identity_sha256": ctx.ident_sha,
            "script_sha256": sha_file(__file__), "fit_splits": splits, "seed": a.seed, "val_ratio": a.val_ratio,
            "questions": len(questions), "question_ids": [q["pair_id"] for q in questions],
            "validation_question_ids": [questions[k]["pair_id"] for k in sorted(val_q)],
            "pairs": len(rows), "pairs_true": int(y.sum()), "pairs_train": int(tr.sum()), "pairs_validation": int(va.sum()),
            "text_format": "<bos>Q: {question} A: {answer}" + (" (answer with a full stop added when it ends without . ! or ?)" if a.faithful else ""),
            "faithful": bool(a.faithful), "rank_rule": rule,
            "sigma_draw": "one random fit question per question" if a.faithful else "one random fit question per pair",
            "position": "last token", "site": "input of the attention output projection, split into heads",
            "layers": L, "heads": H, "head_dim": d, "probe": "sklearn LogisticRegression(random_state=seed, max_iter=1000)",
            "direction": "mass mean shift (true minus false) over all fit pairs, normalised",
            "sigma": "std along the direction of the last-token activations of '<bos>Q: {question} A: {answer} Q: {random fit question}' over all fit pairs (the official code's bank, restricted to the fit questions)",
            "sigma_pairs": "std along the direction of the fit activations themselves (the paper's wording)",
            "ranking": [[l, h, v] for v, l, h in rank[:256]], "seconds": time.time() - t0}
    top = [(l, h) for _, l, h in rank[:48]]
    ratio = [sigma[l, h] / sigma_pairs[l, h] for l, h in top]
    meta["sigma_over_sigma_pairs_top48"] = {"min": float(min(ratio)), "median": float(sorted(ratio)[24]), "max": float(max(ratio)),
                                            "correlation": float(np.corrcoef([sigma[l, h] for l, h in top], [sigma_pairs[l, h] for l, h in top])[0, 1])}
    tmp = out.with_name(".%s.%d.tmp" % (out.name, os.getpid()))
    torch.save({"meta": meta, "accuracy": torch.tensor(acc), "unit": torch.tensor(unit), "sigma": torch.tensor(sigma),
                "sigma_pairs": torch.tensor(sigma_pairs)}, tmp)
    try:
        os.link(tmp, out)
    except FileExistsError:
        raise SystemExit("%s exists" % out)
    finally:
        tmp.unlink()
    write_new(str(out) + ".json", json.dumps(meta, indent=1))
    print("fit: %d questions (%d for validation), %d pairs (%d true), validation accuracy of the best head %.3f, of head no. 48 %.3f; "
          "sigma (end-of-next-question bank) over sigma (the pairs) on the top 48 heads: %s; %.0fs" % (
              len(questions), len(val_q), len(rows), int(y.sum()), rank[0][0], rank[47][0], meta["sigma_over_sigma_pairs_top48"], time.time() - t0), flush=True)


def load_fit(path, model, ident):
    import torch
    f = torch.load(path, map_location="cpu", weights_only=False)
    meta = f["meta"]
    if meta.get("protocol") != PROTOCOL or meta.get("model") != model or meta.get("identity_sha256") != ident:
        raise SystemExit("%s is not an ITI fit of this model and run" % path)
    return f


def ranked(acc, rule):
    """[(accuracy, layer, head)] best first. accuracy_then_layer_head: ties to the lower layer, then head (the first
    fits); numpy_argsort_desc: the official get_top_heads, np.argsort of the flattened [layer, head] array, reversed."""
    import numpy as np
    a = np.asarray(acc, dtype=float)
    L, H = a.shape
    if rule == "numpy_argsort_desc":
        return [(float(a[i // H, i % H]), int(i // H), int(i % H)) for i in np.argsort(a.reshape(L * H))[::-1]]
    if rule == "accuracy_then_layer_head":
        return sorted(((float(a[l, h]), l, h) for l in range(L) for h in range(H)), key=lambda t: (-t[0], t[1], t[2]))
    raise SystemExit("unknown head order %r" % rule)


def head_vectors(bundle, fit, k, alpha, which="endq"):
    """{layer: [heads * head_dim] float32 tensor} = alpha * sigma * direction on the slices of the chosen heads,
    and the list of (layer, head, validation accuracy) chosen. which: "endq" (the code's sigma) or "pairs"."""
    import torch
    acc, unit, sigma = fit["accuracy"], fit["unit"], fit[{"endq": "sigma", "pairs": "sigma_pairs"}[which]]
    L, H = acc.shape
    d = unit.shape[2]
    rank = ranked(acc.numpy() if hasattr(acc, "numpy") else acc, fit["meta"].get("rank_rule", "accuracy_then_layer_head"))
    if not 0 <= k <= len(rank):
        raise SystemExit("K out of range")
    chosen = rank[:k]
    vecs = {}
    for v, l, h in chosen:
        vecs.setdefault(l, torch.zeros(H * d))
        vecs[l][h * d:(h + 1) * d] = float(alpha) * float(sigma[l, h]) * unit[l, h].float()
    return vecs, [(l, h, v) for v, l, h in chosen]


class HeadShift:
    """Forward pre-hook of an attention output projection: add `vec` to its input at the last position of every
    forward (the official `head_output[:, -1, head, :] += alpha * std * direction`). In a cached greedy generation
    the first forward holds the whole prompt (its last column is every row's last prompt token under left padding)
    and every later forward holds one new token."""

    def __init__(self, vec):
        self.vec = vec
        self.calls = 0
        self.widths = []

    def __call__(self, module, args):
        x = args[0]
        self.calls += 1
        self.widths.append(int(x.shape[1]))
        if self.calls > 1 and x.shape[1] != 1:
            raise RuntimeError("a decode forward of width %d: the intervention assumes a key-value cache" % x.shape[1])
        x = x.clone()
        x[:, -1, :] = (x[:, -1, :].float() + self.vec.to(x.device)).to(x.dtype)
        return (x,) + tuple(args[1:])


class Intervention:
    """Context manager: the ITI hooks on the model for one forward or one generation."""

    def __init__(self, bundle, vecs):
        self.bundle, self.vecs = bundle, vecs
        self.hooks, self.handles = {}, []

    def __enter__(self):
        for l, vec in self.vecs.items():
            self.hooks[l] = HeadShift(vec)
            self.handles.append(self.bundle.attn(l).o_proj.register_forward_pre_hook(self.hooks[l]))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        return False

    def audit(self, prompt_width, single_forward=False):
        for l, h in self.hooks.items():
            if not h.calls or h.widths[0] != prompt_width or (single_forward and h.calls != 1):
                raise RuntimeError("the intervention at layer %d did not fire as planned (widths %s)" % (l, h.widths[:3]))
        calls = {h.calls for h in self.hooks.values()}
        if len(calls) > 1:
            raise RuntimeError("the hooks did not fire on the same forwards")
        return calls.pop() if calls else 0


def generate(X, bundle, prompts, vecs, max_new_tokens, batch_size=8):
    """Greedy answers of rendered prompts under the intervention (vecs may be empty = no intervention). Record
    layout of the engine's generation functions: text, token_ids, finished, stop_id, n_tokens (+ forwards)."""
    import torch
    pd = X.pd
    stops = pd.stop_ids(bundle)
    results = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            inputs = bundle.encode([p["prefix_text"] for p in chunk])
            for b, p in enumerate(chunk):
                got = inputs["input_ids"][b][inputs["attention_mask"][b].bool()].tolist()
                if got != p["prefix_ids"]:
                    raise RuntimeError("batched prompt tokens differ from the single-prompt rendering")
            if not bool(inputs["attention_mask"][:, -1].all()):
                raise RuntimeError("left padding expected")
            width = inputs["input_ids"].shape[1]
            with Intervention(bundle, vecs) as iv:
                _, out = X.greedy_generate(bundle, inputs, max_new_tokens)
                forwards = iv.audit(width)
            for b, p in enumerate(chunk):
                ids, finished, stop_id = pd._cut_new_tokens(out[b][width:], stops)
                results.append({"text": bundle.tokenizer.decode(ids, skip_special_tokens=True).strip(), "token_ids": ids,
                                "finished": finished, "stop_id": stop_id, "n_tokens": len(ids), "forwards": forwards})
    return results


def setting(a, fit, chosen):
    return {"k": a.k, "alpha": a.alpha, "sigma": a.sigma, "heads": [[l, h, v] for l, h, v in chosen],
            "layers_touched": sorted({l for l, _, _ in chosen}), "fit_file_sha256": sha_file(a.fit),
            "fit_splits": fit["meta"]["fit_splits"], "positions": "last token of every forward (last prompt token, then every generated token)"}


def run_check(a):
    """Acceptance on the first two tune batches: (1) without hooks the function reproduces the run's frozen no-write
    answers token for token; (2) with alpha = 0 the answers are those of (1); (3) with the requested K and alpha the
    hooks fire on the prefill and on every decode forward, and the logits of the first new token equal, up to
    rounding, those of the same shift written directly on the attention module's OUTPUT, W_O times the vector (o_proj
    is linear and has no bias; for models with a norm after attention, such as Gemma-3, that output is normed before
    it reaches the residual stream, and the two forms are still the same computation)."""
    import torch
    NG, m, X, ctx = open_run(a)
    bundle = ctx.bundle
    fit = load_fit(a.fit, a.model, ctx.ident_sha)
    systems = NG.systems_of(ctx)
    qs = ctx.questions[:16]
    with X.E.fp32_output_projection(bundle, enabled=True):
        prompts = [X.pd.render_prompt(bundle, systems["x"], ctx.user_template, q["question"]) for q in qs]
        frozen = [ctx.look[q["pair_id"]]["answers"]["x"] for q in qs]         # as the engine's verified_base reads them
        plain = generate(X, bundle, prompts, {}, NG.ENGINE_MAX_NEW)
        bad = [q["pair_id"] for q, f, r in zip(qs, frozen, plain)
               if any(r.get(k) != f.get(k) for k in ("text", "token_ids", "finished", "stop_id"))]
        print("check 1: %d of %d no-write answers equal the run's frozen answers token for token" % (len(qs) - len(bad), len(qs)), flush=True)
        vecs0, _ = head_vectors(bundle, fit, a.k, 0.0, a.sigma)
        zero = generate(X, bundle, prompts, vecs0, NG.ENGINE_MAX_NEW)
        same0 = sum(r["token_ids"] == p["token_ids"] for r, p in zip(zero, plain))
        print("check 2: %d of %d answers unchanged with alpha = 0 (%d hooks, %d forwards in the first batch)" % (
            same0, len(qs), len(vecs0), zero[0]["forwards"]), flush=True)
        vecs, chosen = head_vectors(bundle, fit, a.k, a.alpha, a.sigma)
        shifted = generate(X, bundle, prompts, vecs, NG.ENGINE_MAX_NEW)
        changed = sum(r["token_ids"] != p["token_ids"] for r, p in zip(shifted, plain))
        inputs = bundle.encode([p["prefix_text"] for p in prompts[:8]])
        with torch.no_grad():
            with Intervention(bundle, vecs) as iv:
                la = X.forward_logits(bundle, inputs, grad=False, logits_to_keep=1)[:, -1, :].float()
                iv.audit(inputs["input_ids"].shape[1], single_forward=True)
            handles = []
            for l, vec in vecs.items():
                w = bundle.attn(l).o_proj.weight.detach().float()
                delta = (w @ vec.to(w.device)).cpu()

                def post(module, args, out, delta=delta):
                    h = out[0] if isinstance(out, tuple) else out
                    h = h.clone()
                    h[:, -1, :] = (h[:, -1, :].float() + delta.to(h.device)).to(h.dtype)
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

                handles.append(bundle.attn(l).register_forward_hook(post))
            try:
                lb = X.forward_logits(bundle, inputs, grad=False, logits_to_keep=1)[:, -1, :].float()
            finally:
                for h in handles:
                    h.remove()
            l0 = X.forward_logits(bundle, inputs, grad=False, logits_to_keep=1)[:, -1, :].float()
        pa, pb, p0 = (torch.log_softmax(t, -1) for t in (la, lb, l0))
        agree = int((pa.argmax(-1) == pb.argmax(-1)).sum())
        gap_ab = float((pa - pb).abs().max())
        gap_a0 = float((pa - p0).abs().max())
        print("check 3: K %d alpha %s touches layers %s; %d of %d answers changed; first new token: head-space and "
              "block-output forms agree on %d of %d arg-max, largest log-probability gap %.4f (against %.4f between "
              "the intervention and no intervention)" % (a.k, a.alpha, sorted(vecs), changed, len(qs), agree, len(prompts[:8]), gap_ab, gap_a0), flush=True)
    ok = not bad and same0 == len(qs) and agree == len(prompts[:8]) and gap_ab < 0.25 * max(gap_a0, 1e-9)
    print("check %s" % ("PASS" if ok else "FAIL"), flush=True)
    if not ok:
        raise SystemExit(1)


def run_gen(a):
    NG, m, X, ctx = open_run(a)
    bundle = ctx.bundle
    fit = load_fit(a.fit, a.model, ctx.ident_sha)
    systems = NG.systems_of(ctx)
    questions = list(ctx.prepared["splits"][a.split])
    if a.first_batches:
        questions = questions[:a.first_batches * NG.BATCH]
    out = Path(a.out)
    if out.exists():
        raise SystemExit("%s exists" % out)
    vecs, chosen = head_vectors(bundle, fit, a.k, a.alpha, a.sigma)
    t0 = time.time()
    with X.E.fp32_output_projection(bundle, enabled=True):
        prompts = [X.pd.render_prompt(bundle, systems["x"], ctx.user_template, q["question"]) for q in questions]
        recs = generate(X, bundle, prompts, vecs, NG.ENGINE_MAX_NEW)
    rows = [dict(r, pair_id=q["pair_id"], rule_abstain=bool(X.rule_abstains(r["text"]))) for r, q in zip(recs, questions)]
    payload = {"protocol": PROTOCOL, "phase": "gen", "completed": not a.first_batches, "model": a.model, "split": a.split,
               "base_prompt": "x", "system_text": systems["x"], "user_template": ctx.user_template,
               "identity_sha256": ctx.ident_sha, "script_sha256": sha_file(__file__), "engine_sha256": sha_file(X.__file__),
               "max_new_tokens": NG.ENGINE_MAX_NEW, "batch_size": NG.BATCH, "setting": setting(a, fit, chosen),
               "condition": "ITI_k%d_a%s%s" % (a.k, ("%g" % a.alpha).replace(".", "-"), "" if a.sigma == "endq" else "_sp"),
               "question_ids": [q["pair_id"] for q in questions], "rows": rows,
               "summary": {"n": len(rows), "rule_abstain": sum(r["rule_abstain"] for r in rows),
                           "unfinished": sum(not r["finished"] for r in rows), "seconds": time.time() - t0}}
    write_new(out, json.dumps(payload))
    s = payload["summary"]
    print("%s %s %s: %d questions, rule abstain %d, unfinished %d, %.0fs" % (
        a.model, a.split, payload["condition"], s["n"], s["rule_abstain"], s["unfinished"], s["seconds"]), flush=True)


def run_mmlu(a):
    """The first-answer-token probability of the letters A-D at the last prompt token (what cap_mcq_prob.py reads
    through its teacher-forced scorer), under the intervention; K = 0 is the model itself."""
    import torch
    import pyarrow.parquet as pq
    NG, m, X, ctx = open_run(a)
    bundle = ctx.bundle
    systems = NG.systems_of(ctx)
    a.task, a.split, a.only_ids = "mmlu", "all", None
    items, template = NG.load_items(a)
    if a.first_batches:
        items = items[:a.first_batches * NG.BATCH]
    tok = bundle.tokenizer
    letter_ids = {}
    for L in LETTERS:
        cands = []
        for spelling in (L, " " + L):
            ids = tok.encode(spelling, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in cands:
                cands.append(ids[0])
        if not cands:
            raise SystemExit("letter %s has no single-token spelling for this tokenizer" % L)
        letter_ids[L] = cands
    per = {}
    for r in pq.read_table(a.items).to_pylist():
        per.setdefault(r["subject"], []).append(r)
    gold = {i: LETTERS[per[i.split(":")[1]][int(i.split(":")[2])]["answer"]] for i, _ in items}
    if a.k:
        fit = load_fit(a.fit, a.model, ctx.ident_sha)
        vecs, chosen = head_vectors(bundle, fit, a.k, a.alpha, a.sigma)
        st = setting(a, fit, chosen)
    else:
        vecs, st = {}, {"k": 0}
    out = Path(a.out)
    if out.exists():
        raise SystemExit("%s exists" % out)
    recs, t0 = [], time.time()
    with X.E.fp32_output_projection(bundle, enabled=True), torch.no_grad():
        prompts = [X.pd.render_prompt(bundle, systems["x"], template, q) for _, q in items]
        for i in range(0, len(prompts), NG.BATCH):
            chunk = prompts[i:i + NG.BATCH]
            inputs = bundle.encode([p["prefix_text"] for p in chunk])
            if not bool(inputs["attention_mask"][:, -1].all()):
                raise RuntimeError("left padding expected")
            with Intervention(bundle, vecs) as iv:
                logp = torch.log_softmax(X.forward_logits(bundle, inputs, grad=False, logits_to_keep=1)[:, -1, :].float(), -1)
                iv.audit(inputs["input_ids"].shape[1], single_forward=True)
            for b in range(len(chunk)):
                iid = items[i + b][0]
                v = {L: float(torch.logsumexp(logp[b, letter_ids[L]], 0)) for L in LETTERS}
                pick = max(LETTERS, key=lambda L: (v[L], -LETTERS.index(L)))
                recs.append({"id": iid, "logp": v, "pick": pick, "gold": gold[iid], "letter_mass": sum(math.exp(x) for x in v.values())})
    acc = sum(r["pick"] == r["gold"] for r in recs) / len(recs)
    payload = {"protocol": PROTOCOL, "phase": "mmlu", "completed": not a.first_batches, "model": a.model,
               "identity_sha256": ctx.ident_sha, "script_sha256": sha_file(__file__), "items_file_sha256": sha_file(a.items),
               "ids_file_sha256": sha_file(a.ids), "letter_token_ids": letter_ids, "setting": st, "n": len(recs),
               "accuracy": acc, "rows": recs, "seconds": time.time() - t0}
    write_new(out, json.dumps(payload))
    print("%s mmlu K %d alpha %s: accuracy %.4f over %d items, %.0fs" % (a.model, a.k, a.alpha, acc, len(recs), time.time() - t0), flush=True)


def run_gsm8k(a):
    """GSM8K answers under the intervention; rows as the project's own GSM8K passes (id, text, finished, rule_abstain),
    so that diag/cap_score.py can score them."""
    NG, m, X, ctx = open_run(a)
    bundle = ctx.bundle
    systems = NG.systems_of(ctx)
    a.task, a.split, a.only_ids = "gsm8k", "all", None
    items, template = NG.load_items(a)
    if a.first_batches:
        items = items[:a.first_batches * NG.BATCH]
    if a.k:
        fit = load_fit(a.fit, a.model, ctx.ident_sha)
        vecs, chosen = head_vectors(bundle, fit, a.k, a.alpha, a.sigma)
        st = setting(a, fit, chosen)
    else:
        vecs, st = {}, {"k": 0}
    out = Path(a.out)
    if out.exists():
        raise SystemExit("%s exists" % out)
    t0 = time.time()
    with X.E.fp32_output_projection(bundle, enabled=True):
        prompts = [X.pd.render_prompt(bundle, systems["neutral"], template, q) for _, q in items]
        recs = generate(X, bundle, prompts, vecs, a.max_new_tokens)
    rows = [dict(r, id=iid, rule_abstain=bool(X.rule_abstains(r["text"]))) for r, (iid, _) in zip(recs, items)]
    payload = {"protocol": PROTOCOL, "phase": "gsm8k", "completed": not a.first_batches, "model": a.model, "base_prompt": "neutral",
               "system_text": systems["neutral"], "identity_sha256": ctx.ident_sha, "script_sha256": sha_file(__file__),
               "items_file_sha256": sha_file(a.items), "ids_file_sha256": sha_file(a.ids), "max_new_tokens": a.max_new_tokens,
               "batch_size": NG.BATCH, "setting": st, "ids": [i for i, _ in items], "rows": rows, "seconds": time.time() - t0}
    write_new(out, json.dumps(payload))
    print("%s gsm8k K %d alpha %s: %d items, rule abstain %d, unfinished %d, %.0fs" % (
        a.model, a.k, a.alpha, len(rows), sum(r["rule_abstain"] for r in rows), sum(not r["finished"] for r in rows), time.time() - t0), flush=True)


def run_export(a):
    """A judge input with the same question, gold and incorrect lists as an existing judge input of the same model
    and split (--like), and one condition per ITI generation file."""
    like = json.loads(Path(a.like).read_text(encoding="utf-8"))
    gens = [json.loads(Path(p).read_text(encoding="utf-8")) for p in a.gen]
    ids = [r["pair_id"] for r in like["evaluations"]]
    if len(set(ids)) != len(ids) or like.get("question_ids") != ids:
        raise SystemExit("the reference export repeats or misorders the question IDs")
    conds = [g["condition"] for g in gens]
    if len(set(conds)) != len(conds):
        raise SystemExit("two generation files of the same setting")
    for g, p in zip(gens, a.gen):
        if g.get("protocol") != PROTOCOL or g.get("completed") is not True or g.get("model") != like.get("model") \
                or g.get("split") != like.get("split") or g["question_ids"] != ids \
                or [r["pair_id"] for r in g["rows"]] != ids \
                or g.get("identity_sha256") != like.get("identity_sha256"):
            raise SystemExit("%s is not a complete ITI generation of the model, split, run and questions of --like" % p)
    by = [{r["pair_id"]: r for r in g["rows"]} for g in gens]
    rows = []
    for r in like["evaluations"]:
        rows.append({"pair_id": r["pair_id"], "question": r["question"], "gold": r["gold"], "incorrect": r.get("incorrect") or [],
                     "best_answer": r.get("best_answer"),
                     "outputs": {c: d[r["pair_id"]]["text"] for c, d in zip(conds, by)},
                     "finished": {c: bool(d[r["pair_id"]]["finished"]) for c, d in zip(conds, by)},
                     "rule_abstain": {c: bool(d[r["pair_id"]]["rule_abstain"]) for c, d in zip(conds, by)}})
    payload = {"completed": True, "model": like.get("model"), "protocol": PROTOCOL, "export_protocol": PROTOCOL + "_export",
               "split": like.get("split"), "identity_sha256": like.get("identity_sha256"), "conditions": conds,
               "condition_specs": [{"name": g["condition"], "kind": "intervention", "setting": g["setting"],
                                    "base_prompt": g["base_prompt"], "uses_reference_answer": False} for g in gens],
               "source_files": {g["condition"]: {"file": Path(p).name, "sha256": sha_file(p)} for g, p in zip(gens, a.gen)},
               "like": {"file": str(a.like), "sha256": sha_file(a.like)},
               "question_ids": ids, "evaluations": rows}
    out = Path(a.out)
    if out.exists():
        raise SystemExit("%s exists; never overwrite a judge input" % out)
    write_new(out, json.dumps(payload, ensure_ascii=False, indent=1))
    print("exported %s: %d questions x %d conditions" % (out, len(rows), len(conds)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["fit", "check", "gen", "mmlu", "gsm8k", "export"])
    ap.add_argument("--models", type=Path)
    ap.add_argument("--model")
    ap.add_argument("--fit", type=Path)
    ap.add_argument("--fit-splits", default="train")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--faithful", action="store_true", help="fit only: the official probe text, sigma draw and head order")
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--alpha", type=float, default=15.0)
    ap.add_argument("--sigma", choices=["endq", "pairs"], default="endq")
    ap.add_argument("--split", choices=["tune", "test"], default="tune")
    ap.add_argument("--items", type=Path)
    ap.add_argument("--ids", type=Path)
    ap.add_argument("--first-batches", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=512, help="gsm8k only")
    ap.add_argument("--like", type=Path)
    ap.add_argument("--gen", nargs="+", default=[])
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.phase == "export":
        if not (a.like and a.gen and a.out):
            ap.error("export needs --like, --gen and --out")
        run_export(a)
        return
    if not (a.models and a.model):
        ap.error("--models and --model are required")
    if a.phase == "fit":
        if not a.out:
            ap.error("fit needs --out")
        run_fit(a)
    elif a.phase == "check":
        if not a.fit:
            ap.error("check needs --fit")
        run_check(a)
    elif a.phase == "gen":
        if not (a.fit and a.out):
            ap.error("gen needs --fit and --out")
        run_gen(a)
    else:
        if not (a.items and a.ids and a.out and (a.fit or not a.k)):
            ap.error("%s needs --items, --ids, --out and (unless --k 0) --fit" % a.phase)
        (run_mmlu if a.phase == "mmlu" else run_gsm8k)(a)


if __name__ == "__main__":
    main()
