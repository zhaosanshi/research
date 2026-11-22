# -*- coding: utf-8 -*-
"""Exploration (Claude Opus 5.5, 2026-11-22): Contrastive Activation Addition (CAA; Rimsky / Panickssery et al.,
ACL 2024) adapted to this project's evaluation protocol, as an external baseline written on the residual stream.
Plan: discussion/work_queue/2026-11-22_queue_f.md (sections 1-4, with section 7 taking precedence); method facts:
novelty/prior_work/2026-11-22_caa_method_spec.md.

What is computed (fit):
  data       the TruthfulQA TRAINING split of the run's source questions (96 questions, the questions E and ITI use;
             read from the engine's Context as scripts/newsets/iti_baseline.py fit reads them). For every question the
             best correct answer (field best_answer) is paired with EVERY incorrect answer (field incorrect), and each
             pair is asked twice, once with the correct answer as (A) and once as (B):
                 user turn  "{question}\\n\\nChoices:\\n (A) {first}\\n (B) {second}"   (the CAA multiple-choice layout)
             rendered with the model's own chat template (bundle.build_chat, no system prompt, as CAA builds its
             vectors), generation header included, then the assistant turn "(A" or "(B" appended as text.
  token      the answer-letter token, located by tokenizing the same text with and without the letter
             ("...(" against "...(A" / "...(B"): the first position where the two token lists differ. It must be the
             last token, and the A / B versions must have the same length and differ only there. Its text is recorded
             (for Llama-3 it may be the merged token "(A"). This replaces the authors' fixed "second to last" index.
  activation the output of every decoder block (the residual stream after layer l, layer numbers from 0) at that
             token, one sequence per forward (no padding), float32 copy of the model's own dtype.
  vector     per pair and order: positive (ends with the correct letter) minus negative (ends with the incorrect
             letter); averaged first within a question (all its pairs and both orders), then over questions (every
             question has the same weight). Not normalised; the per-layer norm is stored.
  halves     the same activations also give the vector of half0 (the training questions at even positions of the
             split order); half1 (odd positions) is held out for the sign check only. The formal vector uses all 96.

Differences from the paper (reported with every result):
  - data: A / B pairs built from TruthfulQA training questions, not the authors' behaviour datasets (hallucination
    1,000 pairs); no cross-behaviour norm equalisation (one behaviour only);
  - templates: each model's own chat template instead of Llama 2 [INST] tags; letter token found by tokenization;
  - writing: through the project's engine (scripts/e44_eval_extra.py, direction kind md = multiplier x the vector)
    at the core position rule "gen" (last prompt token and every generated token; the question untouched), via
    scripts/review_checks/e44_positions.py gen (generation) and scripts/capability/cap_mcq_sweep.py @gen (MMLU);
  - evaluation: free-form TruthfulQA answers scored right / abstain / wrong, layer and multiplier chosen on the
    tune split; the paper reads A / B letter probabilities and picks layers on its test questions.

The engine loads directions only as "direction records" (dirs/E_L<layer>.pt of a finished run, sealed, bound to the
run identity and its frozen training answers). mirror writes a NEW run directory that the engine accepts unchanged:
copies of the run's frozen_{train,tune,test}.json, and for every layer a record whose md write is exactly the CAA
vector (w_md = w = vector / |vector|, md_norm = gap = |vector|, cosine 1). Fields the loaders require but CAA does
not have are carried over from a real E record of the run and marked in the record's "caa" entry: identity, frozen
source, training_contract (positions "content" there is only the carrier value the position wrappers require; the
CAA write is always made at "gen"; train_ids = the 96 CAA questions, val_ids empty). --kind real copies real E
records byte for byte instead (acceptance 1). A models config pointing at the mirror is written next to it.

Phases (model interpreter, from /workspace/Truth_Decoupling; paths relative to it):
  fit     --models CFG --model NAME --out VEC.pt                         (GPU) vectors + VEC.pt.json
  mirror  --models CFG --model NAME --mirror DIR --kind caa|real [--vectors VEC.pt] [--layers L ...] [--template-layer L]
                                                                           (CPU) the mirror run directory
  check   --models CFG --model NAME --mirror DIR --vectors VEC.pt --out CHECK.json [--first-batches 2] [--layers L ...]
          (GPU) acceptance 2-5: multiplier 0 equals no write token for token (and the record's md write equals the
          vector), the sign check on the held-out half, the question-ID check, the template / position record.
"""
import argparse
import collections
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
PROTOCOL = "caa_vector_v1"
USER_FORMAT = "{question}\n\nChoices:\n (A) {a}\n (B) {b}"
DR100 = "Iteration_01/code/configs/splits/dr_100pairs_20261110.json"


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new_bytes(out, data):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(".%s.%d.tmp" % (out.name, os.getpid()))
    tmp.write_bytes(data)
    try:
        os.link(tmp, out)
    except FileExistsError:
        raise SystemExit("%s exists" % out)
    finally:
        tmp.unlink()


def write_new(out, text):
    write_new_bytes(out, text.encode("utf-8"))


def torch_save_new(out, obj):
    import io
    import torch
    buf = io.BytesIO()
    torch.save(obj, buf)
    write_new_bytes(out, buf.getvalue())


def open_run(a, run_dir=None, first_batches=0, load_model=True):
    """(newsets_gen module, model config, engine module, context or None). run_dir replaces the config's run_dir."""
    import newsets_gen as NG
    m = NG.load_models(a.models, a.model)
    if run_dir is not None:
        m = dict(m, run_dir=str(run_dir))
    X = NG.engine(m)
    ctx = NG.context(X, m, first_batches) if load_model else None
    return NG, m, X, ctx


def candidate_layers(n_layers):
    """Depth 25 % to 75 %, every second layer (both ends included when they fall on the grid)."""
    lo, hi = int(round(0.25 * n_layers)), int(round(0.75 * n_layers))
    return list(range(lo, hi + 1, 2))


def halves(questions):
    return [q for i, q in enumerate(questions) if i % 2 == 0], [q for i, q in enumerate(questions) if i % 2 == 1]


def ab_items(bundle, questions):
    """Every (question, incorrect answer, order) as rendered texts and the located letter token.
    Returns (items, template record). item: pair_id, correct letter, pos_ids, neg_ids, read_ids (prompt + shared stem,
    for the sign check), prefix_len."""
    tok = bundle.tokenizer
    enc = lambda t: list(tok(t, add_special_tokens=False)["input_ids"])  # noqa: E731
    items, seen_letters, stems = [], {"A": set(), "B": set()}, set()
    for q in questions:
        best = q["best_answer"]
        if not isinstance(best, str) or not best.strip() or best not in q["gold"]:
            raise SystemExit("question %s: best_answer missing or not among the correct answers" % q["pair_id"])
        for wrong in q["incorrect"]:
            for correct_letter in ("A", "B"):
                a, b = (best, wrong) if correct_letter == "A" else (wrong, best)
                user = USER_FORMAT.format(question=q["question"], a=a, b=b)
                prefix = bundle.build_chat(None, user, add_generation_prompt=True)
                if prefix.count(user) != 1:
                    raise SystemExit("question %s: the user text does not occur exactly once in the rendering" % q["pair_id"])
                p_ids = enc(prefix)
                open_ids = enc(prefix + "(")
                ids = {L: enc(prefix + "(" + L) for L in "AB"}
                if ids["A"][:len(p_ids)] != p_ids or ids["B"][:len(p_ids)] != p_ids:
                    raise SystemExit("question %s: the rendered prompt is not a token prefix of the answered text" % q["pair_id"])
                pos = {}
                for L in "AB":
                    k = 0
                    while k < min(len(open_ids), len(ids[L])) and open_ids[k] == ids[L][k]:
                        k += 1
                    if k != len(ids[L]) - 1:
                        raise SystemExit("question %s: the letter %s is not the single last differing token" % (q["pair_id"], L))
                    pos[L] = k
                if pos["A"] != pos["B"] or ids["A"][:-1] != ids["B"][:-1] or ids["A"][-1] == ids["B"][-1]:
                    raise SystemExit("question %s: the A and B texts differ elsewhere than in the letter token" % q["pair_id"])
                k = pos["A"]
                if k < len(p_ids):
                    raise SystemExit("question %s: the letter token reaches into the rendered prompt" % q["pair_id"])
                for L in "AB":
                    seen_letters[L].add((ids[L][-1], tok.decode([ids[L][-1]])))
                stems.add(tuple(ids["A"][len(p_ids):k]))
                wrong_letter = "B" if correct_letter == "A" else "A"
                items.append({"pair_id": q["pair_id"], "correct": correct_letter, "pos_ids": ids[correct_letter],
                              "neg_ids": ids[wrong_letter], "read_ids": ids["A"][:k], "prefix_len": len(p_ids),
                              "letter_ids": {L: ids[L][-1] for L in "AB"}})
    if any(len(v) != 1 for v in seen_letters.values()) or len(stems) != 1:
        raise SystemExit("the letter tokens or the shared stem differ between items: %s %s" % (seen_letters, stems))
    stem = list(stems.pop())
    info = {"letter_token": {L: {"id": next(iter(v))[0], "text": next(iter(v))[1]} for L, v in seen_letters.items()},
            "stem_between_prompt_and_letter": {"ids": stem, "text": [tok.decode([t]) for t in stem]}}
    return items, info


def capture(bundle, X, ids):
    """[layers, hidden] float32: every decoder block's output at the last token of one unpadded sequence."""
    import torch
    L = bundle.num_layers
    store, handles = {}, []

    def make(l):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            store[l] = h[0, -1, :].detach().float().cpu()
        return hook

    for l in range(L):
        handles.append(bundle.layer(l).register_forward_hook(make(l)))
    try:
        t = torch.tensor([ids], device=bundle.device)
        X.forward_logits(bundle, {"input_ids": t, "attention_mask": torch.ones_like(t)}, grad=False, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    if len(store) != L:
        raise SystemExit("a decoder block did not run")
    return torch.stack([store[l] for l in range(L)])


def run_fit(a):
    import torch
    NG, m, X, ctx = open_run(a)
    bundle = ctx.bundle
    questions = list(ctx.prepared["splits"]["train"])
    h0, h1 = halves(questions)
    t0 = time.time()
    if len(questions) != 96 or len({q["pair_id"] for q in questions}) != 96:
        raise SystemExit("fit needs all 96 distinct training questions")
    items, info = ab_items(bundle, questions)
    per_q = collections.OrderedDict((q["pair_id"], []) for q in questions)
    with torch.no_grad():
        for n, it in enumerate(items):
            d = capture(bundle, X, it["pos_ids"]) - capture(bundle, X, it["neg_ids"])
            if not bool(torch.isfinite(d).all()):
                raise SystemExit("non-finite activations")
            per_q[it["pair_id"]].append(d)
            if n % 200 == 0:
                print("pair %d of %d, %.0fs" % (n, len(items), time.time() - t0), flush=True)
    used = [pid for pid, v in per_q.items() if v]
    if len(used) != len(questions):
        raise SystemExit("a training question has no usable positive/negative pair")
    qmean = {pid: torch.stack(v).mean(0) for pid, v in per_q.items() if v}
    vec = torch.stack([qmean[p] for p in used]).mean(0)                                  # [layers, hidden]
    used0 = [q["pair_id"] for q in h0 if q["pair_id"] in qmean]
    vec0 = torch.stack([qmean[p] for p in used0]).mean(0)
    counts = collections.Counter(len(v) for v in per_q.values())
    norms, norms0 = vec.norm(dim=-1), vec0.norm(dim=-1)
    qpath = Path(m["source_run"])
    qpath = (NG.ROOT / qpath / "questions.json") if not qpath.is_absolute() else qpath / "questions.json"
    meta = {"protocol": PROTOCOL, "phase": "fit", "model": a.model, "identity_sha256": ctx.ident_sha,
            "script_sha256": sha_file(__file__), "engine_sha256": sha_file(X.__file__),
            "questions_file_sha256": sha_file(qpath), "split": "train",
            "question_ids": [q["pair_id"] for q in questions], "questions_used": used,
            "half0_ids": [q["pair_id"] for q in h0], "half1_ids": [q["pair_id"] for q in h1],
            "half_rule": "training split order: even positions = half0 (sign-check vector), odd = half1 (held out)",
            "pairs": len(items) // 2, "prompts_per_sign": len(items),
            "prompts_per_question_distribution": {str(k): v for k, v in sorted(counts.items())},
            "user_format": USER_FORMAT, "system_prompt": None, "assistant_suffix": "(A or (B", **info,
            "site": "output of every decoder block (residual stream), layer numbers from 0",
            "average": "positive minus negative per (pair, order); mean within question, then mean over questions",
            "normalised": False, "layers": bundle.num_layers, "hidden": bundle.hidden_size,
            "candidate_layers": candidate_layers(bundle.num_layers),
            "norms": [float(x) for x in norms], "norms_half0": [float(x) for x in norms0],
            "seconds": time.time() - t0}
    torch_save_new(a.out, {"meta": meta, "vectors": vec.float(), "vectors_half0": vec0.float()})
    write_new(str(a.out) + ".json", json.dumps(meta, indent=1, ensure_ascii=True))
    print("fit: %d questions, %d pairs, %d prompts per sign, prompts per question %s; letter tokens %s, stem %s; "
          "norm range %.3f .. %.3f (layer %d .. %d); %.0fs" % (
              len(used), meta["pairs"], len(items), meta["prompts_per_question_distribution"],
              {L: v["text"] for L, v in info["letter_token"].items()}, info["stem_between_prompt_and_letter"]["text"],
              float(norms.min()), float(norms.max()), int(norms.argmin()), int(norms.argmax()), time.time() - t0), flush=True)
    for l in range(bundle.num_layers):
        print("  layer %2d norm %8.4f (half0 %8.4f)%s" % (l, float(norms[l]), float(norms0[l]),
                                                           "  candidate" if l in meta["candidate_layers"] else ""), flush=True)


def load_vectors(path, model):
    import torch
    v = torch.load(path, map_location="cpu", weights_only=True)
    meta = v["meta"]
    if meta != json.loads(Path(str(path) + ".json").read_text(encoding="utf-8")):
        raise SystemExit("vector metadata sidecar differs from tensor file")
    for field, norm_field in (("vectors", "norms"), ("vectors_half0", "norms_half0")):
        x = v[field]
        if (x.shape != (meta["layers"], meta["hidden"]) or not torch.isfinite(x).all()
                or not torch.allclose(x.float().norm(dim=-1), torch.tensor(meta[norm_field]), atol=1e-6, rtol=1e-6)):
            raise SystemExit("invalid shape, values or norms in " + field)
    if v["meta"].get("protocol") != PROTOCOL or v["meta"].get("model") != model:
        raise SystemExit("%s is not a CAA vector file of %s" % (path, model))
    return v


def verify_inputs(root, model_config, vectors):
    meta = vectors["meta"]
    src = root / model_config["source_run"] / "questions.json"
    bank = json.loads(src.read_text(encoding="utf-8"))["splits"]
    ids = {split: [q["pair_id"] for q in bank[split]] for split in ("train", "tune", "test")}
    train = ids["train"]
    if (sha_file(src) != meta["questions_file_sha256"] or len(train) != 96
            or any(len(seq) != len(set(seq)) for seq in ids.values())
            or meta["split"] != "train" or meta["questions_used"] != train or meta["question_ids"] != train
            or meta["half0_ids"] != train[::2] or meta["half1_ids"] != train[1::2]):
        raise SystemExit("CAA vector training source, order, complete roster or halves differ")
    dr = set(json.loads((root / DR100).read_text(encoding="utf-8"))["pair_ids"])
    if set(train) & (set(ids["tune"]) | set(ids["test"]) | dr):
        raise SystemExit("CAA train overlaps a held-out split")
    return ids


def verify_bundle(root, a, model_config):
    """CPU-only; the generic E loader cannot distinguish CAA from a real E record."""
    import torch
    v = load_vectors(a.vectors, a.model)
    meta, vec = v["meta"], v["vectors"]
    ids = verify_inputs(root, model_config, v)
    dst = (root / a.mirror).resolve()
    listing = json.loads((dst / "MIRROR.json").read_text(encoding="utf-8"))
    digest = sha_file(a.vectors)
    src = (root / listing["source_run_dir"]).resolve()
    if (listing["kind"] != "caa" or listing["model"] != a.model or listing["protocol"] != PROTOCOL
            or listing["vectors_file_sha256"] != digest or src == dst
            or (root / model_config["run_dir"]).resolve() not in (src, dst)):
        raise SystemExit("wrong CAA mirror, model or vector source")
    for split, roster in ids.items():
        name = "frozen_" + split + ".json"
        f = json.loads((dst / name).read_text(encoding="utf-8"))
        if (listing["frozen_files_sha256"][name] != sha_file(dst / name) or sha_file(src / name) != sha_file(dst / name)
                or f["identity_sha256"] != meta["identity_sha256"] or f["model"] != a.model or not f["completed"]
                or f["split"] != split or f["question_ids"] != roster):
            raise SystemExit("wrong frozen CAA inputs in " + name)
    for l in (a.layers or meta["candidate_layers"]):
        entry = listing["layers"].get(str(l))
        p = dst / "dirs" / ("E_L%d.pt" % l)
        if not entry or entry["file_sha256"] != sha_file(p):
            raise SystemExit("missing or mismatched CAA layer " + str(l))
        rec = torch.load(p, map_location="cpu", weights_only=True)
        if (rec["layer"] != l or rec["identity_sha256"] != meta["identity_sha256"]
                or rec.get("caa", {}).get("vectors_file_sha256") != digest
                or rec["training_contract"]["train_ids"] != ids["train"] or rec["training_contract"]["val_ids"] != []
                or not torch.allclose(rec["md_norm"] * rec["w_md"], vec[l], rtol=1e-6, atol=1e-7)):
            raise SystemExit("CAA mirror is not the intended direction at layer " + str(l))
    return v


def run_mirror(a):
    import copy
    import torch
    NG, m, X, _ = open_run(a, load_model=False)
    E, pd = X.E, X.pd
    src = (NG.ROOT / m["run_dir"]).resolve()
    dst = (NG.ROOT / a.mirror).resolve()
    if dst.exists():
        raise SystemExit("%s exists; a mirror is never overwritten" % a.mirror)
    if dst.name == src.name or (m.get("c_run_dir") and dst.name == Path(m["c_run_dir"]).name):
        raise SystemExit("the mirror needs a directory name of its own")
    if a.kind == "caa":
        if not a.vectors:
            raise SystemExit("--kind caa needs --vectors")
        v = load_vectors(a.vectors, a.model)
        verify_inputs(NG.ROOT, m, v)
    names = ["frozen_train.json", "frozen_tune.json", "frozen_test.json"]
    (dst / "dirs").mkdir(parents=True)
    for n in names:
        shutil.copyfile(src / n, dst / n)
        if sha_file(src / n) != sha_file(dst / n):
            raise SystemExit("copy of %s differs" % n)
    listing = {"protocol": PROTOCOL, "phase": "mirror", "kind": a.kind, "model": a.model, "source_run_dir": m["run_dir"],
               "frozen_files_sha256": {n: sha_file(dst / n) for n in names}, "layers": {}}
    if a.kind == "real":
        if not a.layers:
            raise SystemExit("--kind real needs --layers")
        for l in a.layers:
            shutil.copyfile(src / "dirs" / ("E_L%d.pt" % l), dst / "dirs" / ("E_L%d.pt" % l))
            listing["layers"][str(l)] = {"file_sha256": sha_file(dst / "dirs" / ("E_L%d.pt" % l)), "copied_from_run": True}
    else:
        vec, meta = v["vectors"], v["meta"]
        tl = a.template_layer if a.template_layer is not None else meta["candidate_layers"][len(meta["candidate_layers"]) // 2]
        template = torch.load(src / "dirs" / ("E_L%d.pt" % tl), map_location="cpu", weights_only=False)
        E.validate_direction(template, "E", tl, template["identity_sha256"], False)
        if pd.fingerprint(template["identity"]) != template["identity_sha256"] or template["identity_sha256"] != meta["identity_sha256"]:
            raise SystemExit("the template record and the vectors belong to different run identities")
        ids = meta["questions_used"]
        layers = a.layers or list(range(vec.shape[0]))
        for l in layers:
            x = vec[l].float()
            n = float(x.norm())
            if not n > 0:
                raise SystemExit("layer %d: zero vector" % l)
            u = (x / n).float().contiguous()
            rec = {k: copy.deepcopy(val) for k, val in template.items() if k != "artifact_sha256"}
            tc = dict(rec["training_contract"], layer=l, train_ids=list(ids), val_ids=[], targets_sha256=None)
            rec.update({"layer": l, "w": u.clone(), "w_md": u.clone(), "gap": n, "md_norm": n, "mean_difference_defined": True,
                        "cosine_w_md": 1.0, "difference_along_w": {"defined": False, "reason": "CAA vector, not a DAS direction"},
                        "difference_along_md": {"defined": False, "reason": "CAA vector, not a DAS direction"},
                        "read_difference_abs_along_w": {}, "losses": {}, "skipped": [],
                        "training": {"method": "none: CAA mean difference of A / B letter activations"},
                        "training_contract": tc, "planned_train_ids": list(ids), "planned_val_ids": [], "seconds": 0.0,
                        "caa": {"protocol": PROTOCOL, "vectors_file_sha256": sha_file(a.vectors), "vector_norm": n,
                                "pairs": meta["pairs"], "letter_token": meta["letter_token"],
                                "template_record_layer": tl,
                                "carrier_fields": "identity, frozen_source, training_contract (except layer / train_ids / "
                                                  "val_ids / targets_sha256), e_on, e_off, protocol come from the run's real "
                                                  "E record of the template layer, only so that the engine's loaders accept "
                                                  "the record; training_contract.positions = content is a carrier value: the "
                                                  "CAA vector is written at the rule gen through the position wrappers",
                                "write": "kind md: multiplier x md_norm x w_md = multiplier x the CAA vector"}})
            rec = E.seal(rec)
            E.validate_direction(rec, "E", l, template["identity_sha256"], False)
            E.validate_direction_sources(rec, dst)
            path = dst / "dirs" / ("E_L%d.pt" % l)
            torch_save_new(path, rec)
            back = torch.load(path, map_location="cpu", weights_only=True)      # the export reads it this way
            E.validate_direction(back, "E", l, template["identity_sha256"], False)
            if float((back["w_md"] * back["md_norm"] - x).abs().max()) > 1e-4 * max(1.0, n):
                raise SystemExit("layer %d: the stored md write differs from the vector" % l)
            listing["layers"][str(l)] = {"file_sha256": sha_file(path), "norm": n}
        listing["vectors_file_sha256"] = sha_file(a.vectors)
        listing["template_layer"] = tl
    cfg = json.loads(Path(a.models).read_text(encoding="utf-8"))
    entry = dict(cfg["models"][a.model], run_dir=str(Path(a.mirror)))
    cfg2 = {k: v for k, v in cfg.items() if k != "models"}
    cfg2["models"] = {a.model: entry}
    cfg2["note"] = "written by caa_vector.py mirror: the frozen config entry of this model with run_dir = the mirror"
    write_new(dst.parent / ("models_%s.json" % dst.name), json.dumps(cfg2, indent=1))
    write_new(dst / "MIRROR.json", json.dumps(listing, indent=1))
    print("mirror %s (%s): %d direction records, frozen files copied, models config models_%s.json" % (
        dst.name, a.kind, len(listing["layers"]), dst.name), flush=True)


def sign_check(X, bundle, items, vec, norms, layers, batch=16):
    """{layer: {...}}: probability of the correct letter token at the reading position (rendered prompt plus the shared
    stem before the letter) with no write, +1 and -1 times the half0 vector written at the last prompt token and the
    stem tokens (the rule gen applied to this prompt). Means within question first, then over questions."""
    import torch
    fd = X.fd
    pad = bundle.tokenizer.pad_token_id
    ncols = len(items[0]["read_ids"]) - items[0]["prefix_len"] + 1
    rows = {}

    def batches():
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            w = max(len(it["read_ids"]) for it in chunk)
            ids = torch.full((len(chunk), w), pad, dtype=torch.long)
            att = torch.zeros((len(chunk), w), dtype=torch.long)
            for b, it in enumerate(chunk):
                ids[b, w - len(it["read_ids"]):] = torch.tensor(it["read_ids"])
                att[b, w - len(it["read_ids"]):] = 1
            mask = torch.zeros((len(chunk), w), dtype=torch.bool)
            mask[:, w - ncols:] = True
            yield chunk, {"input_ids": ids.to(bundle.device), "attention_mask": att.to(bundle.device)}, mask

    def probs(layer, mult):
        out = []
        for chunk, inputs, mask in batches():
            if layer is None:
                logits = X.forward_logits(bundle, inputs, grad=False, logits_to_keep=1)
            else:
                hook = fd.ShiftHook(vec[layer], amounts=torch.full((len(chunk),), float(mult) * float(norms[layer])),
                                    mask=mask.to(bundle.device), decode_all=False)
                with X.pd.clamp(bundle, layer, hook):
                    logits = X.forward_logits(bundle, inputs, grad=False, logits_to_keep=1)
                if hook.applied != 1:
                    raise RuntimeError("the sign-check write did not fire exactly once")
            p = torch.softmax(logits[:, -1, :].float(), -1)
            for b, it in enumerate(chunk):
                c = it["letter_ids"][it["correct"]]
                o = it["letter_ids"]["B" if it["correct"] == "A" else "A"]
                out.append((it["pair_id"], float(p[b, c]), float(p[b, c] / (p[b, c] + p[b, o]))))
        return out

    def qmean(vals, k):
        g = collections.OrderedDict()
        for v in vals:
            g.setdefault(v[0], []).append(v[k])
        m = [sum(x) / len(x) for x in g.values()]
        return sum(m) / len(m), m

    with torch.no_grad():
        base = probs(None, 0)
        b_raw, bq_raw = qmean(base, 1)
        b_rel, bq_rel = qmean(base, 2)
        for l in layers:
            r = {"no_write": {"p_correct": b_raw, "p_correct_of_ab": b_rel}}
            for mult in (1, -1):
                s = probs(l, mult)
                raw, q_raw = qmean(s, 1)
                rel, q_rel = qmean(s, 2)
                r["%+d" % mult] = {"p_correct": raw, "p_correct_of_ab": rel, "change": raw - b_raw, "change_of_ab": rel - b_rel,
                                   "questions_up_of_ab": sum(x > y for x, y in zip(q_rel, bq_rel)), "questions": len(q_rel)}
            rows[l] = r
            print("  sign layer %2d | P(correct) %.4f: +1 %+.4f  -1 %+.4f | share of A+B %.4f: +1 %+.4f (up in %d of %d questions)  -1 %+.4f (up in %d)" % (
                l, b_raw, r["+1"]["change"], r["-1"]["change"], b_rel, r["+1"]["change_of_ab"], r["+1"]["questions_up_of_ab"],
                r["+1"]["questions"], r["-1"]["change_of_ab"], r["-1"]["questions_up_of_ab"]), flush=True)
    return {"rows": rows, "written_positions": "last %d columns (last prompt token + stem before the letter)" % ncols,
            "items": len(items)}


def run_check(a):
    import torch
    NG, m, X, ctx = open_run(a, run_dir=a.mirror, first_batches=a.first_batches)
    bundle, fd, pd = ctx.bundle, X.fd, X.pd
    v = verify_bundle(NG.ROOT, a, NG.load_models(a.models, a.model))
    meta = v["meta"]
    if meta["identity_sha256"] != ctx.ident_sha:
        raise SystemExit("the vectors belong to another run identity")
    cands = meta["candidate_layers"]
    layers = a.layers or cands
    out = {"protocol": PROTOCOL, "phase": "check", "model": a.model, "mirror": str(a.mirror), "identity_sha256": ctx.ident_sha,
           "script_sha256": sha_file(__file__), "engine_sha256": sha_file(X.__file__), "vectors_file_sha256": sha_file(a.vectors),
           "first_batches": a.first_batches, "questions": len(ctx.questions)}
    ok = True
    # --- 4. question IDs ---------------------------------------------------------------------------------------
    sp = ctx.prepared["splits"]
    train = {q["pair_id"]: q for q in sp["train"]}
    tune_ids = {q["pair_id"] for q in sp["tune"]}
    test_ids = {q["pair_id"] for q in sp.get("test", [])}
    dr = json.loads((NG.ROOT / DR100).read_text(encoding="utf-8"))["pair_ids"]
    used = meta["questions_used"]
    frozen_train_ids = set(json.loads((NG.ROOT / a.mirror / "frozen_train.json").read_text(encoding="utf-8"))["question_ids"])
    texts = lambda ids, rows: {rows[i]["question"].strip().lower() for i in ids}  # noqa: E731
    allrows = {q["pair_id"]: q for s in sp.values() for q in s}
    ids_check = {"vector_questions": len(used), "all_in_train": set(used) <= set(train), "equal_to_train": set(used) == set(train),
                 "in_frozen_train": set(used) <= frozen_train_ids,
                 "overlap_tune_ids": len(set(used) & tune_ids), "overlap_test_ids": len(set(used) & test_ids),
                 "overlap_dr100_ids": len(set(used) & set(dr)),
                 "overlap_tune_question_text": len(texts(used, allrows) & texts(tune_ids, allrows)),
                 "overlap_test_question_text": len(texts(used, allrows) & texts(test_ids, allrows)),
                 "tune": len(tune_ids), "test": len(test_ids), "dr100": len(dr),
                 "half0_half1_disjoint": not set(meta["half0_ids"]) & set(meta["half1_ids"]),
                 "halves_cover_train": set(meta["half0_ids"]) | set(meta["half1_ids"]) == set(train)}
    good = (ids_check["equal_to_train"] and ids_check["all_in_train"] and ids_check["in_frozen_train"] and ids_check["half0_half1_disjoint"]
            and ids_check["halves_cover_train"] and not any(ids_check[k] for k in ids_check if k.startswith("overlap_")))
    ok &= good
    out["ids"] = ids_check
    print("check 4 (question IDs): %s -> %s" % (json.dumps(ids_check), "PASS" if good else "FAIL"), flush=True)
    # --- 5. template and positions -----------------------------------------------------------------------------
    prompts = ctx.render("x")
    p0 = prompts[0]
    tmpl = {"x_prompt_length_first_question": p0["prompt_length"], "content_start": p0["content_start"],
            "first_write_position": "the last prompt token (index prompt_length - 1 = %d for the first tune question)" % p0["read_position"],
            "first_write_token_text": p0["read_token"],
            "last_five_prompt_tokens": [bundle.tokenizer.decode([t]) for t in p0["prefix_ids"][-5:]],
            "first_decode_step": "the decode forward whose input is the first generated token, at index prompt_length "
                                 "(= %d for the first tune question); every later generated token is written too "
                                 "(the prefill write at the last prompt token already acts on the first generated token)" % p0["prompt_length"],
            "same_last_token_all_questions": len({p["read_token"] for p in prompts}) == 1,
            "candidate_layers": cands, "norms": meta["norms"],
            "letter_token": meta["letter_token"], "stem": meta["stem_between_prompt_and_letter"]}
    out["template"] = tmpl
    print("check 5 (template): first write at the last prompt token %r (same token for all %d questions: %s); last five "
          "prompt tokens %r; first decode step = the first generated token's position (index %d for question 1)" % (
              tmpl["first_write_token_text"], len(prompts), tmpl["same_last_token_all_questions"],
              tmpl["last_five_prompt_tokens"], p0["prompt_length"]), flush=True)
    print("        candidate layers %s; letter tokens %s; stem %s" % (cands, {L: x["text"] for L, x in meta["letter_token"].items()},
                                                                       meta["stem_between_prompt_and_letter"]["text"]), flush=True)
    # --- 2. multiplier 0 and the record's write -----------------------------------------------------------------
    zero = {}
    with X.E.fp32_output_projection(bundle, enabled=True):
        frozen = [ctx.look[q["pair_id"]]["answers"]["x"] for q in ctx.questions]
        plain = pd.generate_prompts(bundle, prompts, max_new_tokens=ctx.args.max_new_tokens, batch_size=ctx.args.batch_size)
        same_plain = sum(all(r.get(k) == f.get(k) for k in ("text", "token_ids", "finished", "stop_id")) for r, f in zip(plain, frozen))
        print("check 2: no-write regeneration equals the frozen x answers on %d of %d" % (same_plain, len(frozen)), flush=True)
        ok &= same_plain == len(frozen)
        for l in layers:
            ws, trained = ctx.writes([{"family": "E", "layer": l, "kind": "md", "beta": 0.0}])
            w1, _ = ctx.writes([{"family": "E", "layer": l, "kind": "md", "beta": 1.0}])
            vec_l = v["vectors"][l].float()
            got = w1[0][1]["direction"].float() * float(w1[0][1]["amount"])
            rec_gap = float((got - vec_l).abs().max())
            recs = fd.generate_shift(bundle, prompts, max_new_tokens=ctx.args.max_new_tokens, batch_size=ctx.args.batch_size,
                                     layer=l, positions="gen", **ws[0][1])
            same = sum(all(r.get(k) == f.get(k) for k in ("text", "token_ids", "finished", "stop_id")) for r, f in zip(recs, frozen))
            pre = sorted({r["write"]["prefill_positions"] for r in recs})
            dec = sorted({r["write"]["decode_steps_total"] for r in recs})
            zero[l] = {"amount": float(ws[0][1]["amount"]), "same_as_no_write": same, "of": len(recs), "trained_positions_carrier": trained,
                       "record_md_write_minus_vector_max_abs": rec_gap, "prefill_positions_written_per_row": pre,
                       "decode_steps_total_per_batch": dec}
            expected_steps = [r["n_tokens"] if r["finished"] else max(0, r["n_tokens"] - 1) for r in recs]
            decode_ok = all(r["write"]["decode_steps_own"] == n for r, n in zip(recs, expected_steps))
            zero[l]["decode_steps_match_generated_rows"] = decode_ok
            good = (same == len(recs) == len(frozen) > 0 and rec_gap <= 1e-4 * max(1.0, float(vec_l.norm()))
                    and pre == [1] and decode_ok and zero[l]["amount"] == 0.0)
            ok &= good
            print("check 2: layer %2d multiplier 0 at gen: %d of %d answers equal no write token for token; prefill positions "
                  "written per row %s; record md write minus vector max %.2e -> %s" % (l, same, len(recs), pre, rec_gap,
                                                                                         "PASS" if good else "FAIL"), flush=True)
    out["zero"] = {str(k): x for k, x in zero.items()}
    # --- 3. sign check on the held-out half ----------------------------------------------------------------------
    held = [train[i] for i in meta["half1_ids"]]
    items, info = ab_items(bundle, held)
    if info["letter_token"] != meta["letter_token"]:
        raise SystemExit("letter tokens of the held-out half differ from the fit")
    with X.E.fp32_output_projection(bundle, enabled=True):
        vec0 = v["vectors_half0"].float()
        sc = sign_check(X, bundle, items, vec0, vec0.norm(dim=-1), list(range(bundle.num_layers)))
    out["sign"] = {"vector": "half0 (%d questions)" % len(meta["half0_ids"]), "held_out": "half1 (%d questions)" % len(held),
                   "written_positions": sc["written_positions"], "items": sc["items"],
                   "rows": {str(k): x for k, x in sc["rows"].items()}}
    cand_rows = [sc["rows"][l] for l in cands]
    up = sum(r["+1"]["change_of_ab"] > 0 for r in cand_rows)
    down = sum(r["-1"]["change_of_ab"] < 0 for r in cand_rows)
    up_raw = sum(r["+1"]["change"] > 0 for r in cand_rows)
    down_raw = sum(r["-1"]["change"] < 0 for r in cand_rows)
    out["sign"]["candidate_summary"] = {"plus_up_share_of_ab": up, "minus_down_share_of_ab": down, "plus_up_raw": up_raw,
                                        "minus_down_raw": down_raw, "candidates": len(cands)}
    print("check 3: candidate layers with +1 raising / -1 lowering the correct letter: share of A+B %d / %d, raw probability "
          "%d / %d, of %d" % (up, down, up_raw, down_raw, len(cands)), flush=True)
    out["passed_2_4"] = bool(ok)
    write_new(a.out, json.dumps(out, indent=1, ensure_ascii=True))
    print("check 2 and 4: %s (check 3 is reported, not gated)" % ("PASS" if ok else "FAIL"), flush=True)
    if not ok:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["fit", "mirror", "check", "verify"])
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out")
    ap.add_argument("--vectors")
    ap.add_argument("--mirror", type=Path, help="mirror run directory, relative to /workspace/Truth_Decoupling")
    ap.add_argument("--kind", choices=["caa", "real"], default="caa")
    ap.add_argument("--layers", type=int, nargs="*", default=[])
    ap.add_argument("--template-layer", type=int)
    ap.add_argument("--first-batches", type=int, default=2)
    a = ap.parse_args()
    if a.mirror is not None and a.mirror.is_absolute():
        ap.error("--mirror is relative to the project root")
    if a.phase == "fit":
        if not a.out:
            ap.error("fit needs --out")
        run_fit(a)
    elif a.phase == "mirror":
        if not a.mirror:
            ap.error("mirror needs --mirror")
        run_mirror(a)
    elif a.phase == "verify":
        if not (a.mirror and a.vectors):
            ap.error("verify needs --mirror and --vectors")
        import newsets_gen as NG
        verify_bundle(NG.ROOT, a, NG.load_models(a.models, a.model))
        print("CAA training inputs and mirror verified")
    else:
        if not (a.mirror and a.vectors and a.out):
            ap.error("check needs --mirror, --vectors and --out")
        run_check(a)


if __name__ == "__main__":
    main()
