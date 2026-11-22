#!/usr/bin/env python3
"""Queue K (Claude Opus 5.5, 2026-11-23; discussion/work_queue/2026-11-23_queue_k.md): the E training of
scripts/e44_factorial_das.py with one added loss term, behind a switch that is off by default.

Why: the stacked method (honest sentence + E + C) turns right answers on ordinary fact questions into denials. Training
writes each question's own coordinate (h_source - h_base) . w, which is small where the sentence changes little, so
nothing constrained the fixed inference write on such questions (summary section 7.4). The added term does.

keep term   keep questions = TriviaQA / PopQA training-split questions the model answers right (alias match) with the
            plain prompt x and no write (file from keepfix_keep_set.py). For each: the prompt x_E2 (the stacked
            method's prompt; system text from the engine's own extra_systems with --keep-prompts-extra), the fixed
            write A * u at layer L at the inference write rule "prompt" (first user-content token .. last prompt
            token; answer positions untouched; core position_mask "prompt"), u = the direction being trained, oriented
            as at inference: sign of mean_d . u, mean_d = the training mean of h(conf_E2) - h(conf) at the read
            position (the vector whose projection is the gap), so the trained write is the write of the aligned
            strength A / gap at inference. Target = the model's own original answer (its x answer token ids; the
            end-of-turn token appended for a complete answer; cap --max-target-tokens, as the training rows);
            loss = teacher-forced cross-entropy, mean over the answer tokens of a row (core rows_loss with loss "ce").
            A = a fixed amount read from an existing generation record (--keep-amount-from: parts[0].amount).
total       per optimizer step: the original question loss (unchanged code, unchanged batch) + lambda * the keep loss
            of ONE keep question, in a separate forward pass. Keep questions are taken in a random order from their
            own generator (seed 1000 * seed + layer + 7777; the training generator is not touched), a new order each
            time the list is used up.
switch      without any --keep-* option, "train" runs scripts/e44_factorial_das.py's main() as it is (the default).
            --keep-set with --keep-lambda 0 runs this loop with the term computed and multiplied by 0 (queue K
            section 2 item 1: the direction must equal the existing one).
identity    no core file and no identity-bearing script changes (this file is new), so the run identity, the frozen
            answers and the C direction of the existing runs stay valid. The keep settings, the keep-set file and
            this file's sha256 go into the direction record ("keep") and its training contract ("keep").
Modes (from the code copy's root, with the model's interpreter):
  train     <driver train arguments> [--keep-set F --keep-lambda L --keep-amount-from G --keep-prompts-extra P
            [--keep-mem-cap-mib N]]
  unittest  <driver train arguments> --keep-set F --keep-amount-from G --keep-prompts-extra P --keep-check-dir RUN
            --keep-report OUT [--keep-mem-cap-mib N]: with the direction RUN/dirs/E_L<layer>.pt, on 4 keep questions
            (the two of each task with the largest keep loss under that direction at A): the keep loss of the training
            code (one row per forward, as in training) against an independent hand computation (own hook on the
            positions content_start .. prompt_length - 1, own log-softmax and mean); amount 0 against the cross-entropy
            without any hook; the keep loss alone differentiated (gradient to the direction finite and nonzero, the
            written positions exactly the prompt rule's, a step against the gradient lowers it: grad_checks); the mean
            keep loss over all keep questions at A and with no write, by class; then one full training step (the
            longest training question's rows + the longest keep row, forward and backward) for the peak memory. No
            direction is written.
  compare   A.pt B.pt [--min-cos C]: elementwise equality, cosine and max abs difference of two direction files (CPU);
            exit 1 when the cosine is below C (default 0.9999)
  aligned   DIR.pt --keep-amount-from G: prints s' = A / gap rounded to 4 decimals (CPU; details on stderr)
  signcheck DIR.pt [--l0 L0.pt] [--min-gap G]: the trained direction's orientation and unit write amount (CPU)
--keep-mem-cap-mib N limits this process's PyTorch allocations to N MiB (torch.cuda.set_per_process_memory_fraction):
an allocation beyond it fails instead of passing a shared card's limit. It changes no computation.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import common  # noqa: E402,F401
import e44_factorial_das as E  # noqa: E402
import e44_eval_extra as X  # noqa: E402
from core import qa_factorial_das as fd  # noqa: E402
from core import qa_prompt_das as pd  # noqa: E402
from core.model_io import forward_logits  # noqa: E402

KEEP_PROTOCOL = "e44_keepfix_v1"
KEEP_SEED_OFFSET = 7777
KEEP_PROMPT = "x_E2"
KEEP_POSITIONS = "prompt"
KEEP_SET_PROTOCOL = "keepfix_keep_set_v1"
OWN = Path(__file__).resolve()
KEEP = {}
KEEP_FLAGS = {"--keep-set": str, "--keep-lambda": float, "--keep-amount-from": str, "--keep-prompts-extra": str,
              "--keep-check-dir": str, "--keep-report": str, "--keep-mem-cap-mib": int}
NEED = {"train": ("--keep-set", "--keep-lambda", "--keep-amount-from", "--keep-prompts-extra"),
        "unittest": ("--keep-set", "--keep-amount-from", "--keep-prompts-extra", "--keep-check-dir", "--keep-report")}
HAND_TOL = 1e-4      # training code vs hand computation, nats per token (one unpadded row each)
ZERO_TOL = 1e-6      # amount 0 vs no hook


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_keep_args(argv):
    """(keep options, the rest for the driver). Every keep flag takes one value and appears at most once."""
    keep, rest, i = {}, [], 0
    while i < len(argv):
        a = argv[i]
        if a in KEEP_FLAGS:
            if a in keep or i + 1 >= len(argv):
                raise SystemExit("%s given twice or without a value" % a)
            keep[a] = KEEP_FLAGS[a](argv[i + 1])
            i += 2
            continue
        if any(a.startswith(k + "=") for k in KEEP_FLAGS):
            raise SystemExit("write keep options as '--flag value', not '--flag=value'")
        rest.append(a)
        i += 1
    return keep, rest


# ----------------------------------------------------------------------------- keep inputs
def keep_amount(path, layer):
    """A from an existing generation record: parts[0] is E at this layer, kind w, written at "prompt", amount =
    beta * gap; every row carries the same amount."""
    g = json.loads(Path(path).read_text(encoding="utf-8"))
    parts = g.get("parts") or []
    if not parts:
        raise SystemExit("%s has no write parts" % path)
    p = parts[0]
    if g.get("positions") != KEEP_POSITIONS or p.get("family") != "E" or p.get("layer") != layer or p.get("kind") != "w":
        raise SystemExit("%s: parts[0] is not an E write at layer %d at %s" % (path, layer, KEEP_POSITIONS))
    a = float(p["amount"])
    if not (math.isfinite(a) and a > 0) or not math.isclose(a, float(p["beta"]) * float(p["gap"]), rel_tol=1e-9):
        raise SystemExit("%s: amount %r is not beta * gap" % (path, a))
    rows = g.get("rows", [])
    for r in rows:
        w = r["write"][0] if isinstance(r["write"], list) else r["write"]
        if not math.isclose(float(w["amount"]), a, rel_tol=1e-6):
            raise SystemExit("%s: a row was written with another amount" % path)
    return a, {"file": str(path), "file_sha256": sha_file(path), "arm": g.get("arm"), "beta": p["beta"], "gap": p["gap"],
               "direction_run": p.get("direction_run"), "direction_file_sha256": p.get("direction_file_sha256"),
               "rows": len(rows)}


def keep_system(meta, cfg, systems, args):
    """The x_E2 system text through the engine's own extra_systems (the text the generation renders)."""
    ns = argparse.Namespace(prompts_extra=Path(KEEP["--keep-prompts-extra"]), prompts=args.prompts)
    extra = X.extra_systems(meta, cfg, systems, ns)
    text = extra.get(KEEP_PROMPT)
    if text is None or text != meta["config"]["qa_system"] + " " + cfg["sentences"]["E2"]:
        raise SystemExit("%s is not the plain prompt plus the honest sentence E2" % KEEP_PROMPT)
    return text


def keep_rows(bundle, meta, cfg, systems, args, eot):
    """([{key, encoded}], skipped, system text, record of the keep set)."""
    path = Path(KEEP["--keep-set"])
    ks = json.loads(path.read_text(encoding="utf-8"))
    if ks.get("protocol") != KEEP_SET_PROTOCOL or ks.get("model") != meta["model"]:
        raise SystemExit("%s is not a keep set of %s" % (path, meta["model"]))
    system = keep_system(meta, cfg, systems, args)
    template = meta["config"]["qa_user"]
    items, skipped, seen = [], [], set()
    for it in ks["items"]:
        key = "%s:%s" % (it["task"], it["id"])
        if key in seen:
            raise SystemExit("keep set repeats %s" % key)
        seen.add(key)
        if it.get("n_tokens") != len(it.get("token_ids") or []):
            raise SystemExit("keep item %s is malformed" % key)
        reason = E.usable_answer(it, eot)
        if reason:
            skipped.append({"key": key, "reason": reason})
            continue
        prompt = pd.render_prompt(bundle, system, template, it["question"])
        try:
            enc = pd.build_target(bundle, prompt, answer_ids=it["token_ids"], finished=it["finished"],
                                  max_target_tokens=args.max_target_tokens, eot_ids=eot)
        except ValueError as exc:
            skipped.append({"key": key, "reason": "unusable_target: %s" % exc})
            continue
        items.append({"key": key, "encoded": enc, "cls": it.get("class")})
    if not items:
        raise SystemExit("no usable keep question")
    by_cls = {}
    for it in items:
        by_cls[it["cls"]] = by_cls.get(it["cls"], 0) + 1
    rec = {"file": str(path), "file_sha256": sha_file(path), "n_items": len(ks["items"]), "n_used": len(items),
           "n_used_by_class": by_cls, "n_skipped": len(skipped),
           "system_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest()}
    return items, skipped, system, rec


def keep_loss(bundle, layer, items, w, amount, grad):
    """(mean over rows of each row's mean answer-token cross-entropy, per-row tensor) with the fixed write
    amount * unit(w) at the "prompt" positions of layer."""
    packed = pd.pack_with_content_mask([it["encoded"] for it in items], bundle.tokenizer.pad_token_id, bundle.device)
    mask = fd.position_mask(packed, KEEP_POSITIONS)
    hook = fd.ShiftHook(w, amounts=torch.full((len(items),), float(amount)), mask=mask, decode_all=False)
    with pd.clamp(bundle, layer, hook):
        logits = forward_logits(bundle, packed["inputs"], grad=grad)
    if hook.applied != 1:
        raise RuntimeError("keep write must fire exactly once per teacher-forced forward")
    logp, row_index, target_ids = fd.answer_token_logprobs(logits, packed)
    per_token = -logp.gather(1, target_ids[:, None]).squeeze(1)
    n_tokens = packed["target_mask"].sum(1).float().to(per_token.device)
    per_row = per_token.new_zeros(len(items)).index_add(0, row_index, per_token) / n_tokens
    return per_row.mean(), per_row


@torch.no_grad()
def keep_eval(bundle, layer, items, w, amount, batch_size):
    """{"all": mean keep loss over all items, <class>: mean over the items of that class} (batches of batch_size;
    padding as in any batched forward)."""
    vals = []
    for i in range(0, len(items), batch_size):
        _, per_row = keep_loss(bundle, layer, items[i:i + batch_size], w, amount, grad=False)
        vals += per_row.tolist()
    out = {"all": sum(vals) / len(vals)}
    for c in sorted({str(it["cls"]) for it in items}):
        v = [x for x, it in zip(vals, items) if str(it["cls"]) == c]
        out[c] = sum(v) / len(v)
    return out


def mem_cap():
    cap = KEEP.get("--keep-mem-cap-mib")
    if cap is None:
        return None
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    if not 0 < cap < total:
        raise SystemExit("--keep-mem-cap-mib %d outside (0, %.0f)" % (cap, total))
    torch.cuda.set_per_process_memory_fraction(cap / total, 0)
    return {"cap_mib": cap, "device_total_mib": round(total)}


def mem_now():
    free, total = torch.cuda.mem_get_info()
    return {"peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2 ** 20),
            "device_used_mib": round((total - free) / 2 ** 20)}


# ----------------------------------------------------------------------------- training
def train_direction_keep(bundle, layer, questions, *, epochs, lr, clip, seed, init, loss, positions, val_questions,
                         log, keep_items, amount, lam, mean_vector):
    """core train_direction (same checks, initialisation, generator, order, schedule, clipping, renormalisation,
    history) plus lam * the keep loss of one keep question per step."""
    if not questions:
        raise ValueError("training needs at least one question")
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in (lr, clip)):
        raise ValueError("learning rate and clipping norm must be finite and positive")
    if bundle.model.training or any(p.requires_grad for p in bundle.model.parameters()):
        raise ValueError("direction training requires an eval-mode frozen model")
    if not (math.isfinite(lam) and lam >= 0):
        raise ValueError("keep lambda must be finite and nonnegative")
    g = torch.Generator(device="cpu").manual_seed(int(seed) * 1000 + int(layer))
    w0 = pd.unit(torch.randn(bundle.hidden_size, generator=g)) if init is None else pd.unit(init)
    w = torch.nn.Parameter(w0.clone().to(bundle.device))
    opt = torch.optim.Adam([w], lr=lr)
    kg = torch.Generator(device="cpu").manual_seed(int(seed) * 1000 + int(layer) + KEEP_SEED_OFFSET)
    md = torch.as_tensor(mean_vector).float().to(bundle.device)
    pending, used = [], []
    total_steps = epochs * len(questions)
    step = 0
    history = []
    t0 = time.time()
    for epoch in range(epochs):
        order = torch.randperm(len(questions), generator=g).tolist()
        sums = {"cause": 0.0, "isolate": 0.0}
        counts = {"cause": 0, "isolate": 0}
        grad_norms = []
        ksum, kn, kneg = 0.0, 0, 0
        for idx in order:
            q = questions[idx]
            kinds = [r["kind"] for r in q["rows"]]
            for group in opt.param_groups:
                group["lr"] = lr * (1.0 - step / total_steps)
            opt.zero_grad(set_to_none=True)
            per_row, _ = fd.rows_loss(bundle, layer, q["rows"], w=w, loss=loss, positions=positions, grad=True)
            value = fd.question_loss(per_row, kinds)
            if not torch.isfinite(value):
                raise RuntimeError("nonfinite training loss at layer %d" % layer)
            if not pending:
                pending = torch.randperm(len(keep_items), generator=kg).tolist()
            k = pending.pop(0)
            sign = 1.0 if float(torch.dot(md, w.detach().float())) >= 0 else -1.0
            kl, _ = keep_loss(bundle, layer, [keep_items[k]], w, sign * amount, grad=True)
            if not torch.isfinite(kl):
                raise RuntimeError("nonfinite keep loss at layer %d" % layer)
            (value + lam * kl).backward()
            if w.grad is None or not torch.isfinite(w.grad).all():
                raise RuntimeError("direction received no finite gradient at layer %d" % layer)
            grad_norms.append(float(torch.nn.utils.clip_grad_norm_([w], clip)))
            opt.step()
            with torch.no_grad():
                w.data = w.data / w.data.norm()
            step += 1
            for kk, v in zip(kinds, per_row.detach().tolist()):
                sums[kk] += v
                counts[kk] += 1
            ksum += float(kl.detach())
            kn += 1
            kneg += sign < 0
            used.append(k)
        record = {"epoch": epoch + 1,
                  "train_cause": sums["cause"] / max(1, counts["cause"]),
                  "train_isolate": (sums["isolate"] / counts["isolate"]) if counts["isolate"] else None,
                  "grad_norm_mean": sum(grad_norms) / len(grad_norms), "grad_norm_max": max(grad_norms),
                  "zero_grad_steps": sum(1 for x in grad_norms if x == 0.0), "seconds": time.time() - t0,
                  "keep_train_mean": ksum / kn, "keep_sign_negative_steps": kneg}
        if val_questions:
            record.update({"val_" + kk: v for kk, v in fd.mean_losses(bundle, layer, val_questions, w=w.detach(), loss=loss,
                                                                      positions=positions).items()})
        history.append(record)
        if log is not None:
            log(json.dumps({"layer": layer, **record}, ensure_ascii=False))
    times = [used.count(i) for i in range(len(keep_items))]
    info = {"history": history, "seed": seed, "lr": lr, "lr_schedule": "linear_to_zero",
            "clip": clip, "epochs": epochs, "n_questions": len(questions),
            "n_val_questions": len(val_questions), "loss": loss, "positions": positions,
            "init": "random" if init is None else "given",
            "init_sha256": pd.fingerprint(w0.tolist()), "seconds": time.time() - t0,
            "keep_steps": len(used), "keep_draws_sha256": pd.fingerprint(used), "keep_last_sign": sign,
            "keep_times_used_min": min(times), "keep_times_used_max": max(times)}
    return w.detach().float().cpu(), info


def run_train_keep(args, meta, prepared, bundle, cfg, systems, ident, ident_sha):
    """scripts/e44_factorial_das.run_train with the keep term (same checks, rows, references, held-out part, record
    layout and files); the additions are marked "keep"."""
    own_sha = sha_file(OWN)
    cap = mem_cap()
    if args.family != "E":
        raise SystemExit("the keep term is defined for the E family")
    lam = float(KEEP["--keep-lambda"])
    user_template = meta["config"]["qa_user"]
    spec = E.family_spec(args.family, args.e_on, getattr(args, 'e_off', None))
    missing = [n for n in spec["prompts"] if n not in systems]
    if missing:
        raise SystemExit("prompts file lacks %s" % missing)
    frozen, frozen_name = E.load_frozen(args, "train", systems, ident_sha, prepared, allow_pilot=E.is_pilot(args))
    questions = E.e43.pick_questions(prepared, "train", args)
    planned_train, planned_val = E.split_train_val(questions, args.val_n, getattr(args, 'split_seed', 42))
    train_ids = {q["pair_id"] for q in planned_train}
    val_ids = {q["pair_id"] for q in planned_val}
    built, skipped = E.build_questions(bundle, spec, questions, frozen, systems, user_template, args.max_target_tokens)
    train_q = [q for q in built if q["pair_id"] in train_ids]
    val_q = [q for q in built if q["pair_id"] in val_ids]
    if args.val_n and not val_q:
        raise ValueError("all preselected held-out questions were unusable; do not silently train without validation")
    E.log("%s (truth levels %s / %s): %d questions usable (%d train, %d held out), skipped %s" % (
        args.family, spec["e_off"] or "none", args.e_on, len(built), len(train_q), len(val_q), skipped))
    if not train_q:
        raise SystemExit("no training questions")
    eot = pd.turn_terminator_ids(bundle)
    if list(eot) != list(frozen["eot_ids"]):
        raise SystemExit("end-of-turn tokens differ from the frozen answers'")
    keep_items, keep_skipped, _, keep_rec = keep_rows(bundle, meta, cfg, systems, args, eot)
    E.log("keep: %d keep questions used, %d skipped, lambda %g" % (len(keep_items), len(keep_skipped), lam))
    ref_logp = {}
    for name in spec["prompts"]:                      # target-prompt distributions do not depend on layer or w
        pieces = fd.reference_logprobs(bundle, [q["refs"][name] for q in built], args.batch_size)
        ref_logp.update({(q["pair_id"], name): piece for q, piece in zip(built, pieces)})
    kw = {"loss": args.loss, "positions": args.positions}
    for layer in args.layers:
        amount, amount_src = keep_amount(KEEP["--keep-amount-from"], layer)
        out = E.dir_path(args, args.family, layer)
        contract = E.training_contract(args, spec, layer, train_q, val_q, frozen, args.run_dir / frozen_name)
        keep_contract = {"protocol": KEEP_PROTOCOL, "lambda": lam, "amount": amount, "amount_source": amount_src,
                         "prompt": KEEP_PROMPT, "positions": KEEP_POSITIONS, "loss": "ce_mean_over_answer_tokens",
                         "keep_rows_per_step": 1, "seed_offset": KEEP_SEED_OFFSET,
                         "orientation": "sign(mean_d . u), mean_d = training mean of h(plus) - h(minus)",
                         "keep_set": keep_rec, "entry": OWN.name, "entry_sha256": own_sha,
                         "max_target_tokens": args.max_target_tokens}
        contract["keep"] = keep_contract
        if out.exists() and not args.overwrite:
            E.validate_direction(torch.load(out, map_location="cpu", weights_only=False), args.family, layer, ident_sha,
                                 E.is_pilot(args), contract)
            E.log("skip L%d (%s verified)" % (layer, out.name))
            continue
        t0 = time.time()
        H = E.attach_layer(bundle, built, spec, layer, args.batch_size, ref_logp)
        tids = {q["pair_id"] for q in train_q}
        train_index = [i for i, q in enumerate(built) if q["pair_id"] in tids]
        plus = torch.cat([H[p][train_index] for p, _ in spec["plus_minus"]])
        minus = torch.cat([H[m][train_index] for _, m in spec["plus_minus"]])
        mean_vector = (plus - minus).mean(0)
        if not bool(torch.isfinite(mean_vector).all()):
            raise ValueError('nonfinite mean-difference reference')
        md_defined = bool(float(mean_vector.norm()) > 0)
        if not md_defined:
            raise ValueError("zero mean difference: the keep write has no inference orientation")
        w_md, md_norm = fd.mean_difference(plus, minus)
        w, info = train_direction_keep(bundle, layer, train_q, epochs=args.epochs, lr=args.lr, clip=args.clip,
                                       seed=args.seed, init=(w_md if args.init == "md" else None), loss=args.loss,
                                       positions=args.positions, val_questions=val_q, log=E.log, keep_items=keep_items,
                                       amount=amount, lam=lam, mean_vector=mean_vector)
        stats = fd.difference_stats(plus, minus, w)
        pre_flip_mean = stats["mean"]
        flipped = stats["mean"] < 0
        if stats["mean"] < 0:                            # orientation only: the write (d . w) w does not depend on the sign
            w = -w
            stats = fd.difference_stats(plus, minus, w)
        # keep: the side the keep term wrote at the last step (sign * u) against the side inference writes (+ saved w)
        sign_check = {"last_step_sign": info["keep_last_sign"], "pre_save_mean_projection": pre_flip_mean,
                      "flipped_at_save": bool(flipped), "saved_gap": stats["mean"],
                      "same_side": (info["keep_last_sign"] > 0) == (not flipped)}
        losses = {}
        for part, qs in (("train", train_q), ("val", val_q)):
            if qs:
                losses[part] = {"no_write": fd.mean_losses(bundle, layer, qs, mode="none", **kw),
                                "whole_difference": fd.mean_losses(bundle, layer, qs, mode="vectors", **kw),
                                "mean_difference_direction": fd.mean_losses(bundle, layer, qs, w=w_md.to(bundle.device), **kw),
                                "trained": fd.mean_losses(bundle, layer, qs, w=w.to(bundle.device), **kw)}
        diag = {"keep_loss_trained_at_A": keep_eval(bundle, layer, keep_items, w.to(bundle.device), amount, args.batch_size),
                "keep_loss_no_write": keep_eval(bundle, layer, keep_items, w.to(bundle.device), 0.0, args.batch_size),
                "keep_loss_opposite_side": keep_eval(bundle, layer, keep_items, w.to(bundle.device), -amount, args.batch_size),
                "batch_size": args.batch_size}
        leak = {}
        for kind in ("cause", "isolate"):                # how much of each row type's read difference lies along w
            vals = [abs(float(torch.dot(r["difference"], w))) for q in train_q for r in q["rows"] if r["kind"] == kind]
            leak[kind] = sum(vals) / max(1, len(vals))
        record = {"protocol": fd.PROTOCOL, "family": args.family, "e_on": args.e_on, "e_off": spec["e_off"], "layer": layer, "model": meta["model"],
                  "w": w, "w_md": w_md, "gap": stats["mean"], "md_norm": md_norm,
                  "mean_difference_defined": md_defined, "cosine_w_md": float(torch.dot(w, w_md)),
                  "difference_along_w": stats, "difference_along_md": fd.difference_stats(plus, minus, w_md),
                  "read_difference_abs_along_w": leak, "losses": losses, "training": info, "skipped": skipped,
                  "identity": ident, "identity_sha256": ident_sha, "pilot": E.is_pilot(args), "frozen_source": frozen_name,
                  "training_contract": contract, "planned_train_ids": [q["pair_id"] for q in planned_train],
                  "planned_val_ids": [q["pair_id"] for q in planned_val], "seconds": time.time() - t0,
                  "keep": dict(keep_contract, diagnostics=diag, sign_check=sign_check, skipped=keep_skipped, memory_cap=cap,
                               memory=mem_now())}
        E.require_current(ident, args)
        if sha_file(OWN) != own_sha:
            raise RuntimeError("this entry changed during training")
        if E.file_sha(args.run_dir / frozen_name) != contract["frozen_file_sha256"]:
            raise RuntimeError("frozen training inputs changed during training")
        record = E.seal(record)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        torch.save(record, tmp)
        tmp.replace(out)
        light = {k: v for k, v in record.items() if k not in ("w", "w_md")}
        for key in ("difference_along_w", "difference_along_md"):
            light[key] = {k: v for k, v in record[key].items() if k != "values"}
        light.update({"direction_file": out.name, "direction_file_sha256": E.file_sha(out)})
        E.save(args.run_dir / "train_logs" / E.artifact_name("%s_L%d.json" % (args.family, layer), E.is_pilot(args)), E.seal(light))
        tr = losses["train"]
        E.log("L%d done in %.0fs: gap %.3f (md norm %.3f, cos %.4f) | cause loss no-write %.4f whole-diff %.4f md %.4f trained %.4f"
              " | isolate trained %.4f | held-out cause %s isolate %s | keep lambda %g: at A %.4f, no write %.4f" % (
                  layer, record["seconds"], record["gap"], md_norm, record["cosine_w_md"], tr["no_write"]["cause"],
                  tr["whole_difference"]["cause"], tr["mean_difference_direction"]["cause"], tr["trained"]["cause"],
                  tr["trained"]["isolate"] or 0.0,
                  ("%.4f" % losses["val"]["trained"]["cause"]) if val_q else "-",
                  ("%.4f" % (losses["val"]["trained"]["isolate"] or 0.0)) if val_q else "-",
                  lam, diag["keep_loss_trained_at_A"]["all"], diag["keep_loss_no_write"]["all"]))


# ----------------------------------------------------------------------------- unit test
@torch.no_grad()
def hand_ce(bundle, layer, enc, w, amount):
    """Independent computation: one unpadded sequence, own hook adding amount * unit(w) at positions
    content_start .. prompt_length - 1 (amount None = no hook), own log-softmax at the predictor positions, mean."""
    ids = list(enc["full_ids"])
    P, c, L = enc["prompt_length"], enc["content_start"], enc["answer_length"]
    dev = bundle.device
    inputs = {"input_ids": torch.tensor([ids], device=dev), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=dev)}
    handle = None
    if amount is not None:
        d = torch.as_tensor(w).to(dev).float()
        u = d / d.norm()

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hf = h.float()
            add = torch.zeros_like(hf)
            add[0, c:P, :] = float(amount) * u
            h2 = (hf + add).to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        handle = bundle.layer(layer).register_forward_hook(hook)
    try:
        logits = forward_logits(bundle, inputs, grad=False)
    finally:
        if handle is not None:
            handle.remove()
    lp = torch.log_softmax(logits[0].float(), dim=-1)
    target = ids[P:P + L]
    vals = [-float(lp[P - 1 + j, t]) for j, t in enumerate(target)]
    return sum(vals) / L


def grad_checks(bundle, layer, four, w, amount):
    """The keep loss alone (preflight added from the GPT review of 2026-11-23, section 2 item 2), on the four
    questions: its gradient to the direction parameter (finite, nonzero, part tangent to the unit sphere); where the
    write lands (layer output with and without the write: the changed positions are content_start .. prompt_length - 1
    exactly, the rest bit-identical, the coordinate added along u about A); and a step of the direction against /
    along the tangent gradient (renormalized as in training) at t = 0.01, 0.03, 0.1. Pass: finite nonzero tangent
    gradient, exact positions, and down < loss < up at t = 0.03 and t = 0.1."""
    dev = bundle.device
    u = pd.unit(w).to(dev)
    wp = torch.nn.Parameter(u.clone())
    for it in four:
        kl, _ = keep_loss(bundle, layer, [it], wp, amount, grad=True)
        (kl / len(four)).backward()
    g = wp.grad.detach().float()
    gt = g - (g @ u) * u

    def mean_at(ww):
        return sum(float(keep_loss(bundle, layer, [it], ww, amount, grad=False)[0]) for it in four) / len(four)
    base = mean_at(u)
    res = {"loss": base, "grad_finite": bool(torch.isfinite(g).all()), "grad_norm": float(g.norm()),
           "grad_tangent_norm": float(gt.norm()), "grad_radial": float(g @ u)}
    pos = []
    for it in four:
        enc = it["encoded"]
        P, c = enc["prompt_length"], enc["content_start"]
        packed = pd.pack_with_content_mask([enc], bundle.tokenizer.pad_token_id, dev)
        if packed["inputs"]["input_ids"].shape[1] != P + enc["answer_length"]:
            raise RuntimeError("a single keep row was padded")
        got = []
        for a in (None, amount):
            cap = {}

            def probe(module, inp, out, cap=cap):
                h = out[0] if isinstance(out, tuple) else out
                cap["h"] = h.detach().float()[0].cpu().clone()
            if a is None:
                hd = bundle.layer(layer).register_forward_hook(probe)
                try:
                    forward_logits(bundle, packed["inputs"], grad=False)
                finally:
                    hd.remove()
            else:
                hook = fd.ShiftHook(u, amounts=torch.full((1,), float(a)), mask=fd.position_mask(packed, KEEP_POSITIONS),
                                    decode_all=False)
                with pd.clamp(bundle, layer, hook):
                    hd = bundle.layer(layer).register_forward_hook(probe)      # after the write hook: sees the written state
                    try:
                        forward_logits(bundle, packed["inputs"], grad=False)
                    finally:
                        hd.remove()
            got.append(cap["h"])
        diff = got[1] - got[0]
        changed = [i for i in range(diff.shape[0]) if bool((diff[i] != 0).any())]
        uc = u.float().cpu()
        along = diff[c:P] @ uc
        pos.append({"key": it["key"], "length": int(diff.shape[0]), "expected_first_last": [c, P - 1],
                    "changed_equals_prompt_rule": changed == list(range(c, P)), "n_changed": len(changed),
                    "rest_bit_identical": bool(torch.equal(got[1][:c], got[0][:c]) and torch.equal(got[1][P:], got[0][P:])),
                    "added_along_u_min": float(along.min()), "added_along_u_max": float(along.max()),
                    "off_axis_rel_max": float(((diff[c:P] - along[:, None] * uc).norm(dim=-1)
                                               / got[0][c:P].norm(dim=-1).clamp_min(1e-12)).max())})
    res["positions"] = pos
    gh = gt / gt.norm() if float(gt.norm()) > 0 else gt
    steps = []
    for t in (0.01, 0.03, 0.1):
        steps.append({"t": t, "down": mean_at(pd.unit(u - t * gh).to(dev)), "up": mean_at(pd.unit(u + t * gh).to(dev))})
    res["steps"] = steps
    res["pass"] = bool(res["grad_finite"] and res["grad_tangent_norm"] > 0
                       and all(p["changed_equals_prompt_rule"] and p["rest_bit_identical"] for p in pos)
                       and all(st["down"] < base < st["up"] for st in steps if st["t"] >= 0.03))
    return res


def run_unittest(args, meta, prepared, bundle, cfg, systems, ident, ident_sha):
    cap = mem_cap()
    torch.cuda.reset_peak_memory_stats()
    report_path = Path(KEEP["--keep-report"])
    if report_path.exists():
        raise SystemExit("%s exists" % report_path)
    if len(args.layers) != 1 or args.family != "E":
        raise SystemExit("unittest: one E layer")
    layer = args.layers[0]
    eot = pd.turn_terminator_ids(bundle)
    items, skipped, system, keep_rec = keep_rows(bundle, meta, cfg, systems, args, eot)
    amount, amount_src = keep_amount(KEEP["--keep-amount-from"], layer)
    dpath = Path(KEEP["--keep-check-dir"]) / "dirs" / ("E_L%d.pt" % layer)
    if E.file_sha(dpath) != amount_src["direction_file_sha256"]:
        raise SystemExit("%s is not the direction of the amount record" % dpath)
    rec = torch.load(dpath, map_location="cpu", weights_only=False)
    w = rec["w"].float()
    # per keep question, the keep loss under the main E at A (batched): the four test questions are the two of each
    # task the main E hurts most (the cases the term is for; a clear signal for the gradient checks)
    vals = []
    for i in range(0, len(items), args.batch_size):
        vals += keep_loss(bundle, layer, items[i:i + args.batch_size], w.to(bundle.device), amount, grad=False)[1].tolist()
    tasks = sorted({it["key"].split(":", 1)[0] for it in items})
    ranked = sorted(range(len(items)), key=lambda i: (-vals[i], i))
    four = [items[i] for t in tasks for i in [j for j in ranked if items[j]["key"].startswith(t + ":")][:2]]
    if len(four) != 4:
        raise SystemExit("unittest needs two keep questions of each task")
    rows, ok = [], True
    for it in four:
        t_a = float(keep_loss(bundle, layer, [it], w.to(bundle.device), amount, grad=False)[0])
        h_a = hand_ce(bundle, layer, it["encoded"], w, amount)
        t_0 = float(keep_loss(bundle, layer, [it], w.to(bundle.device), 0.0, grad=False)[0])
        h_n = hand_ce(bundle, layer, it["encoded"], w, None)
        r = {"key": it["key"], "answer_tokens": it["encoded"]["answer_length"],
             "prompt_positions_written": it["encoded"]["prompt_length"] - it["encoded"]["content_start"],
             "train_code_at_A": t_a, "hand_at_A": h_a, "diff_at_A": t_a - h_a,
             "train_code_at_0": t_0, "hand_no_hook": h_n, "diff_at_0": t_0 - h_n}
        r["pass"] = abs(r["diff_at_A"]) <= HAND_TOL and abs(r["diff_at_0"]) <= ZERO_TOL
        ok &= r["pass"]
        rows.append(r)
        E.log("unittest %s: train code %.6f hand %.6f (diff %.2e) | amount 0 %.6f no hook %.6f (diff %.2e) | %s" % (
            it["key"], t_a, h_a, r["diff_at_A"], t_0, h_n, r["diff_at_0"], "pass" if r["pass"] else "FAIL"))
    batch4 = float(keep_loss(bundle, layer, four, w.to(bundle.device), amount, grad=False)[0])
    over_all = {"main_E_at_A": keep_eval(bundle, layer, items, w.to(bundle.device), amount, args.batch_size),
                "no_write": keep_eval(bundle, layer, items, w.to(bundle.device), 0.0, args.batch_size)}
    E.log("unittest: batch of 4 %.6f (mean of singles %.6f) | all %d keep questions: main E at A %s, no write %s" % (
        batch4, sum(r["train_code_at_A"] for r in rows) / 4, len(items), json.dumps(over_all["main_E_at_A"]),
        json.dumps(over_all["no_write"])))
    grads = grad_checks(bundle, layer, four, w, amount)
    ok &= grads["pass"]
    E.log("unittest keep-loss gradient: %s" % json.dumps(grads))
    # one full training step for the peak memory
    spec = E.family_spec("E", args.e_on, getattr(args, 'e_off', None))
    frozen, _ = E.load_frozen(args, "train", systems, ident_sha, prepared, allow_pilot=False)
    qs = E.e43.pick_questions(prepared, "train", args)
    built, _ = E.build_questions(bundle, spec, qs, frozen, systems, meta["config"]["qa_user"], args.max_target_tokens)
    longest = max(built, key=lambda q: max(r["encoded"]["prompt_length"] + r["encoded"]["answer_length"] for r in q["rows"]))
    width = max(r["encoded"]["prompt_length"] + r["encoded"]["answer_length"] for r in longest["rows"])
    kmax = max(items, key=lambda it: it["encoded"]["prompt_length"] + it["encoded"]["answer_length"])
    ref = {}
    for name in spec["prompts"]:
        ref[(longest["pair_id"], name)] = fd.reference_logprobs(bundle, [longest["refs"][name]], args.batch_size)[0]
    E.attach_layer(bundle, [longest], spec, layer, args.batch_size, ref)
    memory = {"before_step": mem_now()}
    step = {"question": longest["pair_id"], "rows": len(longest["rows"]), "width": width,
            "keep_width": kmax["encoded"]["prompt_length"] + kmax["encoded"]["answer_length"]}
    try:
        wp = torch.nn.Parameter(pd.unit(w).to(bundle.device))
        per_row, _ = fd.rows_loss(bundle, layer, longest["rows"], w=wp, loss=args.loss, positions=args.positions, grad=True)
        value = fd.question_loss(per_row, [r["kind"] for r in longest["rows"]])
        kl, _ = keep_loss(bundle, layer, [kmax], wp, amount, grad=True)
        (value + kl).backward()
        torch.cuda.synchronize()
        step.update(question_loss=float(value.detach()), keep_loss=float(kl.detach()),
                    grad_norm=float(wp.grad.norm()), fits_cap=True)
    except torch.cuda.OutOfMemoryError as exc:
        step.update(fits_cap=False, error=str(exc)[:300])
    memory["after_step"] = mem_now()
    report = {"protocol": KEEP_PROTOCOL + "_unittest", "model": meta["model"], "layer": layer, "amount": amount,
              "amount_source": amount_src, "keep_set": keep_rec, "keep_skipped": skipped, "direction": str(dpath),
              "rows": rows, "tolerances": {"hand": HAND_TOL, "zero": ZERO_TOL}, "batch_of_4": batch4,
              "gradient_check": grads, "four": [it["key"] for it in four],
              "all_keep_questions": over_all, "passed": bool(ok), "step": step, "memory": memory, "memory_cap": cap,
              "entry_sha256": sha_file(OWN), "identity_sha256": ident_sha}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = report_path.with_suffix(report_path.suffix + ".new")
    tmp.write_text(json.dumps(report, indent=1), encoding="utf-8")
    tmp.replace(report_path)
    E.log("unittest memory: %s | step %s" % (json.dumps(memory), json.dumps({k: v for k, v in step.items() if k != "error"})))
    if not ok:
        raise SystemExit("unittest FAILED")
    E.log("unittest PASS")


# ----------------------------------------------------------------------------- CPU modes
def cmd_compare(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--min-cos", type=float, default=0.9999)
    a = ap.parse_args(argv)
    ra = torch.load(a.a, map_location="cpu", weights_only=False)
    rb = torch.load(a.b, map_location="cpu", weights_only=False)
    wa, wb = ra["w"].double(), rb["w"].double()
    out = {"a": a.a, "b": a.b, "elementwise_equal": bool(torch.equal(ra["w"], rb["w"])),
           "cosine": float(wa @ wb / (wa.norm() * wb.norm())), "max_abs_diff": float((wa - wb).abs().max()),
           "gap": [ra["gap"], rb["gap"]], "md_norm": [ra["md_norm"], rb["md_norm"]],
           "cosine_w_md": [ra.get("cosine_w_md"), rb.get("cosine_w_md")], "w_md_equal": bool(torch.equal(ra["w_md"], rb["w_md"])),
           "layer": [ra["layer"], rb["layer"]], "keep_lambda": [(ra.get("keep") or {}).get("lambda"), (rb.get("keep") or {}).get("lambda")],
           "init_sha256_equal": ra["training"]["init_sha256"] == rb["training"]["init_sha256"],
           "contract_equal_without_keep": {k: v for k, v in ra["training_contract"].items() if k != "keep"}
           == {k: v for k, v in rb["training_contract"].items() if k != "keep"}}
    print(json.dumps(out))
    if out["cosine"] < a.min_cos:
        raise SystemExit(1)


def cmd_signcheck(argv):
    """The trained E_k (preflight added from the GPT review of 2026-11-23, section 2 item 1): the unit write amount
    inference uses (record gap) is finite and at least --min-gap and equals the mean of the stored per-question
    projections; the side the keep term wrote at the last step is the side inference writes; the keep loss on that side
    is below the opposite side's. With --l0: lambda 0 and lambda 1 used the same keep draws, keep set, seed, start, A."""
    ap = argparse.ArgumentParser()
    ap.add_argument("direction")
    ap.add_argument("--l0")
    ap.add_argument("--min-gap", type=float, default=0.1)
    a = ap.parse_args(argv)
    r = torch.load(a.direction, map_location="cpu", weights_only=False)
    k = r.get("keep") or {}
    sc, dg = k.get("sign_check") or {}, k.get("diagnostics") or {}
    gap = float(r["gap"])
    vals = r["difference_along_w"]["values"]
    out = {"gap": gap, "gap_from_values": sum(vals) / len(vals), "min_gap": a.min_gap, "sign_check": sc,
           "keep_loss_at_A": dg.get("keep_loss_trained_at_A"), "keep_loss_opposite_side": dg.get("keep_loss_opposite_side"),
           "keep_loss_no_write": dg.get("keep_loss_no_write")}
    ok = (math.isfinite(gap) and gap >= a.min_gap and math.isclose(out["gap_from_values"], gap, rel_tol=1e-6)
          and sc.get("same_side") is True and sc.get("saved_gap") == gap
          and dg["keep_loss_trained_at_A"]["all"] < dg["keep_loss_opposite_side"]["all"])
    if a.l0:
        r0 = torch.load(a.l0, map_location="cpu", weights_only=False)
        k0 = r0.get("keep") or {}
        same = {"keep_draws": r0["training"].get("keep_draws_sha256") == r["training"].get("keep_draws_sha256"),
                "keep_set": (k0.get("keep_set") or {}).get("file_sha256") == (k.get("keep_set") or {}).get("file_sha256"),
                "seed": r0["training"]["seed"] == r["training"]["seed"],
                "init": r0["training"]["init_sha256"] == r["training"]["init_sha256"], "amount": k0.get("amount") == k.get("amount")}
        out["lambda0_vs_lambda1_same"] = same
        out["lambdas"] = [k0.get("lambda"), k.get("lambda")]
        ok = ok and all(same.values())
    out["pass"] = bool(ok)
    print(json.dumps(out))
    if not ok:
        raise SystemExit(1)


def cmd_aligned(argv):
    keep, rest = split_keep_args(argv)
    if len(rest) != 1 or set(keep) != {"--keep-amount-from"}:
        raise SystemExit("aligned DIR.pt --keep-amount-from G")
    r = torch.load(rest[0], map_location="cpu", weights_only=False)
    gap = float(r["gap"])
    amount, _ = keep_amount(keep["--keep-amount-from"], r["layer"])
    s = amount / gap
    text = ("%.4f" % s).rstrip("0").rstrip(".")
    got = float(text) * gap
    print(json.dumps({"amount": amount, "gap": gap, "s_exact": s, "s": text, "amount_at_s": got,
                      "relative_difference": got / amount - 1, "keep_lambda": (r.get("keep") or {}).get("lambda")}), file=sys.stderr)
    print(text)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    mode = sys.argv[1]
    if mode == "compare":
        cmd_compare(sys.argv[2:])
        return
    if mode == "aligned":
        cmd_aligned(sys.argv[2:])
        return
    if mode == "signcheck":
        cmd_signcheck(sys.argv[2:])
        return
    if mode not in NEED:
        raise SystemExit(__doc__)
    keep, rest = split_keep_args(sys.argv[2:])
    if mode == "train" and not keep:
        sys.argv = [E.__file__, "train"] + rest        # switch off: the driver as it is
        E.main()
        return
    missing = [k for k in NEED[mode] if k not in keep]
    extra = sorted(set(keep) - set(NEED[mode]) - {"--keep-mem-cap-mib"})
    if missing or extra:
        raise SystemExit("%s: missing %s, not allowed %s" % (mode, missing, extra))
    KEEP.update(keep)
    E.run_train = run_train_keep if mode == "train" else run_unittest
    sys.argv = [E.__file__, "train"] + rest
    E.main()


if __name__ == "__main__":
    main()
