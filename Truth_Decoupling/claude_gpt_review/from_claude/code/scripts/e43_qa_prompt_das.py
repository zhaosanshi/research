#!/usr/bin/env python3
"""E43: prompt-source clamp DAS for free-form QA (plan 21, fourth version, route 乙; Claude, 2026-11-10).

E candidate   source = the same question under x_E (x plus one truthfulness sentence) vs x, read at
              the last prompt token; forward target = gold best_answer (only when it is a substantive
              answer), reverse target = the model's own frozen answer under x.
C candidate   source = x_caut vs x_conf (the QA-format cautious / confident systems); targets = the
              model's own frozen answers under the two prompts. No hand-written refusal strings.
Write         clamp the coordinate along the learned unit direction to the source coordinate at the
              base sequence's content positions (first user-content token .. end), training and
              inference alike; inference uses the training-set mean coordinate of the chosen side.

Phases
  freeze     greedy answers under x / x_E / x_conf / x_caut for the train (96) and tune (192) questions:
             the prompt-only controls, the frozen targets, refusal counts, C switch pairs, long targets.
  check      one layer, one question: token-position audit, clamp exactness (declared tolerance), gradient
             probes at that layer and at the last layer (sources re-read there).
  train      per layer: source capture, direction training (two-row batch per question), endpoints,
             mean-difference direction with the same operator.
  generate   per layer and arm: clamped greedy generation on a split, compared with the frozen answers
             of the arm's base prompt and with the frozen answers of the clamped side's own prompt.
  summarize  judge-free tables over all generated arms (pilot files excluded unless --include-pilot).

Pilot runs (--n-check / --pair-ids) write *.pilot.* files and are never read by full runs. Every frozen
file and direction record carries an identity fingerprint (model, prompts, caps, terminator, output
projection); later phases refuse inputs whose identity differs. Gold answers reach only E's forward
training target and never a tune/test prompt. The test split is never read here. Judging is a separate
step (e43_export_judge.py -> judge_two_axis_run.py cells / allenai / llm --judge pro / single).
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch

import common  # noqa: F401
from common import EXP_ROOT
from core import qa_prompt_das as pd
from core.model_io import load_model
from core.qa_context import _unusable_answer
from core.qa_precision import fp32_output_projection
from core.readout import rule_abstains

E_SENTENCE = "Answer truthfully and directly. Correct common misconceptions."
PROMPT_NAMES = ("x", "x_E", "x_conf", "x_caut")
NEGATION_RE = re.compile(r"\b(no|not|never|none|nothing|nobody|neither|nor|cannot|without|isn't|aren't|wasn't|"
                         r"weren't|don't|doesn't|didn't|won't|wouldn't|can't|couldn't|shouldn't|hasn't|haven't|"
                         r"hadn't)\b|n't\b", re.IGNORECASE)
FAMILIES = {
    # side "plus" is the side whose coordinate the main deployable arm clamps to.
    "E": {"plus": {"source": "x_E", "base": "x", "target": "gold"},
          "minus": {"source": "x", "base": "x_E", "target": "frozen:x"},
          "questions": "e_train"},
    "C": {"plus": {"source": "x_caut", "base": "x_conf", "target": "frozen:x_caut"},
          "minus": {"source": "x_conf", "base": "x_caut", "target": "frozen:x_conf"},
          "questions": "train"},
}
# arm -> (base prompt, kind). kind: plus/minus = clamp to that side's training mean; perq = this question's own
# plus-side coordinate; md = mean-difference direction; random = norm-matched random direction; none = no write.
ARMS = {
    "E": {"none": ("x", "none"), "main": ("x", "plus"), "recover": ("x_E", "minus"), "perq": ("x", "perq"),
          "md_main": ("x", "md"), "random_main": ("x", "random")},
    "C": {"none": ("x", "none"), "main": ("x", "plus"), "to_conf": ("x", "minus"), "conf_to_caut": ("x_conf", "plus"),
          "caut_to_conf": ("x_caut", "minus"), "md_main": ("x", "md"), "random_main": ("x", "random")},
}


# ----------------------------------------------------------------------------- io
def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def normalize_text(text):
    return " ".join((text or "").casefold().split()).rstrip(".!? ")


def negation_count(texts):
    return sum(1 for t in texts if NEGATION_RE.search(t or ""))


def is_pilot(args):
    return bool(args.pair_ids or args.n_check)


def suffix(args):
    return ".pilot" if is_pilot(args) else ""


# ----------------------------------------------------------------------------- inputs & identity
def load_source(args):
    meta = read(args.source_run / "bank.json")
    prepared = read(args.source_run / "questions.json")
    if "qa_system" not in meta["config"] or "qa_user" not in meta["config"]:
        raise ValueError("source run config lacks qa_system / qa_user")
    return meta, prepared


def prompt_systems(meta, prompts_path):
    cfg = read(prompts_path)["qa"]
    base = meta["config"]["qa_system"]
    return {"x": base, "x_E": base + " " + E_SENTENCE,
            "x_conf": cfg["system_confident"], "x_caut": cfg["system_cautious"]}


def run_identity(meta, systems, args, eot):
    """Everything a frozen answer or a direction depends on besides the question text."""
    ident = {"protocol": pd.PROTOCOL, "model": meta["model"], "model_path": meta["model_config"]["path"],
             "prompt_systems": systems, "user_template": meta["config"]["qa_user"], "eot_ids": list(eot),
             "max_new_tokens": args.max_new_tokens, "max_target_tokens": args.max_target_tokens,
             "output_projection": args.output_projection, "e_sentence": E_SENTENCE}
    return ident, pd.fingerprint(ident)


def require_identity(found, expected_sha, what):
    if found != expected_sha:
        raise SystemExit("%s was produced under a different identity (model / prompts / caps / terminator / output "
                         "projection): %s vs current %s. Use a fresh --run-dir or --overwrite after checking why." % (what, found, expected_sha))


def pick_questions(prepared, which, args):
    qs = prepared["e_train"] if which == "e_train" else prepared["splits"][which]
    if args.pair_ids:
        wanted = set(args.pair_ids)
        qs = [q for q in qs if q["pair_id"] in wanted]
    elif args.n_check:
        qs = qs[:args.n_check]
    if not qs:
        raise ValueError("empty question set %s" % which)
    return qs


def open_model(meta):
    bundle = load_model(meta["model_config"])
    if bundle.hidden_size != meta["hidden_size"] or bundle.num_layers != meta["num_layers"]:
        raise ValueError("loaded model architecture differs from the source run")
    if bundle.cfg.get("path") != meta["model_config"]["path"] or bundle.cfg.get("name") != meta["model_config"]["name"]:
        raise ValueError("loaded model path/name differs from the source run")
    return bundle


def render_all(bundle, systems, user_template, questions, names=PROMPT_NAMES):
    return {name: [pd.render_prompt(bundle, systems[name], user_template, q["question"]) for q in questions]
            for name in names}


def frozen_lookup(frozen):
    return {q["pair_id"]: q for q in frozen["questions"]}


def frozen_path(args, split, pilot):
    return args.run_dir / ("frozen_%s%s.json" % (split, ".pilot" if pilot else ""))


def load_frozen(args, split, identity_sha, allow_pilot):
    """A complete full freeze is preferred; a pilot freeze is accepted only for pilot runs."""
    full = frozen_path(args, split, False)
    if full.exists():
        d = read(full)
        if d.get("completed"):
            require_identity(d.get("identity_sha256"), identity_sha, full.name)
            return d, full.name
    if allow_pilot:
        pilot = frozen_path(args, split, True)
        if pilot.exists():
            d = read(pilot)
            require_identity(d.get("identity_sha256"), identity_sha, pilot.name)
            return d, pilot.name
    raise SystemExit("no usable frozen_%s.json (run the freeze phase%s)" % (split, " or a pilot freeze" if allow_pilot else ""))


def gold_target_text(q):
    """E's forward target: best_answer when it is a substantive answer; otherwise the question is unusable for E."""
    gold = q.get("best_answer")
    if not isinstance(gold, str) or not gold.strip():
        return None, "no_best_answer"
    reason = _unusable_answer(gold)
    if reason is not None:
        return None, "best_answer_%s" % reason
    return gold.strip(), None


# ----------------------------------------------------------------------------- freeze
def run_freeze(args, meta, prepared, bundle, systems, ident, ident_sha):
    user_template = meta["config"]["qa_user"]
    eot = pd.turn_terminator_ids(bundle)
    for split in args.splits:
        out_path = frozen_path(args, split, is_pilot(args))
        if out_path.exists() and read(out_path).get("completed") and not args.overwrite:
            log("skip %s (complete; --overwrite to redo)" % out_path.name)
            continue
        questions = pick_questions(prepared, split, args)
        rendered = render_all(bundle, systems, user_template, questions)
        answers = {}
        for name in PROMPT_NAMES:
            log("freeze %s %s: %d questions" % (split, name, len(questions)))
            answers[name] = pd.generate_prompts(bundle, rendered[name], max_new_tokens=args.max_new_tokens,
                                                batch_size=args.batch_size)
        rows = []
        for i, q in enumerate(questions):
            row = {"pair_id": q["pair_id"], "question": q["question"],
                   "content_start": {n: rendered[n][i]["content_start"] for n in PROMPT_NAMES},
                   "boundary_clean": {n: rendered[n][i]["boundary_clean"] for n in PROMPT_NAMES},
                   "answers": {}}
            for name in PROMPT_NAMES:
                rec = answers[name][i]
                rec["refusal_rule"] = bool(rule_abstains(rec["text"]))
                rec["over_cap"] = rec["n_tokens"] > args.max_target_tokens
                rec["stop_is_eot"] = (rec["stop_id"] in eot) if rec["finished"] else None
                row["answers"][name] = rec
            if split == "train":          # gold stays out of the tune file; it is only E's training target
                gold, reason = gold_target_text(q)
                row["best_answer"] = q.get("best_answer")
                if gold is None:
                    row["gold_target"] = {"usable": False, "reason": reason}
                else:
                    try:
                        t = pd.build_target(bundle, rendered["x"][i], answer_text=gold,
                                            max_target_tokens=args.max_target_tokens, eot_ids=eot)
                        row["gold_target"] = {"usable": True, "n_tokens": t["answer_body_tokens"], "truncated": t["truncated"]}
                    except ValueError as exc:
                        row["gold_target"] = {"usable": False, "reason": "encode_error: %s" % exc}
            rows.append(row)
        ref = {n: sum(r["answers"][n]["refusal_rule"] for r in rows) for n in PROMPT_NAMES}
        summary = {
            "n": len(rows), "refusal_rule": ref,
            "c_switch_conf_answers_caut_refuses": sum(1 for r in rows if not r["answers"]["x_conf"]["refusal_rule"]
                                                      and r["answers"]["x_caut"]["refusal_rule"]),
            "c_both_refuse": sum(1 for r in rows if r["answers"]["x_conf"]["refusal_rule"] and r["answers"]["x_caut"]["refusal_rule"]),
            "c_reverse_switch_conf_refuses_caut_answers": sum(1 for r in rows if r["answers"]["x_conf"]["refusal_rule"]
                                                              and not r["answers"]["x_caut"]["refusal_rule"]),
            "c_same_text": sum(1 for r in rows if normalize_text(r["answers"]["x_conf"]["text"]) == normalize_text(r["answers"]["x_caut"]["text"])),
            "e_prompt_changed_text_vs_x": sum(1 for r in rows if normalize_text(r["answers"]["x_E"]["text"])
                                              != normalize_text(r["answers"]["x"]["text"])),
            "over_cap": {n: sum(r["answers"][n]["over_cap"] for r in rows) for n in PROMPT_NAMES},
            "empty_answers": {n: sum(1 for r in rows if r["answers"][n]["n_tokens"] == 0) for n in PROMPT_NAMES},
            "unfinished": {n: sum(1 for r in rows if not r["answers"][n]["finished"]) for n in PROMPT_NAMES},
            "stop_not_eot": {n: sum(1 for r in rows if r["answers"][n]["stop_is_eot"] is False) for n in PROMPT_NAMES},
            "negation_words": {n: negation_count([r["answers"][n]["text"] for r in rows]) for n in PROMPT_NAMES},
            "boundary_clean_all": all(all(r["boundary_clean"].values()) for r in rows),
        }
        if split == "train":
            usable = [r for r in rows if r["gold_target"].get("usable")]
            summary["gold_usable_for_E"] = len(usable)
            summary["gold_unusable_reasons"] = {}
            for r in rows:
                if not r["gold_target"].get("usable"):
                    k = r["gold_target"]["reason"]
                    summary["gold_unusable_reasons"].setdefault(k, []).append(r["pair_id"])
            summary["gold_over_cap"] = sum(1 for r in usable if r["gold_target"]["truncated"])
            summary["negation_words"]["gold_target"] = negation_count([r["best_answer"] for r in usable])
        save(out_path, {"protocol": pd.PROTOCOL, "phase": "freeze", "split": split, "model": meta["model"],
                        "identity": ident, "identity_sha256": ident_sha, "pilot": is_pilot(args),
                        "prompt_systems": systems, "user_template": user_template, "eot_ids": eot,
                        "max_new_tokens": args.max_new_tokens, "max_target_tokens": args.max_target_tokens,
                        "question_ids": [q["pair_id"] for q in questions], "summary": summary, "questions": rows,
                        "completed": not is_pilot(args)})
        log("freeze %s summary: %s" % (split, json.dumps(summary, ensure_ascii=False)))


# ----------------------------------------------------------------------------- pairs
def target_for(bundle, prompt, spec, q, frozen_q, eot, cap):
    try:
        if spec == "gold":
            gold, reason = gold_target_text(q)
            if gold is None:
                return None, reason
            return pd.build_target(bundle, prompt, answer_text=gold, max_target_tokens=cap, eot_ids=eot), None
        name = spec.split(":", 1)[1]
        rec = frozen_q["answers"][name]
        if rec["n_tokens"] == 0:
            return None, "empty_frozen_answer"
        if rec["finished"] and rec.get("stop_id") not in eot:
            return None, "nonstandard_stop_token_%s" % rec.get("stop_id")
        return pd.build_target(bundle, prompt, answer_ids=rec["token_ids"], finished=rec["finished"],
                               max_target_tokens=cap, eot_ids=eot), None
    except ValueError as exc:
        return None, "unusable_target: %s" % exc


def build_pairs(bundle, family, questions, frozen, systems, user_template, cap):
    """Encoded base sequences for both sides (source hidden states are added per layer)."""
    spec = FAMILIES[family]
    look = frozen_lookup(frozen)
    eot = frozen["eot_ids"]
    pairs, skipped = [], []
    for q in questions:
        fq = look.get(q["pair_id"])
        if fq is None:
            raise ValueError("question %s has no frozen answers" % q["pair_id"])
        pair = {"pair_id": q["pair_id"]}
        reason = None
        for side in ("plus", "minus"):
            s = spec[side]
            base_prompt = pd.render_prompt(bundle, systems[s["base"]], user_template, q["question"])
            source_prompt = pd.render_prompt(bundle, systems[s["source"]], user_template, q["question"])
            encoded, reason = target_for(bundle, base_prompt, s["target"], q, fq, eot, cap)
            if reason:
                break
            pair[side] = {"encoded": encoded, "source_text": source_prompt["prefix_text"],
                          "base": s["base"], "source": s["source"], "target_spec": s["target"]}
        if reason:
            skipped.append({"pair_id": q["pair_id"], "reason": reason})
        else:
            pairs.append(pair)
    return pairs, skipped


def attach_sources(bundle, pairs, layer, batch_size):
    for side in ("plus", "minus"):
        H = pd.capture_prompt_end(bundle, [p[side]["source_text"] for p in pairs], layer, batch_size)
        for p, h in zip(pairs, H):
            p[side]["source_hidden"] = h
    return pairs


def pair_audit(pairs, skipped):
    out = {"n_pairs": len(pairs), "n_skipped": len(skipped), "skipped": skipped}
    for side in ("plus", "minus"):
        enc = [p[side]["encoded"] for p in pairs]
        out[side] = {"truncated": sum(e["truncated"] for e in enc),
                     "terminator_appended": sum(e["terminator_appended"] for e in enc),
                     "target_tokens_mean": sum(e["answer_length"] for e in enc) / max(1, len(enc)),
                     "base": pairs[0][side]["base"] if pairs else None, "source": pairs[0][side]["source"] if pairs else None,
                     "target_spec": pairs[0][side]["target_spec"] if pairs else None}
    return out


# ----------------------------------------------------------------------------- check
def run_check(args, meta, prepared, bundle, systems, ident, ident_sha):
    user_template = meta["config"]["qa_user"]
    layer = args.layers[0] if args.layers else args.layer
    family = args.family or "E"
    q = pick_questions(prepared, FAMILIES[family]["questions"], args)[0]
    eot = pd.turn_terminator_ids(bundle)
    report = {"protocol": pd.PROTOCOL, "phase": "check", "layer": layer, "family": family, "pair_id": q["pair_id"],
              "question": q["question"], "eot_ids": eot, "eot_text": bundle.tokenizer.decode(eot),
              "identity_sha256": ident_sha, "prompts": {}}
    for name in PROMPT_NAMES:
        p = pd.render_prompt(bundle, systems[name], user_template, q["question"])
        ids = p["prefix_ids"]
        report["prompts"][name] = {
            "prompt_tokens": p["prompt_length"], "content_start": p["content_start"], "boundary_clean": p["boundary_clean"],
            "token_before_content": bundle.tokenizer.decode(ids[p["content_start"] - 1:p["content_start"]]),
            "content_first_token": p["content_first_token"], "read_position": p["read_position"], "read_token": p["read_token"],
            "written_prompt_positions": p["prompt_length"] - p["content_start"],
            "system_and_header_tokens_not_written": p["content_start"],
            "not_written_text": bundle.tokenizer.decode(ids[:p["content_start"]]),
            "content_tokens_text": bundle.tokenizer.decode(ids[p["content_start"]:])}
        log("%s: %d tokens, content starts at %d (%r), reads at %d (%r)" % (
            name, p["prompt_length"], p["content_start"], p["content_first_token"], p["read_position"], p["read_token"]))
    try:
        frozen, frozen_name = load_frozen(args, "train", ident_sha, allow_pilot=True)
    except SystemExit:
        log("no usable frozen_train file; generating this question's frozen answers for the check")
        rendered = render_all(bundle, systems, user_template, [q])
        fq = {"pair_id": q["pair_id"], "answers": {}}
        for name in PROMPT_NAMES:
            rec = pd.generate_prompts(bundle, rendered[name], max_new_tokens=args.max_new_tokens, batch_size=1)[0]
            rec["refusal_rule"] = bool(rule_abstains(rec["text"]))
            fq["answers"][name] = rec
        frozen, frozen_name = {"eot_ids": eot, "questions": [fq]}, "(generated now)"
    report["frozen_source"] = frozen_name
    pairs, skipped = build_pairs(bundle, family, [q], frozen, systems, user_template, args.max_target_tokens)
    report["skipped"] = skipped
    out_path = args.run_dir / ("check_%s_L%d.json" % (family, layer))
    if not pairs:
        save(out_path, report)
        raise SystemExit("question skipped: %s" % skipped)
    pair = pairs[0]
    for side in ("plus", "minus"):
        e = pair[side]["encoded"]
        report[side] = {"base": pair[side]["base"], "source": pair[side]["source"], "target_spec": pair[side]["target_spec"],
                        "target_tokens": e["answer_length"], "target_body_tokens": e["answer_body_tokens"],
                        "truncated": e["truncated"], "terminator_appended": e["terminator_appended"],
                        "target_text": e["answer_text"], "target_ids": e["answer_ids"],
                        "target_positions": list(range(e["prompt_length"], e["prompt_length"] + e["answer_length"])),
                        "predictor_positions": list(range(e["prompt_length"] - 1, e["prompt_length"] + e["answer_length"] - 1)),
                        "content_positions": "%d..%d" % (e["content_start"], e["prompt_length"] + e["answer_length"] - 1)}
        log("%s side: base %s, source %s, target %r (%d tokens, truncated=%s, eot=%s)" % (
            side, pair[side]["base"], pair[side]["source"], e["answer_text"], e["answer_length"], e["truncated"], e["terminator_appended"]))
    items = [pair["plus"]["encoded"], pair["minus"]["encoded"]]
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    w = pd.unit(torch.randn(bundle.hidden_size, generator=g))
    report["gradient_probe"], report["clamp_check"] = {}, {}
    for probe_layer in sorted({layer, bundle.num_layers - 1}):
        attach_sources(bundle, pairs, probe_layer, args.batch_size)
        src = torch.stack([pair["plus"]["source_hidden"], pair["minus"]["source_hidden"]])
        report["clamp_check"][str(probe_layer)] = pd.verify_clamp(bundle, probe_layer, items, src, w)
        report["gradient_probe"][str(probe_layer)] = pd.gradient_probe(bundle, probe_layer, items, src, seed=args.seed)
        report["gradient_probe"][str(probe_layer)].update({
            "source_gap_along_random_w": float((src[0] - src[1]) @ w), "source_norms": [float(src[0].norm()), float(src[1].norm())]})
        log("L%d clamp check: %s" % (probe_layer, json.dumps(report["clamp_check"][str(probe_layer)])))
        log("L%d gradient probe: %s" % (probe_layer, json.dumps(report["gradient_probe"][str(probe_layer)])))
    report["passed"] = all(c["passed"] for c in report["clamp_check"].values()) and all(
        g["grad_finite"] and g["grad_norm"] > 0 for g in report["gradient_probe"].values())
    save(out_path, report)
    log("saved %s; passed=%s" % (out_path.name, report["passed"]))
    if not report["passed"]:
        raise SystemExit("check failed; see %s" % out_path)


# ----------------------------------------------------------------------------- train
def dir_path(args, family, layer):
    return args.run_dir / "dirs" / ("%s_L%d%s.pt" % (family, layer, suffix(args)))


def run_train(args, meta, prepared, bundle, systems, ident, ident_sha):
    user_template = meta["config"]["qa_user"]
    frozen, frozen_name = load_frozen(args, "train", ident_sha, allow_pilot=is_pilot(args))
    family = args.family
    questions = pick_questions(prepared, FAMILIES[family]["questions"], args)
    pairs, skipped = build_pairs(bundle, family, questions, frozen, systems, user_template, args.max_target_tokens)
    audit = pair_audit(pairs, skipped)
    log("%s: %d pairs, skipped %s (frozen: %s)" % (family, len(pairs), skipped, frozen_name))
    if not pairs:
        raise SystemExit("no training pairs")
    for layer in args.layers:
        out = dir_path(args, family, layer)
        if out.exists() and not args.overwrite:
            log("skip L%d (%s exists)" % (layer, out.name))
            continue
        t0 = time.time()
        attach_sources(bundle, pairs, layer, args.batch_size)
        H_plus = torch.stack([p["plus"]["source_hidden"] for p in pairs])
        H_minus = torch.stack([p["minus"]["source_hidden"] for p in pairs])
        w, train_log = pd.train_direction(bundle, layer, pairs, epochs=args.epochs, lr=args.lr, clip=args.clip,
                                          seed=args.seed, log=log)
        w_md = pd.mean_difference_direction(H_plus, H_minus)
        per_pair = pd.baseline_ce(bundle, layer, w, pairs)
        record = {
            "protocol": pd.PROTOCOL, "family": family, "layer": layer, "model": meta["model"], "w": w, "w_md": w_md,
            "identity": ident, "identity_sha256": ident_sha, "pilot": is_pilot(args), "frozen_source": frozen_name,
            "sides": {"plus": FAMILIES[family]["plus"], "minus": FAMILIES[family]["minus"]},
            "endpoints": {"plus": pd.coordinates(H_plus, w), "minus": pd.coordinates(H_minus, w)},
            "endpoints_md": {"plus": pd.coordinates(H_plus, w_md), "minus": pd.coordinates(H_minus, w_md)},
            "cosine_w_md": float(torch.dot(w, w_md)),
            "pair_ids": [p["pair_id"] for p in pairs], "pair_audit": audit, "training": train_log,
            "per_pair_ce_after": per_pair, "max_target_tokens": args.max_target_tokens, "seconds": time.time() - t0,
            "source_hidden_sha256": {"plus": pd.fingerprint(H_plus.tolist()), "minus": pd.fingerprint(H_minus.tolist())},
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(record, out)
        light = {k: v for k, v in record.items() if k not in ("w", "w_md", "per_pair_ce_after")}
        light["endpoints"] = {s: {k: v for k, v in c.items() if k != "values"} for s, c in record["endpoints"].items()}
        light["endpoints_md"] = {s: {k: v for k, v in c.items() if k != "values"} for s, c in record["endpoints_md"].items()}
        light["ce_after_mean"] = {"plus": sum(r["ce_plus"] for r in per_pair) / len(pairs),
                                  "minus": sum(r["ce_minus"] for r in per_pair) / len(pairs)}
        save(args.run_dir / "train_logs" / ("%s_L%d%s.json" % (family, layer, suffix(args))), light)
        log("L%d done in %.0fs: endpoints plus %.3f minus %.3f (gap %.3f), final CE %s" % (
            layer, record["seconds"], record["endpoints"]["plus"]["mean"], record["endpoints"]["minus"]["mean"],
            record["endpoints"]["plus"]["mean"] - record["endpoints"]["minus"]["mean"], json.dumps(light["ce_after_mean"])))


# ----------------------------------------------------------------------------- generate
def run_generate(args, meta, prepared, bundle, systems, ident, ident_sha):
    user_template = meta["config"]["qa_user"]
    frozen, frozen_name = load_frozen(args, args.split, ident_sha, allow_pilot=is_pilot(args))
    look = frozen_lookup(frozen)
    family = args.family
    questions = pick_questions(prepared, args.split, args)
    missing = [q["pair_id"] for q in questions if q["pair_id"] not in look]
    if missing:
        raise SystemExit("frozen %s lacks %d of the requested questions, e.g. %s" % (frozen_name, len(missing), missing[:3]))
    rendered = render_all(bundle, systems, user_template, questions)
    for layer in args.layers:
        path = dir_path(args, family, layer)
        if not path.exists():
            raise SystemExit("missing direction %s (train it first; pilot and full runs do not share files)" % path.name)
        record = torch.load(path, map_location="cpu", weights_only=False)
        require_identity(record.get("identity_sha256"), ident_sha, path.name)
        if bool(record.get("pilot")) != is_pilot(args):
            raise SystemExit("%s pilot flag does not match this run" % path.name)
        for arm in args.arms:
            base_name, kind = ARMS[family][arm]
            out_path = args.run_dir / "gen" / ("%s_L%d_%s_%s%s.json" % (family, layer, arm, args.split, suffix(args)))
            if out_path.exists() and not args.overwrite:
                log("skip %s" % out_path.name); continue
            direction, random_direction, targets, target_note = record["w"], None, None, ""
            side = "plus"
            if kind == "none":
                target_note = "no write (zero-write control; must reproduce the frozen answers)"
            elif kind in ("plus", "minus"):
                side = kind
                targets = record["endpoints"][kind]["mean"]
                target_note = "training mean coordinate of side %s (%s)" % (kind, record["sides"][kind]["source"])
            elif kind == "md":
                direction = record["w_md"]
                targets = record["endpoints_md"]["plus"]["mean"]
                target_note = "mean-difference direction, training mean coordinate of side plus"
            elif kind == "random":
                random_direction = pd.fixed_random_direction(bundle.hidden_size, args.seed, layer)
                targets = record["endpoints"]["plus"]["mean"]
                target_note = "random fixed direction, write magnitude matched to |a_plus - z| of the learned direction"
            elif kind == "perq":
                src_name = record["sides"]["plus"]["source"]
                H = pd.capture_prompt_end(bundle, [p["prefix_text"] for p in rendered[src_name]], layer, args.batch_size)
                targets = (H @ pd.unit(record["w"])).tolist()
                target_note = "per-question coordinate of this question's own %s prompt" % src_name
            else:
                raise ValueError(arm)
            side_prompt = record["sides"][side]["source"]          # the prompt whose behaviour the clamp should reproduce
            log("L%d %s: base %s, %s" % (layer, arm, base_name, target_note))
            if kind == "none":
                recs = pd.generate_prompts(bundle, rendered[base_name], max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
            else:
                recs = pd.generate_prompts(bundle, rendered[base_name], max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
                                           layer=layer, direction=direction, targets=targets, random_direction=random_direction)
            rows = []
            for i, (q, rec, prompt) in enumerate(zip(questions, recs, rendered[base_name])):
                fq = look[q["pair_id"]]["answers"]
                base_rec, side_rec = fq[base_name], fq[side_prompt]
                rec.update({"pair_id": q["pair_id"], "question": q["question"], "refusal_rule": bool(rule_abstains(rec["text"])),
                            "baseline_text": base_rec["text"], "baseline_refusal_rule": base_rec["refusal_rule"],
                            "changed": normalize_text(rec["text"]) != normalize_text(base_rec["text"]),
                            "side_prompt": side_prompt, "side_prompt_text": side_rec["text"],
                            "matches_side_prompt": normalize_text(rec["text"]) == normalize_text(side_rec["text"]),
                            "baseline_matches_side_prompt": normalize_text(base_rec["text"]) == normalize_text(side_rec["text"]),
                            "content_start": prompt["content_start"]})
                if isinstance(targets, list):
                    rec["target_value"] = targets[i]
                rows.append(rec)
            n = len(rows)
            summary = {"n": n, "changed": sum(r["changed"] for r in rows),
                       "refusal_before": sum(r["baseline_refusal_rule"] for r in rows),
                       "refusal_after": sum(r["refusal_rule"] for r in rows),
                       "matches_side_prompt": sum(r["matches_side_prompt"] for r in rows),
                       "baseline_matches_side_prompt": sum(r["baseline_matches_side_prompt"] for r in rows),
                       "unfinished": sum(1 for r in rows if not r["finished"])}
            if kind != "none":
                summary.update({"prefill_rel_mean": sum(r["write"]["prefill_rel"] for r in rows) / n,
                                "prefill_gap_abs_mean": sum(r["write"]["prefill_gap_abs"] for r in rows) / n,
                                "decode_rel_mean": sum((r["write"]["decode_rel_mean"] or 0.0) for r in rows) / n})
            save(out_path, {"protocol": pd.PROTOCOL, "phase": "generate", "family": family, "layer": layer, "arm": arm,
                            "split": args.split, "base_prompt": base_name, "side_prompt": side_prompt,
                            "target": targets if not isinstance(targets, list) else "per_question", "target_note": target_note,
                            "direction": {"none": "none", "md": "w_md", "random": "random"}.get(kind, "w"),
                            "random_direction_sha256": None if random_direction is None else pd.fingerprint(random_direction.tolist()),
                            "identity_sha256": ident_sha, "direction_file": path.name, "frozen_source": frozen_name,
                            "pilot": is_pilot(args), "max_new_tokens": args.max_new_tokens,
                            "question_ids": [q["pair_id"] for q in questions], "summary": summary, "rows": rows,
                            "completed": not is_pilot(args)})
            log("L%d %s: %s" % (layer, arm, json.dumps(summary)))
            if kind == "none" and summary["changed"]:
                log("WARNING: zero-write arm changed %d answers; generation is not reproducing the freeze" % summary["changed"])


# ----------------------------------------------------------------------------- summarize
def run_summarize(args):
    rows = []
    for path in sorted((args.run_dir / "gen").glob("*.json")):
        d = read(path)
        if d.get("pilot") and not args.include_pilot:
            continue
        s = d["summary"]
        rows.append({"family": d["family"], "layer": d["layer"], "arm": d["arm"], "split": d["split"], "pilot": bool(d.get("pilot")),
                     "n": s["n"], "changed": s["changed"], "refusal_before": s["refusal_before"], "refusal_after": s["refusal_after"],
                     "matches_side_prompt": s["matches_side_prompt"], "baseline_matches_side_prompt": s["baseline_matches_side_prompt"],
                     "unfinished": s["unfinished"], "prefill_rel_mean": s.get("prefill_rel_mean"), "decode_rel_mean": s.get("decode_rel_mean"),
                     "completed": d.get("completed", False)})
    train = {}
    for path in sorted((args.run_dir / "train_logs").glob("*.json")):
        d = read(path)
        if d.get("pilot") and not args.include_pilot:
            continue
        train["%s_L%d" % (d["family"], d["layer"])] = {"ce_after": d["ce_after_mean"], "gap": d["endpoints"]["plus"]["mean"] - d["endpoints"]["minus"]["mean"],
                                                        "cosine_w_md": d["cosine_w_md"], "n_pairs": d["pair_audit"]["n_pairs"], "seconds": d["seconds"]}
    rows.sort(key=lambda r: (r["family"], r["arm"], r["layer"]))
    fmt = lambda v: "—" if v is None else "%.3f" % v  # noqa: E731
    lines = ["# E43 judge-free summary (%s)" % ("pilot included" if args.include_pilot else "full runs only"), "",
             "| family | arm | layer | n | changed | refusal before → after | = side prompt (baseline =) | unfinished | write prefill / decode (‖Δh‖/‖h‖) | final CE plus / minus | endpoint gap | pairs | complete |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t = train.get("%s_L%d" % (r["family"], r["layer"]), {})
        ce = t.get("ce_after", {})
        lines.append("| %s | %s | %d | %d | %d | %d → %d | %d (%d) | %d | %s / %s | %s | %s | %s | %s%s |" % (
            r["family"], r["arm"], r["layer"], r["n"], r["changed"], r["refusal_before"], r["refusal_after"],
            r["matches_side_prompt"], r["baseline_matches_side_prompt"], r["unfinished"], fmt(r["prefill_rel_mean"]), fmt(r["decode_rel_mean"]),
            ("%.2f / %.2f" % (ce["plus"], ce["minus"])) if ce else "—", ("%.3f" % t["gap"]) if t else "—",
            str(t.get("n_pairs", "—")), "yes" if r["completed"] else "no", " (pilot)" if r["pilot"] else ""))
    (args.run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save(args.run_dir / "summary.json", {"generation": rows, "training": train})
    print("\n".join(lines))


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=("freeze", "check", "train", "generate", "summarize"))
    p.add_argument("--source-run", type=Path, required=True, help="E25 v1u run dir (bank.json, questions.json)")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--prompts", type=Path, default=Path(EXP_ROOT) / "configs" / "prompts" / "default_v1.json")
    p.add_argument("--family", choices=tuple(FAMILIES))
    p.add_argument("--layers", nargs="+", type=int, default=[])
    p.add_argument("--layer", type=int, default=16, help="check phase (or the first of --layers)")
    p.add_argument("--arms", nargs="+", default=["main"])
    p.add_argument("--split", default="tune", choices=("train", "tune"))
    p.add_argument("--splits", nargs="+", default=["train", "tune"], choices=("train", "tune"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--max-target-tokens", type=int, default=24)
    p.add_argument("--output-projection", default="float32", choices=("float32", "native"),
                   help="compute the output projection in float32 (E25/E42 practice) for every forward here")
    p.add_argument("--n-check", type=int, default=0, help="pilot: restrict to the first K questions (writes *.pilot.* files)")
    p.add_argument("--pair-ids", nargs="+", help="pilot: explicit question IDs (writes *.pilot.* files)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--include-pilot", action="store_true", help="summarize: include pilot files")
    args = p.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "summarize":
        run_summarize(args); return
    if args.phase in ("train", "generate") and not (args.family and args.layers):
        raise SystemExit("train/generate need --family and --layers")
    for arm in (args.arms if args.phase == "generate" else []):
        if arm not in ARMS[args.family]:
            raise SystemExit("unknown arm %s for family %s" % (arm, args.family))
    meta, prepared = load_source(args)
    systems = prompt_systems(meta, args.prompts)
    torch.manual_seed(args.seed)
    bundle = open_model(meta)
    eot = pd.turn_terminator_ids(bundle)
    ident, ident_sha = run_identity(meta, systems, args, eot)
    log("identity %s (output projection %s, pilot=%s)" % (ident_sha[:12], args.output_projection, is_pilot(args)))
    with fp32_output_projection(bundle, enabled=args.output_projection == "float32"):
        {"freeze": run_freeze, "check": run_check, "train": run_train, "generate": run_generate}[args.phase](
            args, meta, prepared, bundle, systems, ident, ident_sha)


if __name__ == "__main__":
    main()
