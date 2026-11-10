#!/usr/bin/env python3
"""E44: factorial two-variable interchange training for free-form QA (plan 25; Claude, 2026-11-10).

Two instruction factors, each on / off, give four system prompts for the same question:

                     abstaining forbidden (conf)      abstaining allowed (caut)
  truth level low           conf                            caut              (or conf_M / caut_M with --e-off M)
  truth sentence on         conf_<E>                        caut_<E>          (<E> = E1 or E2, see --e-on)

E candidate   cause rows: conf <-> conf_<E> (only the truth factor differs; abstaining is forbidden on both sides).
              isolate rows: conf <-> caut, conf_<E> <-> caut_<E> (only the OTHER factor differs).
C candidate   cause rows: conf <-> caut and conf_<E> <-> caut_<E>.  isolate rows: conf <-> conf_<E>, caut <-> caut_<E>.
              The low-level names above are replaced by conf_M/caut_M only when explicitly requested.
              No score threshold automatically changes the factor to M.
Row           base prompt + the target prompt's own frozen answer, with the write
                  h' = h + ((h_source - h_base) . w) w        (both states read at the last prompt token)
              at the base's write positions. cause: target = the source prompt (behave like it).
              isolate: target = the base prompt (do not change).
Loss          per-token JSD to the target prompt's own next-token distribution on that answer (default), or CE.
Inference     w     add beta * g * w, g = the training mean of (h_plus - h_minus) . w   (no source prompt needed)
              md    add beta * (the training-mean difference vector)                 (same operator, no training)
              rand  add beta * g along a fixed random axis
              wq    add beta * ((h_plus - h_minus) . w) w for THIS question       (interchange with the question's own
              dq    add beta * (h_plus - h_minus) for THIS question                 two prompts; dq = the whole difference)

Phases
  freeze     greedy answers of every prompt in the prompts file for the train (96) and tune (192) questions:
             the prompt-only controls (step 0), the frozen targets, refusal / length / sentence-end counts.
  train      per layer: read states, cached target distributions, direction training with a held-out part of the
             training questions, reference losses (no write / whole difference / mean-difference direction).
  generate   per layer and arm: shifted greedy generation on a split, compared with the frozen answers.
  mc         per layer and arm: log-probability of the dataset's true vs false reference answers under the same
             shift (judge-free discrimination readout; references never enter training or prompts), and of the
             sentence "I don't know." (continuous abstention readout). --bases alone scores prompts without a write.
  summarize  judge-free tables.

Arms are written  <base prompt>.<w|md|rand|wq|dq>.<p|m><strength>   e.g.  x.w.p1  conf.wq.p1  conf_E1.w.m1  x.md.p2  x.rand.p1
Pilot runs (--n-check / --pair-ids) write t.* files and are never read by full runs. The test split is
never read here. Judging is a separate step (e44_export_judge.py -> judge_two_axis_run.py cells / allenai / single).
"""
import argparse
import math
import re
import time
from pathlib import Path

import torch

import common  # noqa: F401
from common import EXP_ROOT
import e43_qa_prompt_das as e43
from core import qa_factorial_das as fd
from core import qa_prompt_das as pd
from core.experiment_provenance import implementation_identity
from core.qa_precision import fp32_output_projection
from core.qa_scoring import reference_answers
from core.readout import rule_abstains

ARTIFACT_PROTOCOL = "e44_factorial_v1"
FAMILIES = ("E", "C")
ARM_RE = re.compile(r"^(?P<base>[A-Za-z0-9_]+)\.(?P<dir>w|md|rand|wq|dq|iso)\.(?P<sign>[pm])(?P<beta>\d+(?:\.\d+)?)$")
PER_QUESTION = ("wq", "dq", "iso")
ABSTAIN_TEXT = "I don't know."      # scored under every write: a continuous abstention readout

read, save, log, seal, file_sha = e43.read, e43.save, e43.log, e43.seal, e43.file_sha
normalize_text, negation_count = e43.normalize_text, e43.negation_count
is_pilot = e43.is_pilot


def artifact_name(stem, pilot=False):
    """Pilot files are explicitly t.* and never collide with full artifacts."""
    return ("t." if pilot else "") + stem


# ----------------------------------------------------------------------------- prompts and families
def load_prompts(meta, base_prompts_path, e44_path):
    """name -> system text, built from the three base systems and the sentences of the E44 prompts file."""
    cfg = read(e44_path)
    if cfg.get("version") != ARTIFACT_PROTOCOL:
        raise ValueError("prompts file version %r is not %s" % (cfg.get("version"), ARTIFACT_PROTOCOL))
    qa = read(base_prompts_path)["qa"]
    bases = {"neutral": meta["config"]["qa_system"], "confident": qa["system_confident"], "cautious": qa["system_cautious"]}
    systems = {}
    for name, spec in cfg["prompts"].items():
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise ValueError("prompt name %r must be alphanumeric / underscore" % name)
        if spec["base"] not in bases or any(s not in cfg["sentences"] for s in spec["add"]):
            raise ValueError("prompt %s refers to an unknown base or sentence" % name)
        systems[name] = " ".join([bases[spec["base"]]] + [cfg["sentences"][s] for s in spec["add"]])
    if len(set(systems.values())) != len(systems):
        raise ValueError("two prompt names render the same system text")
    return cfg, systems


def normalize_e_off(value):
    if value in (None, "", "none"):
        return None
    if value != "M":
        raise ValueError("E44 low truth level must be none or explicit M")
    return value


def family_spec(family, e_on, e_off=None):
    """Rows of one family. cause / isolate entries are (base, source, target) prompt names;
    plus_minus lists the (plus, minus) prompt pairs whose mean difference defines the inference shift."""
    if e_on not in ("E1", "E2"):
        raise ValueError("E44 high truth level supports E1/E2; M is only an explicit low-level hypothesis")
    e_off = normalize_e_off(e_off)
    low = "" if e_off is None else "_" + e_off
    conf, caut, conf_e, caut_e = "conf" + low, "caut" + low, "conf_" + e_on, "caut_" + e_on
    both = lambda a, b: [(a, b), (b, a)]  # noqa: E731
    if family == "E":
        cause = [(b, s, s) for b, s in both(conf, conf_e)]
        isolate = [(b, s, b) for b, s in both(conf, caut) + both(conf_e, caut_e)]
        plus_minus = [(conf_e, conf)]
    elif family == "C":
        cause = [(b, s, s) for b, s in both(conf, caut) + both(conf_e, caut_e)]
        isolate = [(b, s, b) for b, s in both(conf, conf_e) + both(caut, caut_e)]
        plus_minus = [(caut, conf), (caut_e, conf_e)]
    else:
        raise ValueError("family must be E or C")
    names = sorted({n for row in cause + isolate for n in row})
    return {"family": family, "e_on": e_on, "e_off": e_off, "cause": cause, "isolate": isolate,
            "plus_minus": plus_minus, "prompts": names,
            "factor_pairs": {"E": [(conf_e, conf), (caut_e, caut)], "C": [(caut, conf), (caut_e, conf_e)]}}


def arm_target_prompt(family, e_on, base, direction, beta, e_off=None):
    """The prompt whose behaviour an in-distribution unit-strength arm should reproduce (None otherwise)."""
    if direction == "iso":
        question_pair(family_spec(family, e_on, e_off), base)
        return base if beta == 1.0 else None
    if direction == "rand" or abs(abs(beta) - 1.0) > 1e-9:
        return None
    for plus, minus in family_spec(family, e_on, e_off)["plus_minus"]:
        if base == minus and beta > 0:
            return plus
        if base == plus and beta < 0:
            return minus
    return None


def parse_arm(arm, systems):
    m = ARM_RE.match(arm)
    if not m:
        raise SystemExit("arm %r is not <base>.<w|md|rand|wq|dq>.<p|m><strength>" % arm)
    if m["base"] not in systems:
        raise SystemExit("arm %r: unknown base prompt %s" % (arm, m["base"]))
    beta = float(m["beta"]) * (1.0 if m["sign"] == "p" else -1.0)
    if m["dir"] == "iso" and beta != 1.0:
        raise SystemExit("isolate diagnostics require p1 (actual other-factor source minus base)")
    if beta == 0:
        raise SystemExit("arm %r: zero strength is the no-write control (use the frozen answers)" % arm)
    return m["base"], m["dir"], beta


def arm_file_token(arm):
    return arm.replace(".", "-")


# ----------------------------------------------------------------------------- identity
def load_source(args):
    """E44 uses train/tune, not E43's gold-filtered e_train.

    Explicit pilots may omit an unrequested tune split, avoiding any need to
    carry test records into a three-question execution check.
    """
    meta = read(args.source_run / 'bank.json')
    prepared = read(args.source_run / 'questions.json')
    if 'qa_system' not in meta.get('config', {}) or 'qa_user' not in meta.get('config', {}):
        raise ValueError('source bank lacks QA prompt configuration')
    splits = prepared.get('splits', {})
    train = e43._unique_questions(splits.get('train'))
    tune_rows = splits.get('tune', [])
    if not isinstance(tune_rows, list):
        raise ValueError('tuning cohort must be a list')
    phase = getattr(args, 'phase', None)
    requested = set(getattr(args, 'splits', [])) if phase == 'freeze' else (
        {getattr(args, 'split', None)} if phase in ('generate', 'mc') else set())
    if not tune_rows:
        if not is_pilot(args) or 'tune' in requested:
            raise ValueError('tuning cohort is empty; only an unrequested pilot split may be omitted')
        tune = {}
    else:
        tune = e43._unique_questions(tune_rows)
    if set(train).intersection(tune):
        raise ValueError('training and tuning question IDs overlap')
    return meta, prepared


def run_identity(meta, systems, cfg, args, eot, bundle):
    """E43's identity (source data, model files, tokenizer, runtime) with this experiment's code and prompts."""
    ident, _ = e43.run_identity(meta, systems, args, eot, bundle=bundle)
    root = Path(EXP_ROOT)
    ident.pop("e_sentence", None)
    ident.update({"protocol": fd.PROTOCOL, "artifact_protocol": ARTIFACT_PROTOCOL,
                  "implementation": implementation_identity(__file__, root),
                  "imported_scripts_sha256": {"e43_qa_prompt_das.py": file_sha(root / "scripts" / "e43_qa_prompt_das.py")},
                  "prompts_config": cfg, "max_ref_tokens": args.max_ref_tokens})
    return ident, pd.fingerprint(ident)


def require_current(ident, args):
    root = Path(EXP_ROOT)
    if ident["source_files_sha256"] != {n: file_sha(args.source_run / n) for n in ("bank.json", "questions.json")}:
        raise RuntimeError("source bank/questions changed during execution")
    if (ident["implementation"] != implementation_identity(__file__, root)
            or ident["imported_scripts_sha256"]["e43_qa_prompt_das.py"] != file_sha(root / "scripts" / "e43_qa_prompt_das.py")):
        raise RuntimeError("E44 implementation changed during execution")


# ----------------------------------------------------------------------------- frozen answers
def frozen_path(args, split, pilot):
    return args.run_dir / artifact_name("frozen_%s.json" % split, pilot)


def validate_frozen(d, questions, systems, identity_sha=None, require_full=False):
    e43.validate_artifact(d, "frozen answers")
    if identity_sha is not None:
        e43.require_identity(d["identity_sha256"], identity_sha, "frozen answers")
    if d.get("phase") != "freeze" or d.get("split") not in ("train", "tune"):
        raise ValueError("invalid frozen phase or split")
    if not d.get('execution_complete'):
        raise ValueError('frozen generation is incomplete')
    if require_full and (d.get("pilot") or not d.get("completed")):
        raise ValueError("a pilot/incomplete freeze cannot stand for a full cohort")
    expected = e43._unique_questions(questions)
    rows = e43._unique_questions(d.get("questions"))
    if list(rows) != list(expected) or d.get("question_ids") != list(expected) or d.get("cohort_sha256") != pd.fingerprint(questions):
        raise ValueError("frozen question cohort/order/text differs from the requested cohort")
    if d.get("prompt_systems") != systems:
        raise ValueError("frozen prompts differ from the prompts file")
    if (d.get('prompt_systems') != d['identity'].get('prompt_systems')
            or d.get('eot_ids') != d['identity'].get('eot_ids')):
        raise ValueError('frozen prompt/termination fields differ from their identity')
    for pid, row in rows.items():
        if row["question"] != expected[pid]["question"] or set(row.get("answers", {})) != set(systems):
            raise ValueError("frozen answers belong to a different question or prompt set")
        for rec in row["answers"].values():
            if (not isinstance(rec.get("text"), str) or not isinstance(rec.get("token_ids"), list)
                    or type(rec.get("finished")) is not bool or rec.get("n_tokens") != len(rec["token_ids"])):
                raise ValueError("frozen answer record is malformed")
    return d


def load_frozen(args, split, systems, identity_sha, prepared, allow_pilot):
    full = frozen_path(args, split, False)
    if full.exists():
        d = read(full)
        if d.get('split') != split:
            raise ValueError('frozen file declares a different split')
        if not d.get("completed"):
            raise ValueError("full freeze exists but is incomplete; do not silently substitute a pilot")
        return validate_frozen(d, prepared["splits"][split], systems, identity_sha, require_full=True), full.name
    if allow_pilot:
        pilot = frozen_path(args, split, True)
        if pilot.exists():
            d = validate_frozen(read(pilot), e43.pick_questions(prepared, split, args), systems, identity_sha)
            if not d.get("pilot") or d.get('split') != split:
                raise ValueError("pilot filename lacks pilot identity")
            return d, pilot.name
    raise FileNotFoundError("no usable frozen_%s.json (run the freeze phase%s)" % (split, " or a pilot freeze" if allow_pilot else ""))


def text_stats(texts):
    n = max(1, len(texts))
    return {"refusal_rule": sum(1 for t in texts if rule_abstains(t)),
            "ends_with_period": sum(1 for t in texts if (t or "").rstrip().endswith((".", "!", "?"))),
            "mean_words": sum(len((t or "").split()) for t in texts) / n,
            "negation_words": negation_count(texts)}


def run_freeze(args, meta, prepared, bundle, cfg, systems, ident, ident_sha):
    user_template = meta["config"]["qa_user"]
    eot = pd.turn_terminator_ids(bundle)
    for split in args.splits:
        out_path = frozen_path(args, split, is_pilot(args))
        questions = e43.pick_questions(prepared, split, args)
        if out_path.exists() and not args.overwrite:
            validate_frozen(read(out_path), questions, systems, ident_sha, require_full=not is_pilot(args))
            log("skip %s (verified; --overwrite to redo)" % out_path.name)
            continue
        rendered = {n: [pd.render_prompt(bundle, systems[n], user_template, q["question"]) for q in questions] for n in systems}
        answers = {}
        for name in systems:
            log("freeze %s %s: %d questions" % (split, name, len(questions)))
            answers[name] = pd.generate_prompts(bundle, rendered[name], max_new_tokens=args.max_new_tokens,
                                                batch_size=args.batch_size)
        rows = []
        for i, q in enumerate(questions):
            row = {"pair_id": q["pair_id"], "question": q["question"], "answers": {},
                   "content_start": {n: rendered[n][i]["content_start"] for n in systems},
                   "boundary_clean": {n: rendered[n][i]["boundary_clean"] for n in systems}}
            for name in systems:
                rec = answers[name][i]
                rec["refusal_rule"] = bool(rule_abstains(rec["text"]))
                rec["over_cap"] = rec["n_tokens"] > args.max_target_tokens
                rec["stop_is_eot"] = (rec["stop_id"] in eot) if rec["finished"] else None
                row["answers"][name] = rec
            rows.append(row)
        summary = {"n": len(rows), "per_prompt": {}, "pairs": {},
                   "boundary_clean_all": all(all(r["boundary_clean"].values()) for r in rows)}
        for name in systems:
            texts = [r["answers"][name]["text"] for r in rows]
            s = text_stats(texts)
            s.update({"unfinished": sum(1 for r in rows if not r["answers"][name]["finished"]),
                      "over_cap": sum(r["answers"][name]["over_cap"] for r in rows),
                      "empty": sum(1 for r in rows if r["answers"][name]["n_tokens"] == 0),
                      "stop_not_eot": sum(1 for r in rows if r["answers"][name]["stop_is_eot"] is False)})
            summary["per_prompt"][name] = s
        for a, b in cfg.get("step0_pairs", []):
            ra = [r["answers"][a] for r in rows]
            rb = [r["answers"][b] for r in rows]
            summary["pairs"]["%s->%s" % (a, b)] = {
                "changed_text": sum(1 for x, y in zip(ra, rb) if normalize_text(x["text"]) != normalize_text(y["text"])),
                "a_answers_b_refuses": sum(1 for x, y in zip(ra, rb) if not x["refusal_rule"] and y["refusal_rule"]),
                "a_refuses_b_answers": sum(1 for x, y in zip(ra, rb) if x["refusal_rule"] and not y["refusal_rule"])}
        require_current(ident, args)
        save(out_path, seal({"protocol": fd.PROTOCOL, "phase": "freeze", "split": split, "model": meta["model"],
                             "identity": ident, "identity_sha256": ident_sha, "pilot": is_pilot(args),
                             "prompt_systems": systems, "user_template": user_template, "eot_ids": eot,
                             "max_new_tokens": args.max_new_tokens, "max_target_tokens": args.max_target_tokens,
                             "question_ids": [q["pair_id"] for q in questions], "cohort_sha256": pd.fingerprint(questions),
                             "summary": summary, "questions": rows, "completed": not is_pilot(args), "execution_complete": True}))
        for name, s in summary["per_prompt"].items():
            log("freeze %s %-8s refusals %3d | sentence-end %3d | words %.1f | negation %3d | unfinished %d" % (
                split, name, s["refusal_rule"], s["ends_with_period"], s["mean_words"], s["negation_words"], s["unfinished"]))


# ----------------------------------------------------------------------------- training rows
def usable_answer(rec, eot):
    if rec["n_tokens"] == 0:
        return "empty_frozen_answer"
    if rec["finished"] and rec.get("stop_id") not in eot:
        return "nonstandard_stop_token_%s" % rec.get("stop_id")
    return None


def build_questions(bundle, spec, questions, frozen, systems, user_template, cap):
    """Per question: the rendered prompts and the rows (base + target answer) of the family; references are keyed
    by target prompt. A question is skipped when any prompt of the family has an unusable frozen answer."""
    look = e43.frozen_lookup(frozen)
    eot = frozen["eot_ids"]
    built, skipped = [], []
    for q in questions:
        fq = look.get(q["pair_id"])
        if fq is None or fq.get("question") != q["question"]:
            raise ValueError("question %s has no matching frozen answers" % q["pair_id"])
        reason = next((r for r in (usable_answer(fq["answers"][n], eot) for n in spec["prompts"]) if r), None)
        if reason:
            skipped.append({"pair_id": q["pair_id"], "reason": reason})
            continue
        prompts = {n: pd.render_prompt(bundle, systems[n], user_template, q["question"]) for n in spec["prompts"]}
        try:
            refs = {n: pd.build_target(bundle, prompts[n], answer_ids=fq["answers"][n]["token_ids"],
                                       finished=fq["answers"][n]["finished"], max_target_tokens=cap, eot_ids=eot)
                    for n in spec["prompts"]}
            rows = []
            for kind in ("cause", "isolate"):
                for base, source, target in spec[kind]:
                    rec = fq["answers"][target]
                    rows.append({"kind": kind, "base": base, "source": source, "target": target,
                                 "encoded": pd.build_target(bundle, prompts[base], answer_ids=rec["token_ids"],
                                                            finished=rec["finished"], max_target_tokens=cap, eot_ids=eot)})
        except ValueError as exc:
            skipped.append({"pair_id": q["pair_id"], "reason": "unusable_target: %s" % exc})
            continue
        built.append({"pair_id": q["pair_id"], "prompts": prompts, "refs": refs, "rows": rows})
    return built, skipped


def attach_layer(bundle, built, spec, layer, batch_size, ref_logp):
    """Read every prompt of the family at the last prompt token of `layer`; fill each row's difference and reference."""
    H = {n: pd.capture_prompt_end(bundle, [q["prompts"][n]["prefix_text"] for q in built], layer, batch_size)
         for n in spec["prompts"]}
    for i, q in enumerate(built):
        for row in q["rows"]:
            row["difference"] = (H[row["source"]][i] - H[row["base"]][i]).float().cpu()
            row["ref_logp"] = ref_logp[(q["pair_id"], row["target"])]
            row["ref_target_ids"] = list(q["refs"][row["target"]]["answer_ids"])
    return H


def split_train_val(built, val_n, seed):
    if type(val_n) is not int or val_n < 0:
        raise ValueError("held-out count must be a nonnegative integer")
    if val_n == 0:
        return list(built), []
    if len(built) <= val_n:
        raise ValueError("requested held-out questions leave no training data; explicitly choose a smaller --val-n for a pilot")
    g = torch.Generator(device="cpu").manual_seed(int(seed) * 7919 + 13)
    order = torch.randperm(len(built), generator=g).tolist()
    val = set(order[:val_n])
    return [q for i, q in enumerate(built) if i not in val], [q for i, q in enumerate(built) if i in val]


# ----------------------------------------------------------------------------- train
def dir_path(args, family, layer):
    return args.run_dir / "dirs" / artifact_name("%s_L%d.pt" % (family, layer), is_pilot(args))


def training_contract(args, spec, layer, train_q, val_q, frozen, frozen_file):
    return {"family": spec["family"], "e_on": spec["e_on"], "e_off": spec["e_off"], "layer": layer, "seed": args.seed, "epochs": args.epochs,
            "lr": args.lr, "clip": args.clip, "loss": args.loss, "positions": args.positions, "init": args.init,
            "val_n": args.val_n, 'split_seed': getattr(args, 'split_seed', 42), "pilot": is_pilot(args),
            "rows": {k: [list(r) for r in spec[k]] for k in ("cause", "isolate", "plus_minus")},
            "train_ids": [q["pair_id"] for q in train_q], "val_ids": [q["pair_id"] for q in val_q],
            "targets_sha256": pd.fingerprint([[r["encoded"] for r in q["rows"]] for q in train_q + val_q]),
            "frozen_record_sha256": frozen["artifact_sha256"], "frozen_file_sha256": file_sha(frozen_file)}


def validate_direction(record, family, layer, identity_sha, pilot, contract=None):
    e43.validate_artifact(record, "direction")
    e43.require_identity(record["identity_sha256"], identity_sha, "direction")
    if record.get("family") != family or record.get("layer") != layer or bool(record.get("pilot")) != pilot:
        raise ValueError("direction family/layer/pilot mismatch")
    if not isinstance(record.get("training_contract"), dict):
        raise ValueError("direction lacks a training contract")
    tc = record['training_contract']
    if (tc.get('family') != family or tc.get('layer') != layer or tc.get('e_on') != record.get('e_on') or normalize_e_off(tc.get('e_off')) != normalize_e_off(record.get('e_off'))
            or bool(tc.get('pilot')) != pilot or tc.get('positions') not in fd.POSITIONS):
        raise ValueError('direction metadata differs from its training contract')
    spec = family_spec(family, record['e_on'], record.get('e_off'))
    expected_rows = {k: [list(row) for row in spec[k]] for k in ('cause', 'isolate', 'plus_minus')}
    if tc.get('rows') != expected_rows:
        raise ValueError('direction was trained under a different factorial row definition')
    train_ids, val_ids = tc.get('train_ids', []), tc.get('val_ids', [])
    if (not train_ids or len(train_ids) != len(set(train_ids)) or len(val_ids) != len(set(val_ids))
            or set(train_ids).intersection(val_ids)):
        raise ValueError('invalid or overlapping direction training/held-out IDs')
    if contract is not None and record["training_contract"] != contract:
        raise ValueError("direction trained with different seed, parameters, rows, targets or frozen inputs")
    for name in ("w", "w_md"):
        w = record.get(name)
        if (not isinstance(w, torch.Tensor) or w.ndim != 1 or not bool(torch.isfinite(w).all())
                or not math.isclose(float(w.norm()), 1.0, rel_tol=1e-5, abs_tol=1e-5)):
            raise ValueError("direction vector must be finite and unit length")
    if not (math.isfinite(record.get("gap", float("nan"))) and math.isfinite(record.get("md_norm", float("nan")))):
        raise ValueError("direction record lacks finite shift sizes")
    return record


def _source_path(run_dir, subdir, name):
    if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('invalid source artifact filename')
    root = (Path(run_dir) / subdir).resolve()
    path = root / name
    if path.resolve().parent != root:
        raise ValueError('source artifact escapes its run directory')
    return path


def validate_direction_sources(record, run_dir):
    path = _source_path(run_dir, '', record.get('frozen_source'))
    contract = record['training_contract']
    if file_sha(path) != contract.get('frozen_file_sha256'):
        raise ValueError('direction training freeze changed')
    frozen = e43.validate_artifact(read(path), 'direction training freeze')
    if (frozen.get('phase') != 'freeze' or frozen.get('split') != 'train'
            or frozen.get('identity_sha256') != record['identity_sha256']
            or frozen.get('artifact_sha256') != contract.get('frozen_record_sha256')):
        raise ValueError('direction training freeze content identity differs')
    ids = set(frozen.get('question_ids', []))
    if not set(contract['train_ids'] + contract['val_ids']) <= ids:
        raise ValueError('direction train/held-out ID is foreign to its freeze')
    return frozen


def run_train(args, meta, prepared, bundle, cfg, systems, ident, ident_sha):
    user_template = meta["config"]["qa_user"]
    spec = family_spec(args.family, args.e_on, getattr(args, 'e_off', None))
    missing = [n for n in spec["prompts"] if n not in systems]
    if missing:
        raise SystemExit("prompts file lacks %s" % missing)
    frozen, frozen_name = load_frozen(args, "train", systems, ident_sha, prepared, allow_pilot=is_pilot(args))
    questions = e43.pick_questions(prepared, "train", args)
    # Freeze the split before target eligibility filtering; a skipped held-out
    # row must never be replaced with a previously training question or vice versa.
    planned_train, planned_val = split_train_val(questions, args.val_n, getattr(args, 'split_seed', 42))
    train_ids = {q["pair_id"] for q in planned_train}
    val_ids = {q["pair_id"] for q in planned_val}
    built, skipped = build_questions(bundle, spec, questions, frozen, systems, user_template, args.max_target_tokens)
    train_q = [q for q in built if q["pair_id"] in train_ids]
    val_q = [q for q in built if q["pair_id"] in val_ids]
    if args.val_n and not val_q:
        raise ValueError("all preselected held-out questions were unusable; do not silently train without validation")
    log("%s (truth levels %s / %s): %d questions usable (%d train, %d held out), skipped %s" % (
        args.family, spec["e_off"] or "none", args.e_on, len(built), len(train_q), len(val_q), skipped))
    if not train_q:
        raise SystemExit("no training questions")
    ref_logp = {}
    for name in spec["prompts"]:                      # target-prompt distributions do not depend on layer or w
        pieces = fd.reference_logprobs(bundle, [q["refs"][name] for q in built], args.batch_size)
        ref_logp.update({(q["pair_id"], name): piece for q, piece in zip(built, pieces)})
    kw = {"loss": args.loss, "positions": args.positions}
    for layer in args.layers:
        out = dir_path(args, args.family, layer)
        contract = training_contract(args, spec, layer, train_q, val_q, frozen, args.run_dir / frozen_name)
        if out.exists() and not args.overwrite:
            validate_direction(torch.load(out, map_location="cpu", weights_only=False), args.family, layer, ident_sha,
                               is_pilot(args), contract)
            log("skip L%d (%s verified)" % (layer, out.name))
            continue
        t0 = time.time()
        H = attach_layer(bundle, built, spec, layer, args.batch_size, ref_logp)
        train_ids = {q["pair_id"] for q in train_q}
        train_index = [i for i, q in enumerate(built) if q["pair_id"] in train_ids]
        plus = torch.cat([H[p][train_index] for p, _ in spec["plus_minus"]])
        minus = torch.cat([H[m][train_index] for _, m in spec["plus_minus"]])
        mean_vector = (plus - minus).mean(0)
        if not bool(torch.isfinite(mean_vector).all()):
            raise ValueError('nonfinite mean-difference reference')
        md_defined = bool(float(mean_vector.norm()) > 0)
        if md_defined:
            w_md, md_norm = fd.mean_difference(plus, minus)
        else:
            # A zero mean does not imply zero per-question differences. Keep
            # random-init DAS evaluable; the fixed mean-vector control is zero.
            # The unit placeholder only makes zero-shift storage well-defined.
            w_md = torch.zeros_like(mean_vector); w_md[0] = 1.
            md_norm = 0.0
            if args.init == 'md':
                raise ValueError('mean-difference initialization is undefined at this layer (zero mean)')
        w, info = fd.train_direction(bundle, layer, train_q, epochs=args.epochs, lr=args.lr, clip=args.clip, seed=args.seed,
                                     init=(w_md if args.init == "md" else None), val_questions=val_q, log=log, **kw)
        stats = fd.difference_stats(plus, minus, w)
        if stats["mean"] < 0:                            # orientation only: the write (d . w) w does not depend on the sign
            w = -w
            stats = fd.difference_stats(plus, minus, w)
        losses = {}
        for part, qs in (("train", train_q), ("val", val_q)):
            if qs:
                losses[part] = {"no_write": fd.mean_losses(bundle, layer, qs, mode="none", **kw),
                                "whole_difference": fd.mean_losses(bundle, layer, qs, mode="vectors", **kw),
                                "mean_difference_direction": fd.mean_losses(bundle, layer, qs, w=w_md.to(bundle.device), **kw) if md_defined else
                                    {**fd.mean_losses(bundle, layer, qs, mode='none', **kw), 'reference_status': 'zero_mean_no_write_not_a_direction'},
                                "trained": fd.mean_losses(bundle, layer, qs, w=w.to(bundle.device), **kw)}
        leak = {}
        for kind in ("cause", "isolate"):                # how much of each row type's read difference lies along w
            vals = [abs(float(torch.dot(r["difference"], w))) for q in train_q for r in q["rows"] if r["kind"] == kind]
            leak[kind] = sum(vals) / max(1, len(vals))
        record = {"protocol": fd.PROTOCOL, "family": args.family, "e_on": args.e_on, "e_off": spec["e_off"], "layer": layer, "model": meta["model"],
                  "w": w, "w_md": w_md, "gap": stats["mean"], "md_norm": md_norm,
                  "mean_difference_defined": md_defined, "cosine_w_md": float(torch.dot(w, w_md)) if md_defined else None,
                  "difference_along_w": stats, "difference_along_md": fd.difference_stats(plus, minus, w_md) if md_defined else
                      {"defined": False, "reason": "zero_mean_difference", "values": []},
                  "read_difference_abs_along_w": leak, "losses": losses, "training": info, "skipped": skipped,
                  "identity": ident, "identity_sha256": ident_sha, "pilot": is_pilot(args), "frozen_source": frozen_name,
                  "training_contract": contract, "planned_train_ids": [q["pair_id"] for q in planned_train],
                  "planned_val_ids": [q["pair_id"] for q in planned_val], "seconds": time.time() - t0}
        require_current(ident, args)
        if file_sha(args.run_dir / frozen_name) != contract["frozen_file_sha256"]:
            raise RuntimeError("frozen training inputs changed during training")
        record = seal(record)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        torch.save(record, tmp)
        tmp.replace(out)
        light = {k: v for k, v in record.items() if k not in ("w", "w_md")}
        for key in ("difference_along_w", "difference_along_md"):
            light[key] = {k: v for k, v in record[key].items() if k != "values"}
        light.update({"direction_file": out.name, "direction_file_sha256": file_sha(out)})
        save(args.run_dir / "train_logs" / artifact_name("%s_L%d.json" % (args.family, layer), is_pilot(args)), seal(light))
        tr = losses["train"]
        log("L%d done in %.0fs: gap %.3f (md norm %.3f, cos %s) | cause loss no-write %.4f whole-diff %.4f md %.4f trained %.4f"
            " | isolate trained %.4f | held-out cause %s isolate %s" % (
                layer, record["seconds"], record["gap"], md_norm, _fmt(record["cosine_w_md"]), tr["no_write"]["cause"],
                tr["whole_difference"]["cause"], tr["mean_difference_direction"]["cause"], tr["trained"]["cause"],
                tr["trained"]["isolate"] or 0.0,
                ("%.4f" % losses["val"]["trained"]["cause"]) if val_q else "-",
                ("%.4f" % (losses["val"]["trained"]["isolate"] or 0.0)) if val_q else "-"))


# ----------------------------------------------------------------------------- shared by generate / mc
def load_direction(args, family, layer, ident_sha):
    path = dir_path(args, family, layer)
    if not path.exists():
        raise SystemExit("missing direction %s (train it first; pilot and full runs do not share files)" % path.name)
    record = torch.load(path, map_location="cpu", weights_only=False)
    validate_direction(record, family, layer, ident_sha, is_pilot(args))
    if record['e_on'] != args.e_on or normalize_e_off(record.get('e_off')) != normalize_e_off(getattr(args, 'e_off', None)):
        raise ValueError('requested E factor differs from the trained direction; use the matching --e-on')
    validate_direction_sources(record, args.run_dir)
    return record, path


def _validate_rows(d, questions, what):
    expected = e43._unique_questions(questions)
    rows = e43._unique_questions(d.get('rows', []))
    if (list(rows) != list(expected) or d.get('question_ids') != list(expected)
            or d.get('cohort_sha256') != pd.fingerprint(questions)):
        raise ValueError(what + ' cohort/order/references differ')
    if any(rows[pid]['question'] != expected[pid]['question'] for pid in expected):
        raise ValueError(what + ' row belongs to a different question')
    return rows


def validate_generation(d, frozen, questions, run_dir):
    """One verified generation and all actual parents, shared by resume/export/summary."""
    e43.validate_artifact(d, 'generation')
    if d.get('phase') != 'generate' or d.get('split') != frozen.get('split'):
        raise ValueError('generation split/phase differs from freeze')
    if not d.get('execution_complete') or not d.get('baseline_reproduction_verified'):
        raise ValueError('generation execution/baseline verification is incomplete')
    e43.require_identity(d['identity_sha256'], frozen['identity_sha256'], 'generation')
    if d.get('frozen_record_sha256') != frozen['artifact_sha256']:
        raise ValueError('generation frozen content identity differs')
    rows = _validate_rows(d, questions, 'generation')
    base, direction, beta = parse_arm(d['arm'], frozen['prompt_systems'])
    if (d.get('base_prompt') != base or d.get('direction') != direction or d.get('beta') != beta):
        raise ValueError('generation metadata differs from its arm')
    frozen_file = _source_path(run_dir, '', d.get('frozen_source'))
    path = _source_path(run_dir, 'dirs', d.get('direction_file'))
    if file_sha(frozen_file) != d.get('frozen_file_sha256') or file_sha(path) != d.get('direction_file_sha256'):
        raise ValueError('generation source file changed')
    record = torch.load(path, map_location='cpu', weights_only=True)
    validate_direction(record, d['family'], d['layer'], d['identity_sha256'], bool(d['pilot']))
    validate_direction_sources(record, run_dir)
    if (record['artifact_sha256'] != d.get('direction_record_sha256') or record['e_on'] != d.get('e_on') or normalize_e_off(record.get('e_off')) != normalize_e_off(d.get('e_off'))
            or record['training_contract']['positions'] != d.get('positions')):
        raise ValueError('generation direction/positions/E factor changed')
    fc = e43.frozen_lookup(frozen)
    for pid, row in rows.items():
        if (not isinstance(row.get('text'), str) or not isinstance(row.get('token_ids'), list)
                or type(row.get('finished')) is not bool or row.get('n_tokens') != len(row['token_ids'])):
            raise ValueError('malformed generation answer')
        if row.get('baseline_text') != fc[pid]['answers'][base]['text']:
            raise ValueError('generation comparison baseline differs')
    contract = d.get('generation_contract', {})
    if any(contract.get(k) != d.get(k) for k in ('family', 'layer', 'arm', 'split', 'pilot', 'e_on', 'e_off', 'positions',
                                               'identity_sha256', 'cohort_sha256', 'direction_file_sha256', 'frozen_file_sha256')):
        raise ValueError('generation contract differs from its actual record')
    return d


def validate_mc(d, questions, run_dir):
    e43.validate_artifact(d, 'reference scoring')
    if d.get('phase') != 'mc' or d.get('kind') not in ('baseline', 'arm') or not d.get('execution_complete'):
        raise ValueError('reference scoring phase/incomplete artifact')
    _validate_rows(d, questions, 'reference scoring')
    contract = d.get('mc_contract', {})
    if any(contract.get(k) != d.get(k) for k in ('split', 'pilot', 'cohort_sha256', 'identity_sha256')):
        raise ValueError('reference scoring contract differs')
    if d['kind'] == 'baseline':
        if contract.get('prompt') != d.get('prompt') or d.get('prompt') not in d['identity']['prompt_systems']:
            raise ValueError('reference baseline prompt differs')
        base_rows = None
    else:
        for k in ('family', 'layer', 'arm', 'e_on', 'e_off', 'positions', 'direction_file_sha256', 'baseline_file_sha256'):
            if contract.get(k) != d.get(k):
                raise ValueError('reference arm contract differs: ' + k)
        path = _source_path(run_dir, 'dirs', d.get('direction_file'))
        if file_sha(path) != d['direction_file_sha256']:
            raise ValueError('reference scoring direction source changed')
        record = torch.load(path, map_location='cpu', weights_only=True)
        validate_direction(record, d['family'], d['layer'], d['identity_sha256'], bool(d['pilot']))
        validate_direction_sources(record, run_dir)
        if record['e_on'] != d['e_on'] or normalize_e_off(record.get('e_off')) != normalize_e_off(d.get('e_off')) or record['training_contract']['positions'] != d['positions']:
            raise ValueError('reference scoring E factor/positions differ from training')
        base, direction, beta = parse_arm(d['arm'], d['identity']['prompt_systems'])
        if (d.get('base_prompt') != base or d.get('direction') != direction or d.get('beta') != beta):
            raise ValueError('reference scoring arm metadata differs')
        base_path = _source_path(run_dir, 'mc', d.get('baseline_file'))
        if file_sha(base_path) != d['baseline_file_sha256']:
            raise ValueError('reference scoring baseline source changed')
        baseline = read(base_path)
        if baseline.get('kind') != 'baseline' or baseline.get('prompt') != base or baseline.get('identity_sha256') != d['identity_sha256']:
            raise ValueError('reference scoring baseline comes from another prompt or identity')
        validate_mc(baseline, questions, run_dir)
        base_rows = baseline['rows']
    if d.get('summary') != mc_summary(d['rows'], base_rows):
        raise ValueError('reference scoring summary differs from complete rows')
    return d


def question_pair(spec, base):
    """The actual factor edge for this base, preserving the other factor level."""
    canonical = family_spec(spec['family'], spec['e_on'], spec.get('e_off'))
    pairs = canonical['factor_pairs'][spec['family']]
    for plus, minus in pairs:
        if base in (plus, minus):
            return plus, minus
    raise ValueError('per-question arms require a factorial base; neutral x has no declared interchange edge')


def read_pair(spec, base, kind="effect"):
    if kind != "iso":
        return question_pair(spec, base)
    other = {"family": "C" if spec["family"] == "E" else "E", "e_on": spec["e_on"], 'e_off': spec.get('e_off')}
    plus, minus = question_pair(other, base)
    return (plus if base == minus else minus), base


def question_differences(bundle, spec, rendered, layer, batch_size, base, kind="effect"):
    """[N, hidden] from the actual edge at this base's other-factor level, never an average of distinct edges."""
    plus, minus = read_pair(spec, base, kind)
    H = {n: pd.capture_prompt_end(bundle, [p['prefix_text'] for p in rendered[n]], layer, batch_size)
         for n in (plus, minus)}
    return (H[plus] - H[minus]).float().cpu()


def arm_write(record, direction, beta, hidden_size, seed, layer, D=None):
    """Keyword arguments of an arm's write for generate_shift / score_answers.
    w / md / rand: one constant for every question. wq / dq: from each question's own two prompts (D, [N, hidden])."""
    if direction == "w":
        return {"direction": record["w"], "amount": beta * record["gap"]}
    if direction == "md":
        return {"direction": record["w_md"], "amount": beta * record["md_norm"]}
    if direction == "rand":
        return {"direction": fd.fixed_random_direction(hidden_size, seed, layer), "amount": beta * record["gap"],
                'reference_direction': record['w']}
    if D is None:
        raise ValueError("per-question arms need the question differences")
    if direction in ("wq", "iso"):
        return {"direction": record["w"], "amount": (beta * (D @ record["w"])).tolist()}
    if direction == "dq":
        return {"vectors": beta * D}
    raise ValueError(direction)


def write_note(write):
    if "vectors" in write:
        return "whole per-question difference, mean norm %.3f" % float(write["vectors"].norm(dim=-1).mean())
    a = write["amount"]
    return ("per-question coordinate, mean %+.3f" % (sum(a) / len(a))) if isinstance(a, list) else ("%+.3f" % a)


def verified_base(args, bundle, rendered, look, questions, name, cache):
    """The zero-write control: plain generation must reproduce the frozen answers of this prompt exactly."""
    if name not in cache:
        recs = pd.generate_prompts(bundle, rendered[name], max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
        bad = [q["pair_id"] for q, rec in zip(questions, recs)
               if any(rec.get(k) != look[q["pair_id"]]["answers"][name].get(k) for k in ("text", "token_ids", "finished", "stop_id"))]
        if len(recs) != len(questions) or bad:
            raise RuntimeError("zero-write generation does not reproduce frozen %s answers: %s" % (name, bad[:8]))
        cache[name] = recs
    return cache[name]


# ----------------------------------------------------------------------------- generate
def run_generate(args, meta, prepared, bundle, cfg, systems, ident, ident_sha):
    user_template = meta["config"]["qa_user"]
    frozen, frozen_name = load_frozen(args, args.split, systems, ident_sha, prepared, allow_pilot=is_pilot(args))
    frozen_sha = file_sha(args.run_dir / frozen_name)
    look = e43.frozen_lookup(frozen)
    questions = e43.pick_questions(prepared, args.split, args)
    arms = [(arm,) + parse_arm(arm, systems) for arm in args.arms]
    per_question = any(a[2] in PER_QUESTION for a in arms)
    rendered = {}

    def render(names):
        for n in names:
            if n not in rendered:
                rendered[n] = [pd.render_prompt(bundle, systems[n], user_template, q["question"]) for q in questions]

    render(sorted({a[1] for a in arms}))
    cache = {}
    for layer in args.layers:
        record, path = load_direction(args, args.family, layer, ident_sha)
        positions = record["training_contract"]["positions"]
        e_on, e_off = record["e_on"], record.get("e_off")
        direction_sha = file_sha(path)
        D = {}
        if per_question:
            spec = family_spec(args.family, e_on, e_off)
            for base, kind in sorted({(a[1], 'iso' if a[2] == 'iso' else 'effect') for a in arms if a[2] in PER_QUESTION}):
                render(read_pair(spec, base, kind))
                D[(base, kind)] = question_differences(bundle, spec, rendered, layer, args.batch_size, base, kind)
        for arm, base, direction, beta in arms:
            out_path = args.run_dir / "gen" / artifact_name("%s_L%d_%s_%s.json" % (args.family, layer, arm_file_token(arm), args.split), is_pilot(args))
            contract = {"family": args.family, "layer": layer, "arm": arm, "split": args.split, "pilot": is_pilot(args),
                        "seed": args.seed, "e_on": record["e_on"], "e_off": record.get("e_off"), "positions": positions, "cohort_sha256": pd.fingerprint(questions), "direction_file_sha256": direction_sha,
                        "frozen_file_sha256": frozen_sha, "identity_sha256": ident_sha}
            if out_path.exists() and not args.overwrite:
                previous = validate_generation(read(out_path), frozen, questions, args.run_dir)
                if previous.get("generation_contract") != contract:
                    raise ValueError("generation was made with a different seed, arm, subset or direction")
                log("skip %s (verified)" % out_path.name)
                continue
            verified_base(args, bundle, rendered, look, questions, base, cache)
            write = arm_write(record, direction, beta, bundle.hidden_size, args.seed, layer, D.get((base, "iso" if direction == "iso" else "effect")))
            target_prompt = arm_target_prompt(args.family, e_on, base, direction, beta, e_off)
            log("L%d %s: base %s, %s: %s%s" % (layer, arm, base, direction, write_note(write),
                                              (" (should behave like %s)" % target_prompt) if target_prompt else ""))
            recs = fd.generate_shift(bundle, rendered[base], max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
                                     layer=layer, positions=positions, **write)
            if len(recs) != len(questions):
                raise RuntimeError("generation returned the wrong number of answers")
            rows = []
            for q, rec in zip(questions, recs):
                fq = look[q["pair_id"]]["answers"]
                base_rec = fq[base]
                rec.update({"pair_id": q["pair_id"], "question": q["question"], "refusal_rule": bool(rule_abstains(rec["text"])),
                            "baseline_text": base_rec["text"], "baseline_refusal_rule": base_rec["refusal_rule"],
                            "changed": rec["text"] != base_rec["text"],
                            "changed_normalized": normalize_text(rec["text"]) != normalize_text(base_rec["text"])})
                if target_prompt:
                    t = fq[target_prompt]
                    rec.update({"target_prompt_text": t["text"],
                                "matches_target_prompt": normalize_text(rec["text"]) == normalize_text(t["text"]),
                                "baseline_matches_target_prompt": normalize_text(base_rec["text"]) == normalize_text(t["text"])})
                rows.append(rec)
            n = len(rows)
            before, after = text_stats([r["baseline_text"] for r in rows]), text_stats([r["text"] for r in rows])
            summary = {"n": n, "changed": sum(r["changed"] for r in rows),
                       "changed_normalized": sum(r["changed_normalized"] for r in rows),
                       "before": before, "after": after, "unfinished": sum(1 for r in rows if not r["finished"]),
                       "write": write_note(write), "amount_mean": sum(r["write"]["amount"] for r in rows) / n,
                       "prefill_rel_mean": sum(r["write"]["prefill_rel"] for r in rows) / n}
            if target_prompt:
                summary.update({"matches_target_prompt": sum(r["matches_target_prompt"] for r in rows),
                                "baseline_matches_target_prompt": sum(r["baseline_matches_target_prompt"] for r in rows)})
            require_current(ident, args)
            if file_sha(path) != direction_sha or file_sha(args.run_dir / frozen_name) != frozen_sha:
                raise RuntimeError("direction or frozen source changed during generation")
            save(out_path, seal({"protocol": fd.PROTOCOL, "phase": "generate", "family": args.family, "e_on": e_on, "e_off": e_off, "layer": layer,
                                 "arm": arm, "split": args.split, "base_prompt": base, "direction": direction, "beta": beta,
                                 "positions": positions, "target_prompt": target_prompt,
                                 "read_pair": list(read_pair(family_spec(args.family, e_on, e_off), base, "iso" if direction == "iso" else "effect")) if direction in PER_QUESTION else None,
                                 "identity": ident, "identity_sha256": ident_sha, "direction_file": path.name,
                                 "direction_file_sha256": direction_sha, "direction_record_sha256": record["artifact_sha256"],
                                 "frozen_source": frozen_name, "frozen_file_sha256": frozen_sha,
                                 "frozen_record_sha256": frozen["artifact_sha256"], "cohort_sha256": pd.fingerprint(questions),
                                 "generation_contract": contract, "baseline_reproduction_verified": True, "pilot": is_pilot(args),
                                 "max_new_tokens": args.max_new_tokens, "question_ids": [q["pair_id"] for q in questions],
                                 "summary": summary, "rows": rows, "completed": not is_pilot(args), "execution_complete": True}))
            log("L%d %s: changed %d | refusals %d -> %d | sentence-end %d -> %d | words %.1f -> %.1f%s | write/|h| %.3f" % (
                layer, arm, summary["changed_normalized"], before["refusal_rule"], after["refusal_rule"],
                before["ends_with_period"], after["ends_with_period"], before["mean_words"], after["mean_words"],
                (" | = %s %d (base %d)" % (target_prompt, summary["matches_target_prompt"], summary["baseline_matches_target_prompt"]))
                if target_prompt else "", summary["prefill_rel_mean"]))


# ----------------------------------------------------------------------------- mc (true vs false references)
def candidate_lists(q):
    """(true references, false references) of a question; None when the question has no substantive true reference."""
    true, only_refusal = reference_answers(q["gold"], question=q["question"])
    false = [a for a in (q.get("incorrect") or []) if isinstance(a, str) and a.strip()]
    if only_refusal or not true or not false:
        return None
    return list(true), false


def score_split(args, bundle, prompts, questions, eot, layer=None, write=None, positions="content"):
    """Score complete reference sets. Missing/truncated candidates never change the task silently."""
    if len(prompts) != len(questions):
        raise ValueError('scoring prompts and questions must have identical lengths')
    rows = []
    for i, (q, prompt) in enumerate(zip(questions, prompts)):
        cands = candidate_lists(q)
        true, false = cands if cands is not None else ([], [])
        kw = {}
        if write is not None and "vectors" in write:
            kw = {"vector": write["vectors"][i]}
        elif write is not None:
            a = write["amount"]
            kw = {"direction": write["direction"], "amount": a[i] if isinstance(a, list) else a}
            if 'reference_direction' in write:
                kw['reference_direction'] = write['reference_direction']
        texts = [ABSTAIN_TEXT] + true + false
        recs = fd.score_answers(bundle, prompt, texts, eot_ids=eot, max_target_tokens=args.max_ref_tokens,
                                batch_size=args.mc_batch_size, layer=layer, positions=positions, **kw)
        if len(recs) != len(texts):
            raise ValueError('candidate scorer returned a different number of candidates')
        recs = [dict(rec, text=text) for text, rec in zip(texts, recs)]
        t, f = recs[1:1 + len(true)], recs[1 + len(true):]
        complete = bool(cands) and all(_complete_candidate(rec) for rec in t + f)
        best = q.get('best_answer')
        best = best.strip() if isinstance(best, str) else None
        best_index = true.index(best) if best in true else None
        metrics = fd.choice_metrics([r['sum_logp'] for r in t], [r['sum_logp'] for r in f],
                                    best_true_index=best_index) if complete else None
        abstain = recs[0]['sum_logp'] if _complete_candidate(recs[0]) else None
        rows.append({'pair_id': q['pair_id'], 'question': q['question'], 'scored': metrics is not None,
                     'unscorable_reason': None if metrics is not None else ('no_complete_reference_set' if cands else 'no_substantive_true_or_false_candidates'),
                     'abstain_logp': abstain, 'abstain_scored': abstain is not None,
                     'abstain_record': recs[0], 'true': t, 'false': f, 'best_answer': best,
                     **(metrics or {})})
    return rows


def _finite_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _complete_candidate(rec):
    return not rec.get('truncated') and not rec.get('skipped') and _finite_number(rec.get('sum_logp'))


def _mean(values):
    return sum(values) / len(values) if values else None


def _fmt(value):
    return '—' if value is None else '%.4f' % value


def mc_summary(rows, base_rows=None):
    lookup = e43._unique_questions([dict(r, question=r.get('question', r['pair_id'])) for r in rows])
    scored = [r for r in rows if r['scored']]
    abstain = [r['abstain_logp'] for r in rows if _finite_number(r.get('abstain_logp'))]
    standard = [r['standard_mc1'] for r in scored if r.get('standard_mc1') is not None]
    s = {'readout_protocol': 'e44_complete_candidates_max_true_v2',
         'mc1_definition': 'legacy_any_true_top1_alias_not_standard_truthfulqa_mc1',
         'n_scored': len(scored), 'n_unscored': len(rows) - len(scored),
         'n_abstain_scored': len(abstain), 'n_abstain_unscored': len(rows) - len(abstain),
         'abstain_logp_mean': _mean(abstain),
         'any_true_top1': _mean([r.get('any_true_top1', r['mc1']) for r in scored]),
         'mc1': _mean([r['mc1'] for r in scored]), 'mc2': _mean([r['mc2'] for r in scored]),
         'max_true_margin_mean': _mean([r['margin'] for r in scored]),
         'margin_mean': _mean([r['margin'] for r in scored]),
         'standard_mc1': _mean(standard), 'n_standard_mc1': len(standard),
         'truncated_candidates': sum(bool(c.get('truncated')) for r in rows for c in r.get('true', []) + r.get('false', [])),
         'skipped_candidates': sum(bool(c.get('skipped')) for r in rows for c in r.get('true', []) + r.get('false', []))}
    if base_rows is not None:
        every = e43._unique_questions([dict(r, question=r.get('question', r['pair_id'])) for r in base_rows])
        if list(lookup) != list(every):
            raise ValueError('MC comparison cohorts/order differ')
        abstain_pairs = [(every[r['pair_id']]['abstain_logp'], r['abstain_logp']) for r in rows
                         if _finite_number(every[r['pair_id']].get('abstain_logp')) and _finite_number(r.get('abstain_logp'))]
        both = [(every[r['pair_id']], r) for r in scored if every[r['pair_id']]['scored']]
        s.update({'n_abstain_paired': len(abstain_pairs),
                  'abstain_logp_change_mean': _mean([a - b for b, a in abstain_pairs]),
                  'n_paired': len(both), 'margin_change_mean': _mean([a['margin'] - b['margin'] for b, a in both]),
                  'margin_up': sum(a['margin'] > b['margin'] for b, a in both),
                  'to_correct': sum(not b['mc1'] and a['mc1'] for b, a in both),
                  'to_wrong': sum(b['mc1'] and not a['mc1'] for b, a in both),
                  'mc2_change_mean': _mean([a['mc2'] - b['mc2'] for b, a in both])})
    return s


def run_mc(args, meta, prepared, bundle, cfg, systems, ident, ident_sha):
    user_template = meta["config"]["qa_user"]
    questions = e43.pick_questions(prepared, args.split, args)
    eot = pd.turn_terminator_ids(bundle)
    arms = [(arm,) + parse_arm(arm, systems) for arm in args.arms]
    per_question = any(a[2] in PER_QUESTION for a in arms)
    unknown = [n for n in args.bases if n not in systems]
    if unknown:
        raise SystemExit("--bases: unknown prompt names %s" % unknown)
    bases = sorted({a[1] for a in arms} | set(args.bases))        # --bases alone = the prompt-only readout of step 0
    rendered = {n: [pd.render_prompt(bundle, systems[n], user_template, q["question"]) for q in questions] for n in bases}
    base_rows = {}
    for name in bases:                                   # no-write reference of every base prompt, once per split
        path = args.run_dir / "mc" / artifact_name("base_%s_%s.json" % (name, args.split), is_pilot(args))
        contract = {"prompt": name, "split": args.split, "pilot": is_pilot(args), "cohort_sha256": pd.fingerprint(questions),
                    "identity_sha256": ident_sha, "mc_batch_size": args.mc_batch_size}
        if path.exists() and not args.overwrite:
            d = validate_mc(read(path), questions, args.run_dir)
            if d.get("mc_contract") != contract:
                raise ValueError("mc baseline was made for a different prompt, subset or identity")
        else:
            rows = score_split(args, bundle, rendered[name], questions, eot)
            require_current(ident, args)
            d = seal({"protocol": fd.PROTOCOL, "phase": "mc", "kind": "baseline", "prompt": name, "split": args.split,
                      "identity": ident, "identity_sha256": ident_sha, "mc_contract": contract, "pilot": is_pilot(args),
                      "summary": mc_summary(rows), "rows": rows, "completed": not is_pilot(args), "execution_complete": True,
                      "question_ids": [q["pair_id"] for q in questions], "cohort_sha256": pd.fingerprint(questions)})
            save(path, d)
            log("mc base %s: %s" % (name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d["summary"].items()}))
        base_rows[name] = d["rows"]
    for layer in (args.layers if arms else []):
        record, dpath = load_direction(args, args.family, layer, ident_sha)
        positions = record["training_contract"]["positions"]
        direction_sha = file_sha(dpath)
        D = {}
        if per_question:
            spec = family_spec(args.family, record["e_on"], record.get("e_off"))
            for base, kind in sorted({(a[1], 'iso' if a[2] == 'iso' else 'effect') for a in arms if a[2] in PER_QUESTION}):
                for n in read_pair(spec, base, kind):
                    if n not in rendered:
                        rendered[n] = [pd.render_prompt(bundle, systems[n], user_template, q["question"]) for q in questions]
                D[(base, kind)] = question_differences(bundle, spec, rendered, layer, args.batch_size, base, kind)
        for arm, base, direction, beta in arms:
            path = args.run_dir / "mc" / artifact_name("%s_L%d_%s_%s.json" % (args.family, layer, arm_file_token(arm), args.split), is_pilot(args))
            contract = {"family": args.family, "layer": layer, "arm": arm, "split": args.split, "pilot": is_pilot(args),
                        "seed": args.seed, "e_on": record["e_on"], "e_off": record.get("e_off"), "positions": positions, "cohort_sha256": pd.fingerprint(questions), "direction_file_sha256": direction_sha,
                        "identity_sha256": ident_sha, "mc_batch_size": args.mc_batch_size,
                        "baseline_file_sha256": file_sha(args.run_dir / "mc" / artifact_name("base_%s_%s.json" % (base, args.split), is_pilot(args)))}
            if path.exists() and not args.overwrite:
                if validate_mc(read(path), questions, args.run_dir).get("mc_contract") != contract:
                    raise ValueError("mc arm was made with a different seed, arm, subset or direction")
                log("skip %s (verified)" % path.name)
                continue
            write = arm_write(record, direction, beta, bundle.hidden_size, args.seed, layer, D.get((base, "iso" if direction == "iso" else "effect")))
            rows = score_split(args, bundle, rendered[base], questions, eot, layer=layer, write=write, positions=positions)
            summary = mc_summary(rows, base_rows[base])
            require_current(ident, args)
            if (file_sha(dpath) != direction_sha or file_sha(args.run_dir / "mc" / artifact_name("base_%s_%s.json" % (base, args.split), is_pilot(args))) != contract["baseline_file_sha256"]):
                raise RuntimeError("direction or reference baseline changed during scoring")
            save(path, seal({"protocol": fd.PROTOCOL, "phase": "mc", "kind": "arm", "family": args.family, "e_on": record["e_on"], "e_off": record.get("e_off"),
                             "layer": layer, "arm": arm, "split": args.split, "base_prompt": base, "direction": direction,
                             "beta": beta, "write": write_note(write), "positions": positions, "identity": ident,
                             "identity_sha256": ident_sha, "direction_file": dpath.name,
                             "baseline_file": artifact_name("base_%s_%s.json" % (base, args.split), is_pilot(args)),
                             "baseline_file_sha256": contract["baseline_file_sha256"], "direction_file_sha256": direction_sha,
                             "mc_contract": contract, "pilot": is_pilot(args), "summary": summary, "rows": rows,
                             "question_ids": [q["pair_id"] for q in questions], "cohort_sha256": pd.fingerprint(questions),
                             "completed": not is_pilot(args), "execution_complete": True}))
            log("L%d %s reference readout: any-true top1 %s MC2 %s | max-true margin change %s | flips %d/%d (n %d) | log p(%s) change %s" % (
                layer, arm, _fmt(summary["any_true_top1"]), _fmt(summary["mc2"]), _fmt(summary["margin_change_mean"]), summary["to_correct"],
                summary["to_wrong"], summary["n_paired"], ABSTAIN_TEXT, _fmt(summary["abstain_logp_change_mean"])))


# ----------------------------------------------------------------------------- summarize
def run_summarize(args):
    lines = ["# E44 judge-free summary (%s)" % ("pilot included" if args.include_pilot else "full runs only"), ""]
    keep = lambda d: args.include_pilot or not d.get("pilot")  # noqa: E731
    logs = [d for d in (read(p) for p in sorted((args.run_dir / "train_logs").glob("*.json"))) if keep(d)]
    gens = [d for d in (read(p) for p in sorted((args.run_dir / "gen").glob("*.json"))) if keep(d)]
    mcs = [d for d in (read(p) for p in sorted((args.run_dir / "mc").glob("*.json"))) if keep(d)]
    source = read(args.source_run / 'questions.json')
    for d in logs + gens + mcs:
        e43.validate_artifact(d, "summary input")
        require_current(d['identity'], args)
        if not d.get('pilot') and d.get('phase') in ('generate', 'mc') and not d.get('completed'):
            raise ValueError('incomplete full artifact cannot enter a summary')
    for d in logs:
        path = _source_path(args.run_dir, 'dirs', d.get('direction_file'))
        if file_sha(path) != d.get('direction_file_sha256'):
            raise ValueError('training log direction source changed')
        record = torch.load(path, map_location='cpu', weights_only=True)
        validate_direction(record, d['family'], d['layer'], d['identity_sha256'], bool(d['pilot']))
        validate_direction_sources(record, args.run_dir)
        for key in ('family', 'layer', 'e_on', 'e_off', 'training_contract', 'losses', 'training'):
            if d.get(key) != record.get(key):
                raise ValueError('training log differs from its direction: ' + key)
    for d in gens:
        frozen_path_ = _source_path(args.run_dir, '', d.get('frozen_source'))
        frozen = read(frozen_path_)
        cohort = {q['pair_id']: q for q in source['splits'][d['split']]}
        questions = [cohort[pid] for pid in d['question_ids']]
        parent_questions = [cohort[pid] for pid in frozen['question_ids']]
        validate_frozen(frozen, parent_questions, frozen['prompt_systems'])
        validate_generation(d, frozen, questions, args.run_dir)
    for d in mcs:
        cohort = {q['pair_id']: q for q in source['splits'][d['split']]}
        validate_mc(d, [cohort[pid] for pid in d['question_ids']], args.run_dir)
    if len({d["identity_sha256"] for d in logs + gens + mcs}) > 1:
        raise ValueError("cannot summarize artifacts from different identities")
    assignments = sorted({(d['family'], d['layer'], d.get('e_off') or 'none', d['e_on'])
                          for d in logs + gens + mcs if 'family' in d})
    lines += ['## Explicit factor assignments (different pairs are different hypotheses)', '',
              '| family | layer | low truth level | high truth level |', '|---|---|---|---|']
    for family, layer, low, high in assignments:
        lines.append('| %s | %d | %s | %s |' % (family, layer, low, high))
    lines.append('')
    f = lambda v: "—" if v is None else "%.4f" % v  # noqa: E731
    lines += ["## training (loss per answer token; held-out questions never received a gradient)", "",
              "| family | layer | gap | md norm | cos(w, md) | cause: no write / whole diff / md / trained | isolate: md / trained | held-out cause: no write / md / trained | held-out isolate: md / trained |",
              "|---|---|---|---|---|---|---|---|---|"]
    for d in sorted(logs, key=lambda d: (d["family"], d["layer"])):
        tr, va = d["losses"]["train"], d["losses"].get("val")
        lines.append("| %s | %d | %.3f | %.3f | %s | %s / %s / %s / %s | %s / %s | %s | %s |" % (
            d["family"], d["layer"], d["gap"], d["md_norm"], _fmt(d["cosine_w_md"]), f(tr["no_write"]["cause"]),
            f(tr["whole_difference"]["cause"]), f(tr["mean_difference_direction"]["cause"]), f(tr["trained"]["cause"]),
            f(tr["mean_difference_direction"]["isolate"]), f(tr["trained"]["isolate"]),
            ("%s / %s / %s" % (f(va["no_write"]["cause"]), f(va["mean_difference_direction"]["cause"]), f(va["trained"]["cause"]))) if va else "—",
            ("%s / %s" % (f(va["mean_difference_direction"]["isolate"]), f(va["trained"]["isolate"]))) if va else "—"))
    lines += ["", "## generation", "",
              "| family | layer | arm | split | n | changed | refusals | sentence-end | words | = target prompt (base =) | write/‖h‖ |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in sorted(gens, key=lambda d: (d["family"], d["arm"], d["split"], d["layer"])):
        s = d["summary"]
        lines.append("| %s | %d | %s | %s | %d | %d | %d → %d | %d → %d | %.1f → %.1f | %s | %.3f |" % (
            d["family"], d["layer"], d["arm"], d["split"], s["n"], s["changed_normalized"], s["before"]["refusal_rule"],
            s["after"]["refusal_rule"], s["before"]["ends_with_period"], s["after"]["ends_with_period"],
            s["before"]["mean_words"], s["after"]["mean_words"],
            ("%d (%d)" % (s["matches_target_prompt"], s["baseline_matches_target_prompt"])) if "matches_target_prompt" in s else "—",
            s["prefill_rel_mean"]))
    lines += ["", "## reference ranking (max-true margin / any-true top1 are not standard MC1) and log p(\"%s\")" % ABSTAIN_TEXT, "",
              "| family | layer | arm | split | scored | any-true top1 | MC2 | max-true margin change | to correct / to wrong | log p(abstain) change |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for d in sorted(mcs, key=lambda d: (d.get("family", ""), d.get("arm", d.get("prompt", "")), d["split"], d.get("layer", -1))):
        s = d["summary"]
        if d["kind"] == "baseline":
            lines.append("| — | — | no write, %s | %s | %d | %s | %s | — | — | (level %s) |" % (
                d["prompt"], d["split"], s["n_scored"], _fmt(s["any_true_top1"]), _fmt(s["mc2"]), _fmt(s["abstain_logp_mean"])))
        else:
            lines.append("| %s | %d | %s | %s | %d | %s | %s | %s | %d / %d | %s |" % (
                d["family"], d["layer"], d["arm"], d["split"], s["n_scored"], _fmt(s["any_true_top1"]), _fmt(s["mc2"]), _fmt(s["margin_change_mean"]),
                s["to_correct"], s["to_wrong"], _fmt(s["abstain_logp_change_mean"])))
    (args.run_dir / artifact_name('summary.md', args.include_pilot)).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=("freeze", "train", "generate", "mc", "summarize"))
    p.add_argument("--source-run", type=Path, required=True, help="E25 v1u run dir (bank.json, questions.json)")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--prompts", type=Path, default=Path(EXP_ROOT) / "configs" / "prompts" / "default_v1.json")
    p.add_argument("--prompts-e44", type=Path, default=Path(EXP_ROOT) / "configs" / "prompts" / "e44_factorial_v1.json")
    p.add_argument("--family", choices=FAMILIES)
    p.add_argument("--e-on", default="E1", help="high truth-level sentence (E1/E2)")
    p.add_argument("--e-off", default="none", choices=("none", "M"), help="explicit low truth level; M is a separate hypothesis, never an automatic fallback")
    p.add_argument("--layers", nargs="+", type=int, default=[])
    p.add_argument("--arms", nargs="+", default=[])
    p.add_argument("--bases", nargs="+", default=[], help="mc: also (or only) score these prompts without any write")
    p.add_argument("--split", default="tune", choices=("train", "tune"))
    p.add_argument("--splits", nargs="+", default=["train", "tune"], choices=("train", "tune"))
    p.add_argument("--loss", default="jsd", choices=fd.LOSSES)
    p.add_argument("--positions", default="content", choices=fd.POSITIONS)
    p.add_argument("--init", default="random", choices=("random", "md"))
    p.add_argument("--val-n", type=int, default=16, help="training questions held out from the gradient (loss reported only)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument('--split-seed', type=int, default=42, help='fixed holdout partition seed, independent of optimization seed')
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--mc-batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--max-target-tokens", type=int, default=24)
    p.add_argument("--max-ref-tokens", type=int, default=48, help="mc: cap on a reference answer's tokens")
    p.add_argument("--output-projection", default="float32", choices=("float32", "native"))
    p.add_argument("--n-check", type=int, default=0, help="pilot: the first K questions (writes t.* files)")
    p.add_argument("--pair-ids", nargs="+", help="pilot: explicit question IDs (writes t.* files)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--include-pilot", action="store_true", help="summarize: include pilot files")
    args = p.parse_args()
    if min(args.batch_size, args.mc_batch_size, args.epochs, args.max_new_tokens, args.max_target_tokens, args.max_ref_tokens) < 1 or args.n_check < 0 or args.val_n < 0:
        raise SystemExit("counts/caps must be positive; --n-check and --val-n nonnegative")
    if not (math.isfinite(args.lr) and args.lr > 0 and math.isfinite(args.clip) and args.clip > 0):
        raise SystemExit("learning rate and gradient clip must be finite and positive")
    if args.pair_ids and args.n_check:
        raise SystemExit("choose --pair-ids or --n-check, not both")
    if len(args.layers) != len(set(args.layers)) or len(args.arms) != len(set(args.arms)):
        raise SystemExit("duplicate layers or arms")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "summarize":
        run_summarize(args)
        return
    if args.phase in ("train", "generate") and not (args.family and args.layers):
        raise SystemExit("train / generate need --family and --layers")
    if args.phase == "generate" and not args.arms:
        raise SystemExit("generate needs --arms")
    if args.phase == "mc" and not ((args.family and args.layers and args.arms) or (args.bases and not args.arms)):
        raise SystemExit("mc needs --family, --layers and --arms, or --bases alone (prompt-only readout)")
    meta, prepared = load_source(args)
    if any(l < 0 or l >= meta["num_layers"] for l in args.layers):
        raise SystemExit("layer index outside the source model")
    cfg, systems = load_prompts(meta, args.prompts, args.prompts_e44)
    if args.e_on not in ("E1", "E2") or args.e_on not in cfg["sentences"]:
        raise SystemExit("--e-on must be E1/E2")
    e_off = normalize_e_off(args.e_off)
    if e_off is not None and e_off not in cfg["sentences"]:
        raise SystemExit("unknown explicit --e-off sentence")
    for arm in args.arms:
        parse_arm(arm, systems)
    torch.manual_seed(args.seed)
    bundle = e43.open_model(meta)
    eot = pd.turn_terminator_ids(bundle)
    ident, ident_sha = run_identity(meta, systems, cfg, args, eot, bundle)
    log("identity %s (output projection %s, pilot=%s)" % (ident_sha[:12], args.output_projection, is_pilot(args)))
    with fp32_output_projection(bundle, enabled=args.output_projection == "float32"):
        {"freeze": run_freeze, "train": run_train, "generate": run_generate, "mc": run_mc}[args.phase](
            args, meta, prepared, bundle, cfg, systems, ident, ident_sha)


if __name__ == "__main__":
    main()
