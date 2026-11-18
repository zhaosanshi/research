#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prompt ladder helper (Iteration_01, Claude Opus 5.5, 2026-11-18; DRAFT, not reviewed by GPT, never run on a GPU).

Goal: the E write of a finished E44 run, exactly as in its condition "must-answer prompt conf + E at strength s"
(E_L<L>_conf-w-p<s>), with only the system prompt replaced by new prompts P1, P2, P3 (a ladder of firmer prompts for the
questions the model leaves blank after the E write; conf is the last rung).

Why code and not a config file: scripts/e44_eval_extra.py already generates with a run's own trained direction on
evaluation-only prompts while verifying the parent run (identity, frozen answers, direction and source files), but it
builds those prompts only as <one of the three base systems> + <sentences of the run's e44_factorial_v1.json>, and that
file is part of the run identity (identity.prompts_config), so no new sentence can be added to it without breaking the
identity check; the new texts are not such a combination. It also writes only into <run>/eval_extra/, which already holds
conf--E<L>-w-p<s> files of the E+C selection made with another prompts file (the engine would refuse them as foreign).
So this helper runs e44_eval_extra.py UNCHANGED (its file, the driver and core stay byte-identical; the parent identity is
the run's own) with three in-memory changes:
  1. extra_systems: the evaluation-only prompts are read verbatim from a flat JSON {name: system text} (--prompts-extra),
     with the original's checks (name pattern, not a prompt of the run, no system text twice);
  2. Context: its output directory is --out instead of <run>/eval_extra;
  3. run_export: the same output directory (one exact source replacement that must match once, else nothing runs).
For --split test it first applies scripts/review_checks/e44_split_test.py's own patch() (split choices, frozen split,
driver export), as every test-split run of these scripts did. Everything else -- direction, layer, amount = strength x
gap, content positions including decode steps, chat template, greedy decoding, max new tokens, batches, all checks and
the file layout -- is the engine's own code. The engine never overwrites: an existing output is verified and skipped, or
refused when it differs.

usage (in the container, -w <code copy>, PYTHONPATH=<code copy>, PYTHONDONTWRITEBYTECODE=1, python -B):
  ladder_extra.py check --code C --out O --split tune|test        apply the changes, print hashes, run nothing
  ladder_extra.py extra --code C --out O generate|export <e44_eval_extra.py arguments, with --prompts-extra LADDER>
                        (only the exact options the launcher uses are accepted; no abbreviations)
  ladder_extra.py dryrun --code C --source-run S --run-dir R --prompts-extra LADDER --layer L [--strengths 2]
                        CPU, tokenizer only, writes nothing: identity fields, rendering, written span, write amount
  ladder_extra.py reproduce MINE DRIVER                          a control file must equal the driver's file
  ladder_extra.py bind --code C --run-dir R --source-run S --grid G --layer L --strength 2 --split tune|test
                        [--subset F] [--write F]                 the candidate set of THIS run and split, rebuilt from the
                        run's own E+C grid export (read-only); --subset: F must hold exactly that set; --write: write it
  ladder_extra.py subset EXPORT SUBSET WORK --split tune|test    the export restricted to SUBSET's pair_ids
  ladder_extra.py verdicts --work W --judge pro|allenai --subset F   every expected cell judged exactly once
  ladder_extra.py show [--n 8] GEN ...                           answers of generation files, side by side
(2026-11-18, GPT review B1/S2/S3: 'candidates', which read one judged file and accepted a question twice or missing, is
replaced by 'bind'; 'verdicts' checks the full expected cell set; options are an exact whitelist per phase.)
"""
import argparse
import collections
import hashlib
import importlib.util
import inspect
import json
import math
import re
import sys
from pathlib import Path

NAME_RE = re.compile(r"[A-Za-z0-9_]+")
ARM_RE = re.compile(r"^(?P<base>[A-Za-z0-9_]+):E(?P<layer>\d+)\.w\.p(?P<beta>\d+(?:\.\d+)?)$")
STRENGTH_RE = re.compile(r"^\d+(?:\.\d+)?$")
CONTROL = "conf"                                     # the last rung: the run's own must-answer prompt
EXPORT_OLD = '    out_dir = run_dir / "eval_extra"\n'
EXPORT_NEW = '    out_dir = _LADDER_OUT(run_dir)\n'
SAME_FIELDS = ("token_ids", "finished", "stop_id", "text")     # as the genconf check of ff_explore_ec_select.sh
COMMON_OPTIONS = ("--source-run", "--run-dir", "--e-on", "--batch-size", "--max-new-tokens", "--max-target-tokens",
                  "--seed", "--prompts-extra", "--split")
ALLOWED = {"generate": COMMON_OPTIONS + ("--arms", "--first-batches"),     # exact names: the engine's parser would also
           "export": COMMON_OPTIONS + ("--gen", "--name")}                  # take a unique abbreviation (--overw)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def say(msg):
    print("[ladder] " + msg, flush=True)


def _no_duplicate_keys(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise SystemExit("duplicate JSON key %r" % k)
        out[k] = v
    return out


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=_no_duplicate_keys)


def write_new(path, text):
    """Write a file that must not exist yet, or must already hold exactly this text (never overwritten)."""
    path = Path(path)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise SystemExit("%s exists and differs: not overwritten" % path)
        return "kept (identical)"
    path.write_text(text, encoding="utf-8")
    return "written"


# ----------------------------------------------------------------------------- the ladder file
def ladder_prompts(path):
    """A ladder file is a flat JSON object {prompt name: system text}, in ladder order."""
    raw = read_json(path)
    if not isinstance(raw, dict) or not raw:
        raise SystemExit("%s: a ladder file is a nonempty JSON object {name: system text}" % path)
    for name, text in raw.items():
        if not NAME_RE.fullmatch(name):
            raise SystemExit("%s: prompt name %r: letters, digits and _ only" % (path, name))
        if not isinstance(text, str) or not text.strip() or text != text.strip() or any(ord(c) < 32 for c in text):
            raise SystemExit("%s: prompt %s: the text must be one nonempty line without outer spaces" % (path, name))
    if len(set(raw.values())) != len(raw):
        raise SystemExit("%s: two ladder prompts have the same text" % path)
    return raw


def ladder_systems(meta, cfg, systems, args):
    """In-memory replacement of e44_eval_extra.extra_systems (same signature): name -> system text of the evaluation-only
    prompts, verbatim from the ladder file given as --prompts-extra; the original's checks are kept."""
    extra = ladder_prompts(args.prompts_extra)
    clash = sorted(set(extra) & set(systems))
    if clash:
        raise SystemExit("ladder prompt names %s are prompts of the run already" % clash)
    every = dict(systems, **extra)
    if len(set(every.values())) != len(every):
        raise SystemExit("a ladder prompt renders the same system text as a prompt of the run")
    return dict(extra)


# ----------------------------------------------------------------------------- the engine and its in-memory changes
def import_engine(code):
    """(code root, driver module E, engine module Y) imported from this code copy only."""
    code = Path(code).resolve()
    scripts = code / "scripts"
    if not (scripts / "e44_eval_extra.py").is_file():
        raise SystemExit("%s is not a code copy (no scripts/e44_eval_extra.py)" % code)
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import common                                    # noqa: E402  (puts the code root on sys.path, as for the scripts)
    import e44_eval_extra as Y                       # noqa: E402
    E = Y.E
    if Path(common.EXP_ROOT).resolve() != code:
        raise SystemExit("common.EXP_ROOT %s is not the code copy %s" % (common.EXP_ROOT, code))
    for m in (Y, E, common):
        if Path(m.__file__).resolve().parent != scripts:
            raise SystemExit("%s was imported from %s, not from %s" % (m.__name__, m.__file__, scripts))
    return code, E, Y


def split_test_module(code):
    path = code / "scripts" / "review_checks" / "e44_split_test.py"
    spec = importlib.util.spec_from_file_location("e44_split_test", path)
    S = importlib.util.module_from_spec(spec)
    sys.modules["e44_split_test"] = S
    spec.loader.exec_module(S)
    return S


def apply_changes(code, E, Y, out, split):
    """The split widening (test only, e44_split_test.py's own patch) and the three ladder changes. Returns that module."""
    S = split_test_module(code)
    if split == "test":
        S.patch()
    elif split != "tune":
        raise SystemExit("--split must be tune or test")
    out = Path(out).resolve()

    def ladder_out(run_dir):
        run = Path(run_dir).resolve()
        if out == run or run in out.parents or out in run.parents:
            raise SystemExit("--out %s and the parent run %s must be separate directories" % (out, run))
        if not out.is_dir():
            raise SystemExit("--out %s does not exist (the launcher creates it)" % out)
        return out

    Y._LADDER_OUT = ladder_out
    Y.extra_systems = ladder_systems
    parent = Y.Context

    class LadderContext(parent):
        """e44_eval_extra.Context, writing into --out (the parent's __init__ does not use out_dir)."""

        def __init__(self, args):
            super().__init__(args)
            self.out_dir = ladder_out(args.run_dir)

    Y.Context = LadderContext
    old = Y.run_export
    lines, first = inspect.getsourcelines(old)
    src = "".join(lines)
    if src.count(EXPORT_OLD) != 1:
        raise SystemExit("e44_eval_extra.run_export: expected exactly one %r, found %d" % (EXPORT_OLD, src.count(EXPORT_OLD)))
    new = src.replace(EXPORT_OLD, EXPORT_NEW)
    exec(compile("\n" * (first - 1) + new, Y.__file__, "exec"), Y.__dict__)     # blank lines keep the file's line numbers
    if Y.run_export is old or "_LADDER_OUT" not in Y.run_export.__code__.co_names:
        raise SystemExit("e44_eval_extra.run_export was not recompiled with the ladder output directory")
    say("e44_eval_extra.extra_systems -> ladder file; Context -> output dir %s; run_export: output dir replaced "
        "(source sha256 %s -> %s)" % (out, hashlib.sha256(src.encode()).hexdigest()[:12], hashlib.sha256(new.encode()).hexdigest()[:12]))
    for name, path in (("helper", Path(__file__).resolve()), ("e44_eval_extra.py", Y.__file__), ("e44_factorial_das.py", E.__file__),
                       ("e44_split_test.py", S.__file__)):
        say("%s sha256 %s" % (name, sha(path)))
    return S


def option_values(rest):
    """{option: [values]} of an engine command line; options in --opt=value form are refused (unambiguous checks)."""
    vals, i = {}, 0
    while i < len(rest):
        tok = rest[i]
        if not tok.startswith("--"):
            raise SystemExit("unexpected argument %r" % tok)
        if "=" in tok:
            raise SystemExit("give %s as '%s value', not with '='" % (tok.split("=")[0], tok.split("=")[0]))
        if tok in vals:
            raise SystemExit("%s given twice" % tok)
        j = i + 1
        while j < len(rest) and not rest[j].startswith("--"):
            j += 1
        vals[tok] = rest[i + 1:j]
        i = j
    return vals


def one(vals, opt):
    v = vals.get(opt)
    if not v or len(v) != 1:
        raise SystemExit("give exactly one value for %s" % opt)
    return v[0]


def run_extra(argv):
    """extra --code C --out O generate|export <engine arguments>"""
    if len(argv) < 5 or argv[0] != "--code" or argv[2] != "--out":
        raise SystemExit("usage: extra --code CODE --out OUT generate|export <e44_eval_extra.py arguments>")
    code_arg, out_arg, phase, rest = argv[1], argv[3], argv[4], argv[5:]
    if phase not in ("generate", "export"):
        raise SystemExit("extra: phase must be generate or export (got %s)" % phase)
    vals = option_values(rest)
    bad = sorted(o for o in vals if o not in ALLOWED[phase])
    if bad:
        raise SystemExit("extra %s: %s not accepted (exact option names only: %s)" % (phase, bad, " ".join(ALLOWED[phase])))
    for opt in ("--source-run", "--run-dir", "--e-on"):
        one(vals, opt)
    split = one(vals, "--split")
    ladder_file = one(vals, "--prompts-extra")
    ladder = ladder_prompts(ladder_file)
    names = list(ladder)
    if phase == "generate":
        arms = vals.get("--arms") or []
        if not arms:
            raise SystemExit("generate needs --arms")
        layers = set()
        for arm in arms:
            m = ARM_RE.match(arm)
            if not m or m["base"] not in names + [CONTROL]:
                raise SystemExit("arm %r: only <ladder prompt or %s>:E<layer>.w.p<strength> (one E write, learned "
                                 "direction, toward the truth sentence)" % (arm, CONTROL))
            layers.add(m["layer"])
        if len(layers) != 1:
            raise SystemExit("all arms of one call write at one layer (got %s)" % sorted(layers))
    else:
        if not vals.get("--gen"):
            raise SystemExit("export needs the generation files named (--gen ...): it never collects files by itself")
        one(vals, "--name")
    code, E, Y = import_engine(code_arg)
    S = apply_changes(code, E, Y, out_arg, split)
    if split == "test":
        S.require_test("extra", phase, S.split_values(rest))
    say("prompts file %s sha256 %s: %s" % (ladder_file, sha(ladder_file), json.dumps(ladder, ensure_ascii=False)))
    sys.argv = [Y.__file__, phase] + rest
    Y.main()
    if split == "test" and not S._hits:
        raise SystemExit("the split option was never widened (this should not happen)")


def run_check(a):
    code, E, Y = import_engine(a.code)
    apply_changes(code, E, Y, a.out, a.split)
    widened = ("train", "tune", "test") in E.validate_frozen.__code__.co_consts
    ok = {"extra_systems is the ladder reader": Y.extra_systems is ladder_systems,
          "Context writes into --out": Y.Context.__name__ == "LadderContext",
          "run_export writes into --out": "_LADDER_OUT" in Y.run_export.__code__.co_names,
          "frozen test split accepted only for test": widened == (a.split == "test")}
    for k, v in ok.items():
        say("check %-42s %s" % (k, "ok" if v else "FAILED"))
    if not all(ok.values()):
        raise SystemExit(1)
    say("check passed for --split %s (nothing was run)" % a.split)


# ----------------------------------------------------------------------------- dry run (CPU, tokenizer only, no writes)
def run_dryrun(a):
    code, E, Y = import_engine(a.code)
    from core import model_io
    from core import qa_prompt_das as pd
    for s in a.strengths:
        if not STRENGTH_RE.match(s) or not float(s) > 0:
            raise SystemExit("strength %r is not a positive number like 2 or 0.5" % s)
    ns = argparse.Namespace(source_run=Path(a.source_run), run_dir=Path(a.run_dir), phase="generate", split="tune",
                            splits=["tune"], n_check=0, pair_ids=None, prompts_extra=Path(a.prompts_extra))
    prompts = Path(a.prompts or code / "configs" / "prompts" / "default_v1.json")
    prompts_e44 = Path(a.prompts_e44 or code / "configs" / "prompts" / "e44_factorial_v1.json")
    meta, prepared = E.load_source(ns)
    cfg, systems = E.load_prompts(meta, prompts, prompts_e44)
    extra = ladder_systems(meta, cfg, systems, ns)
    if CONTROL not in systems:
        raise SystemExit("the run has no prompt %s" % CONTROL)
    every = dict(systems, **extra)
    names = [CONTROL] + list(extra)
    say("ladder %s (sha256 %s): %s, then %s" % (a.prompts_extra, sha(a.prompts_extra)[:12], " -> ".join(extra), CONTROL))
    for n in names:
        say("  %-5s %s" % (n, every[n]))
    say("  (plain prompt x of the run: %s)" % systems["x"])
    # 1. identity fields of the run that need no model: prompts, prompt config, source files, user turn, questions
    src_sha = {n: sha(ns.source_run / n) for n in ("bank.json", "questions.json")}
    frozen = {}
    for split in ("tune", "test"):
        fz = E.read(ns.run_dir / ("frozen_%s.json" % split))
        ident = fz["identity"]
        checks = {"prompt systems": fz["prompt_systems"] == systems == ident["prompt_systems"],
                  "prompts config": ident["prompts_config"] == cfg,
                  "source files": ident["source_files_sha256"] == src_sha,
                  "user template": ident["user_template"] == meta["config"]["qa_user"] == fz["user_template"],
                  "question ids": fz["question_ids"] == [q["pair_id"] for q in prepared["splits"][split]],
                  "complete": fz.get("completed") is True and fz.get("split") == split and not fz.get("pilot")}
        bad = [k for k, v in checks.items() if not v]
        if bad:
            raise SystemExit("frozen_%s.json differs from this code copy / source in: %s" % (split, bad))
        frozen[split] = fz
        say("frozen_%s.json: identity %s, %d questions; prompts, prompt config, source files, user turn and question "
            "order equal this code copy and source" % (split, fz["identity_sha256"][:12], len(fz["question_ids"])))
    if frozen["tune"]["identity_sha256"] != frozen["test"]["identity_sha256"]:
        raise SystemExit("tune and test freezes have different identities")
    # 2. the tokenizer alone, prepared as core.model_io.load_model prepares it (no model weights are loaded)
    from transformers import AutoTokenizer
    mc = meta["model_config"]
    tok = AutoTokenizer.from_pretrained(mc["path"], trust_remote_code=bool(mc.get("trust_remote_code", True)))
    if mc.get("extra_special_tokens"):
        model_io.register_extra_special_tokens(tok, mc)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    backend = getattr(tok, "backend_tokenizer", None)
    got = {"class": type(tok).__name__, "chat_template": getattr(tok, "chat_template", None), "pad_token_id": tok.pad_token_id,
           "eos_token_id": tok.eos_token_id, "padding_side": tok.padding_side, "all_special_ids": list(tok.all_special_ids),
           "backend_sha256": pd.fingerprint(backend.to_str()) if backend is not None else None}
    want = frozen["test"]["identity"]["tokenizer"]
    diff = [k for k in got if got[k] != want.get(k)]
    if diff:
        raise SystemExit("this tokenizer differs from the run's identity in %s" % diff)
    say("tokenizer %s (%s): equal to the run's identity in %s" % (got["class"], mc["path"], ", ".join(got)))

    class TokenizerOnly:                             # all pd.render_prompt needs of a bundle
        build_chat = model_io.ModelBundle.build_chat

    b = TokenizerOnly()
    b.tokenizer, b.cfg = tok, mc
    user = meta["config"]["qa_user"]
    for split in ("tune", "test"):
        qs = prepared["splits"][split]
        look = {r["pair_id"]: r for r in frozen[split]["questions"]}
        bad, written = [], []
        for q in qs:
            r = {n: pd.render_prompt(b, every[n], user, q["question"]) for n in names}
            fr = look[q["pair_id"]]
            c = r[CONTROL]
            if c["content_start"] != fr["content_start"][CONTROL] or c["boundary_clean"] != fr["boundary_clean"][CONTROL]:
                bad.append((q["pair_id"], CONTROL, "content start differs from the frozen record"))
            span = c["prefix_ids"][c["content_start"]:]
            written.append(len(span))
            for n in names[1:]:
                if not r[n]["boundary_clean"]:
                    bad.append((q["pair_id"], n, "the user content does not start on a token boundary"))
                if r[n]["prefix_ids"][r[n]["content_start"]:] != span:
                    bad.append((q["pair_id"], n, "the written prompt span differs from conf's"))
        if bad:
            raise SystemExit("%s: %d problems, e.g. %s" % (split, len(bad), bad[:5]))
        say("%s: %d questions x %d prompts rendered; conf's content start = its frozen record on every question; under "
            "every ladder prompt the written prompt span (user turn to the end, %d-%d tokens) is token-identical to "
            "conf's, so only the unwritten system part differs" % (split, len(qs), len(names), min(written), max(written)))
        q = qs[0]
        r = {n: pd.render_prompt(b, every[n], user, q["question"]) for n in names}
        say("%s, first question %s: %r" % (split, q["pair_id"], q["question"]))
        for n in names:
            say("  %-5s %3d prompt tokens, written from token %d (%d written prompt positions), read token %r" % (
                n, r[n]["prompt_length"], r[n]["content_start"], r[n]["prompt_length"] - r[n]["content_start"], r[n]["read_token"]))
        first = names[1]
        say("  rendered prompt under %s (text between the markers):\n>>>>\n%s<<<<" % (first, r[first]["prefix_text"]))
        say("  its token ids: %s" % r[first]["prefix_ids"])
    # 3. the write: the run's direction file, amount = strength x gap, against the driver's own files of the condition
    import torch
    dpath = ns.run_dir / "dirs" / ("E_L%d.pt" % a.layer)
    rec = torch.load(dpath, map_location="cpu", weights_only=False)
    gap, positions = float(rec["gap"]), rec["training_contract"]["positions"]
    say("direction %s (sha256 %s): family %s, layer %s, e_on %s, positions %s, gap %.6f" % (
        dpath, sha(dpath)[:12], rec["family"], rec["layer"], rec["e_on"], positions, gap))
    if rec["family"] != "E" or rec["layer"] != a.layer:
        raise SystemExit("direction file family/layer differs")
    for s in a.strengths:
        amount = float(s) * gap
        for split in ("tune", "test"):
            dfile = ns.run_dir / "gen" / ("E_L%d_conf-w-p%s_%s.json" % (a.layer, s.replace(".", "-"), split))
            d = E.read(dfile)
            ok = (math.isclose(d["summary"]["amount_mean"], amount, rel_tol=1e-9) and d["beta"] == float(s)
                  and d["positions"] == positions and d["identity_sha256"] == frozen[split]["identity_sha256"]
                  and d["direction_file_sha256"] == sha(dpath) and d.get("completed") is True and d["summary"]["n"] == len(prepared["splits"][split]))
            say("  strength %s: amount %+.6f; driver file %s: amount %+.6f, %d answers, same identity and direction: %s" % (
                s, amount, dfile.name, d["summary"]["amount_mean"], d["summary"]["n"], "yes" if ok else "NO"))
            if not ok:
                raise SystemExit("the driver file of the condition does not match this direction / strength")
    # 4. what STAGE=gen will make
    tokn = lambda s: s.replace(".", "-")  # noqa: E731
    for split in ("tune", "test"):
        for s in a.strengths:
            ctrl = "%s:E%d.w.p%s" % (CONTROL, a.layer, s)
            say("%s s=%s: control %s -> gen/%s_%s.json (= driver gen/E_L%d_conf-w-p%s_%s.json token for token)" % (
                split, s, ctrl, Y.arm_token(ctrl), split, a.layer, tokn(s), split))
            for n in extra:
                arm = "%s:E%d.w.p%s" % (n, a.layer, s)
                say("%s s=%s: %s -> gen/%s_%s.json and base_%s_%s.json; conditions baseline_%s, E_L%d_%s-w-p%s" % (
                    split, s, arm, Y.arm_token(arm), split, n, split, n, a.layer, n, tokn(s)))
    say("dry run passed (nothing written)")


# ----------------------------------------------------------------------------- reproduction check of the control arm
def run_reproduce(a):
    mine, drv = read_json(a.mine), read_json(a.driver)
    parts = mine.get("parts") or []
    if mine.get("phase") != "generate" or drv.get("phase") != "generate" or len(parts) != 1:
        raise SystemExit("reproduce compares one single-write engine file with one driver file")
    p = parts[0]
    rm, rd = mine["rows"], drv["rows"]
    partial = mine.get("partial") is True
    amounts = {r["write"]["amount"] for r in rd}
    checks = {"base prompt": mine.get("base_prompt") == drv.get("base_prompt") == CONTROL,
              "write": p.get("family") == drv.get("family") == "E" and p.get("kind") == drv.get("direction") == "w",
              "split": mine.get("split") == drv.get("split"),
              "identity": mine.get("identity_sha256") == drv.get("identity_sha256"),
              "frozen file": mine.get("frozen_file_sha256") == drv.get("frozen_file_sha256"),
              "direction file": p.get("direction_file_sha256") == drv.get("direction_file_sha256"),
              "layer / strength": p.get("layer") == drv.get("layer") and p.get("beta") == drv.get("beta"),
              "positions": mine.get("positions") == drv.get("positions"),
              "max new tokens": mine.get("max_new_tokens") == drv.get("max_new_tokens"),
              "driver complete": drv.get("completed") is True and not drv.get("pilot"),
              "mine complete" if not partial else "mine partial": (mine.get("completed") is True) != partial,
              "row count": (0 < len(rm) <= len(rd)) if partial else len(rm) == len(rd),
              "question order": [r["pair_id"] for r in rm] == [r["pair_id"] for r in rd[:len(rm)]],
              "amount": len(amounts) == 1 and math.isclose(p.get("amount", float("nan")), next(iter(amounts)), rel_tol=1e-9)
                        and all(math.isclose(r["write"]["amount"], p["amount"], rel_tol=1e-9) for r in rm)}
    bad = [k for k, v in checks.items() if not v]
    if bad:
        raise SystemExit("%s vs %s: %s" % (Path(a.mine).name, Path(a.driver).name, bad))
    differ = [(r["pair_id"], [f for f in SAME_FIELDS if r[f] != d[f]]) for r, d in zip(rm, rd) if any(r[f] != d[f] for f in SAME_FIELDS)]
    print("- %s: %d / %d identical with the driver file %s (%s%s; amount %+.6f)" % (
        Path(a.mine).name, len(rm) - len(differ), len(rm), Path(a.driver).name, "/".join(SAME_FIELDS),
        "; first %d driver rows" % len(rm) if partial else "", p["amount"]), flush=True)
    if differ:
        raise SystemExit("not reproduced, e.g. %s" % differ[:5])


# ----------------------------------------------------------------------------- judge inputs
def run_subset(a):
    """ff_test_subset_judge.sh's subset step for either split (that script takes test exports only)."""
    raw = Path(a.export).read_bytes()
    d = json.loads(raw)
    if d.get("completed") is not True or d.get("split") != a.split:
        raise SystemExit("%s is not a completed export of the %s split" % (a.export, a.split))
    sraw = Path(a.subset).read_bytes()
    ids = json.loads(sraw)["pair_ids"]
    if not ids or len(set(ids)) != len(ids):
        raise SystemExit("the subset is empty or repeats a pair id")
    have = set(d["question_ids"])
    missing = [i for i in ids if i not in have]
    if missing:
        raise SystemExit("%d subset question(s) are not in the export, e.g. %s" % (len(missing), missing[:3]))
    keep = set(ids)
    out = dict(d)
    out["question_ids"] = [q for q in d["question_ids"] if q in keep]
    out["evaluations"] = [e for e in d["evaluations"] if e["pair_id"] in keep]
    if [e["pair_id"] for e in out["evaluations"]] != out["question_ids"]:
        raise SystemExit("the export's evaluations are not in the order of its question ids")
    work = Path(a.work)
    if not work.is_dir():
        raise SystemExit("work dir %s does not exist" % work)
    st = write_new(work / "judge_input.json", json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    info = {"subset_of": str(a.export), "source_sha256": hashlib.sha256(raw).hexdigest(), "subset_file": str(a.subset),
            "subset_sha256": hashlib.sha256(sraw).hexdigest(), "n_questions": len(out["question_ids"]),
            "n_source_questions": len(d["question_ids"]), "conditions": d["conditions"]}
    write_new(work / "subset_info.json", json.dumps(info, ensure_ascii=False, indent=1) + "\n")
    print("%d of %d questions kept, %d conditions (judge_input.json %s)" % (
        len(out["question_ids"]), len(d["question_ids"]), len(d["conditions"]), st), flush=True)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ids_sha(ids):
    """sha256 of a question set: its sorted ids as compact JSON."""
    return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode("utf-8")).hexdigest()


def run_bind(a):
    """B1: the candidate questions of THIS run and split (blank after E alone at the ladder strength on the plain prompt),
    rebuilt from the run's own E+C grid export, read-only. No judge label and no gold answer is read: blank = the run's
    rule detector on the raw answer text, or an empty answer. The population is the questions the judge has rows for;
    the only questions allowed to be missing there are the special-gold ones (their only reference is a refusal)."""
    if not STRENGTH_RE.match(a.strength):
        raise SystemExit("strength %r is not a number like 2 or 0.5" % a.strength)
    code = Path(a.code).resolve()
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    from core.readout import rule_abstains           # the run's own rule detector (the one that made the stored flags)
    run, grid = Path(a.run_dir).resolve(), Path(a.grid).resolve()
    if grid.parent != run / "eval_extra":
        raise SystemExit("--grid %s is not a work dir in this run's eval_extra/ (%s)" % (grid, run))
    cond = "E_L%d_x-w-p%s" % (a.layer, a.strength.replace(".", "-"))
    bad = []

    def need(ok, msg):
        if not ok:
            bad.append(msg)
        return ok

    def stop():
        if bad:
            raise SystemExit("bind %s %s: %d problem(s): %s" % (run.name, a.split, len(bad), "; ".join(bad[:8])))

    export_path, frozen_path, cells_path = grid / "judge_input.json", run / ("frozen_%s.json" % a.split), grid / "judge2" / "cells.jsonl"
    d, fz = read_json(export_path), read_json(frozen_path)
    questions = read_json(Path(a.source_run) / "questions.json")["splits"][a.split]
    order = [q["pair_id"] for q in questions]
    text_of = {q["pair_id"]: q["question"] for q in questions}
    # 1. the export is this run's, of this split, over all of its generated questions in order
    frozen_sha = sha(frozen_path)
    need(d.get("completed") is True and d.get("split") == a.split, "export is not a completed export of the %s split" % a.split)
    need(d.get("identity_sha256") == fz.get("identity_sha256"), "export identity differs from the run's frozen answers")
    need(d.get("question_ids") == order == fz.get("question_ids"), "export question ids differ from the split's (order included)")
    need(d.get("sources", {}).get("frozen") == frozen_path.name and
         d.get("sources", {}).get("sha256", {}).get(str(frozen_path)) == frozen_sha, "export was made from other frozen answers")
    # 2. the condition is E alone on the plain prompt x at this strength, with this run's direction file
    dfile = run / "dirs" / ("E_L%d.pt" % a.layer)
    dsha = sha(dfile)
    specs = [s for s in d.get("condition_specs", []) if s.get("name") == cond]
    if need(cond in d.get("conditions", []) and len(specs) == 1, "export has no single condition %s" % cond):
        s0 = specs[0]
        parts = s0.get("parts") or [{}]
        p0 = parts[0]
        need(len(parts) == 1 and s0.get("base_condition") == "baseline_x" and s0.get("arm") == "x.w.p" + a.strength
             and p0.get("family") == "E" and p0.get("layer") == a.layer and p0.get("kind") == "w"
             and p0.get("beta") == float(a.strength) and p0.get("direction_file_sha256") == dsha
             and p0.get("direction_run") == run.name, "%s is not E alone at %s with this run's direction file" % (cond, a.strength))
    stop()
    # 3. every generated question exactly once with its own question text; blank recomputed from the raw answer
    evs = d["evaluations"]
    need([e.get("pair_id") for e in evs] == order, "export rows are not every question of the split exactly once, in order")
    stop()
    blank, rule_of, empty_of = {}, {}, {}
    for e in evs:
        p, text = e["pair_id"], e.get("outputs", {}).get(cond)
        if not need(isinstance(text, str), "%s: no answer text" % p):
            continue
        need(e.get("question") == text_of[p], "%s: question text differs from the source run" % p)
        rule_of[p], empty_of[p] = bool(rule_abstains(text)), not text.strip()
        need(e.get("rule_abstain", {}).get(cond) is rule_of[p], "%s: the export's stored rule flag differs from the rule detector" % p)
        blank[p] = rule_of[p] or empty_of[p]
    stop()
    # 4. cells and judged rows of the condition
    cells = [c for c in read_jsonl(cells_path) if c.get("condition") == cond]
    cn = collections.Counter(c.get("pair_id") for c in cells)
    need(set(cn) == set(order) and all(n == 1 for n in cn.values()), "cells.jsonl: not every question exactly once for %s" % cond)
    stop()
    cmap, ev = {c["pair_id"]: c for c in cells}, {e["pair_id"]: e for e in evs}
    for p in order:
        c = cmap[p]
        need(c.get("question") == ev[p]["question"] and c.get("prediction") == ev[p]["outputs"][cond]
             and c.get("rule_abstain") is rule_of[p], "%s: its cell differs from the export" % p)
    special = [p for p in order if cmap[p].get("gold_is_abstention") is True]
    population = [p for p in order if cmap[p].get("gold_is_abstention") is not True]
    judged = {}
    for j in ("pro", "allenai"):
        f = grid / "judge2" / ("single_%s.jsonl" % j)
        if not f.exists():
            continue
        rows = [r for r in read_jsonl(f) if r.get("condition") == cond]
        seen = collections.Counter(r.get("pair_id") for r in rows)
        dup = sorted(p for p, n in seen.items() if n > 1)
        missing, extra = sorted(set(population) - set(seen)), sorted(set(seen) - set(population))
        need(not (dup or missing or extra), "single_%s.jsonl, %s: questions twice %s, missing %s, outside the population %s" % (
            j, cond, dup[:3], missing[:3], extra[:3]))
        nv = []
        for r in rows:
            p = r.get("pair_id")
            if p not in cmap:
                continue
            need(r.get("cell_id") == p + "|" + cond and r.get("input_sha") == cmap[p].get("input_sha"),
                 "single_%s %s: the row is not this question's cell" % (j, p))
            need(r.get("rule_abstain") is rule_of[p] and r.get("empty_output") is empty_of[p],
                 "single_%s %s: the stored rule / empty flags differ from the raw answer" % (j, p))
            if r.get("status") != "scored" and not (r.get("basis") == "empty_output_no_judge_vote" and r.get("empty_output") is True):
                nv.append(p)
        judged[j] = {"file": str(f), "sha256": sha(f), "rows": len(rows), "no_verdict_nonempty": nv}
    need(bool(judged), "the grid has no judged rows (judge2/single_pro.jsonl or single_allenai.jsonl)")
    stop()
    cand = [p for p in population if blank[p]]
    binding = {"protocol": "promptladder_candidates_v1", "run_dir": str(run), "split": a.split, "condition": cond,
               "rule": "blank = core.readout.rule_abstains(raw answer) or an empty answer; no judge label or gold answer read",
               "rule_code_sha256": sha(code / "core" / "readout.py"),
               "source": {"grid": str(grid), "export_sha256": sha(export_path), "cells_sha256": sha(cells_path),
                          "judged": {j: {k: v for k, v in x.items() if k != "no_verdict_nonempty"} for j, x in judged.items()},
                          "identity_sha256": d["identity_sha256"], "frozen_sha256": frozen_sha, "direction_file_sha256": dsha},
               "population": {"generated": len(order), "special_gold_removed": special, "n": len(population),
                              "ids_sha256": ids_sha(population),
                              "why_removed": "only reference answer is a refusal (cells.jsonl gold_is_abstention); the judge has no rows for them",
                              "no_verdict_nonempty_kept": {j: x["no_verdict_nonempty"] for j, x in judged.items()}},
               "candidates": {"n": len(cand), "ids_sha256": ids_sha(cand)}}
    say("bind %s %s, %s: generated %d, special gold removed %d %s, population %d (sha256 %s), candidates %d (sha256 %s)" % (
        run.name, a.split, cond, len(order), len(special), special, len(population), ids_sha(population)[:12], len(cand),
        ids_sha(cand)[:12]))
    for j, x in judged.items():
        say("  %s rows: %d, every population question once, stored rule/empty flags = the raw answers; kept although "
            "without a verdict (non-empty answer): %d %s" % (j, x["rows"], len(x["no_verdict_nonempty"]), x["no_verdict_nonempty"][:5]))
    if not cand:
        say("no question is blank after E alone: nothing needs re-asking on %s" % a.split)
    if a.subset:
        sub = read_json(a.subset)
        ids = sub.get("pair_ids") if isinstance(sub, dict) else None
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            raise SystemExit("%s: no list of pair_ids" % a.subset)
        rep = sorted({i for i in ids if ids.count(i) > 1})
        extra, missing = sorted(set(ids) - set(cand)), sorted(set(cand) - set(ids))
        if rep or extra or missing:
            raise SystemExit("SUBSET %s is not this run's %s candidate set: %d ids given, %d rebuilt; repeated %s; not "
                             "candidates %d %s; candidates missing %d %s" % (a.subset, a.split, len(ids), len(cand), rep[:3],
                                                                            len(extra), extra[:3], len(missing), missing[:3]))
        binding["subset"] = {"file": str(a.subset), "sha256": sha(a.subset)}
        say("SUBSET equals the rebuilt candidate set exactly (%d questions)" % len(cand))
    if a.write:
        say("candidate file %s: %s" % (a.write, write_new(a.write, json.dumps({"pair_ids": cand, "binding": binding},
                                                                              ensure_ascii=False, indent=1) + "\n")))
    print("BINDING " + json.dumps(binding, ensure_ascii=False, sort_keys=True, separators=(",", ":")), flush=True)


def run_verdicts(a):
    """S2: every expected cell (the work dir's questions x its exported conditions) has exactly one judged row. Empty
    answers (never judged) and judge-declined cells are listed; a missing, repeated, foreign or unverdicted row stops."""
    work = Path(a.work)
    d = read_json(work / "judge_input.json")
    qids, conds = d["question_ids"], d["conditions"]
    sub = read_json(a.subset)["pair_ids"]
    if len(set(qids)) != len(qids) or [e["pair_id"] for e in d["evaluations"]] != qids:
        raise SystemExit("%s/judge_input.json: question ids repeat or differ from its rows" % work)
    if sorted(sub) != sorted(qids) or len(set(sub)) != len(sub):
        raise SystemExit("%s holds other questions than the subset %s" % (work, a.subset))
    exp = {"%s|%s" % (q, c) for q in qids for c in conds}
    cells = read_jsonl(work / "judge2" / "cells.jsonl")
    cc = collections.Counter(c.get("cell_id") for c in cells)
    if set(cc) != exp or any(n != 1 for n in cc.values()):
        raise SystemExit("%s/judge2/cells.jsonl is not the expected cell set (%d rows, %d expected)" % (work, len(cells), len(exp)))
    special = [c["cell_id"] for c in cells if c.get("gold_is_abstention")]
    if special:
        raise SystemExit("special-gold cells in the subset (the binding excludes such questions): %s" % special[:3])
    shas = {c["cell_id"]: c.get("input_sha") for c in cells}
    rows = read_jsonl(work / "judge2" / ("single_%s.jsonl" % a.judge))
    rc = collections.Counter(r.get("cell_id") for r in rows)
    dup, missing, extra = sorted(k for k, n in rc.items() if n > 1), sorted(exp - set(rc)), sorted(set(rc) - exp)
    wrong = sorted(r["cell_id"] for r in rows if r.get("cell_id") in shas and r.get("input_sha") != shas[r["cell_id"]])
    empty = [r["cell_id"] for r in rows if r.get("status") != "scored" and r.get("basis") == "empty_output_no_judge_vote"
             and r.get("empty_output") is True]
    declined = [r["cell_id"] for r in rows if r.get("status") != "scored" and r.get("basis") == "pro_declined"]
    listed = set(empty) | set(declined)
    other = [r.get("cell_id") for r in rows if r.get("status") != "scored" and r.get("cell_id") not in listed]
    print("verdicts %s (%s): %d expected cells (%d questions x %d conditions), %d rows; scored %d, empty answers %d "
          "(abstentions, never judged), declined %d, other without verdict %d; missing %d, repeated %d, foreign %d, "
          "other input %d" % (work.name, a.judge, len(exp), len(qids), len(conds), len(rows),
                              sum(r.get("status") == "scored" for r in rows), len(empty), len(declined), len(other),
                              len(missing), len(dup), len(extra), len(wrong)), flush=True)
    if declined:
        print("  declined (no table until judge2/excluded_questions.json names their questions with a reason): %s" % declined)
    if missing or dup or extra or wrong or other:
        raise SystemExit("%s (%s): missing %s, repeated %s, foreign %s, other input %s, without verdict %s" % (
            work, a.judge, missing[:3], dup[:3], extra[:3], wrong[:3], other[:3]))


def run_show(a):
    files = [read_json(f) for f in a.gen]
    rows = [{r["pair_id"]: r for r in d["rows"]} for d in files]
    order = [r["pair_id"] for r in files[0]["rows"]][:a.n]
    for pid in order:
        print("\n- %s %s" % (pid, rows[0][pid]["question"]))
        for d, look in zip(files, rows):
            r = look.get(pid)
            if r is None:
                print("    %-22s (not in this file)" % d.get("arm"))
                continue
            print("    %-22s no write: %-40s | E write: %s" % (d.get("arm"), json.dumps(r["baseline_text"], ensure_ascii=False),
                                                              json.dumps(r["text"], ensure_ascii=False)))


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "extra":
        return run_extra(sys.argv[2:])
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", allow_abbrev=False)                  # S3: no abbreviated options anywhere
    c.add_argument("--code", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--split", required=True, choices=("tune", "test"))
    d = sub.add_parser("dryrun", allow_abbrev=False)
    d.add_argument("--code", required=True)
    d.add_argument("--source-run", required=True)
    d.add_argument("--run-dir", required=True)
    d.add_argument("--prompts-extra", required=True)
    d.add_argument("--layer", required=True, type=int)
    d.add_argument("--strengths", nargs="+", default=["2"])
    d.add_argument("--prompts")
    d.add_argument("--prompts-e44")
    r = sub.add_parser("reproduce", allow_abbrev=False)
    r.add_argument("mine")
    r.add_argument("driver")
    b = sub.add_parser("bind", allow_abbrev=False)
    b.add_argument("--code", required=True)
    b.add_argument("--run-dir", required=True)
    b.add_argument("--source-run", required=True)
    b.add_argument("--grid", required=True)
    b.add_argument("--layer", required=True, type=int)
    b.add_argument("--strength", required=True)
    b.add_argument("--split", required=True, choices=("tune", "test"))
    b.add_argument("--subset")
    b.add_argument("--write")
    s = sub.add_parser("subset", allow_abbrev=False)
    s.add_argument("export")
    s.add_argument("subset")
    s.add_argument("work")
    s.add_argument("--split", required=True, choices=("tune", "test"))
    v = sub.add_parser("verdicts", allow_abbrev=False)
    v.add_argument("--work", required=True)
    v.add_argument("--judge", required=True, choices=("pro", "allenai"))
    v.add_argument("--subset", required=True)
    w = sub.add_parser("show", allow_abbrev=False)
    w.add_argument("--n", type=int, default=8)
    w.add_argument("gen", nargs="+")
    a = p.parse_args()
    {"check": run_check, "dryrun": run_dryrun, "reproduce": run_reproduce, "bind": run_bind, "subset": run_subset,
     "verdicts": run_verdicts, "show": run_show}[a.cmd](a)


if __name__ == "__main__":
    main()
