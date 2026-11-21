# -*- coding: utf-8 -*-
"""Confidence of the model in its frozen NO-WRITE answers under any prompt of the run (Claude Fable 5.2, 2026-11-21).

Why: the selection comparators so far rank the answers of "must-answer + E" (the strong form) or of the must-answer
prompt alone (targets conf / x of ff_explore_ec_confidence.py). The prompt-only baselines of the main table (conf_E2,
caut_E2 and the run's other prompts, diag/prompt_sweep_table.py) had no confidence, so "the best prompt, keeping only
its most confident answers" - no write at all - could not be compared with N1.

This entry runs ff_explore_ec_confidence.py's main() UNCHANGED (scoring, the greedy check and its 98% line, the output
format), once per named prompt, with two in-memory changes:
  1. its target list is replaced by ONE no-write target, ("prompt:<name>", <name>, [], "frozen", None): the frozen
     answers of that prompt, re-run teacher-forced under the same prompt with nothing written;
  2. the engine's Context is built once and handed to every call (the model is loaded once, not once per prompt).
One file per prompt, because the script saves nothing when a target misses its greedy check: a smoke run on Llama
(48 tune questions per prompt) gave reproduction rates between 97.8% and 99.6%, i.e. around the line. The line is not
moved. A prompt that misses it at the scoring batch of 8 is scored once more with batch 1, the fallback the script's
own --batch help names ("1 only if a target misses the 98% greedy check at 8"), into a file with the suffix _b1; a
prompt that misses it at both has no confidence file and is reported as such.
The files of the confidence script, the engine and the driver stay byte-identical. --el / --cl / --es / --ks are still
required by the script's argument parser and are recorded in the output's "spec"; no write is built from them.
For --split test the in-memory widenings of e44_split_test.py are applied first, exactly as ff_test_confidence.py does.
usage (from the code copy's root, with the model's own interpreter):
  python scripts/review_checks/ff_prompt_confidence.py PREFIX --bases conf_E2 caut_E2 [...] [--split test]
         --source-run S --run-dir E_RUN --e-on E2 [--c-run-dir C_RUN --c-e-on E1] --el 15 --cl 16 --es 1 --ks 1
         [--first-batches K]
output: <E_RUN>/eval_extra/confidence_<PREFIX>_<name>.json (or ..._<name>_b1.json), one target "prompt:<name>" each
        (read by diag/prompt_conf_table.py). Exit status 1 if a prompt has no file at the end.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE):          # <code copy>/scripts (engine, driver), then review_checks (the wrapper, the script)
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import e44_split_test as S  # noqa: E402

GREEDY_MISS = "teacher-forced arg-max reproduces only"


def take_bases(argv):
    """(prompt names, argv without the --bases option)."""
    if argv.count("--bases") != 1:
        raise SystemExit("give --bases PROMPT [PROMPT ...] exactly once")
    i = argv.index("--bases")
    j = i + 1
    while j < len(argv) and not argv[j].startswith("-"):
        j += 1
    bases = argv[i + 1:j]
    if not bases or len(set(bases)) != len(bases):
        raise SystemExit("--bases needs at least one prompt name, each named once")
    return bases, argv[:i] + argv[j:]


def main():
    bases, argv = take_bases(sys.argv[1:])
    if not argv or argv[0].startswith("-"):
        raise SystemExit("the first argument is the prefix of the output names")
    prefix, rest = argv[0], argv[1:]
    if "--batch" in rest:
        raise SystemExit("--batch is set by this entry: 8, then 1 for a prompt that misses the greedy check at 8")
    vals = S.split_values(rest)
    split = vals.get("--split", ["tune"])
    if "--splits" in vals or split not in (["tune"], ["test"]):
        raise SystemExit("give '--split tune' (the default) or '--split test', and no --splits (got %s)" % vals)
    if split == ["test"]:
        S.patch()
    import ff_explore_ec_confidence as C
    if Path(C.__file__).resolve().parent != HERE:
        raise SystemExit("imported %s, expected the confidence script next to this entry" % C.__file__)
    print("[prompt-confidence] split %s, prompts %s | confidence script sha256 %s, entry sha256 %s" % (
        split[0], " ".join(bases), S.sha(C.__file__), S.sha(__file__)), flush=True)

    real_context, cache = C.X.Context, {}

    def context_once(a):
        if "ctx" not in cache:
            cache["ctx"] = real_context(a)
        return cache["ctx"]

    C.X.Context = context_once
    outcome = {}
    for b in bases:
        C.build_targets = lambda el, cl, es, ks, split="tune", b=b: [("prompt:%s" % b, b, [], "frozen", None)]
        for batch in ("8", "1"):
            name = "%s_%s" % (prefix, b) + ("" if batch == "8" else "_b1")
            sys.argv = [C.__file__, name] + rest + ["--batch", batch]
            try:
                C.main()
                outcome[b] = "ok, scoring batch %s" % batch
                break
            except RuntimeError as e:
                if GREEDY_MISS not in str(e):
                    raise
                outcome[b] = "NO FILE: greedy check missed at batch %s (%s)" % (batch, e)
                print("[prompt-confidence] %s: %s" % (b, outcome[b]), flush=True)
            except SystemExit as e:
                if not str(e).endswith(" exists"):
                    raise
                outcome[b] = "already there (%s)" % e
                break
    print("[prompt-confidence] summary, split %s:" % split[0], flush=True)
    for b in bases:
        print("[prompt-confidence]   %-8s %s" % (b, outcome[b]), flush=True)
    if any(v.startswith("NO FILE") for v in outcome.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
