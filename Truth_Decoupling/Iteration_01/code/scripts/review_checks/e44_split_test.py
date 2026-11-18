# -*- coding: utf-8 -*-
"""Evaluation-only entry to the E44 scripts on the TEST split (Claude, 2026-11-13 23:xx; from_claude/45 section 2,
赵磊 20:4x "可以统一重验").

The driver (e44_factorial_das.py), the two-write engine (e44_eval_extra.py) and the judge export (e44_export_judge.py)
refuse the test split by design: it was never read while the method and the rules were developed. This wrapper runs them
UNCHANGED -- their files, hence the run identity that every artifact is checked against, stay byte-identical (the identity
covers core/*.py, the driver, e43_qa_prompt_das.py and DEPENDENCY_SCRIPTS; scripts/review_checks/ is not part of it) --
with exactly three in-memory changes, each of which only widens the accepted split set by "test":
  1. argparse: an option --split / --splits declared with choices ("train", "tune") also takes "test";
  2. e44_factorial_das.validate_frozen: the accepted frozen split ("train", "tune") also takes "test" (every other
     check of the frozen answers is the original code, which the driver, the engine and the export all call);
  3. e44_export_judge.export: the exportable split ("train", "tune") also takes "test".
Changes 2 and 3 are source replacements that must match exactly once; otherwise the wrapper stops before anything runs.
Only the phases that evaluate are allowed (training stays on the train split, by the driver's own code), and the split
named on the command line must be test (the scripts' default is tune).
usage (from the code copy's root, like the scripts themselves):
  python scripts/review_checks/e44_split_test.py check                       (apply the changes, print hashes, run nothing)
  python scripts/review_checks/e44_split_test.py driver freeze   --splits test  <driver arguments>
  python scripts/review_checks/e44_split_test.py driver generate --split test   <driver arguments>
  python scripts/review_checks/e44_split_test.py extra  selfcheck|generate|export --split test <engine arguments>
  python scripts/review_checks/e44_split_test.py export --split test <export arguments>
"""
import argparse
import hashlib
import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

OLD, NEW = ("train", "tune"), ("train", "tune", "test")
PHASES = {"driver": ("freeze", "generate"), "extra": ("selfcheck", "generate", "export"), "export": None}
_hits = []
_orig_add = argparse._ActionsContainer.add_argument


def _add_argument(self, *names, **kw):
    if any(n in ("--split", "--splits") for n in names) and kw.get("choices") is not None and tuple(kw["choices"]) == OLD:
        kw["choices"] = NEW
        _hits.append(names)
    return _orig_add(self, *names, **kw)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def widen(module, func_name, old, new):
    """Recompile one function of a module with OLD replaced by NEW (exactly one occurrence), in the module's namespace."""
    fn = getattr(module, func_name)
    lines, first = inspect.getsourcelines(fn)
    src = "".join(lines)
    if src.count(old) != 1:
        raise SystemExit("%s.%s: expected exactly one occurrence of %r, found %d" % (module.__name__, func_name, old, src.count(old)))
    # blank lines in front keep the file's line numbers in tracebacks
    exec(compile("\n" * (first - 1) + src.replace(old, new), module.__file__, "exec"), module.__dict__)
    fresh = getattr(module, func_name)
    if fresh is fn or NEW not in fresh.__code__.co_consts or OLD in fresh.__code__.co_consts:
        raise SystemExit("recompiled %s.%s does not carry the widened split set" % (module.__name__, func_name))
    print("[split-test] widened %s.%s (source sha256 %s -> %s)" % (
        module.__name__, func_name, hashlib.sha256(src.encode()).hexdigest()[:12],
        hashlib.sha256(src.replace(old, new).encode()).hexdigest()[:12]), flush=True)


def split_values(argv):
    """{option: [values]} for --split / --splits on the command line (both 'opt value' and 'opt=value' forms)."""
    vals, i = {}, 0
    while i < len(argv):
        a = argv[i]
        if a in ("--split", "--splits"):
            j = i + 1
            while j < len(argv) and not argv[j].startswith("-"):
                vals.setdefault(a, []).append(argv[j])
                j += 1
                if a == "--split":
                    break
            if j == i + 1:
                raise SystemExit("%s without a value" % a)
            i = j
            continue
        if a.startswith("--split=") or a.startswith("--splits="):
            opt, v = a.split("=", 1)
            vals.setdefault(opt, []).append(v)
        i += 1
    return vals


def require_test(which, phase, vals):
    """The option the phase actually reads must name test, alone; the other split option must be absent (red team 23:21:
    'driver generate --splits test' would otherwise pass and run on tune)."""
    reads = "--splits" if (which == "driver" and phase == "freeze") else "--split"
    other = "--split" if reads == "--splits" else "--splits"
    if vals.get(reads) != ["test"] or other in vals:
        raise SystemExit("%s %s reads %s: give exactly '%s test' and no %s (got %s)" % (which, phase or "", reads, reads, other, vals or "nothing"))


def patch():
    argparse._ActionsContainer.add_argument = _add_argument
    import e44_factorial_das as E                     # imports common (code root on sys.path) and the core
    widen(E, "validate_frozen", 'd.get("split") not in ("train", "tune")', 'd.get("split") not in ("train", "tune", "test")')
    import e44_export_judge as X
    widen(X, "export", 'if split not in ("train", "tune"):', 'if split not in ("train", "tune", "test"):')
    import e44_eval_extra as Y
    for m in (E, X, Y):
        print("[split-test] %s sha256 %s (unchanged file)" % (Path(m.__file__).name, sha(m.__file__)), flush=True)
    print("[split-test] wrapper sha256 %s" % sha(__file__), flush=True)
    return E, X, Y


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("check",) + tuple(PHASES):
        raise SystemExit(__doc__)
    which, rest = sys.argv[1], sys.argv[2:]
    E, X, Y = patch()
    if which == "check":
        if rest:
            raise SystemExit("check takes no arguments")
        return
    allowed = PHASES[which]
    if allowed is not None and (not rest or rest[0] not in allowed):
        raise SystemExit("%s: phase must be one of %s (got %s)" % (which, allowed, rest[:1]))
    require_test(which, rest[0] if allowed is not None else None, split_values(rest))
    target = {"driver": E, "extra": Y, "export": X}[which]
    sys.argv = [target.__file__] + rest
    target.main()
    if not _hits:
        raise SystemExit("the split option was never declared: nothing was widened (this should not happen)")


if __name__ == "__main__":
    main()
