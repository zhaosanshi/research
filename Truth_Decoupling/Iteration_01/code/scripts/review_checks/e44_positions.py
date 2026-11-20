# -*- coding: utf-8 -*-
"""Evaluation-only entry (Claude Fable 5.2, 2026-11-20): write the trained directions at another position rule than the
one they were trained with. A diagnostic of WHERE the write has to act, not a method setting.

Background (novelty/ours/2026-11-19_capability_preservation_design.md, section 8.2): on MMLU the damage of the E write
almost disappears when the write starts at the last prompt token (the question tokens stay untouched). Whether the
TruthfulQA gain survives that change was never measured; the engine accepts only the training rule of a direction.

The engine (e44_eval_extra.py) runs unchanged, with three in-memory changes:
  1. Context.writes returns POS in place of the training rule, after checking that the training rule is "content"
     (core rules: content = first user-content token .. end; gen = last prompt token .. end; prompt = first
     user-content token .. last prompt token);
  2. arm_token appends "-at-<POS>", so generation files (and the names of two-write conditions) never collide with
     the content-position files of the same arms;
  3. run_export accepts a generation whose positions are POS (the original line demands the training rule).
Change 3 is a source replacement that must match exactly once. On the test split the widenings of e44_split_test.py are
applied first (that file unchanged). Only generate and export are allowed, and only single-write arms are generated:
the two-write acceptance record (selfcheck) is kept for the training rule alone.
A single-write condition keeps its usual name in the export (E_L<layer>_<base>-<kind>-<sign><strength>): give every
POS its own export name and work dir, and never merge its cells with content-position cells without renaming.
usage (from the code copy's root, with the model's own interpreter):
  python scripts/review_checks/e44_positions.py POS generate|export --split tune|test <engine arguments>
"""
import hashlib
import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

ALLOWED_POS = ("gen", "prompt")
OLD_LINE = 'rec["training_contract"]["positions"] != d["positions"] or'


def recompile(module, func_name, old, new):
    """One function of a module with OLD replaced by NEW (exactly one occurrence), compiled in the module's namespace."""
    fn = getattr(module, func_name)
    lines, first = inspect.getsourcelines(fn)
    src = "".join(lines)
    if src.count(old) != 1:
        raise SystemExit("%s.%s: expected exactly one occurrence of %r, found %d" % (module.__name__, func_name, old, src.count(old)))
    exec(compile("\n" * (first - 1) + src.replace(old, new), module.__file__, "exec"), module.__dict__)
    if getattr(module, func_name) is fn:
        raise SystemExit("recompiling %s.%s changed nothing" % (module.__name__, func_name))
    print("[positions] recompiled %s.%s (source sha256 %s -> %s)" % (
        module.__name__, func_name, hashlib.sha256(src.encode()).hexdigest()[:12],
        hashlib.sha256(src.replace(old, new).encode()).hexdigest()[:12]), flush=True)


def split_of(rest):
    for i, a in enumerate(rest):
        if a == "--split" and i + 1 < len(rest):
            return rest[i + 1]
        if a.startswith("--split="):
            return a.split("=", 1)[1]
    return "tune"


def arms_of(rest):
    out, on = [], False
    for a in rest:
        if a == "--arms":
            on = True
        elif a.startswith("--"):
            on = False
        elif on:
            out.append(a)
    return out


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ALLOWED_POS or sys.argv[2] not in ("generate", "export"):
        raise SystemExit(__doc__)
    pos, rest = sys.argv[1], sys.argv[2:]
    split = split_of(rest)
    if rest[0] == "generate" and any("+" in a for a in arms_of(rest)):
        raise SystemExit("only single-write arms: the two-write acceptance record is for the training rule")
    if split == "test":
        import e44_split_test as S
        _, _, Y = S.patch()
        S.require_test("extra", rest[0], S.split_values(rest))
    else:
        import e44_eval_extra as Y
    original_writes, original_token = Y.Context.writes, Y.arm_token

    def writes(self, parts):
        out, trained = original_writes(self, parts)
        if trained != "content":
            raise SystemExit("the directions were trained with positions=%s; this entry only moves content-position directions" % trained)
        return out, pos

    Y.Context.writes = writes
    Y.arm_token = lambda arm: original_token(arm) + "-at-" + pos
    recompile(Y, "run_export", OLD_LINE, 'd["positions"] != %r or' % pos)
    print("[positions] writes at %r instead of the training rule; e44_eval_extra.py sha256 %s (unchanged file); wrapper sha256 %s" % (
        pos, hashlib.sha256(Path(Y.__file__).read_bytes()).hexdigest(), hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), flush=True)
    sys.argv = [Y.__file__] + rest
    Y.main()


if __name__ == "__main__":
    main()
