# -*- coding: utf-8 -*-
"""Exploration entry (Claude Opus 5.5, 2026-11-21): run scripts/newsets/newsets_gen.py with the E write at another
position rule than the one the direction was trained with. A diagnostic of WHERE the write has to act on long
generations (GSM8K), not a method setting.

Why: at the main-table settings (Llama-3.1-8B E16 at 3.5, Gemma-3-12B E24 at 2.5) GSM8K accuracy falls by 19 and 14
points, while MMLU (one read position) moves by under 1 point. The training rule "content" writes every generated
token as well; a GSM8K solution has hundreds of them. The core rule "prompt" writes the question (first user-content
token .. last prompt token) and leaves every decode step untouched (core/qa_factorial_das.py: _prefill_mask,
generate_shift / generate_two with decode_all = positions != "prompt").

newsets_gen.py runs unchanged, with three in-memory changes:
  1. the engine's Context.writes returns POS in place of the training rule, after checking that the training rule is
     "content" (as scripts/review_checks/e44_positions.py does for the TruthfulQA engine);
  2. plan_arm accepts POS where it demands "content" (a source replacement that must match exactly once);
  3. arm_token appends "-at-<POS>" to arms with a write, so the output file never collides with the content-rule file
     of the same arm. The output contract records "positions": POS.
Only single-write arms are allowed (two-write arms have an acceptance record for the training rule only).
usage (host side, with the model's own interpreter, like newsets_gen.py):
  python -I Iteration_01/code/scripts/newsets/ns_positions.py prompt|gen gen <newsets_gen.py gen arguments>
"""
import hashlib
import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import newsets_gen as G  # noqa: E402

ALLOWED_POS = ("prompt", "gen")
OLD = 'if positions != "content":'


def recompile(module, func_name, old, new):
    fn = getattr(module, func_name)
    lines, first = inspect.getsourcelines(fn)
    src = "".join(lines)
    if src.count(old) != 1:
        raise SystemExit("%s.%s: expected exactly one occurrence of %r, found %d" % (module.__name__, func_name, old, src.count(old)))
    exec(compile("\n" * (first - 1) + src.replace(old, new), module.__file__, "exec"), module.__dict__)
    if getattr(module, func_name) is fn:
        raise SystemExit("recompiling %s.%s changed nothing" % (module.__name__, func_name))


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
    if len(sys.argv) < 3 or sys.argv[1] not in ALLOWED_POS or sys.argv[2] != "gen":
        raise SystemExit(__doc__)
    pos, rest = sys.argv[1], sys.argv[2:]
    arms = arms_of(rest)
    if not arms or any("+" in a for a in arms) or any(":" not in a for a in arms):
        raise SystemExit("give single-write arms only (base:direction), no two-write arms and no no-write arm")
    real_engine = G.engine

    def engine(m):
        X = real_engine(m)
        original = X.Context.writes

        def writes(self, parts):
            out, trained = original(self, parts)
            if trained != "content":
                raise SystemExit("the directions were trained with positions=%s; this entry only moves content-position directions" % trained)
            return out, pos

        X.Context.writes = writes
        print("[ns-positions] writes at %r instead of the training rule; engine %s sha256 %s (unchanged file)" % (
            pos, X.__file__, hashlib.sha256(Path(X.__file__).read_bytes()).hexdigest()[:16]), flush=True)
        return X

    G.engine = engine
    recompile(G, "plan_arm", OLD, 'if positions != %r:' % pos)
    original_token = G.arm_token
    G.arm_token = lambda arm: original_token(arm) + ("-at-" + pos if ":" in arm else "")
    print("[ns-positions] newsets_gen.py sha256 %s (unchanged file); entry sha256 %s" % (
        hashlib.sha256(Path(G.__file__).read_bytes()).hexdigest()[:16], hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]), flush=True)
    sys.argv = [G.__file__] + rest
    G.main()


if __name__ == "__main__":
    main()
