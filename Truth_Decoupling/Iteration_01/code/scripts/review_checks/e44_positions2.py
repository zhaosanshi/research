# -*- coding: utf-8 -*-
"""Evaluation-only entry (Claude Opus 5.5, 2026-11-21): e44_positions.py (unchanged, imported) extended to TWO-WRITE
arms, so that N1 (E on every question, blanks re-asked with E + C) can be generated with both writes at another
position rule than the training rule. A diagnostic of where the writes have to act, not a method setting.

Why: with the write rule "prompt" (question tokens and assistant header; no decode step written) single E keeps its
TruthfulQA effect on Gemma-3-12B tune (+42 against +45 at the training rule), while the training rule costs 14 to 19
GSM8K points at the main-table settings. N1 needs the E + C arms at the same rule.

What it adds to e44_positions.py (whose three in-memory changes are applied by calling its main() logic below):
  1. two-write arms are allowed for generate, and the phase selfcheck is allowed;
  2. the engine's acceptance record of a two-write pair is kept apart: Context.selfcheck_path gets the suffix
     "-at-<POS>" (the record itself already stores "positions", which require_selfcheck compares), so the record of
     the training rule is never overwritten;
  3. both writes of a two-write arm act at POS (core generate_two takes one rule for the two hooks).
The engine, the driver and e44_positions.py stay byte-identical. Two-write conditions are exported under
"X_<arm token>-at-<POS>" (e44_positions.py's arm_token change), single-write ones under their usual names: keep every
POS in its own work dir.
usage (from the code copy's root, with the model's own interpreter):
  python scripts/review_checks/e44_positions2.py POS selfcheck|generate|export --split tune|test <engine arguments>
"""
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import e44_positions as P  # noqa: E402


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in P.ALLOWED_POS or sys.argv[2] not in ("selfcheck", "generate", "export"):
        raise SystemExit(__doc__)
    pos, rest = sys.argv[1], sys.argv[2:]
    split = P.split_of(rest)
    if split == "test":
        import e44_split_test as S
        _, _, Y = S.patch()
        S.require_test("extra", rest[0], S.split_values(rest))
    else:
        import e44_eval_extra as Y
    original_writes, original_token, original_check = Y.Context.writes, Y.arm_token, Y.Context.selfcheck_path

    def writes(self, parts):
        out, trained = original_writes(self, parts)
        if trained != "content":
            raise SystemExit("the directions were trained with positions=%s; this entry only moves content-position directions" % trained)
        return out, pos

    def selfcheck_path(self, descs):
        path = original_check(self, descs)
        return path.with_name("%s-at-%s%s" % (path.stem, pos, path.suffix))

    Y.Context.writes = writes
    Y.Context.selfcheck_path = selfcheck_path
    Y.arm_token = lambda arm: original_token(arm) + "-at-" + pos
    P.recompile(Y, "run_export", P.OLD_LINE, 'd["positions"] != %r or' % pos)
    print("[positions2] writes at %r instead of the training rule (two-write arms allowed, acceptance record *-at-%s); "
          "e44_eval_extra.py sha256 %s (unchanged file); e44_positions.py sha256 %s; this entry sha256 %s" % (
              pos, pos, hashlib.sha256(Path(Y.__file__).read_bytes()).hexdigest()[:16],
              hashlib.sha256(Path(P.__file__).read_bytes()).hexdigest()[:16], hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]), flush=True)
    sys.argv = [Y.__file__] + rest
    Y.main()


if __name__ == "__main__":
    main()
