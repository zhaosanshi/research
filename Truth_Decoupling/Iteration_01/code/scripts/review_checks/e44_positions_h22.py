# -*- coding: utf-8 -*-
"""Evaluation-only entry (Claude Opus 5.5, 2026-11-22, queue I): scripts/review_checks/e44_positions2.py of the model's
own code copy (unchanged, imported from the working directory) for arms on the prompt x_E2
(configs/prompts/e44_eval_extra_h22_v1.json, passed with --prompts-extra), with ONE in-memory change:
  the engine's acceptance record of a two-write pair (Context.selfcheck_path) gets the extra suffix "-h22" (before
  e44_positions2.py adds "-at-<POS>"), so that the record of these runs (base prompt x_E2, the new prompt file) never
  overwrites the existing records of the same direction pair (base prompt x, file selfcheck_<pair>-at-prompt.json).
Everything else - write rule, two-write arms, export - is e44_positions2.py as it is.
Run from the root of the model's code copy (the engine, e44_positions.py and e44_positions2.py are imported from there,
not from this file's directory):
  python <this file> prompt selfcheck|generate|export --split tune|test <engine arguments> --prompts-extra <h22 file>
"""
import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != HERE]
CWD = Path(os.getcwd()).resolve()
for p in (CWD / "scripts", CWD / "scripts" / "review_checks"):
    sys.path.insert(0, str(p))
import e44_eval_extra as Y        # noqa: E402  (the model's own engine, the same module e44_split_test.patch returns)
import e44_positions2 as P2       # noqa: E402

if Path(Y.__file__).resolve().parent != CWD / "scripts" or Path(P2.__file__).resolve().parent != CWD / "scripts" / "review_checks":
    raise SystemExit("imported the engine or e44_positions2.py from outside the working directory's code copy")
if "--prompts-extra" not in sys.argv or not any(a.endswith("e44_eval_extra_h22_v1.json") for a in sys.argv):
    raise SystemExit("pass --prompts-extra .../e44_eval_extra_h22_v1.json")
_original = Y.Context.selfcheck_path


def selfcheck_path(self, descs):
    path = _original(self, descs)
    return path.with_name("%s-h22%s" % (path.stem, path.suffix))


Y.Context.selfcheck_path = selfcheck_path
print("[positions-h22] selfcheck records *-h22-at-<POS>; engine %s sha256 %s; e44_positions2.py sha256 %s; this entry sha256 %s" % (
    Y.__file__, hashlib.sha256(Path(Y.__file__).read_bytes()).hexdigest()[:16],
    hashlib.sha256(Path(P2.__file__).read_bytes()).hexdigest()[:16], hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]), flush=True)

if __name__ == "__main__":
    P2.main()
