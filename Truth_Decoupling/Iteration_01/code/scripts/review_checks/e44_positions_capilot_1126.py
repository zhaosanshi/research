# -*- coding: utf-8 -*-
"""Evaluation-only entry (Claude Opus 5.5, executor si, 2026-11-26; plan discussion/work_queue/2026-11-26_queue_o.md):
scripts/review_checks/e44_positions2.py of the model's own code copy (unchanged, imported from the working directory),
tune split only, with ONE in-memory change (the pattern of e44_positions_h22.py):
  the engine's acceptance record of a two-write pair (Context.selfcheck_path) gets the extra suffix "-capilot" (before
  e44_positions2.py adds "-at-<POS>"). Reason: the existing record of each main-table pair at the prompt rule
  (selfcheck_<pair>-at-prompt.json) was last written on the TEST split, and the engine accepts a record only for the
  split it was made on; a tune selfcheck under the usual name would overwrite that test record. With the suffix the
  tune record of this pilot is a new file and the existing record stays byte-identical.
Everything else - write rule, two-write arms, condition names (X_<arm token>-at-<POS>), export - is e44_positions2.py
as it is, so the exported conditions carry the same names as the main table's E + C conditions.
Run from the root of the model's code copy:
  python <this file> prompt selfcheck|generate|export --split tune <engine arguments>
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
import e44_eval_extra as Y        # noqa: E402  (the model's own engine)
import e44_positions2 as P2       # noqa: E402

if Path(Y.__file__).resolve().parent != CWD / "scripts" or Path(P2.__file__).resolve().parent != CWD / "scripts" / "review_checks":
    raise SystemExit("imported the engine or e44_positions2.py from outside the working directory's code copy")
_args = sys.argv[1:]
_split = [a for i, a in enumerate(_args) if i > 0 and _args[i - 1] == "--split"] + [a.split("=", 1)[1] for a in _args if a.startswith("--split=")]
if _split != ["tune"]:
    raise SystemExit("this entry runs on the tune split only (pass --split tune once)")
if "--prompts-extra" in _args:
    raise SystemExit("this entry uses the code copy's default prompts file")
_original = Y.Context.selfcheck_path


def selfcheck_path(self, descs):
    path = _original(self, descs)
    return path.with_name("%s-capilot%s" % (path.stem, path.suffix))


Y.Context.selfcheck_path = selfcheck_path
print("[positions-capilot] selfcheck records *-capilot-at-<POS>; engine %s sha256 %s; e44_positions2.py sha256 %s; this entry sha256 %s" % (
    Y.__file__, hashlib.sha256(Path(Y.__file__).read_bytes()).hexdigest()[:16],
    hashlib.sha256(Path(P2.__file__).read_bytes()).hexdigest()[:16], hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]), flush=True)

if __name__ == "__main__":
    P2.main()
