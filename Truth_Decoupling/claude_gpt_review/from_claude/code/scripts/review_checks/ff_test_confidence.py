# -*- coding: utf-8 -*-
"""Evaluation-only entry of ff_explore_ec_confidence.py on the TEST split (Claude, 2026-11-14; from_claude/45 R6: the
equal-N comparison "E + the model's own confidence" on the 529 test questions).

The engine (e44_eval_extra.py) refuses the test split by design. This entry applies exactly the in-memory widenings of
e44_split_test.py (its patch(): argparse choices, e44_factorial_das.validate_frozen, e44_export_judge.export; each a
source replacement that must match exactly once; the driver, engine and export files stay byte-identical, so the run
identity is unchanged), then runs the confidence script's main() unchanged. The command line must name --split test,
alone (the confidence script's default is tune).
usage (from the code copy's root, with the model's own interpreter):
  python scripts/review_checks/ff_test_confidence.py OUT_NAME --split test <ff_explore_ec_confidence.py arguments>
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE):          # <code copy>/scripts (engine, driver), then review_checks (the wrapper, the script)
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import e44_split_test as S  # noqa: E402


def main():
    argv = sys.argv[1:]
    vals = S.split_values(argv)
    if vals.get("--split") != ["test"] or "--splits" in vals:
        raise SystemExit("give exactly '--split test' and no --splits (got %s)" % (vals or "nothing"))
    S.patch()
    import ff_explore_ec_confidence as C
    if Path(C.__file__).resolve().parent != HERE:
        raise SystemExit("imported %s, expected the confidence script next to this entry" % C.__file__)
    print("[split-test] confidence script sha256 %s, entry sha256 %s" % (S.sha(C.__file__), S.sha(__file__)), flush=True)
    sys.argv = [C.__file__] + argv
    C.main()


if __name__ == "__main__":
    main()
