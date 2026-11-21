# -*- coding: utf-8 -*-
"""Confidence of the model in its must-answer + E answers when E was written at another position rule (Claude Opus 5.5,
2026-11-21). For the strong comparator of the main table at the write rule "prompt": must-answer prompt + E at the same
layer and strength, keep the N most confident answers (N = N1's answers).

ff_explore_ec_confidence.py runs UNCHANGED (scoring, the greedy check at 98%, the first-difference check against the
no-write answers, the output format), with two in-memory changes:
  1. its target list is replaced by one target per strength, ("conf_E<s>", "conf", ["E<el>.w.p<s>"],
     "conf--E<el>-w-p<s>-at-<POS>_<split>.json", ("frozen", [])): the must-answer answers generated with E at POS
     (scripts/review_checks/e44_positions.py), re-run teacher-forced with E at POS;
  2. the core write mask used for scoring is POS instead of the training rule: core position_mask is wrapped so that a
     request for "content" returns the POS mask (the only mask the confidence script asks for). Under "prompt" this
     writes the prompt content positions and no answer position, as the generation did (no decode step written).
The greedy check then tests that the teacher-forced pass with E at POS reproduces the answers generated with E at POS.
For --split test the widenings of e44_split_test.py are applied first, as ff_test_confidence.py does.
usage (from the code copy's root, with the model's own interpreter):
  python scripts/review_checks/ff_posrule_confidence.py POS OUT_NAME --split test --source-run S --run-dir E_RUN --e-on E2
         [--c-run-dir C_RUN --c-e-on E1] --el 16 --cl 19 --es 3.5 --ks 1 [--first-batches K]
output: <E_RUN>/eval_extra/confidence_<OUT_NAME>.json with targets conf_E<s>
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import e44_split_test as S  # noqa: E402

ALLOWED_POS = ("prompt", "gen")


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ALLOWED_POS:
        raise SystemExit(__doc__)
    pos, argv = sys.argv[1], sys.argv[2:]
    vals = S.split_values(argv)
    split = vals.get("--split", ["tune"])
    if "--splits" in vals or split not in (["tune"], ["test"]):
        raise SystemExit("give '--split tune' (the default) or '--split test', and no --splits (got %s)" % vals)
    if split == ["test"]:
        S.patch()
    import ff_explore_ec_confidence as C
    if Path(C.__file__).resolve().parent != HERE:
        raise SystemExit("imported %s, expected the confidence script next to this entry" % C.__file__)

    def targets(el, cl, es, ks, split="tune"):
        out = []
        for s in es:
            e = "E%d.w.p%s" % (el, s)
            out.append(("conf_E%s" % s, "conf", [e], "conf--%s-at-%s_%s.json" % (C.part_token([e]), pos, split), ("frozen", [])))
        return out

    C.build_targets = targets
    fd = C.fd
    original_mask = fd.position_mask

    def mask(packed, positions):
        if positions != "content":
            raise SystemExit("the confidence script asked for the %r mask; this entry only replaces 'content'" % positions)
        return original_mask(packed, pos)

    fd.position_mask = mask
    print("[posrule-confidence] split %s, writes scored at %r; confidence script sha256 %s, entry sha256 %s" % (
        split[0], pos, S.sha(C.__file__), S.sha(__file__)), flush=True)
    sys.argv = [C.__file__] + argv
    C.main()


if __name__ == "__main__":
    main()
