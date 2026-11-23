# Iteration_01 (Claude Opus 5.5, executor F, 2026-11-24; plan discussion/work_queue/2026-11-24_queue_m.md section 1):
# sign check of the C write of the E44 engine on the first N tune questions of one layer, read-only, numbers only.
# "toward caution" = C_L<l>_x-w-p0-25 must not answer more questions than the frozen answers (no write);
# "toward answering" = C_L<l>_x-w-m0-25 must not answer fewer. Answered = the rule detector does not flag the answer
# and the answer is not empty (an empty answer counts as an abstention, as the judges count it).
# usage: python3 crow_sign.py GEN_DIR LAYER [N]     exit 0 = signs as expected, 1 = a violation, 2 = bad input
import json
import os
import sys


def answered(text, refusal):
    return (not refusal) and bool(text.strip())


def main():
    if len(sys.argv) not in (3, 4):
        print(__doc__ if __doc__ else "usage: crow_sign.py GEN_DIR LAYER [N]")
        return 2
    gen, layer = sys.argv[1], int(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) == 4 else 16
    ok = True
    for arm, want in (("p0-25", "le"), ("m0-25", "ge")):
        p = os.path.join(gen, "C_L%d_x-w-%s_tune.json" % (layer, arm))
        if not os.path.exists(p):
            print("missing %s" % p)
            return 2
        d = json.load(open(p, encoding="utf-8"))
        if not d.get("completed") or d.get("pilot"):
            print("%s is not a complete full-run file" % p)
            return 2
        rows = d["rows"]
        b16 = sum(answered(r["baseline_text"], r["baseline_refusal_rule"]) for r in rows[:n])
        a16 = sum(answered(r["text"], r["refusal_rule"]) for r in rows[:n])
        ball = sum(answered(r["baseline_text"], r["baseline_refusal_rule"]) for r in rows)
        aall = sum(answered(r["text"], r["refusal_rule"]) for r in rows)
        good = a16 <= b16 if want == "le" else a16 >= b16
        ok &= good
        print("L%d x.w.%s: first %d questions answered %d -> %d; all %d questions %d -> %d [%s]" % (
            layer, arm.replace("-", "."), n, b16, a16, len(rows), ball, aall, "ok" if good else "SIGN VIOLATION"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
