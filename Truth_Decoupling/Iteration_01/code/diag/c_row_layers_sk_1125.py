# Iteration_01 (Claude Opus 5.5, executor yin, 2026-11-25 22 h; plan discussion/work_queue/2026-11-24_queue_m.md, progress
# entry 2026-11-25 22:11 part one, rules v4): the per-layer C row table of c_row_layers.py (imported, unchanged; every
# number comes from its rows_for) for ONE run dir and ONE tune work dir, for a given list of layers. Read-only, no model,
# no judge call. Written for gemma3-27b layers 0-17 trained with the cause rows and the isolate rows back-propagated
# separately (results root Iteration_01/results/ff/gemma3/27b_splitkinds), plus the overlap layers 18, 20, 31.
# Two things differ from c_row_layers2.py: the model list is one entry (name Gemma-3-27B, the given run dir, the given
# layers, no proS dirs), and every tune layer's work dir is <run>/judge_tune_<NAME>.
# modes (scoring container, from /workspace/Truth_Decoupling):
#   python3 -B Iteration_01/diag/c_row_layers_sk_1125.py tune RUN NAME LAYERS [CHECKS_FILE]
#       LAYERS comma-separated. CHECKS_FILE (the out_split_kinds_checks_... results of crow2_checks_sk_1125.sh) is
#       printed in the header.
#   python3 -B Iteration_01/diag/c_row_layers_sk_1125.py check RUN NAME LAYERS TABLE
#       every row this reader prints for LAYERS against the rows of the same (model, layer, arm) in TABLE (an existing
#       printout, e.g. diag/out_c_row_layers2_from18_gemma3-27b_20261125.txt), character by character. Exit 1 on any
#       difference or missing row.
import sys

sys.path.insert(0, "Iteration_01/diag")
import c_row_layers as C  # noqa: E402

NAME = "Gemma-3-27B"


def setup(run, work, layers):
    C.MODELS = ((NAME, run, layers, (), (31,), 31),)
    C.work_of = lambda r, split, layer, own: "%s/judge_tune_%s" % (r, work)


def parse_layers(s):
    out = [int(x) for x in s.split(",") if x.strip()]
    if not out or len(set(out)) != len(out):
        raise SystemExit("LAYERS must be distinct comma-separated layer numbers")
    return out


def check(run, layers, table):
    want = {}
    for line in open(table, encoding="utf-8"):
        if line.startswith("| %s | " % NAME):
            c = [x.strip() for x in line.split("|")]
            want.setdefault((c[2], c[3]), line.rstrip("\n"))
    same, bad = 0, []
    for layer in layers:
        for r in C.rows_for(NAME, run, layer, "tune", ()):
            c = [x.strip() for x in r.split("|")]
            w = want.get((c[2], c[3]))
            if w == r:
                same += 1
            else:
                bad.append("L%s %s: %s" % (c[2], c[3], "missing in the table" if w is None else "differs"))
    print("check of layers %s against %s: rows identical %d; differing or missing %d" % (layers, table, same, len(bad)))
    for b in bad:
        print("  DIFF %s" % b)
    return 1 if bad or not same else 0


def main():
    a = sys.argv[1:]
    if len(a) not in (4, 5) or a[0] not in ("tune", "check") or (a[0] == "check" and len(a) != 5):
        print("usage: c_row_layers_sk_1125.py tune RUN NAME LAYERS [CHECKS_FILE] | check RUN NAME LAYERS TABLE")
        return 2
    mode, run, work, layers = a[0], a[1], a[2], parse_layers(a[3])
    if not work.replace("_", "").isalnum():
        print("NAME must be alphanumeric / underscore")
        return 2
    setup(run, work, layers)
    if mode == "check":
        return check(run, layers, a[4])
    print("# C row of the free-form 2x2 by layer, Gemma-3-27B layers trained with the cause part and the isolate part "
          "back-propagated separately: plain prompt, C at 0.25, DeepSeek Pro (c_row_layers_sk_1125.py over c_row_layers.py; "
          "run %s, tune work dir judge_tune_%s)\n" % (run, work))
    print("All layers below (%s) were trained with the patched code copy Iteration_01/code_splitkinds_20261125, switch "
          "--split-kinds: the cause rows and the isolate rows of a question are back-propagated one after the other and "
          "the gradients added (the same gradient in exact arithmetic), in the separate results root. Layers 18, 20, 31 "
          "are overlap layers: the main root's table (layers 18-61, trained in one pass) has its own rows for them.\n" % (
              ", ".join(str(x) for x in layers)))
    if len(a) == 5:
        print("Four pre-run checks (%s):\n" % a[4])
        for line in open(a[4], encoding="utf-8"):
            print("    " + line.rstrip("\n"))
        print()
    print("Tolerance 3%% of the questions (tune: 6 of 192 beyond, 5 inside; test: 16 of 523 beyond). Paired bootstrap %d "
          "resamples, seed %d. 'status' = met / not_met / incomplete (incomplete = no random arm, nothing else failed).\n" % (
              C.T.BOOT, C.T.SEED))
    C.block("tune")
    return 0


if __name__ == "__main__":
    sys.exit(main())
