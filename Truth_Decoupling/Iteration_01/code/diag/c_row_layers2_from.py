# Iteration_01 (Claude Opus 5.5, executor gui, 2026-11-24 21:4x; plan discussion/work_queue/2026-11-24_queue_m.md,
# progress entry 21:35 point 1): the reader c_row_layers2.py (imported, unchanged) for a PARTIAL second stage of one
# model, in which the layers below MINL were not trained (gemma3-27b: one card cannot hold the training of layers 0-18).
# Read-only, no model, no judge call. Two things differ from c_row_layers2.py, nothing else:
#   (a) the second-stage tune work dir is <run>/judge_tune_<NAME> (the crow2_judge.sh export name of the partial run,
#       e.g. crow2_C_from20) instead of judge_tune_crow2_C;
#   (b) a tune layer below MINL that has no direction file <run>/dirs/C_L<l>.pt prints one row
#       "NOT TRAINED (one card cannot hold the training)" instead of "no Pro verdicts yet"; the header says so.
# modes (scoring container, from /workspace/Truth_Decoupling):
#   python3 -B Iteration_01/diag/c_row_layers2_from.py MINL NAME tune KEY
#   python3 -B Iteration_01/diag/c_row_layers2_from.py MINL NAME check ARCHIVE KEY
#       the check of c_row_layers2.py (stage-one tune rows of ARCHIVE, cell by cell), with (a) and (b) in place.
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import c_row_layers2 as C2  # noqa: E402

C = C2.C
OLD_ROWS_FOR = C.rows_for
MINL = None
NOTE = "NOT TRAINED (one card cannot hold the training)"


def rows_for(name, run, layer, split, own):
    if split == "tune" and layer < MINL and not os.path.exists(os.path.join(run, "dirs", "C_L%d.pt" % layer)):
        return ["| %s | %d | - | - | %s |" % (name, layer, NOTE)]
    return OLD_ROWS_FOR(name, run, layer, split, own)


def main():
    global MINL
    a = sys.argv[1:]
    if len(a) < 4 or not a[0].isdigit() or not a[1].replace("_", "").isalnum() or a[2] not in ("tune", "check"):
        print("usage: c_row_layers2_from.py MINL NAME tune KEY | MINL NAME check ARCHIVE KEY")
        return 2
    MINL, name, mode = int(a[0]), a[1], a[2]
    archive, keys = (a[3], a[4:]) if mode == "check" else (None, a[3:])
    if len(keys) != 1 or keys[0] not in C2.KEYS:
        print("exactly one KEY of %s" % sorted(C2.KEYS))
        return 2
    C2.CROW2 = "judge_tune_%s" % name
    C2.install(keys)
    C.rows_for = rows_for
    run = C.MODELS[0][1]
    out = [l for l in range(C2.num_layers(run)) if l < MINL and not os.path.exists(os.path.join(run, "dirs", "C_L%d.pt" % l))]
    if mode == "check":
        return C2.check(archive)
    print("# C row of the free-form 2x2 by layer, PARTIAL second stage: plain prompt, C at 0.25, DeepSeek Pro "
          "(c_row_layers2_from.py over c_row_layers2.py / c_row_layers.py; tune work dir %s where it holds the layer)\n" % C2.CROW2)
    print("%s: layers %s were NOT TRAINED (one card cannot hold the training of these layers; plan queue_m, 2026-11-24 "
          "21:35). Their rows below say so; they are not results.\n" % (C2.KEYS[keys[0]], (
              "%d-%d" % (out[0], out[-1]) if out and out == list(range(out[0], out[-1] + 1)) else out) if out else "none"))
    print("Tolerance 3%% of the questions (tune: 6 of 192 beyond, 5 inside; test: 16 of 523 beyond). Paired bootstrap %d "
          "resamples, seed %d. 'status' = met / not_met / incomplete (incomplete = no random arm, nothing else failed).\n" % (
              C.T.BOOT, C.T.SEED))
    C.block("tune")
    return 0


if __name__ == "__main__":
    sys.exit(main())
