# Iteration_01 (Claude Opus 5.5, executor H, 2026-11-24; plan discussion/work_queue/2026-11-24_queue_m.md section 4.2):
# the C row of the free-form 2x2 per model, per layer, per arm, DeepSeek Pro only, second stage. Read-only, no model,
# no judge call. Everything is computed by c_row_layers.py (imported, unchanged); only two things differ:
#   (a) the tune layer list of each model is all its layers (num_layers of the model config of its code copy);
#   (b) on the tune split a layer's work dir is <run>/judge_tune_crow2_C when that dir's judge input holds this layer's
#       arms; otherwise what c_row_layers.work_of returns -- except for a layer that stage one never listed (it has no
#       stage-one work dir): that one is pointed at judge_tune_crow2_C as well, so it prints "no Pro verdicts yet" until
#       the crow2 Pro pass is there.
# modes (scoring container, from /workspace/Truth_Decoupling):
#   python3 -B Iteration_01/diag/c_row_layers2.py tune|test|all [KEY ...]     KEY = llama q9 q27 g12 g27 (default all)
#   python3 -B Iteration_01/diag/c_row_layers2.py check ARCHIVE [KEY ...]
#       every tune row of the stage-one printout ARCHIVE (c_row_layers_five_models_20261124.txt) against this reader,
#       cell by cell; the columns "beats random A / B", "status A / B" and "arm minus random" are reported apart (they
#       change once the random arms are judged). Exit 1 if any other cell differs or a row is missing.
import json
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import c_row_layers as C  # noqa: E402

CROW2 = "judge_tune_crow2_C"
CFG = {C.LL: "code_e45_20261112/configs/models/llama3.1-8b.json",
       C.Q9: "code_ff_explore_20261113/configs/models/qwen3.5-9b.json",
       C.Q27: "code_ff_explore_20261113b/configs/models/qwen3.8-27b-dm.json",
       C.G12: "Iteration_01/code/configs/models/gemma3-12b.json",
       C.G27: "Iteration_01/code/configs/models/gemma3-27b.json"}
KEYS = {"llama": "Llama-3.1-8B", "q9": "Qwen3.5-9B", "q27": "Qwen3.8-27B", "g12": "Gemma-3-12B", "g27": "Gemma-3-27B"}
STAGE1 = {run: set(tl) for _, run, tl, _, _, _ in C.MODELS}
OLD_WORK_OF = C.work_of
RANDOM_COLS = (13, 15, 16)   # beats random A / B, status A / B, arm minus random (index in the "|"-split row)
_CONDS = {}


def num_layers(run):
    return int(json.load(open(CFG[run]))["num_layers"])


def crow2_conditions(run):
    if run not in _CONDS:
        p = os.path.join(run, CROW2, "judge_input.json")
        _CONDS[run] = set(json.load(open(p, encoding="utf-8"))["conditions"]) if os.path.exists(p) else set()
    return _CONDS[run]


def work_of(run, split, layer, own):
    if split == "tune":
        if C.T.cond("C", layer, "x.w.p0.25") in crow2_conditions(run) or layer not in STAGE1[run]:
            return "%s/%s" % (run, CROW2)
    return OLD_WORK_OF(run, split, layer, own)


def install(keys):
    names = {KEYS[k] for k in keys}
    C.MODELS = tuple((name, run, list(range(num_layers(run))), own, xl, n1)
                     for name, run, tl, own, xl, n1 in C.MODELS if name in names)
    C.work_of = work_of


def cells(row):
    return [c.strip() for c in row.split("|")]


def check(archive):
    rows, inblock = [], False
    for line in open(archive, encoding="utf-8"):
        if line.startswith("## "):
            inblock = line.startswith("## tune split")
        elif inblock and line.startswith("| ") and not line.startswith("| model |"):
            rows.append(line.rstrip("\n"))
    names = {m[0] for m in C.MODELS}
    rows = [r for r in rows if cells(r)[1] in names]
    own = {m[0]: (m[1], m[3]) for m in C.MODELS}
    mine = {}
    for name, layer in sorted({(cells(r)[1], int(cells(r)[2])) for r in rows}):
        run, o = own[name]
        for r in C.rows_for(name, run, layer, "tune", o):
            c = cells(r)
            mine[(c[1], c[2], c[3])] = c
    same_all = same_rest = 0
    bad, rand_changed = [], []
    for r in rows:
        a = cells(r)
        b = mine.get((a[1], a[2], a[3]))
        if b is None or len(b) != len(a):
            bad.append("%s L%s %s: row missing or different shape" % (a[1], a[2], a[3]))
            continue
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        if not diff:
            same_all += 1
        if not [i for i in diff if i not in RANDOM_COLS]:
            same_rest += 1
            if diff:
                rand_changed.append("%s L%s %s" % (a[1], a[2], a[3]))
        else:
            bad.append("%s L%s %s: columns %s differ" % (a[1], a[2], a[3], [i for i in diff if i not in RANDOM_COLS]))
    print("check against %s (tune rows of %s)" % (archive, ", ".join(sorted(names))))
    print("archive rows %d; identical in every column %d; identical outside the random / status columns %d; "
          "of these, random / status columns now filled %d; other differences %d" % (
              len(rows), same_all, same_rest, len(rand_changed), len(bad)))
    for x in bad[:40]:
        print("  DIFF %s" % x)
    return 1 if bad else 0


def main():
    if len(sys.argv) < 2:
        print("usage: c_row_layers2.py tune|test|all [KEY ...] | check ARCHIVE [KEY ...]")
        return 2
    mode = sys.argv[1]
    if mode == "check":
        archive, keys = sys.argv[2], sys.argv[3:] or list(KEYS)
    else:
        archive, keys = None, sys.argv[2:] or list(KEYS)
    bad = [k for k in keys if k not in KEYS]
    if bad or mode not in ("tune", "test", "all", "check"):
        print("unknown mode or key: %s %s" % (mode, bad))
        return 2
    install(keys)
    if mode == "check":
        return check(archive)
    print("# C row of the free-form 2x2 by layer, all layers: plain prompt, C at 0.25, DeepSeek Pro (c_row_layers2.py over "
          "c_row_layers.py; tune work dir judge_tune_crow2_C where it holds the layer)\n")
    print("Tolerance 3%% of the questions (tune: 6 of 192 beyond, 5 inside; test: 16 of 523 beyond). Paired bootstrap %d "
          "resamples, seed %d. 'status' = met / not_met / incomplete (incomplete = no random arm, nothing else failed).\n" % (
              C.T.BOOT, C.T.SEED))
    for split in (("tune", "test") if mode == "all" else (mode,)):
        C.block(split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
