#!/usr/bin/env python3
"""Read-only: per layer of a C-row run, what the training of the C direction ended with and how large the write of the
C arms at 0.25 was (main session, Claude Fable 5.2, 2026-11-26; plan discussion/work_queue/2026-11-24_queue_m.md).

For each layer that has a train log:
  * gap: mean difference of the two prompts along the trained direction (the unit of the write strength);
  * the validation losses of the cause part and of the isolate part after the first and after the last epoch;
  * for the arms x.w.p0.25 (toward caution), x.w.m0.25 (toward answering) and x.rand.p0.25 (random direction, same
    size): the number of the tune answers whose text differs from the answer without a write, the mean size of the
    write and its size relative to the hidden state (the generation's own summary fields changed, amount_mean,
    prefill_rel_mean).
Nothing is computed from model outputs here; the numbers are copied from train_logs/C_L<l>.json and gen/C_L<l>_*.json.

usage (from the project root, no card):
  python3 -B Iteration_01/diag/c_layer_training_look_1126.py LABEL=RUN_DIR[:LAYERS] [LABEL=RUN_DIR[:LAYERS] ...]
LAYERS: comma-separated layers and A-B ranges; default: every layer with a train log.
"""
import json
import os
import re
import sys

ARMS = (("caution", "x-w-p0-25"), ("answering", "x-w-m0-25"), ("random", "x-rand-p0-25"))


def load(path):
    with open(path) as f:
        return json.load(f)


def layers_of(root, spec):
    have = sorted(int(m.group(1)) for m in (re.match(r"^C_L(\d+)\.json$", n) for n in os.listdir(os.path.join(root, "train_logs"))) if m)
    if not spec:
        return have
    want = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            want.extend(range(int(a), int(b) + 1))
        else:
            want.append(int(part))
    missing = [l for l in want if l not in have]
    if missing:
        raise SystemExit("no train log for layers %s under %s" % (missing, root))
    return want


def row(label, root, layer):
    d = load(os.path.join(root, "train_logs", "C_L%d.json" % layer))
    hist = d["training"]["history"]
    first, last = hist[0], hist[-1]
    cells = ["%s" % label, "%d" % layer, "%.2f" % d["gap"], "%d / %d" % (first["epoch"], last["epoch"]),
             "%.4f -> %.4f" % (first["val_cause"], last["val_cause"]), "%.4f -> %.4f" % (first["val_isolate"], last["val_isolate"])]
    rel = None
    for _, token in ARMS:
        path = os.path.join(root, "gen", "C_L%d_%s_tune.json" % (layer, token))
        if not os.path.isfile(path):
            cells.append("-")
            continue
        g = load(path)
        s = g["summary"]
        if not g.get("completed") or s["n"] != 192:
            raise SystemExit("%s is not a complete tune file of 192 rows" % path)
        cells.append("%d" % s["changed"])
        rel = s["prefill_rel_mean"] if rel is None else rel
    cells.append("-" if rel is None else "%.4f" % rel)
    return "| " + " | ".join(cells) + " |"


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    print("# C direction by layer: validation losses after training and how many of the 192 tune answers a write of 0.25 changes as text")
    print("# (c_layer_training_look_1126.py; numbers copied from train_logs/ and gen/ of the run directories named below)")
    for arg in sys.argv[1:]:
        label, rest = arg.split("=", 1)
        root, _, spec = rest.partition(":")
        print("# %s = %s%s" % (label, root, (" layers " + spec) if spec else ""))
    print()
    print("| run | layer | gap | epochs first / last | validation loss, cause part: first -> last epoch | validation loss, isolate part: first -> last epoch "
          "| answers changed as text, toward caution | toward answering | random direction (+) | write relative to the hidden state |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for arg in sys.argv[1:]:
        label, rest = arg.split("=", 1)
        root, _, spec = rest.partition(":")
        for layer in layers_of(root, spec):
            print(row(label, root, layer))


if __name__ == "__main__":
    main()
