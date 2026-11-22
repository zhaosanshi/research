# Iteration_01 analysis (Claude Opus 5.5, 2026-11-22), read-only, no model, no judge call. Exploration.
# GSM8K (1,319 items, rule scoring with diag/cap_score.py's own functions, batch 8, neutral system prompt) under the CAA
# baseline at its chosen points: scripts/newsets/ns_positions.py gen gen (newsets_gen.py unchanged) with the arm
# neutral:E<l>.md.p<m> on the mirror run directory (baselines/caa/<MODEL>/models_run_caa.json), files
# baselines/caa/<MODEL>/gsm8k/neutral--E<l>-md-p<m>-at-gen.json, against the model's usual no-write pass at the same
# token cap. Same lines and evidence wording as diag/iti_gsm8k.py (whose line() is reused); the project's single E passes
# are printed for reference, as there.
# usage (in the scoring container, from the project root, for pyarrow):
#   docker exec -u 1000:1001 -w /workspace/Truth_Decoupling truth_dec python3 -B Iteration_01/diag/caa_gsm8k.py MODEL [--cap N] LAYER:MULT [LAYER:MULT ...]
import glob
import json
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import cap_score as CS
import iti_gsm8k as IG

IDS = IG.IDS
BASE = "Iteration_01/baselines/caa"


def read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    argv = sys.argv[1:]
    cap = 512
    if "--cap" in argv:
        i = argv.index("--cap")
        cap = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    if len(argv) < 2:
        raise SystemExit("usage: caa_gsm8k.py MODEL [--cap N] LAYER:MULT [LAYER:MULT ...]")
    model, points = argv[0], argv[1:]
    root = "Iteration_01/capability/gen_main%s/gsm8k" % ("" if cap == 512 else str(cap))
    ids = read(IDS)["ids"]
    gold = CS.gold_of("gsm8k")
    g0 = read("%s/%s/b8/neutral.json" % (root, model))
    c0 = g0.get("contract") or {}
    if not (g0.get("completed") and c0.get("full_pass") and c0.get("model") == model and c0.get("max_new_tokens") == cap
            and c0.get("batch_size") == 8 and [r["id"] for r in g0["rows"]] == ids):
        raise SystemExit("the usual no-write pass is not the complete %d-token, batch-8 pass of this model" % cap)
    base = {r["id"]: CS.judge_row("gsm8k", r, gold(r["id"])) for r in g0["rows"]}
    meta = read("%s/%s/vectors.pt.json" % (BASE, model))
    cfg = read("%s/%s/models_run_caa.json" % (BASE, model))
    listing = read(cfg["models"][model]["run_dir"] + "/MIRROR.json")
    n = len(ids)
    print("# GSM8K, %d items, rule scoring, token cap %d; %s: no write accuracy %.1f%%, no answer %.1f%% (%s); changes are against this pass" % (
        n, cap, model, 100.0 * sum(base[i][0] == "R" for i in ids) / n, 100.0 * sum(base[i][0] == "X" for i in ids) / n, IG.reasons(base)))
    for pt in points:
        layer, m = int(pt.split(":")[0]), float(pt.split(":")[1])
        name = "neutral--E%d-md-p%s-at-gen.json" % (layer, ("%g" % m).replace(".", "-"))
        g = read("%s/%s/gsm8k/%s" % (BASE, model, name))
        c = g.get("contract") or {}
        parts = c.get("parts") or []
        if not (g.get("completed") is True and c.get("full_pass") is True and c.get("model") == model and c.get("task") == "gsm8k"
                and c.get("max_new_tokens") == cap and c.get("batch_size") == 8 and c.get("positions") == "gen"
                and c.get("ids") == ids and [r["id"] for r in g["rows"]] == ids and c.get("ids_file_sha256") == CS.sha_file(IDS)
                and c.get("system_text") == c0.get("system_text") and c.get("arm") == "neutral:E%d.md.p%s" % (layer, "%g" % m)):
            raise SystemExit("%s is not a complete GSM8K pass of this model under the same prompt and cap at the rule gen" % name)
        if (len(parts) != 1 or parts[0]["kind"] != "md" or parts[0]["layer"] != layer
                or parts[0]["direction_file_sha256"] != listing["layers"][str(layer)]["file_sha256"]
                or abs(parts[0]["amount"] - m * meta["norms"][layer]) > 1e-5 * max(1.0, m * meta["norms"][layer])):
            raise SystemExit("%s does not write the CAA vector of layer %d times %g" % (name, layer, m))
        judged = {r["id"]: CS.judge_row("gsm8k", r, gold(r["id"])) for r in g["rows"]}
        IG.line("CAA layer %d multiplier %g (rule gen)" % (layer, m), judged, base, ids, g["rows"])
    for p in sorted(glob.glob("%s/%s/b8/neutral--E*.json" % (root, model))):
        g = read(p)
        c = g.get("contract") or {}
        if not (g.get("completed") and c.get("full_pass") and c.get("max_new_tokens") == cap and [r["id"] for r in g["rows"]] == ids):
            continue
        judged = {r["id"]: CS.judge_row("gsm8k", r, gold(r["id"])) for r in g["rows"]}
        IG.line("the project's %s at %s" % (c["arm"].split(":")[1], c.get("positions") or "content"), judged, base, ids, g["rows"])
    print("(file names: %s)" % ", ".join(os.path.basename(p) for p in sorted(glob.glob("%s/%s/b8/neutral--E*.json" % (root, model)))))


if __name__ == "__main__":
    main()
