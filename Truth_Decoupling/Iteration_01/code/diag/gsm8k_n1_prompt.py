# Iteration_01 analysis (Claude Fable 5.2, 2026-11-21), no model, no judge call.
# GSM8K, the full N1 at the write rule "prompt" from actual answers (GPT's review bounded it; this runs it): N1 changes
# only the items that are BLANK under single E (the rule detector's abstention or an empty answer). Those items are
# re-asked with E + C at 0.25 / 0.5 / 1 / 2, both writes at "prompt" (launchers/explore_20261120/gsm8k_n1_pos.sh), and
# N1 takes the first non-blank answer up to the cap of the item's kind - an item the model left blank without any
# write (own blank) or one it answered without a write and left blank under E (E's blank) - with the caps chosen on
# the TruthfulQA tune split at "prompt" (the main table's). An E's blank is released only with an answer that differs
# from the no-write answer (the veto of N1, texts compared after diag/release_origin.norm).
# Two modes:
#   ids MODEL_KEY [--cap N]     writes the blank ids under single E at "prompt" to <dir>/n1_blank_ids_at-prompt.json and
#                               prints their number (the launcher generates nothing when it is 0)
#   table [--cap-of MODEL N]    accuracy of no write, single E and N1 with paired changes (in the container: pyarrow)
# usage (from the project root): python3 -B Iteration_01/diag/gsm8k_n1_prompt.py ids llama3.1-8b
#                                python3 -B Iteration_01/diag/gsm8k_n1_prompt.py table
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "Iteration_01/diag")
sys.path.insert(0, "Iteration_01/code/scripts/newsets")
import newsets_gen as G

GOLD = "data_external/gsm8k_20261119/main/test-00000-of-00001.parquet"

# model key: (E layer, strength, C layer, cap for E's blanks, cap for own blanks, token cap of the passes)
SET = {
    "llama3.1-8b": (16, "3.5", 19, "1", "2", 512),
    "qwen3.5-9b": (15, "2", 16, "0.25", "0.5", 1024),
    "qwen3.8-27b": (28, "1.75", 33, "0.5", "1", 1024),   # 1,024 since the night of 2026-11-21: at 512 its solutions are cut off
    "gemma3-12b": (24, "2.5", 25, "2", "2", 512),
    "gemma3-27b": (28, "1", 31, "1", "2", 512),
}
LAD = ("0.25", "0.5", "1", "2")
IDS = "Iteration_01/capability/sets/gsm8k_1319_20261119.json"


def tok(arm):
    return arm.replace(":", "--").replace("+", "-and-").replace(".", "-")


def root(key):
    cap = SET[key][5]
    return "Iteration_01/capability/%s/gsm8k/%s/b8" % ("gen_main" if cap == 512 else "gen_main%d" % cap, key)


def load_arm(key, arm, pos, ids, subset=None):
    stem = tok(arm) + ("-at-" + pos if pos else "")
    if subset is not None:
        stem = "s" + G.ids_digest(ids)[:12] + "." + stem
    path = Path(root(key)) / (stem + ".json")
    g = json.loads(path.read_text(encoding="utf-8"))
    c = g["contract"]
    rows = g["rows"]
    if g.get("completed") is not True or len(set(ids)) != len(ids) or [r["id"] for r in rows] != ids:
        raise SystemExit("%s: incomplete, duplicate or foreign rows" % path)
    # With plain names disabled, the wrapper suffixes arm_token. The generator
    # uses that token inside a double-write condition name, but not a single one.
    condition = G.condition_name(arm, c["parts"])
    if "+" in arm and pos:
        condition += "-at-" + pos
    expected = dict(protocol=G.PROTOCOL, gen_version=G.GEN_VERSION, task="gsm8k", split="all",
                    model=key, ids=ids, condition=condition, items_file_sha256=G.sha_file(GOLD),
                    arm=arm, base_prompt="neutral", positions=pos, batch_size=8,
                    full_pass=subset is None, max_new_tokens=SET[key][5],
                    ids_file_sha256=G.sha_file(IDS),
                    only_ids_sha256=G.sha_file(subset) if subset is not None else None)
    if any(c.get(k) != v for k, v in expected.items()) or any(g.get(k) != v for k, v in c.items()):
        raise SystemExit("%s: wrong generation contract" % path)
    cfg = json.loads(Path("Iteration_01/code/configs/newsets/frozen_models_20261119.json").read_text())["models"][key]
    if c["model_config"] != cfg:
        raise SystemExit("%s: wrong frozen model setup" % path)
    parts = c["parts"]
    desc = [] if ":" not in arm else arm.split(":")[1].split("+")
    if len(parts) != len(desc):
        raise SystemExit("%s: number of writes differs from the arm" % path)
    for part, token in zip(parts, desc):
        layer_token, kind, signed = token.split(".", 2)
        family, layer = layer_token[0], int(layer_token[1:])
        beta = float(signed[1:]) * (-1 if signed.startswith("m") else 1)
        run = cfg.get("c_run_dir", cfg["run_dir"]) if family == "C" else cfg["run_dir"]
        direction = Path(run) / "dirs" / ("%s_L%d.pt" % (family, layer))
        if (part["family"], part["layer"], part["kind"], part["beta"], part["direction_file_sha256"]) != (
                family, layer, kind, beta, G.sha_file(direction)):
            raise SystemExit("%s: wrong direction, layer or strength" % path)
    return c, {r["id"]: r for r in g["rows"]}


def frozen_ids():
    with open(IDS, encoding="utf-8") as f:
        m = json.load(f)
    ids = m["ids"]
    if len(ids) != 1319 or len(set(ids)) != 1319 or m["source_file_sha256"] != G.sha_file(GOLD):
        raise SystemExit("incorrect GSM8K input or frozen 1,319-item list")
    return ids


def blank(r):
    return bool(r["rule_abstain"]) or not r["text"].strip()


def e_arm(key):
    el, s = SET[key][0], SET[key][1]
    return "neutral:E%d.w.p%s" % (el, s)


def ids_mode(key):
    order = frozen_ids()
    _, e = load_arm(key, e_arm(key), "prompt", order)
    b = [i for i in order if blank(e[i])]
    with open(root(key) + "/n1_blank_ids_at-prompt.json", "w", encoding="utf-8") as f:
        json.dump(b, f)
    el, s, cl, _, _, cap = SET[key]
    print(len(b), el, s, cl, cap, root(key))          # read by the launcher


def table():
    import cap_score as CS
    import release_origin as ro
    gold = CS.gold_of("gsm8k")
    order = frozen_ids()
    print("# GSM8K, 1,319 items: the full N1 at the prompt rule from actual re-asks (changes against no write, paired)")
    tot = {}
    for key, (el, s, cl, cap_e, cap_own, cap) in SET.items():
        try:
            bc, base = load_arm(key, "neutral", None, order)
            ec, e = load_arm(key, e_arm(key), "prompt", order)
            same = ("identity_sha256", "frozen_file_sha256", "model_config", "system_text", "user_template")
            if any(bc[k] != ec[k] for k in same):
                raise SystemExit("%s: base and E belong to different runs" % key)
        except FileNotFoundError as err:
            print("\n## %s: %s" % (key, err))
            continue
        b = [i for i in order if blank(e[i])]
        re = {}
        if b:
            subset = Path(root(key)) / "n1_blank_ids_at-prompt.json"
            if not subset.is_file():
                print("\n## %s: pending blank-ID list" % key)
                continue
            if json.loads(subset.read_text(encoding="utf-8")) != b:
                raise SystemExit("%s: saved re-ask IDs differ from E blanks" % key)
            required = [k for k in LAD if any(float(k) <= float(cap_own if blank(base[i]) else cap_e) for i in b)]
            for k in required:
                arm = "neutral:E%d.w.p%s+C%d.w.m%s" % (el, s, cl, k)
                try:
                    c, re[k] = load_arm(key, arm, "prompt", b, subset)
                except FileNotFoundError:
                    break
                parts = c["parts"]
                if (any(c[x] != ec[x] for x in (*same, "engine_sha256")) or len(parts) != 2
                        or parts[0] != ec["parts"][0] or parts[1]["family"] != "C"
                        or parts[1]["kind"] != "w" or parts[1]["layer"] != cl or parts[1]["beta"] != -float(k)):
                    raise SystemExit("%s %s: foreign re-ask direction or run" % (key, arm))
            if len(re) != len(required):
                print("\n## %s: %d of %d required re-ask strengths ready" % (key, len(re), len(required)))
                continue
        n1, notes = dict(e), []
        for i in b:
            own = blank(base[i])
            lad = LAD[:LAD.index(cap_own if own else cap_e) + 1]
            k = next((k for k in lad if not blank(re[k][i])), None)
            if k is None:
                notes.append("%s (%s blank): not released up to %s" % (i, "own" if own else "E's", lad[-1]))
                continue
            if not own and ro.norm(re[k][i]["text"]) == ro.norm(base[i]["text"]):
                notes.append("%s (E's blank): released at %s with the no-write answer again, kept blank (veto)" % (i, k))
                continue
            n1[i] = re[k][i]
            notes.append("%s (%s blank): released at C %s, %s" % (i, "own" if own else "E's", k, CS.judge_row("gsm8k", re[k][i], gold(i))[0]))
        st = {name: {i: CS.judge_row("gsm8k", rows[i], gold(i))[0] for i in order} for name, rows in (("no write", base), ("single E", e), ("N1", n1))}
        n = len(order)
        acc = {name: 100.0 * sum(v[i] == "R" for i in order) / n for name, v in st.items()}
        dn = CS.ci([(st["N1"][i] == "R") - (st["no write"][i] == "R") for i in order])
        de = CS.ci([(st["single E"][i] == "R") - (st["no write"][i] == "R") for i in order])
        print("\n## %s (E%d at %s, caps E's %s / own %s, token cap %d): no write %.1f%% | single E %.1f%% (%+.1f [%+.1f, %+.1f]) | N1 %.1f%% (%+.1f [%+.1f, %+.1f]) | blank under E %d" % (
            key, el, s, cap_e, cap_own, cap, acc["no write"], acc["single E"], 100 * de[0], 100 * de[1], 100 * de[2],
            acc["N1"], 100 * dn[0], 100 * dn[1], 100 * dn[2], len(b)))
        for x in notes:
            print("    " + x)
        tot[key] = [(st["N1"][i] == "R") - (st["no write"][i] == "R") for i in order]
    if len(tot) == len(SET):
        avg = [sum(tot[k][j] for k in tot) / len(tot) for j in range(len(order))]
        m, lo, hi = CS.ci(avg)
        print("\n## five models, average change of N1 against no write: %+.2f [%+.2f, %+.2f] points (items resampled, the five models together)" % (100 * m, 100 * lo, 100 * hi))


if __name__ == "__main__":
    if sys.argv[1:2] == ["ids"]:
        ids_mode(sys.argv[2])
    elif sys.argv[1:2] == ["table"]:
        table()
    else:
        raise SystemExit(__doc__ or "usage: ids MODEL | table")
