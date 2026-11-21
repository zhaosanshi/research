"""CPU-only review of the five frozen GSM8K settings (2026-11-21, 89aedc1).

From Truth_Decoupling in the scoring container:
    python3 -B PATH/2026-11-21_gsm_bounds.py [--trace-check]

Reads the existing neutral single-E files and their common baseline, verifies
input identities and reuses the frozen parser. N1 is bounded, not generated.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("Iteration_01")
CODE = ROOT / "code"
CAP = ROOT / "capability"
MODELS_CFG = CODE / "configs/newsets/frozen_models_20261119.json"
GOLD = Path("data_external/gsm8k_20261119/main/test-00000-of-00001.parquet")
IDS = CAP / "sets/gsm8k_1319_20261119.json"
# model: (layer, E strength, cap tokens, C cap of own blanks, cap of E-created blanks).
# These are the selected rows of main_table_prompt_budget1_20261121.txt.
SETTINGS = {
    "llama3.1-8b": (16, "3.5", 512, 2, 1),
    "qwen3.5-9b": (15, "2", 1024, .5, .25),
    "qwen3.8-27b": (28, "1.75", 512, 1, .5),
    "gemma3-12b": (24, "2.5", 512, 2, 2),
    "gemma3-27b": (28, "1", 512, 2, 1),
}
RUNGS = (.25, .5, 1, 2)
ROUNDS = 4000
SEED = 20261121


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(ok, message):
    if not ok:
        raise ValueError(message)


def score(row, target, parser):
    blank = bool(row["rule_abstain"]) or not row["text"].strip()
    answer, why = (None, "refusal") if blank else parser.gsm8k_parse(row["text"], row["finished"])
    return [int(answer is not None and abs(answer - target) < 1e-6), int(answer is None),
            int(blank), int(not row["finished"]), int(why == "truncated")]


def bounds(weights, differences, divisor=1):
    """Resample the same question index for all models; report percentage points."""
    scale = 100.0 / (len(differences) * divisor)
    sums = (weights @ differences) * scale
    lo, hi = np.percentile(sums, [2.5, 97.5])
    return "Δ%+.3f [%+.3f,%+.3f]" % (sum(differences) * scale, lo, hi)


def audit_gsm():
    import pyarrow.parquet as pq

    sys.path.insert(0, str((CODE / "scripts/capability").resolve()))
    import cap_prompts

    manifest = read(IDS)
    ids = manifest["ids"]
    ids_sha, gold_sha = sha(IDS), sha(GOLD)
    verify(manifest["source_file_sha256"] == gold_sha and len(ids) == len(set(ids)) == 1319, "GSM8K identity")
    original = pq.read_table(GOLD, columns=["answer"]).to_pylist()
    targets = [cap_prompts.gsm8k_gold(original[int(q.split(":")[1])]["answer"]) for q in ids]
    n = len(ids)
    weights = np.random.default_rng(SEED).multinomial(n, np.full(n, 1 / n), ROUNDS)
    all_deltas = []
    for model, (layer, strength, cap, own_cap, new_cap) in SETTINGS.items():
        root = CAP / ("gen_main" if cap == 512 else f"gen_main{cap}") / "gsm8k" / model / "b8"
        old_arm = "neutral:E%d.w.p%s" % (layer, strength)
        stem = "neutral--E%d-w-p%s" % (layer, strength.replace(".", "-"))
        plans = [("neutral", "neutral", None), (stem, old_arm, "content"), (stem + "-at-prompt", old_arm, "prompt")]
        states, contracts = [], []
        for fname, arm, rule in plans:
            g = read(root / (fname + ".json"))
            c = g["contract"]
            expect = dict(model=model, task="gsm8k", split="all", arm=arm, base_prompt="neutral",
                          positions=rule, max_new_tokens=cap, batch_size=8, full_pass=True,
                          ids=ids, ids_file_sha256=ids_sha, items_file_sha256=gold_sha)
            verify(g.get("completed") is True and g.get("gen_version") == 2, "incomplete or foreign " + fname)
            verify(all(c.get(k) == v for k, v in expect.items()), "wrong condition " + fname)
            verify(all(g[k] == v for k, v in c.items()), "top-level metadata differs: " + fname)
            verify([r["id"] for r in g["rows"]] == ids, "wrong order or duplicates: " + fname)
            if rule:
                part, = c["parts"]
                verify(part["family"] == "E" and part["layer"] == layer and part["beta"] == float(strength), "wrong E")
            states.append(np.array([score(row, target, cap_prompts) for row, target in zip(g["rows"], targets)], dtype=int))
            contracts.append(c)
        for key in ("identity_sha256", "frozen_file_sha256", "system_text", "user_template"):
            verify(len({c[key] for c in contracts}) == 1, "incompatible inputs: " + key)
        verify(contracts[1]["parts"] == contracts[2]["parts"], "changed direction or write amount")
        x, old, prompt = states
        accuracy = prompt[:, 0] - x[:, 0]
        improvement = prompt[:, 0] - old[:, 0]
        no_answer = prompt[:, 1] - x[:, 1]
        all_deltas.append((accuracy, improvement, no_answer))
        blank = prompt[:, 2].astype(bool)
        verify(not prompt[blank, 0].any(), "candidate blank already scored correct")
        own = int(x[blank, 2].sum())
        count = int(blank.sum())
        passes = own * RUNGS.index(own_cap) + (count - own) * RUNGS.index(new_cap) + count
        print(model, "counts [right, no answer, rule blank, reached token cap, truncated without final answer]")
        print(" baseline", x.sum(axis=0).tolist(), "content", old.sum(axis=0).tolist(), "prompt", prompt.sum(axis=0).tolist())
        print(" prompt-base accuracy", bounds(weights, accuracy), "no answer", bounds(weights, no_answer),
              "prompt-content accuracy", bounds(weights, improvement))
        print(" N1 possible Δaccuracy in this sample [%+.3f,%+.3f] points; at most %d extra generations" %
              (100 * accuracy.sum() / n, 100 * (accuracy.sum() + count) / n, passes))
        both = (old[:, 1] == 0) & (prompt[:, 1] == 0)
        print(" prompt-content net change in number correct: both parsed", int(improvement[both].sum()),
              "on", int(both.sum()), "items; other items", int(improvement[~both].sum()))
        print(" optimistic N1 if every candidate is correctly answered:", bounds(weights, accuracy + blank.astype(int)))
        print(" candidate IDs", [ids[i] for i in np.flatnonzero(blank)], "own", own, "E-created", count - own)
    for count in (3, 5):
        columns = zip(*all_deltas[:count])
        print("first", count, "models, avg prompt-base accuracy / prompt-content accuracy / prompt-base no answer",
              *[bounds(weights, sum(col), count) for col in columns])


def trace_check():
    config = read(MODELS_CFG)["models"]
    for name in SETTINGS:
        run = Path(config[name]["run_dir"]) / "eval_extra"
        files = rows = writes = paired_files = paired_hooks = old_decode_hooks = 0
        for path in sorted((run / "gen").glob("*-at-prompt_test.json")):
            g = read(path)
            if g["partial"]:
                continue
            verify(g["positions"] == "prompt" and g["completed"] and g["execution_complete"], path.name)
            verify(g["question_ids"] == [r["pair_id"] for r in g["rows"]] and len(set(g["question_ids"])) == len(g["rows"]), path.name)
            previous = path.with_name(path.name.replace("-at-prompt_test.json", "_test.json"))
            old = read(previous) if previous.exists() else None
            if old is not None:
                verify(old["positions"] == "content" and old["completed"] and
                       all(old[k] == g[k] for k in ("question_ids", "arm", "identity_sha256", "parts")), previous.name)
                verify(old["question_ids"] == [r["pair_id"] for r in old["rows"]], previous.name)
                paired_files += 1
            for i, r in enumerate(g["rows"]):
                hooks = r["write"] if isinstance(r["write"], list) else [r["write"]]
                if old is not None:
                    before = old["rows"][i]["write"]
                    old_hooks = before if isinstance(before, list) else [before]
                    verify(len(old_hooks) == len(hooks), previous.name)
                    for a, b in zip(old_hooks, hooks):
                        verify(all(a[k] == b[k] for k in ("prefill_positions", "prefill_stored_norm")), previous.name)
                        paired_hooks += 1
                        old_decode_hooks += int(a["decode_steps_own"] > 0)
                verify(len(hooks) == len(g["parts"]) > 0, path.name)
                verify(len({h["prefill_positions"] for h in hooks}) == 1, path.name)
                for h in hooks:
                    verify(h["prefill_positions"] > 0 and h["prefill_stored_norm"] > 0 and
                           h["decode_steps_total"] == h["decode_steps_own"] == 0, path.name)
                writes += len(hooks)
            files += 1
            rows += len(g["rows"])
        verify(files > 0, f"no test files for {name}")
        print(name, files, "full test arms", rows, "rows", writes, "hook records; no decode writes;",
              paired_files, "matched content-rule arms", paired_hooks, "equal prefill records,",
              old_decode_hooks, "matched content-rule records with nonzero decode writes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-check", action="store_true")
    args = parser.parse_args()
    audit_gsm()
    if args.trace_check:
        trace_check()
