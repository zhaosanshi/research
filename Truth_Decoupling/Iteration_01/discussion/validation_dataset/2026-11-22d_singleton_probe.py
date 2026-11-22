"""Read archived Llama candidate pairs without accessing labels or printing wording.
Host: python3 this_file.py prepare (CPU) or run-card1 (guarded). No generation or judge calls.
Run from the archive root Truth_Decoupling. run-card1 sends this script to the scoring
container only after both production launchers share the verified card-1 reservation.
"""
import argparse
import fcntl
from pathlib import Path
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, "Iteration_01/diag")
import selfjudge_probe as SP

MODEL = "llama3.1-8b"
ARCHIVE = Path(SP.OUT) / MODEL
SOURCES = (("TQA", "tq_release", "tq_forced"), ("QA2", "qq_release", "qq_forced"))
HASH = {
    "tq_release": "c0229163424f191651dcff0c70088a69e8ca7cab4386d7b611df7e16f3353ec9",
    "tq_forced": "cf61e8e921bb08abbb9c5e10a55ee86c499d67a539455f78f1e49e368f50e300",
    "qq_release": "9cafb48fb3b57d36ceb914fb0076162b3e9cccb15df53cd1112deb052fe34888",
    "qq_forced": "1bce4026298f69e1530efaeabc66165af4180f21727667527d3e3e291dd0f30f",
}
CONDS = ("baseline", "E_a3")
CHECK_CARD = "1"


def read_scores(src, task):
    items = ARCHIVE / (src + "_items.json")
    probe = ARCHIVE / (src + "_probe.npz")
    doc = SP.read_json(items)
    meta = SP.read_json(ARCHIVE / (src + "_probe.json"))
    if (doc["source"] != src or doc["model"] != MODEL or meta["model"] != MODEL
            or meta["items_sha256"] != SP.sha256_file(items) or HASH[src] != SP.sha256_file(probe)):
        raise ValueError(src)
    selected = {}
    with np.load(probe, allow_pickle=False) as z:
        rows = [x for x in doc["items"] if x["answered"]]
        if z["ids"].tolist() != [x["id"] for x in rows] or len(set(z["ids"])) != len(rows):
            raise ValueError(src + " row identity")
        for i, x in enumerate(rows):
            q = x["id"]
            if task == "TQA":
                split, q = q.split("|", 1)
                if split != "test":
                    continue
            elif x["pilot"]:
                continue
            stmt = SP.statement(x["question"], x["answer"])
            if (SP.sha1(x["answer"]) != x["answer_sha1"] or z["answer_sha1"][i] != x["answer_sha1"]
                    or z["statement_sha1"][i] != SP.sha1(stmt)):
                raise ValueError(src + " altered text")
            selected[q] = (x["question"], stmt, np.array([z[c + "__margin"][i] for c in CONDS]))
    return selected


def choose_pairs():
    # Do not use Pro labels, answer length, gap magnitude or gain to choose examples.
    by_task = []
    for task, inc, forced in SOURCES:
        a, b = read_scores(inc, task), read_scores(forced, task)
        found = []
        for q in sorted(set(a) & set(b)):
            if a[q][0] != b[q][0]:
                raise ValueError((task, q, "question mismatch"))
            delta = b[q][2] - a[q][2]
            if a[q][1] != b[q][1] and (delta[0] > 0) != (delta[1] > 0):
                found.append((task, q, (a[q][1], b[q][1]), np.stack((a[q][2], b[q][2]))))
        by_task.append(found)
    # First eligible ID in each bank, then the next TQA ID, then the next QA2 ID.
    selected = [items[i] for i in range(max(map(len, by_task))) for items in by_task if i < len(items)][:3]
    if len(selected) != 3 or len({s[0] for s in selected}) != 2:
        raise ValueError("three examples spanning both banks are required")
    return selected


def print_row(task, q, tag, both):
    delta = both[1] - both[0]
    print(task, q, tag, "forced-minus-C native / written", *(f"{g:+.12g}" for g in delta),
          "choices", *("forced" if g > 0 else "C" for g in delta), flush=True)


def score_singletons(selected):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != CHECK_CARD or os.environ.get("TD_SINGLETON_LOCK") != "held":
        raise SystemExit("invoke run-card1 on the host")
    cfg = SP.MODELS[MODEL]
    code = Path(cfg["code"]).resolve()
    sys.path[:0] = [str(code / "scripts"), str(code)]
    from common import load_split, tf_prompt
    from core.config import load_experiment, load_model_config
    from core.directions import load_das_direction, mean_diff_direction
    from core.intervene import SteerSpec, next_token_logits
    from core.io_utils import set_seed
    from core.model_io import capture_last_token, load_model
    from core.readout import build_label_token_map, label_logprobs_from_logits
    ref = SP.read_json(cfg["ref"] + ".json")
    for name, checksum in ref["code_sha256"].items():
        if SP.sha256_file(code / "scripts" / name) != checksum:
            raise ValueError(name)
    exp = load_experiment("E2b_tqa")
    set_seed(exp.get("seed", 42))
    bundle = load_model(load_model_config(MODEL))
    prompts = exp["prompts"]
    el = cfg["tf_layer"]
    bs = exp.get("batch_size", 8)
    est, refs = [], []
    for ds in exp["eval_datasets"]:
        est += load_split(ds, "calib") + load_split(ds, "dev")
        refs += load_split(ds, exp.get("eval_split", "test"))
    activated = []
    for allow in (False, True):
        texts = [tf_prompt(bundle, prompts, r["text"], allow_idk=allow) for r in est]
        h, _ = capture_last_token(bundle, texts, [el], components=("resid",), batch_size=bs)
        activated.append(h[("resid", el)].numpy().astype(np.float64))
    y = np.array([bool(r["label"]) for r in est])
    hb, hi = activated
    md, _ = mean_diff_direction(hb[y], hb[~y])
    direction, _ = load_das_direction(cfg["das"], "E", el, variant="tf-ce-bidir", align_to=md)
    c = (hi[y].mean(0) + hi[~y].mean(0)) / 2
    specs = (None, [SteerSpec(direction, el, mode="amplify", alpha=3.0, center=c)])
    ref_texts = [tf_prompt(bundle, prompts, r["text"], allow_idk=True) for r in refs[:bs]]
    token_map = build_label_token_map(bundle.tokenizer, ref_texts[0], prompts["statement"]["label_variants"])
    if {k: list(map(int, v)) for k, v in token_map.items()} != SP.read_json(ARCHIVE / "probe.json")["label_map"]:
        raise ValueError("different label token map")

    def margins(texts, spec):
        logits = next_token_logits(bundle, bundle.encode(texts), spec).numpy()
        probs = [label_logprobs_from_logits(row, token_map, use_abstain=True) for row in logits]
        return np.array([p["True"] - p["False"] for p in probs])

    with np.load(cfg["ref"] + ".npz", allow_pickle=False) as archive:
        for cond, spec in zip(CONDS, specs):
            if not np.allclose(margins(ref_texts, spec), archive[cond + "__margin"][:bs], rtol=0, atol=SP.TOL):
                raise ValueError("loaded operator disagrees with reference batch")
    for task, q, stmts, old in selected:
        prompts_ab = [tf_prompt(bundle, prompts, stmt, allow_idk=True) for stmt in stmts]
        repeated = []
        for rep, order in enumerate(((0, 1), (1, 0)), 1):
            scores = np.empty((2, 2))
            for i in order:
                for j, spec in enumerate(specs):
                    scores[i, j] = margins([prompts_ab[i]], spec)[0]
            if not np.isfinite(scores).all():
                raise ValueError("nonfinite singleton score")
            print_row(task, q, f"singleton round {rep}", scores)
            repeated.append(scores)
        choices = [(s[1] - s[0] > 0).tolist() for s in (old, *repeated)]
        print(task, q, "repeat choices agree", choices[1] == choices[2],
              "both preserve archive choices", choices[0] == choices[1] == choices[2],
              "max repeated score difference", float(np.abs(repeated[0] - repeated[1]).max()), flush=True)


def reserved_run():
    import re
    base = Path("Iteration_01/logs/newsets_20261119")
    lock_path = Path(SP.OUT) / ".gpu1.lock"
    def check_launchers():
        for name in ("selfjudge_1122.sh", "selfjudge_d_1122.sh"):
            text = (base / name).read_text()
            if ("LOCK=$HOST/Iteration_01/newsets/selfjudge/.gpu1.lock" not in text
                    or not re.search(r'^exec 9>"\$LOCK"(?:[ \t]+\|\|[^\n]+)?[ \t]*$', text, re.M)
                    or "flock -n 9 ||" not in text):
                raise SystemExit(f"{name}: no verified shared GPU-1 reservation; CPU input check only")
    check_launchers()
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_launchers()
        used = int(subprocess.check_output(["nvidia-smi", "-i", CHECK_CARD, "--query-gpu=memory.used",
                                            "--format=csv,noheader,nounits"], text=True).strip())
        if used + 17000 > 66560:
            raise SystemExit("insufficient reserved space on GPU 1")
        cmd = ["docker", "exec", "-i", "-u", "1000:1001", "-w", "/workspace/Truth_Decoupling"]
        base_path = "/workspace/Truth_Decoupling"
        env = {"TD_SINGLETON_LOCK": "held", "CUDA_VISIBLE_DEVICES": CHECK_CARD, "OMP_NUM_THREADS": "16",
               "TD_DATA_DIR": base_path + "/code/data/processed", "TD_MODEL_ROOT": base_path + "/models",
               "TD_RESULTS_DIR": base_path + "/Iteration_01/newsets/selfjudge/_e45_unused",
               "HF_HOME": base_path + "/hf_cache", "HOME": base_path + "/hf_cache", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        for k, v in env.items():
            cmd += ["-e", k + "=" + v]
        subprocess.run(cmd + ["truth_dec", "python3", "-B", "-", "score"], input=Path(__file__).read_bytes(), check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run-card1", "score"))
    action = parser.parse_args().action
    selected = choose_pairs()
    for task, q, _, old in selected:
        print_row(task, q, "archive", old)
    if action == "run-card1":
        reserved_run()
    elif action == "score":
        score_singletons(selected)


if __name__ == "__main__":
    main()
