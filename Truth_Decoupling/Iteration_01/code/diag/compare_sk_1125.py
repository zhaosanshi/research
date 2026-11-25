# Iteration_01 (Claude Opus 5.5, executor yin, 2026-11-25 22 h; plan discussion/work_queue/2026-11-24_queue_m.md, progress
# entry 2026-11-25 22:11 part one, rules v4): CPU helper of crow2_checks_sk_1125.sh. Read-only, no model, no judge call.
# Numbers only (no question or answer text is printed). Modes (scoring container, from /workspace/Truth_Decoupling):
#   frozen A B    two frozen_<split>.json: same split, question order, prompts; per (question, prompt) the token ids, text,
#                 finished, stop id compared. Exit 1 if any record differs.
#   mc A B        two mc/base_<prompt>_<split>.json: question order and every field of every row compared exactly (floats
#                 with ==), summary and mc_contract compared without identity_sha256. Exit 1 if anything differs.
#   cos A.pt B.pt two direction files: cosine of w (and of w_md). Always exit 0 (a number to record).
#   losses A.json B.json  two train logs: last epoch of training.history and losses train/val "trained"; prints both and
#                 whether they are equal. Always exit 0.
#   contract A.json B.json  two train logs: the training_contract keys that differ (layer included). Always exit 0.
#   answers A.json B.json  two generation files (gen/C_L<l>_<arm>_tune.json): rows, answers identical as text, answered
#                 counts by the crow_sign.py rule. Exit 1 if the question order differs.
import json
import math
import sys


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def frozen(a, b):
    A, B = load(a), load(b)
    bad = []
    for k in ("phase", "split", "question_ids", "cohort_sha256", "prompt_systems", "user_template", "eot_ids",
              "max_new_tokens", "max_target_tokens", "completed", "execution_complete", "summary"):
        if A.get(k) != B.get(k):
            bad.append(k)
    qa, qb = A["questions"], B["questions"]
    n = tok = txt = full = 0
    prompts = sorted(qa[0]["answers"]) if qa else []
    if [q["pair_id"] for q in qa] != [q["pair_id"] for q in qb]:
        bad.append("question order")
    else:
        for x, y in zip(qa, qb):
            if set(x["answers"]) != set(y["answers"]):
                bad.append("prompt set of a question")
                continue
            if x.get("content_start") != y.get("content_start") or x.get("boundary_clean") != y.get("boundary_clean"):
                bad.append("content_start / boundary_clean of a question")
            for p in x["answers"]:
                r, s = x["answers"][p], y["answers"][p]
                n += 1
                tok += r["token_ids"] == s["token_ids"]
                txt += r["text"] == s["text"]
                full += r == s
    print("frozen split %s: questions %d, prompts %d; records compared %d; token ids identical %d; text identical %d; "
          "whole record identical %d; header fields differing %s; identity_sha256 %s vs %s" % (
              A.get("split"), len(qa), len(prompts), n, tok, txt, full, sorted(set(bad)) or "none",
              A.get("identity_sha256", "")[:16], B.get("identity_sha256", "")[:16]))
    return 0 if (not bad and n and tok == n and txt == n and full == n) else 1


def mc(a, b):
    A, B = load(a), load(b)
    bad = []
    for k in ("phase", "kind", "prompt", "split", "question_ids", "cohort_sha256", "pilot", "completed",
              "execution_complete", "summary"):
        if A.get(k) != B.get(k):
            bad.append(k)
    ca = {k: v for k, v in A.get("mc_contract", {}).items() if k != "identity_sha256"}
    cb = {k: v for k, v in B.get("mc_contract", {}).items() if k != "identity_sha256"}
    if ca != cb:
        bad.append("mc_contract")
    ra, rb = A["rows"], B["rows"]
    same = 0
    maxd = 0.0
    if [r["pair_id"] for r in ra] != [r["pair_id"] for r in rb]:
        bad.append("row order")
    else:
        for x, y in zip(ra, rb):
            same += x == y
            for k, v in x.items():
                w = y.get(k)
                if isinstance(v, float) and isinstance(w, float) and math.isfinite(v) and math.isfinite(w):
                    maxd = max(maxd, abs(v - w))
    print("baseline readout prompt %s split %s: rows %d; rows identical in every field %d; largest top-level float "
          "difference %.3g; header fields differing %s" % (A.get("prompt"), A.get("split"), len(ra), same, maxd,
                                                           sorted(set(bad)) or "none"))
    return 0 if (not bad and ra and same == len(ra)) else 1


def cos(a, b):
    import torch
    A = torch.load(a, map_location="cpu", weights_only=False)
    B = torch.load(b, map_location="cpu", weights_only=False)
    out = []
    for k in ("w", "w_md"):
        x, y = A[k].double(), B[k].double()
        out.append("%s %.6f" % (k, float(torch.dot(x, y) / (x.norm() * y.norm()))))
    print("cosine layer %s vs layer %s: %s; backward %s vs %s" % (
        A.get("layer"), B.get("layer"), "; ".join(out), A["training_contract"].get("backward", "-"),
        B["training_contract"].get("backward", "-")))
    return 0


def last(d):
    h = d["training"]["history"][-1]
    t = d["losses"]["train"]["trained"]
    v = d["losses"].get("val", {}).get("trained", {})
    return {"epoch": h.get("epoch"), "hist_train_cause": h.get("train_cause"), "hist_train_isolate": h.get("train_isolate"),
            "hist_val_cause": h.get("val_cause"), "hist_val_isolate": h.get("val_isolate"),
            "trained_train_cause": t.get("cause"), "trained_train_isolate": t.get("isolate"),
            "trained_val_cause": v.get("cause"), "trained_val_isolate": v.get("isolate")}


def losses(a, b):
    A, B = last(load(a)), last(load(b))
    parts = []
    for k in A:
        x, y = A[k], B[k]
        if isinstance(x, float) and isinstance(y, float):
            parts.append("%s %.6f / %.6f (diff %.2g)" % (k, x, y, x - y))
        else:
            parts.append("%s %s / %s" % (k, x, y))
    print("final losses (first file / second file): %s; all equal: %s" % ("; ".join(parts), "yes" if A == B else "no"))
    return 0


def contract(a, b):
    A, B = load(a)["training_contract"], load(b)["training_contract"]
    diff = sorted(k for k in set(A) | set(B) if A.get(k) != B.get(k))
    print("training_contract: %d / %d keys; differing %s" % (len(A), len(B), diff or "none"))
    return 0


def answered(text, refusal):
    return (not refusal) and bool(text.strip())


def answers(a, b):
    A, B = load(a), load(b)
    ra, rb = A["rows"], B["rows"]
    if [r["pair_id"] for r in ra] != [r["pair_id"] for r in rb]:
        print("answers: question order differs")
        return 1
    same = sum(x["text"] == y["text"] for x, y in zip(ra, rb))
    na = sum(answered(r["text"], r["refusal_rule"]) for r in ra)
    nb = sum(answered(r["text"], r["refusal_rule"]) for r in rb)
    base_same = sum(x["baseline_text"] == y["baseline_text"] for x, y in zip(ra, rb))
    print("answers %s: rows %d; identical as text %d; answered (crow_sign rule) first %d second %d; baseline text "
          "identical %d" % (a.rsplit("/", 1)[-1], len(ra), same, na, nb, base_same))
    return 0


MODES = {"frozen": frozen, "mc": mc, "cos": cos, "losses": losses, "contract": contract, "answers": answers}


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in MODES:
        print("usage: compare_sk_1125.py frozen|mc|cos|losses|contract|answers A B")
        return 2
    return MODES[sys.argv[1]](sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    sys.exit(main())
