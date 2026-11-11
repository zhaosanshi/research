# -*- coding: utf-8 -*-
"""Read-only debug (changes nothing, writes nothing): the natural phase of e44_eval_extra.py (4438a282) stopped with
'natural read position is misaligned for tqa:277'. Per training question this prints the reference read-position
coordinate (prompt-only forward, training-cohort batches of 8 - what the check uses), the same coordinate with batch 1,
and the teacher-forced read-position coordinate of natural_rows (batches of 8 over all questions), so that numerical
batch noise can be told apart from a real misalignment. Run inside the container on one GPU."""
import argparse, sys
from pathlib import Path
CODE = Path("/workspace/Truth_Decoupling/review_e44_claude/claude_gpt_review/from_claude/code")
sys.path.insert(0, str(CODE / "scripts"))
import torch  # noqa: E402
import e44_eval_extra as X  # noqa: E402
pd, E = X.pd, X.E
ROOT = "/workspace/Truth_Decoupling"
P = Path(X.EXP_ROOT) / "configs" / "prompts"
a = argparse.Namespace(
    phase="natural", source_run=Path(ROOT + "/results_qa_das_v1u/llama3.1-8b/full"),
    run_dir=Path(ROOT + "/results_e44_factorial_das/llama3.1-8b"), prompts=P / "default_v1.json",
    prompts_e44=P / "e44_factorial_v1.json", prompts_extra=P / "e44_eval_extra_v1.json", e_on="E1", e_off="none",
    c_run_dir=None, c_e_on="E1", c_e_off="none", arms=[], bases=[], family="E", layers=[10], pairs=["conf_E1:conf"],
    answers=["minus", "plus"], split="train", seed=42, batch_size=8, max_new_tokens=48, max_target_tokens=24,
    max_ref_tokens=48, output_projection="float32", first_batches=0, gen=None, prompts_out=None, name="judge_input",
    overwrite=False, n_check=0, pair_ids=None)
ctx = X.Context(a)
plus, minus = "conf_E1", "conf"
with E.fp32_output_projection(ctx.bundle, enabled=True):
    for layer in (10, 8):
        rec, _, _ = ctx.record("E", layer)
        w = pd.unit(torch.as_tensor(rec["w"]).float())
        train_ids = set(rec["training_contract"].get("train_ids") or rec.get("planned_train_ids") or [])
        idx = [i for i, q in enumerate(ctx.questions) if q["pair_id"] in train_ids]
        tp = [ctx.render(plus)[i]["prefix_text"] for i in idx]
        tm = [ctx.render(minus)[i]["prefix_text"] for i in idx]
        ref8 = ((pd.capture_prompt_end(ctx.bundle, tp, layer, 8) - pd.capture_prompt_end(ctx.bundle, tm, layer, 8)).float() @ w).tolist()
        ref1 = ((pd.capture_prompt_end(ctx.bundle, tp, layer, 1) - pd.capture_prompt_end(ctx.bundle, tm, layer, 1)).float() @ w).tolist()
        ids = [ctx.questions[i]["pair_id"] for i in idx]
        r8, r1 = dict(zip(ids, ref8)), dict(zip(ids, ref1))
        print("\n## layer %d: stored gap %.4f; reference mean batch 8 %.4f, batch 1 %.4f; |batch8 - batch1| max %.4f" % (
            layer, float(rec["gap"]), sum(ref8) / len(ref8), sum(ref1) / len(ref1), max(abs(x - y) for x, y in zip(ref8, ref1))))
        for source in ("minus", "plus"):
            name = plus if source == "plus" else minus
            answers = []
            for q in ctx.questions:
                r = ctx.look[q["pair_id"]]["answers"][name]
                answers.append(None if E.usable_answer(r, ctx.eot) else r)
            rows, skipped = X.natural_rows(ctx.bundle, layer, ctx.render(plus), ctx.render(minus), answers, ctx.eot, 24, 8)
            out = []
            for q, row in zip(ctx.questions, rows):
                pid = q["pair_id"]
                if pid not in r8 or row is None:
                    continue
                got = float(row[0][row[1] - 1] @ w)
                tol = 0.02 * abs(r8[pid]) + 1e-3
                out.append((abs(got - r8[pid]), pid, r8[pid], r1[pid], got, abs(got - r1[pid]), tol))
            out.sort(reverse=True)
            over = [o for o in out if o[0] > o[6]]
            print("answers=%s: %d questions checked, skipped %s; |natural - ref8| max %.4f mean %.4f; |natural - ref1| max %.4f; over the 2%%+1e-3 line: %d" % (
                source, len(out), skipped, out[0][0], sum(o[0] for o in out) / len(out), max(o[5] for o in out), len(over)))
            for o in out[:6]:
                print("   %s ref8 %.4f ref1 %.4f natural %.4f |nat-ref8| %.4f |nat-ref1| %.4f tol %.4f" % (o[1], o[2], o[3], o[4], o[0], o[5], o[6]))
