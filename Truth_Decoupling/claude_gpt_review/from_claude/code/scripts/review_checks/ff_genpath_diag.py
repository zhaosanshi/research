#!/usr/bin/env python3
"""Generation-path diagnostics for the free-form E44 code on any model (Claude, 2026-11-13; read-only bug hunt).

Uses the E44 code itself (core/qa_factorial_das.py, core/qa_prompt_das.py) and an existing trained direction. Writes only
the JSON report given by --out. Checks:
  1 rendering   tail of the rendered prompt, first content token, read token, number of think tags
  2 pipeline    generate_shift at batch 8 must equal the pipeline's own generation file (same code, same batch)
  3 batch       the same write at batch 1 vs batch 8 (texts); no write at batch 1 vs the frozen answers (batch 8)
  4 teacher     teacher-force prompt + generated ids with the same write at the same positions (one row, no padding):
                argmax at every answer position must equal the generated token. Also without any write.
                A low rate means the cached decode path and the full forward disagree.
"""
import argparse, json, os, sys
from pathlib import Path

CODE = Path(os.environ.get("TD_CODE", "/workspace/Truth_Decoupling/code_newmodels_ff_20261112"))
sys.path[:0] = [str(CODE / "scripts"), str(CODE)]
import torch  # noqa: E402
import e43_qa_prompt_das as e43  # noqa: E402
import e44_factorial_das as drv  # noqa: E402
from core import qa_factorial_das as fd  # noqa: E402
from core import qa_prompt_das as pd  # noqa: E402
from core.qa_precision import fp32_output_projection  # noqa: E402


@torch.no_grad()
def teacher_match(bundle, prompt, gen_ids, layer=None, write=None, positions="content"):
    """(matched, total, first_mismatch) for one rendered prompt and its generated ids (incl. the stop id if any)."""
    ids = list(prompt["prefix_ids"]) + list(gen_ids)
    P, T = len(prompt["prefix_ids"]), len(ids)
    x = torch.tensor([ids], device=bundle.device)
    att = torch.ones_like(x)
    hook = None
    if write is not None:
        if positions != "content":
            raise SystemExit("only content positions are handled here")
        mask = torch.zeros((1, T), dtype=torch.bool, device=bundle.device)
        mask[0, prompt["content_start"]:] = True
        hook = fd.ShiftHook(write["direction"], amounts=torch.tensor([float(write["amount"])]), mask=mask, decode_all=False)
        with pd.clamp(bundle, layer, hook):
            logits = bundle.model(input_ids=x, attention_mask=att, use_cache=False).logits
        if hook.applied != 1:
            raise RuntimeError("hook did not fire once")
    else:
        logits = bundle.model(input_ids=x, attention_mask=att, use_cache=False).logits
    pred = logits[0, P - 1:T - 1].argmax(-1).tolist()
    tgt = ids[P:]
    m = sum(int(a == b) for a, b in zip(pred, tgt))
    first = next((k for k, (a, b) in enumerate(zip(pred, tgt)) if a != b), None)
    return m, len(tgt), first


def gen_ids_of(rec, stops):
    ids = list(rec["token_ids"])
    if rec.get("finished") and rec.get("stop_id") is not None:
        ids = ids + [int(rec["stop_id"])]
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--family", default="E")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--arm", required=True, help="e.g. x.w.p1 (w arms only)")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    meta = e43.read(a.source_run / "bank.json")
    prepared = e43.read(a.source_run / "questions.json")
    cfg, systems = drv.load_prompts(meta, CODE / "configs" / "prompts" / "default_v1.json",
                                    CODE / "configs" / "prompts" / "e44_factorial_v1.json")
    base, direction, beta = drv.parse_arm(a.arm, systems)
    if direction != "w":
        raise SystemExit("only fixed-shift w arms")
    qs = prepared["splits"]["tune"][:a.n]
    frozen = e43.frozen_lookup(e43.read(a.run_dir / "frozen_tune.json"))
    genfile = e43.read(a.run_dir / "gen" / ("%s_L%d_%s_tune.json" % (a.family, a.layer, drv.arm_file_token(a.arm))))
    pipe = {r["pair_id"]: r for r in genfile["rows"]}
    record = torch.load(a.run_dir / "dirs" / ("%s_L%d.pt" % (a.family, a.layer)), map_location="cpu", weights_only=False)
    positions = record["training_contract"]["positions"]
    write = drv.arm_write(record, direction, beta, meta["hidden_size"], 42, a.layer)
    torch.manual_seed(42)
    bundle = e43.open_model(meta)
    stops = pd.stop_ids(bundle)
    report = {"model": meta["model"], "layer": a.layer, "arm": a.arm, "n": len(qs), "positions": positions,
              "amount": float(write["amount"]), "stops": sorted(stops)}
    rendered = [pd.render_prompt(bundle, systems[base], meta["config"]["qa_user"], q["question"]) for q in qs]
    r0 = rendered[0]
    report["rendering"] = {"prefix_tail": r0["prefix_text"][-120:], "content_first_token": r0["content_first_token"],
                           "read_token": r0["read_token"], "think_open": r0["prefix_text"].count("<think>"),
                           "think_close": r0["prefix_text"].count("</think>"), "prompt_length": r0["prompt_length"],
                           "content_start": r0["content_start"]}
    with fp32_output_projection(bundle, enabled=True):
        g8 = fd.generate_shift(bundle, rendered, max_new_tokens=48, batch_size=8, layer=a.layer, positions=positions, **write)
        g1 = fd.generate_shift(bundle, rendered, max_new_tokens=48, batch_size=1, layer=a.layer, positions=positions, **write)
        n1 = pd.generate_prompts(bundle, rendered, max_new_tokens=48, batch_size=1)
        same_pipe = sum(int(g["text"] == pipe[q["pair_id"]]["text"]) for g, q in zip(g8, qs))
        same_b = sum(int(x["text"] == y["text"]) for x, y in zip(g8, g1))
        same_nw = sum(int(x["text"] == frozen[q["pair_id"]]["answers"][base]["text"]) for x, q in zip(n1, qs))
        tm = [teacher_match(bundle, p, gen_ids_of(g, stops), layer=a.layer, write=write, positions=positions) for p, g in zip(rendered, g8)]
        tn = [teacher_match(bundle, p, gen_ids_of(frozen[q["pair_id"]]["answers"][base], stops)) for p, q in zip(rendered, qs)]
    report["pipeline_equal_batch8"] = "%d/%d" % (same_pipe, len(qs))
    report["write_batch1_equal_batch8"] = "%d/%d" % (same_b, len(qs))
    report["nowrite_batch1_equal_frozen"] = "%d/%d" % (same_nw, len(qs))
    report["teacher_write_tokens"] = "%d/%d" % (sum(m for m, _, _ in tm), sum(t for _, t, _ in tm))
    report["teacher_write_rows_exact"] = "%d/%d" % (sum(int(m == t) for m, t, _ in tm), len(tm))
    report["teacher_nowrite_tokens"] = "%d/%d" % (sum(m for m, _, _ in tn), sum(t for _, t, _ in tn))
    report["teacher_nowrite_rows_exact"] = "%d/%d" % (sum(int(m == t) for m, t, _ in tn), len(tn))
    report["examples"] = [{"pair_id": q["pair_id"], "batch8": x["text"], "batch1": y["text"],
                           "teacher_first_mismatch": t[2]} for q, x, y, t in zip(qs, g8, g1, tm)
                          if x["text"] != y["text"] or t[0] != t[1]][:8]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, ensure_ascii=False, indent=1))
    for e in report["examples"]:
        print("  ", e)


if __name__ == "__main__":
    main()
