# Iteration_01 analysis (Claude Opus 5.5, 2026-11-22), exploration. A check asked for by the read-only review of the
# ITI baseline (2026-11-22 10 h): the official ITI code reads its probes at the last token of "Q: {question} A:
# {choice}" where the choices (TruthfulQA multiple-choice targets) end with a full stop, so that last token is
# mostly the full stop; scripts/newsets/iti_baseline.py fits on the generation split's reference answers, which mostly
# end on a word. This refits ITI with a full stop appended to every answer that does not end with . ! or ? (the
# sigma bank's texts get the same answers), writes the fit to a separate file, and compares it with the existing fit:
# overlap of the top-K head sets, cosine of the directions on the heads the existing fit chose, sigma ratio.
# iti_baseline.py itself is not changed: its pair_text is replaced in memory only.
# usage (model interpreter, from /workspace/Truth_Decoupling):
#   python -I iti_period_check.py fit --models CFG --model MODEL --out OUT.pt           (GPU, under a minute)
#   python -I iti_period_check.py compare --base FIT.pt --other OUT.pt                    (CPU)
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
NEWSETS = HERE.parent / "scripts" / "newsets"
if not NEWSETS.is_dir():                                   # the server keeps diag/ next to code/
    NEWSETS = HERE.parent / "code" / "scripts" / "newsets"
sys.path.insert(0, str(NEWSETS))
import iti_baseline as IB  # noqa: E402

ENDS = (".", "!", "?")


def fit(a):
    original = IB.pair_text
    changed = {"n": 0, "of": 0}

    def with_stop(bundle, question, answer, next_question=None):
        changed["of"] += 1
        if not answer.rstrip().endswith(ENDS):
            answer = answer.rstrip() + "."
            changed["n"] += 1
        return original(bundle, question, answer, next_question)

    IB.pair_text = with_stop
    out = Path(a.out)
    if out.exists():
        raise SystemExit("%s exists" % out)
    IB.run_fit(SimpleNamespace(models=a.models, model=a.model, out=str(out), fit_splits=a.fit_splits,
                               val_ratio=a.val_ratio, seed=a.seed))
    Path(str(out) + ".note.txt").write_text(
        "diagnostic fit: a full stop was appended to %d of %d answer texts (both banks); everything else as "
        "iti_baseline.py fit\n" % (changed["n"], changed["of"]), encoding="utf-8")
    print("full stop appended to %d of %d answer texts" % (changed["n"], changed["of"]), flush=True)


def compare(a):
    import torch
    base = torch.load(a.base, map_location="cpu", weights_only=False)
    other = torch.load(a.other, map_location="cpu", weights_only=False)

    def ranking(f):
        acc = f["accuracy"]
        L, H = acc.shape
        return [(l, h) for _, l, h in sorted(((float(acc[l, h]), l, h) for l in range(L) for h in range(H)),
                                             key=lambda t: (-t[0], t[1], t[2]))]

    rb, ro = ranking(base), ranking(other)
    for k in (16, 32, 48, 64):
        top = rb[:k]
        cos = [float(torch.dot(base["unit"][l, h].double(), other["unit"][l, h].double())) for l, h in top]
        ratio = [float(other["sigma"][l, h] / base["sigma"][l, h]) for l, h in top]
        print("K %2d: heads shared %d of %d | cosine of the directions on the existing fit's heads: min %.3f, median %.3f | sigma (with full stop) / sigma (existing): median %.3f" % (
            k, len(set(top) & set(ro[:k])), k, min(cos), sorted(cos)[len(cos) // 2], sorted(ratio)[len(ratio) // 2]))
    print("best validation accuracy: existing %.3f, with full stop %.3f" % (float(base["accuracy"].max()), float(other["accuracy"].max())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["fit", "compare"])
    ap.add_argument("--models", type=Path)
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--fit-splits", default="train")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base")
    ap.add_argument("--other")
    a = ap.parse_args()
    if a.phase == "fit":
        if not (a.models and a.model and a.out):
            ap.error("fit needs --models, --model and --out")
        fit(a)
    else:
        if not (a.base and a.other):
            ap.error("compare needs --base and --other")
        compare(a)


if __name__ == "__main__":
    main()
