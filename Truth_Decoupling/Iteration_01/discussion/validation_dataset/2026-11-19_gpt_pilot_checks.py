"""Read-only review fixtures for f6adbe4. Run with python3 -B; uses synthetic data."""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[2] / "code"
sys.path[:0] = [str(CODE / "diag"), str(CODE / "scripts/newsets"), str(CODE / "scripts/capability")]
import cap_prompts
import newsets_gap
import newsets_gen
import newsets_score as score


def exported_answer(root):
    bank = root / "bank.jsonl"
    bank.write_text(json.dumps(dict(id="q1", question="Synthetic question", gold=["gold"],
                                    split="test", pro100=True)))
    src = root / "gen"
    src.mkdir()
    rows = [dict(id=q, text=text, finished=True, rule_abstain=False)
            for q, text in [("q1", "first"), ("q1", "last"), ("foreign", "other")]]
    artifact = dict(protocol=newsets_gen.PROTOCOL, task="popqa", split="test",
                    full_pass=True, completed=False, items_file_sha256=newsets_gen.sha_file(bank),
                    model="other-model", identity_sha256="i", frozen_file_sha256="f",
                    script_sha256="s", engine_sha256="e", arm="x", condition="baseline_x",
                    parts=[], base_prompt="x", rows=rows)
    (src / "x.json").write_text(json.dumps(artifact))
    dest = root / "out.json"
    args = SimpleNamespace(items=bank, gen_dir=src, only_arms=["x"], out=dest,
                           task="popqa", split="test", pro100=True)
    newsets_gen.run_export(args)
    out = json.loads(dest.read_text())
    return dict(completed=out["completed"], ids=out["question_ids"],
                answer=out["evaluations"][0]["outputs"]["baseline_x"])


def one_question_gate():
    # Keep the real nine-arm policy and bootstrap. Replace only model configuration / file readers.
    names = list(score.ORDER)
    models = {name: dict(el=11, cl=19, caps=dict(e_blank=0.25, own_blank=0.25)) for name in names}
    def tables(task, model, split, *unused):
        q = task + ":q"
        shell = score.NewSplit(11, 19, {}, {}, [])
        st = {(q, c): "R" for c in score.need_test(shell)}
        st[q, "baseline_x"] = st[q, shell.cE("2")] = "X"
        return st, {k: v == "X" for k, v in st.items()}, {k: ("IDK" if v == "X" else "gold") for k, v in st.items()}
    def pro(task, model, split):
        return (*tables(task, model, split), "synthetic")
    def conf(task, model, split, arm):
        return {task + ":q": {"mean_logp": -0.1}}
    with patch.object(score, "models", return_value=models), patch.object(score, "alias_tables", tables), \
         patch.object(score, "pro_tables", pro), patch.object(score, "conf_rows", conf):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            score.report_c3(names)
    return [s for s in out.getvalue().splitlines() if "Pro 1, alias 1" in s or "gate 1 reading (" in s]


def gap_labels(root):
    path = root / "judge2"
    path.mkdir()
    rows = [dict(condition="baseline_conf", status="scored", pair_id="popqa:q1", label=k)
            for k in ("A", "B")]
    (path / "single_pro.jsonl").write_text("\n".join(map(json.dumps, rows)))
    return newsets_gap.labels_of(root, "popqa")


def main():
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        print("export:", exported_answer(work))
        print("duplicate D2 verdicts:", gap_labels(work))
    qs = SimpleNamespace(qs=["q1", "q2"])
    conf = {"q1": {"mean_logp": -0.1}, "q2": {"mean_logp": None}}
    _, kept = score.top_n(qs, ["R", "X"], [True, False], conf, 2)
    print("target answers=2, retained:", sum(kept))
    print("gate:", one_question_gate())
    for text in ("B", "The answer is not D.", "The answer is B or C."):
        print("MMLU:", repr(text), cap_prompts.mmlu_pick(text))
    for text in ("The answer is 7.", "We used 7 boxes and next we must", "The answer is 1/2."):
        print("GSM8K:", repr(text), cap_prompts.gsm8k_pick(text))


if __name__ == "__main__":
    main()
