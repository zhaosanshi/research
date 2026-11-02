"""Verify E24 scoring repairs without API calls or model generation."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
CODE = ROOT.parent / "from_claude/code"
sys.path.insert(0, str(CODE))
from core.qa_scoring import judged_outcome, reference_answers, rule_conflicts_with_judge


def main():
    assert judged_outcome("wrong") == "wrong"
    assert judged_outcome("correct") == "correct"
    assert judged_outcome("unresolved") == "unresolved"
    assert rule_conflicts_with_judge("wrong", True)
    assert reference_answers(["I have no comment", "The origin is unknown", "Unknown"])[0] == ["The origin is unknown", "Unknown"]
    assert reference_answers(["I have no comment"])[1]
    script = CODE / "scripts/e24_real_qa.py"
    with tempfile.TemporaryDirectory() as tmp:
        for model, expected in (("llama3.1-8b", (127, 222, 172)), ("qwen3-8b", (157, 197, 167))):
            source = ROOT / "analysis_data/real_qa_20261102" / model / "real_qa_fixed.json"
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            prefix = Path(tmp) / model
            command = [sys.executable, str(script), "--model", model, "--phase", "rescore", "--input-file", str(source),
                       "--output-prefix", str(prefix), "--n-boot", "20"]
            run = subprocess.run(command, capture_output=True, text=True)
            assert run.returncode == 0, run.stderr
            assert "zero API and generation calls" in run.stdout
            report = json.loads(Path(str(prefix) + ".json").read_text())
            assert report["n_resolved_all_conditions"] == 521
            rates = report["rates"]["C_abst"]["rates"]
            assert tuple(round(rates[k] * 521) for k in ("correct", "wrong", "abstain")) == expected
            assert report["gold_policy"] == "cached_original_cohort"
            assert hashlib.sha256(source.read_bytes()).hexdigest() == before
            assert subprocess.run(command, capture_output=True).returncode != 0

        # A changed reference list must invalidate cached grades in API mode.
        spec = importlib.util.spec_from_file_location("e24", script)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        items = []
        for i, gold in enumerate((["I have no comment", "The origin is unknown"], ["Paris", "I have no comment"])):
            items.append({"idx": i, "pair_id": "test:%d" % i, "question": "test question", "gold": gold,
                          "outputs": {c: "answer" for c in module.CONDS}, "rule_abstain": {c: True for c in module.CONDS},
                          "judge": {c: {"label": "wrong", "raw": "B"} for c in module.CONDS},
                          "outcome": {c: "abstain" for c in module.CONDS}, "gold_is_abstention": i == 0})
        source = Path(tmp) / "fixture.json"
        source.write_text(json.dumps({"model": "llama3.1-8b", "evaluations": items, "E": {}, "C": {}, "intervention_position": "test"}))
        class FakeJudge:
            name = "fixture_only"
            n_calls = 0
            n_cache_hits = 0
            def qa_grade(self, question, gold, prediction, incorrect=None):
                assert gold == ["The origin is unknown"]
                self.n_calls += 1
                return {"label": "correct", "raw": "A"}
        judge = FakeJudge()
        args = SimpleNamespace(model="llama3.1-8b", tag="fixture", phase="judge", judge_exp="unused",
                               input_file=str(source), output_prefix=str(Path(tmp) / "new_gold"), rejudge=False, n_boot=20)
        with patch.object(module, "load_experiment", return_value={"judge": {}}), \
             patch("core.judge.judge_from_config", return_value=judge), contextlib.redirect_stdout(io.StringIO()):
            module.phase_judge(args)
        assert judge.n_calls == 3
        scored = json.loads(Path(args.output_prefix + "_items.json").read_text())
        assert scored["evaluations"][0]["judge_reference_answers"] == ["The origin is unknown"]
        assert not scored["evaluations"][0]["gold_is_abstention"]
    print("PASS: semantic-label precedence, unresolved preservation, objective-unknown gold, cached CPU rescore, input preservation, cache invalidation on changed gold")


if __name__ == "__main__":
    main()
