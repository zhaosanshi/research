"""CPU tests for scripts/e44_eval_extra.py (evaluation-only additions to E44).

Same toy as tests/test_qa_factorial_das.py: a randomly initialised 2-layer Llama with the REAL Llama-3.1 tokenizer.
Nothing here measures model quality; it checks the arithmetic of the split writes, the two-write generation against
single writes and against a full forward, the alignment of the read-only state differences, and the wiring to the
finished-run loaders and the judge export. Set TD_TEST_TOKENIZER to a local tokenizer directory.

Run:  python3 -m pytest tests/test_e44_eval_extra.py -q      (from code/)
"""
import json
import os
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
for p in (str(CODE), str(CODE / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from core import qa_factorial_das as fd  # noqa: E402
from core import qa_prompt_das as pd  # noqa: E402
from core.model_io import ModelBundle, forward_logits  # noqa: E402
import e44_eval_extra as X  # noqa: E402

TOKENIZER = os.environ.get("TD_TEST_TOKENIZER", "/workspace/Truth_Decoupling/models/Llama-3.1-8B-Instruct")
CONF = "You are a knowledgeable assistant. Always answer with a short, definite answer of a few words. Never say that you do not know."
E1 = "Answer truthfully and directly. Correct common misconceptions."
USER = "Question: {question}\nAnswer:"
Q1 = "What happens if you eat watermelon seeds?"
Q2 = "Who wrote Hamlet, the famous play about a Danish prince?"


@pytest.fixture(scope="module")
def bundle():
    if not Path(TOKENIZER).exists():
        pytest.skip("tokenizer directory %s not available" % TOKENIZER)
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    model = LlamaForCausalLM(cfg).eval()
    model.generation_config.eos_token_id = [128001, 128008, 128009]
    model.generation_config.pad_token_id = tok.pad_token_id
    for p in model.parameters():
        p.requires_grad_(False)
    return ModelBundle(model, tok, {"name": "tiny-llama", "chat": {"supports_system": True, "thinking_stub": ""},
                                   "layers_attr": "model.layers", "attn_attr": "self_attn"})


def _rand(n, seed):
    return torch.randn(n, generator=torch.Generator().manual_seed(seed))


# ----------------------------------------------------------------------------- arms and the split write (no model)
def test_arm_grammar():
    known = {"x": "", "conf": "", "conf_F": ""}
    assert X.parse_arm("conf:E10.wperp.p2", known) == ("conf", [{"family": "E", "layer": 10, "kind": "wperp", "beta": 2.0}])
    base, parts = X.parse_arm("x:E10.w.p2+C31.w.m0.5", known)
    assert base == "x" and [(p["family"], p["layer"], p["kind"], p["beta"]) for p in parts] == [("E", 10, "w", 2.0), ("C", 31, "w", -0.5)]
    assert X.arm_token("x:E10.w.p2+C31.w.m0.5") == "x--E10-w-p2-and-C31-w-m0-5"
    for bad in ("x.w.p1", "nope:E1.w.p1", "x:E1.w.p0", "x:E1.zz.p1", "x:C31.w.m1+E10.w.p2", "x:E3.w.p1+C3.w.p1",
                "x:E1.w.p1+C2.w.p1+C3.w.p1", "x:E1.w.p1:C2.w.p1"):
        with pytest.raises(SystemExit):
            X.parse_arm(bad, known)


def test_the_two_parts_add_up_to_the_learned_write():
    w, w_md = pd.unit(_rand(64, 1)), pd.unit(_rand(64, 2) + 0.8 * pd.unit(_rand(64, 1)))
    rec = {"w": w, "w_md": w_md, "gap": 0.652, "md_norm": 1.3, "mean_difference_defined": True}
    s = X.split_learned_write(rec)
    assert abs(float(s["perp"] @ s["w_md"])) < 1e-9 and abs(float(s["perp"].norm()) - 1) < 1e-9
    assert abs(s["cos"] ** 2 + s["perp_norm"] ** 2 - 1) < 1e-9
    total = s["par_amount"] * s["w_md"] + s["perp_amount"] * s["perp"]
    assert float((total - 0.652 * w.double()).norm()) < 1e-6                  # wpar + wperp = the w write
    for beta in (2.0, -0.5):
        par, perp = X.part_write(rec, "wpar", beta, 64, 42, 1), X.part_write(rec, "wperp", beta, 64, 42, 1)
        both = par["amount"] * par["direction"].double() + perp["amount"] * perp["direction"].double()
        assert float((both - beta * 0.652 * w.double()).norm()) < 1e-6
        mdg = X.part_write(rec, "mdg", beta, 64, 42, 1)
        assert mdg["amount"] == beta * 0.652 and float((mdg["direction"] - w_md).norm()) < 1e-6
    with pytest.raises(ValueError):
        X.split_learned_write(dict(rec, mean_difference_defined=False))
    with pytest.raises(ValueError):
        X.split_learned_write(dict(rec, w_md=w))                               # nothing orthogonal left


# ----------------------------------------------------------------------------- two writes at two layers
def _two(bundle):
    prompts = [pd.render_prompt(bundle, CONF, USER, Q1), pd.render_prompt(bundle, CONF, USER, "Why?"),
               pd.render_prompt(bundle, CONF, USER, Q2)]
    w0, w1 = pd.unit(_rand(bundle.hidden_size, 30)), pd.unit(_rand(bundle.hidden_size, 31))
    return prompts, {"direction": w0, "amount": 4.0}, {"direction": w1, "amount": -3.0}


@pytest.mark.parametrize("positions", ["content", "gen", "prompt"])
def test_two_writes_reduce_to_one_write_and_to_no_write(bundle, positions):
    prompts, a, b = _two(bundle)
    kw = {"max_new_tokens": 5, "batch_size": 2, "positions": positions}
    zero = lambda w: {"direction": w["direction"], "amount": 0.0}  # noqa: E731
    plain = fd.generate_shift(bundle, prompts, max_new_tokens=5, batch_size=2)
    assert X.same_answers(X.generate_two(bundle, prompts, [(0, zero(a)), (1, zero(b))], **kw), plain)
    only_a = fd.generate_shift(bundle, prompts, layer=0, **a, **kw)
    only_b = fd.generate_shift(bundle, prompts, layer=1, **b, **kw)
    first = X.generate_two(bundle, prompts, [(0, a), (1, zero(b))], **kw)
    second = X.generate_two(bundle, prompts, [(0, zero(a)), (1, b)], **kw)
    assert X.same_answers(first, only_a) and X.same_answers(second, only_b)
    for r2, r1 in zip(first, only_a):
        assert r2["write"][0]["layer"] == 0 and r2["write"][1]["layer"] == 1
        assert r2["write"][0]["amount"] == r1["write"]["amount"] == 4.0 and r2["write"][1]["amount"] == 0.0
        assert r2["write"][0]["prefill_positions"] == r1["write"]["prefill_positions"]
        assert abs(r2["write"][0]["prefill_stored_norm"] - r1["write"]["prefill_stored_norm"]) < 1e-6
        assert r2["write"][0]["decode_steps_own"] == r1["write"]["decode_steps_own"]
    with pytest.raises(ValueError):
        X.generate_two(bundle, prompts, [(1, a), (1, b)], **kw)                # same layer is another method
    with pytest.raises(ValueError):
        X.generate_two(bundle, prompts, [(0, a)], **kw)


@pytest.mark.parametrize("positions", ["content", "gen"])
def test_two_cached_writes_equal_one_full_forward_with_both_writes(bundle, positions):
    """Both hooks during cached decoding must equal one teacher-forced forward that carries both writes on the same
    tokens; the lower layer acts first because the model runs it first."""
    prompts, a, b = _two(bundle)
    recs = X.generate_two(bundle, prompts, [(0, a), (1, b)], max_new_tokens=5, batch_size=2, positions=positions)
    eot = pd.turn_terminator_ids(bundle)
    checked = 0
    for prompt, rec in zip(prompts, recs):
        assert [w["amount"] for w in rec["write"]] == [4.0, -3.0]
        assert rec["write"][0]["prefill_positions"] == rec["write"][1]["prefill_positions"] == (
            1 if positions == "gen" else prompt["prompt_length"] - prompt["content_start"])
        assert rec["write"][0]["decode_steps_own"] == rec["write"][1]["decode_steps_own"]
        if not rec["token_ids"]:
            continue
        item = pd.build_target(bundle, prompt, answer_ids=rec["token_ids"], finished=False, max_target_tokens=64, eot_ids=eot)
        packed = pd.pack_with_content_mask([item], bundle.tokenizer.pad_token_id, "cpu")
        mask = fd.position_mask(packed, positions)
        h0 = fd.ShiftHook(a["direction"], amounts=torch.tensor([a["amount"]]), mask=mask, decode_all=False)
        h1 = fd.ShiftHook(b["direction"], amounts=torch.tensor([b["amount"]]), mask=mask, decode_all=False)
        with pd.clamp(bundle, 0, h0), pd.clamp(bundle, 1, h1):
            logits = forward_logits(bundle, packed["inputs"], grad=False)
        logp, _, target_ids = fd.answer_token_logprobs(logits, packed)
        assert logp.argmax(-1).tolist() == target_ids.tolist() == rec["token_ids"]
        checked += 1
    assert checked
    single = fd.generate_shift(bundle, prompts, max_new_tokens=5, batch_size=2, layer=0, positions=positions, **a)
    assert any(x["token_ids"] != y["token_ids"] for x, y in zip(recs, single)) or all(
        abs(r["write"][1]["prefill_stored_norm"]) > 0 for r in recs)            # the second write is really applied


# ----------------------------------------------------------------------------- read-only state differences
def test_natural_rows_are_aligned_and_identical_prompts_give_zero(bundle):
    eot = pd.turn_terminator_ids(bundle)
    qs = [Q1, Q2, "Why?"]
    plus = [pd.render_prompt(bundle, CONF + " " + E1, USER, q) for q in qs]
    minus = [pd.render_prompt(bundle, CONF, USER, q) for q in qs]
    ids = lambda t: bundle.tokenizer(t, add_special_tokens=False)["input_ids"]  # noqa: E731
    answers = [{"token_ids": ids("Nothing happens"), "finished": True}, {"token_ids": ids("William Shakespeare wrote it"), "finished": True}, None]
    rows, skipped = X.natural_rows(bundle, 1, plus, minus, answers, eot, 24, 2)
    assert rows[2] is None and skipped == {"unusable answer": 1}
    read_plus = pd.capture_prompt_end(bundle, [p["prefix_text"] for p in plus[:2]], 1, 2)
    read_minus = pd.capture_prompt_end(bundle, [p["prefix_text"] for p in minus[:2]], 1, 2)
    for i in range(2):
        D, n = rows[i]
        m = minus[i]
        assert n == m["prompt_length"] - m["content_start"]
        assert D.shape == (n + len(answers[i]["token_ids"]) + len(eot), bundle.hidden_size)
        # the row at index n-1 is the last prompt token: the place where training read the difference
        assert torch.allclose(D[n - 1], read_plus[i] - read_minus[i], atol=1e-4)
        assert float(D.norm()) > 0
    same, _ = X.natural_rows(bundle, 1, minus, minus, answers, eot, 24, 2)
    assert float(same[0][0].abs().max()) == 0.0 and float(same[1][0].abs().max()) == 0.0
    u = pd.unit(read_plus[0] - read_minus[0])
    st = X.segment_stats(rows, {"u": u})
    want = [float((rows[i][0][rows[i][1] - 1] @ u)) for i in range(2)]
    assert st["read"]["questions"] == 2 and abs(st["read"]["u"]["mean"] - sum(want) / 2) < 1e-5
    assert st["question"]["positions"] == sum(rows[i][1] - 1 for i in range(2))
    assert st["answer"]["positions"] == sum(len(answers[i]["token_ids"]) + len(eot) for i in range(2))
    # a prompt pair whose content spans differ (another question) is refused, not silently misaligned
    other, skipped = X.natural_rows(bundle, 1, plus[:1], [pd.render_prompt(bundle, CONF, USER, Q2)], answers[:1], eot, 24, 2)
    assert other == [None] and skipped == {"content spans differ": 1}


# ----------------------------------------------------------------------------- wiring to a finished run
def test_script_end_to_end_on_the_toy_model(bundle, tmp_path, monkeypatch, capsys):
    """Parent run made by the unchanged driver; then every phase of this script. Plumbing only."""
    import e43_qa_prompt_das as e43
    import e44_factorial_das as drv
    src, run = tmp_path / "source", tmp_path / "run"
    src.mkdir()
    topics = ["the moon", "green tea", "old maps", "river stones", "night trains", "paper kites", "copper wire",
              "winter wheat", "glass beads", "sea salt", "mountain goats", "city bells", "red clay", "tin roofs",
              "salt marshes", "wool socks"]
    qs = [{"pair_id": "t:%d" % i, "question": "What is true about %s?" % t, "gold": ["It is ordinary", "Nothing special"],
           "incorrect": ["It is magic", "It grants wishes"], "best_answer": "It is ordinary"} for i, t in enumerate(topics)]
    (src / "bank.json").write_text(json.dumps({
        "model": "tiny-llama", "model_config": {"name": "tiny-llama", "path": str(src)}, "hidden_size": bundle.hidden_size,
        "num_layers": bundle.num_layers, "config": {"qa_system": "You are a knowledgeable assistant.", "qa_user": USER}}))
    (src / "questions.json").write_text(json.dumps({"splits": {"train": qs[:8], "tune": qs[8:]}, "e_train": qs[:8]}))
    bundle._e43_model_files = {}
    monkeypatch.setattr(e43, "open_model", lambda meta: bundle)
    common = ["--source-run", str(src), "--run-dir", str(run), "--max-new-tokens", "5", "--batch-size", "4"]

    def parent(*argv, run_dir=None):
        base = common if run_dir is None else ["--source-run", str(src), "--run-dir", str(run_dir), "--max-new-tokens", "5", "--batch-size", "4"]
        monkeypatch.setattr(sys, "argv", ["e44_factorial_das.py", argv[0], *base, *argv[1:]])
        drv.main()
        return capsys.readouterr().out

    def call(*argv):
        monkeypatch.setattr(sys, "argv", ["e44_eval_extra.py", argv[0], *common, *argv[1:]])
        X.main()
        return capsys.readouterr().out

    parent("freeze")
    parent("train", "--family", "E", "--layers", "0", "1", "--epochs", "2", "--val-n", "2")
    parent("train", "--family", "C", "--layers", "1", "--epochs", "2", "--val-n", "2")
    before = {p.name: p.read_bytes() for p in list(run.glob("*.json")) + list((run / "dirs").glob("*.pt"))}

    out = call("selfcheck", "--arms", "x:E0.w.p2+C1.w.m1", "--first-batches", "1")
    assert "selfcheck passed on 4 questions" in out
    with pytest.raises(SystemExit):
        call("selfcheck", "--arms", "x:E0.w.p2")                                # a single write has nothing to check
    arms = ["conf:E1.w.p2", "conf:E1.wpar.p2", "conf:E1.wperp.p2", "conf:E1.mdg.p1", "conf:E1.md.p1", "conf_F:E1.w.p2",
            "x:E0.w.p2+C1.w.m0.5", "x:E0.rand.p1+C1.w.m0.5"]
    out = call("generate", "--arms", *arms)
    assert "base conf_F (evaluation-only prompt)" in out and "E0 w" in out and "C1 w" in out
    g = json.loads((run / "eval_extra" / "gen" / "x--E0-w-p2-and-C1-w-m0-5_tune.json").read_text())
    assert g["completed"] and len(g["rows"]) == 8 and [p["layer"] for p in g["parts"]] == [0, 1]
    assert all(len(r["write"]) == 2 for r in g["rows"]) and g["parts"][1]["amount"] < 0 < g["parts"][0]["amount"]
    frozen = json.loads((run / "frozen_tune.json").read_text())
    assert all(r["baseline_text"] == f["answers"]["x"]["text"] for r, f in zip(g["rows"], frozen["questions"]))
    one = json.loads((run / "eval_extra" / "gen" / "conf--E1-w-p2_tune.json").read_text())
    rec = torch.load(run / "dirs" / "E_L1.pt", map_location="cpu", weights_only=False)
    assert abs(one["parts"][0]["amount"] - 2 * rec["gap"]) < 1e-9
    parent("generate", "--family", "E", "--layers", "1", "--split", "tune", "--arms", "conf.w.p2")
    same = json.loads((run / "gen" / "E_L1_conf-w-p2_tune.json").read_text())
    assert [r["token_ids"] for r in one["rows"]] == [r["token_ids"] for r in same["rows"]]   # kind w = the driver's arm
    par = json.loads((run / "eval_extra" / "gen" / "conf--E1-wpar-p2_tune.json").read_text())["parts"][0]
    perp = json.loads((run / "eval_extra" / "gen" / "conf--E1-wperp-p2_tune.json").read_text())["parts"][0]
    assert abs(par["amount"] ** 2 + perp["amount"] ** 2 - (2 * rec["gap"]) ** 2) < 1e-6
    f = json.loads((run / "eval_extra" / "gen" / "conf_F--E1-w-p2_tune.json").read_text())
    base_f = json.loads((run / "eval_extra" / "base_conf_F_tune.json").read_text())
    assert f["base_kind"] == "extra" and base_f["system"].endswith("Write the answer as one complete sentence.")
    assert [r["baseline_text"] for r in f["rows"]] == [r["text"] for r in base_f["rows"]]
    assert "skip conf--E1-w-p2_tune.json (verified)" in call("generate", "--arms", "conf:E1.w.p2")
    with pytest.raises(ValueError):                                             # another seed is another contract
        call("generate", "--arms", "x:E0.rand.p1+C1.w.m0.5", "--seed", "7")
    with pytest.raises(SystemExit):                                             # another batch size is another parent identity
        monkeypatch.setattr(sys, "argv", ["e44_eval_extra.py", "generate", "--source-run", str(src), "--run-dir", str(run),
                                          "--max-new-tokens", "5", "--batch-size", "2", "--arms", "conf:E1.w.p2"])
        X.main()
    capsys.readouterr()

    out = call("natural", "--family", "E", "--layers", "1", "--pairs", "conf_E1:conf", "caut_E1:caut", "--split", "train")
    assert "read-position check: mean coordinate" in out
    nat = json.loads((run / "eval_extra" / "natural_E_train.json").read_text())
    layer = nat["layers"]["1"]
    assert abs(layer["read_position_check"]["mean_coordinate"] - rec["gap"]) <= 0.02 * abs(rec["gap"]) + 1e-3
    assert set(layer["pairs"]) == {"conf_E1:conf|answer=minus", "conf_E1:conf|answer=plus", "caut_E1:caut|answer=minus", "caut_E1:caut|answer=plus"}
    seg = layer["pairs"]["conf_E1:conf|answer=minus"]["segments"]
    assert seg["read"]["questions"] > 0 and set(seg["answer"]) >= {"w", "w_md", "perp", "difference_norm_mean"}
    out = call("natural", "--family", "C", "--layers", "1", "--pairs", "caut:conf", "--split", "train")
    assert "read-position check not run" in out                                 # C needs both of its training pairs

    # prompt controls without an arm, and a C direction taken from another finished run of the same identity
    out = call("generate", "--bases", "conf_E1_F")
    assert "base conf_E1_F (evaluation-only prompt)" in out and (run / "eval_extra" / "base_conf_E1_F_tune.json").exists()
    with pytest.raises(SystemExit):
        call("generate", "--bases", "nope")
    run_c = tmp_path / "run_c"
    parent("freeze", run_dir=run_c)
    parent("train", "--family", "C", "--layers", "1", "--epochs", "1", "--val-n", "2", "--e-off", "M", run_dir=run_c)
    with pytest.raises(ValueError):                                             # that run's truth levels must be named
        call("generate", "--arms", "caut:E0.w.p1+C1.w.m1", "--c-run-dir", str(run_c))
    call("generate", "--arms", "caut:E0.w.p1+C1.w.m1", "--c-run-dir", str(run_c), "--c-e-off", "M")
    mixed = json.loads((run / "eval_extra" / "gen" / "caut--E0-w-p1-and-C1-w-m1_tune.json").read_text())
    assert [p["direction_run"] for p in mixed["parts"]] == ["run", "run_c"] and mixed["parts"][1]["e_off"] == "M"
    other = torch.load(run_c / "dirs" / "C_L1.pt", map_location="cpu", weights_only=False)
    assert abs(mixed["parts"][1]["amount"] + other["gap"]) < 1e-9 and other["artifact_sha256"] == mixed["parts"][1]["direction_record_sha256"]
    with pytest.raises(SystemExit):                                             # export must be told where that direction lives
        call("export", "--name", "no_c_run")
    arms = arms + ["caut:E0.w.p1+C1.w.m1"]

    call("export", "--name", "judge_input", "--c-run-dir", str(run_c), "--prompts-out", "conf_E1_F")
    payload = json.loads((run / "eval_extra" / "judge_tune" / "judge_input.json").read_text())
    assert {"baseline_x", "baseline_conf", "baseline_conf_F", "baseline_conf_E1_F", "baseline_caut", "X_x--E0-w-p2-and-C1-w-m0-5",
            "X_conf_F--E1-w-p2", "X_caut--E0-w-p1-and-C1-w-m1"} <= set(payload["conditions"])
    assert len(payload["conditions"]) == 5 + len(arms) and len(payload["evaluations"]) == 8      # x conf conf_F caut conf_E1_F
    assert set(payload["evaluations"][0]["outputs"]) == set(payload["conditions"])
    spec = {s["name"]: s for s in payload["condition_specs"]}
    assert spec["X_conf_F--E1-w-p2"]["base_condition"] == "baseline_conf_F"
    call("export", "--name", "judge_input", "--c-run-dir", str(run_c), "--prompts-out", "conf_E1_F")   # identical content is reusable
    # the parent run's own files were only read
    assert before == {p.name: p.read_bytes() for p in list(run.glob("*.json")) + list((run / "dirs").glob("*.pt"))}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
