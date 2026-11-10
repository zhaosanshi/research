"""CPU tests for core/qa_factorial_das.py and the E44 driver wiring.

Same toy as tests/test_qa_prompt_das.py: a randomly initialised 2-layer Llama with the REAL Llama-3.1 tokenizer,
so prompts, positions, masks, hooks, gradients and cached generation run on the true token layout. Nothing here
measures model quality. Set TD_TEST_TOKENIZER to a local tokenizer directory (default: the server path).

Run:  python3 -m pytest tests/test_qa_factorial_das.py -q      (from code/)
"""
import math
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

TOKENIZER = os.environ.get("TD_TEST_TOKENIZER", "/workspace/Truth_Decoupling/models/Llama-3.1-8B-Instruct")
CONF = "You are a knowledgeable assistant. Always answer with a short, definite answer of a few words. Never say that you do not know."
CAUT = "You are a knowledgeable assistant. Answer with a short answer of a few words. If you do not know the answer, reply exactly 'I don't know'."
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


def _ids(bundle, text):
    return bundle.tokenizer(text, add_special_tokens=False)["input_ids"]


def _rand(n, seed):
    return torch.randn(n, generator=torch.Generator().manual_seed(seed))


# ----------------------------------------------------------------------------- hook arithmetic (no model)
def test_shift_adds_exact_amount_only_at_masked_positions():
    h = _rand(2 * 5 * 8, 1).reshape(2, 5, 8)
    u = pd.unit(_rand(8, 2))
    mask = torch.tensor([[False, False, True, True, True], [False, True, True, False, True]])
    hook = fd.ShiftHook(3.0 * u, amounts=torch.tensor([1.5, -0.5]), mask=mask, decode_all=False)   # direction is normalised inside
    out = hook(None, None, (h.clone(),))[0]
    delta = out - h
    assert torch.equal(delta[~mask], torch.zeros_like(delta[~mask]))
    assert torch.allclose(delta[0][mask[0]], (1.5 * u).expand(3, 8), atol=1e-6)
    assert torch.allclose(delta[1][mask[1]], (-0.5 * u).expand(3, 8), atol=1e-6)
    assert torch.allclose((out @ u - h @ u)[mask], torch.tensor([1.5, 1.5, 1.5, -0.5, -0.5, -0.5]), atol=1e-5)
    assert hook.applied == 1 and hook.stats()["prefill"]["positions"] == [3, 3]


def test_difference_mode_writes_the_projection_and_identical_states_write_nothing():
    h = _rand(2 * 4 * 8, 3).reshape(2, 4, 8)
    u = pd.unit(_rand(8, 4))
    d = torch.stack([_rand(8, 5), torch.zeros(8)])                     # row 1: source state == base state
    mask = torch.ones(2, 4, dtype=torch.bool)
    out = fd.ShiftHook(u, difference=d, mask=mask, decode_all=False)(None, None, (h.clone(),))[0]
    assert torch.allclose(out[0] - h[0], ((d[0] @ u) * u).expand(4, 8), atol=1e-6)
    assert torch.equal(out[1], h[1])                                   # exact no-op, not merely small
    flipped = fd.ShiftHook(-u, difference=d, mask=mask, decode_all=False)(None, None, (h.clone(),))[0]
    assert torch.allclose(flipped, out, atol=1e-6)                     # the write does not depend on the sign of w


def test_random_axis_gets_the_same_coordinate_and_vectors_are_written_whole():
    h = _rand(1 * 3 * 8, 6).reshape(1, 3, 8)
    u, r, d = pd.unit(_rand(8, 7)), pd.unit(_rand(8, 8)), _rand(8, 9)[None]
    mask = torch.ones(1, 3, dtype=torch.bool)
    rnd = fd.ShiftHook(u, amounts=torch.tensor([0.8]), mask=mask, write_direction=r, decode_all=False)(None, None, (h.clone(),))[0]
    assert torch.allclose(rnd - h, (0.8 * r).expand(1, 3, 8), atol=1e-6)
    whole = fd.ShiftHook(vectors=d, mask=mask, decode_all=False)(None, None, (h.clone(),))[0]
    assert torch.allclose(whole - h, d.expand(1, 3, 8), atol=1e-6)
    with pytest.raises(ValueError):
        fd.ShiftHook(u, amounts=torch.tensor([1.0]), difference=d)
    with pytest.raises(ValueError):
        fd.ShiftHook(amounts=torch.tensor([1.0]))


def test_gradient_reaches_the_direction_through_amount_and_axis():
    h = _rand(1 * 3 * 8, 10).reshape(1, 3, 8)
    w = torch.nn.Parameter(pd.unit(_rand(8, 11)))
    d = _rand(8, 12)[None]
    out = fd.ShiftHook(w, difference=d, mask=torch.ones(1, 3, dtype=torch.bool), decode_all=False)(None, None, (h,))[0]
    (out * _rand(8, 13)).sum().backward()
    assert w.grad is not None and torch.isfinite(w.grad).all() and float(w.grad.norm()) > 0


def test_jsd_properties():
    logp = torch.log_softmax(_rand(3 * 7, 14).reshape(3, 7), -1)
    logq = torch.log_softmax(_rand(3 * 7, 15).reshape(3, 7), -1)
    assert torch.allclose(fd.jsd(logp, logp), torch.zeros(3), atol=1e-6)
    assert torch.allclose(fd.jsd(logp, logq), fd.jsd(logq, logp), atol=1e-6)
    assert (fd.jsd(logp, logq) > 0).all() and (fd.jsd(logp, logq) <= math.log(2) + 1e-6).all()
    a = torch.log(torch.tensor([[1 - 1e-9, 1e-9]]))
    b = torch.log(torch.tensor([[1e-9, 1 - 1e-9]]))
    assert abs(float(fd.jsd(a, b)) - math.log(2)) < 1e-4


def test_choice_metrics():
    # A missing candidate changes the scoring problem; do not silently drop it.
    assert fd.choice_metrics([-3.0, -1.0, None], [-2.0, -5.0]) is None
    m = fd.choice_metrics([-3.0, -1.0], [-2.0, -5.0])
    assert m["mc1"] == 1 and abs(m["margin"] - 1.0) < 1e-9
    mass_t, mass_f = math.exp(-3) + math.exp(-1), math.exp(-2) + math.exp(-5)
    assert abs(m["mc2"] - mass_t / (mass_t + mass_f)) < 1e-9
    assert fd.choice_metrics([-4.0], [-1.0])["mc1"] == 0
    assert fd.choice_metrics([None], [-1.0]) is None and fd.choice_metrics([-1.0], []) is None


# ----------------------------------------------------------------------------- with the toy model
def _prompts(bundle, question=Q1):
    return {"conf": pd.render_prompt(bundle, CONF, USER, question), "conf_E1": pd.render_prompt(bundle, CONF + " " + E1, USER, question),
            "caut": pd.render_prompt(bundle, CAUT, USER, question)}


def _rows(bundle, layer=1, question=Q1):
    """One cause row (base conf, source/target conf_E1), one isolate row (base conf, source caut, target conf)
    and one row whose source is the base itself."""
    eot = pd.turn_terminator_ids(bundle)
    P = _prompts(bundle, question)
    ans = {"conf": _ids(bundle, "Nothing happens"), "conf_E1": _ids(bundle, "The seeds pass through you")}
    H = {n: pd.capture_prompt_end(bundle, [P[n]["prefix_text"]], layer, 1)[0] for n in P}
    target = lambda base, name: pd.build_target(bundle, P[base], answer_ids=ans[name], finished=True, eot_ids=eot)  # noqa: E731
    refs = fd.reference_logprobs(bundle, [target("conf", "conf"), target("conf_E1", "conf_E1")], batch_size=2)
    return [
        {"kind": "cause", "encoded": target("conf", "conf_E1"), "difference": H["conf_E1"] - H["conf"], "ref_logp": refs[1]},
        {"kind": "isolate", "encoded": target("conf", "conf"), "difference": H["caut"] - H["conf"], "ref_logp": refs[0]},
        {"kind": "isolate", "encoded": target("conf", "conf"), "difference": H["conf"] - H["conf"], "ref_logp": refs[0]},
    ]


def test_reference_rows_and_the_no_write_loss(bundle):
    rows = _rows(bundle)
    assert rows[0]["ref_logp"].shape == (rows[0]["encoded"]["answer_length"], len(bundle.tokenizer))
    none, stats = fd.rows_loss(bundle, 1, rows, mode="none")
    assert stats is None
    assert float(none[0]) > 0                         # another prompt's distribution on its own answer: a gap to close
    assert abs(float(none[1])) < 1e-6 and abs(float(none[2])) < 1e-6      # the base's own distribution: nothing to change
    ce, _ = fd.rows_loss(bundle, 1, rows, mode="none", loss="ce")
    assert (ce > 0).all()


def test_identical_source_and_base_leave_the_forward_untouched(bundle):
    rows = _rows(bundle)[2:]
    w = pd.unit(_rand(bundle.hidden_size, 20))
    packed = pd.pack_with_content_mask([rows[0]["encoded"]], bundle.tokenizer.pad_token_id, "cpu")
    plain = forward_logits(bundle, packed["inputs"], grad=False)
    hook = fd.ShiftHook(w, difference=rows[0]["difference"][None], mask=packed["content_mask"], decode_all=False)
    with pd.clamp(bundle, 1, hook):
        again = forward_logits(bundle, packed["inputs"], grad=False)
    assert hook.applied == 1 and torch.equal(plain, again)
    loss, _ = fd.rows_loss(bundle, 1, rows, w=w)
    assert abs(float(loss[0])) < 1e-6


def test_position_masks(bundle):
    rows = _rows(bundle)
    packed = pd.pack_with_content_mask([r["encoded"] for r in rows[:2]], bundle.tokenizer.pad_token_id, "cpu")
    content, gen, prompt = (fd.position_mask(packed, k) for k in ("content", "gen", "prompt"))
    assert not (gen & ~content).any() and gen.sum() < content.sum()
    assert not (prompt & packed["target_mask"]).any() and torch.equal(prompt | packed["target_mask"], content)
    for row, r in enumerate(rows[:2]):
        assert int(gen[row].sum()) == r["encoded"]["answer_length"] + 1          # the last prompt token + every answer token
        assert int((gen[row] & prompt[row]).sum()) == 1                           # they share only the last prompt token
    with pytest.raises(ValueError):
        fd.position_mask(packed, "everything")


def test_training_runs_records_history_and_keeps_unit_norm(bundle):
    q = lambda pid, question: {"pair_id": pid, "rows": _rows(bundle, question=question)[:2]}  # noqa: E731
    train, val = [q("a", Q1), q("b", Q2)], [q("c", "Why is the sky blue?")]
    w, info = fd.train_direction(bundle, 1, train, epochs=3, lr=1e-2, clip=1.0, seed=5, val_questions=val)
    assert abs(float(w.norm()) - 1.0) < 1e-5 and len(info["history"]) == 3
    assert all(h["grad_norm_mean"] > 0 and h["zero_grad_steps"] == 0 for h in info["history"])
    assert all(h["val_cause"] is not None and h["val_isolate"] is not None for h in info["history"])
    assert info["lr_schedule"] == "linear_to_zero" and info["init"] == "random"
    same, _ = fd.train_direction(bundle, 1, train, epochs=3, lr=1e-2, clip=1.0, seed=5, val_questions=val)
    assert torch.allclose(w, same, atol=1e-6)                                   # the seed fixes the run
    started, info2 = fd.train_direction(bundle, 1, train, epochs=1, lr=1e-3, seed=5, init=torch.ones(bundle.hidden_size))
    assert info2["init"] == "given" and abs(float(started.norm()) - 1.0) < 1e-5
    losses = fd.mean_losses(bundle, 1, train, w=w)
    assert losses["cause"] > 0 and losses["isolate"] >= 0
    assert fd.mean_losses(bundle, 1, train, mode="none")["isolate"] < 1e-6
    assert fd.mean_losses(bundle, 1, train, mode="vectors")["cause"] > 0
    with pytest.raises(ValueError):
        fd.question_loss(torch.zeros(2), ["isolate", "isolate"])


def test_mean_difference_and_difference_stats():
    plus = torch.tensor([[2.0, 0.0], [4.0, 0.0]])
    minus = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    u, n = fd.mean_difference(plus, minus)
    assert torch.allclose(u * n, torch.tensor([3.0, -1.0]), atol=1e-6)
    s = fd.difference_stats(plus, minus, torch.tensor([1.0, 0.0]))
    assert abs(s["mean"] - 3.0) < 1e-6 and s["share_positive"] == 1.0 and len(s["values"]) == 2


@pytest.mark.parametrize("positions", ["content", "gen", "prompt"])
def test_cached_generation_equals_a_full_shifted_forward(bundle, positions):
    """The write during cached decoding must equal the write of one full teacher-forced forward on the same tokens."""
    prompts = [pd.render_prompt(bundle, CONF, USER, Q1), pd.render_prompt(bundle, CONF, USER, "Why?")]
    w = pd.unit(_rand(bundle.hidden_size, 30))
    plain = fd.generate_shift(bundle, prompts, max_new_tokens=5, batch_size=2)
    assert all("write" not in r for r in plain)
    zero = fd.generate_shift(bundle, prompts, max_new_tokens=5, batch_size=2, layer=1, direction=w, amount=0.0, positions=positions)
    assert [r["token_ids"] for r in zero] == [r["token_ids"] for r in plain]
    recs = fd.generate_shift(bundle, prompts, max_new_tokens=5, batch_size=2, layer=1, direction=w, amount=4.0, positions=positions)
    eot = pd.turn_terminator_ids(bundle)
    for prompt, rec in zip(prompts, recs):
        assert rec["write"]["amount"] == 4.0 and rec["write"]["prefill_positions"] == (
            1 if positions == "gen" else prompt["prompt_length"] - prompt["content_start"])
        assert (rec["write"]["decode_steps_total"] == 0) == (positions == "prompt")
        if not rec["token_ids"]:
            continue
        item = pd.build_target(bundle, prompt, answer_ids=rec["token_ids"], finished=False, max_target_tokens=64, eot_ids=eot)
        packed = pd.pack_with_content_mask([item], bundle.tokenizer.pad_token_id, "cpu")
        hook = fd.ShiftHook(w, amounts=torch.tensor([4.0]), mask=fd.position_mask(packed, positions), decode_all=False)
        with pd.clamp(bundle, 1, hook):
            logits = forward_logits(bundle, packed["inputs"], grad=False)
        logp, _, target_ids = fd.answer_token_logprobs(logits, packed)
        assert logp.argmax(-1).tolist() == target_ids.tolist() == rec["token_ids"]


def test_score_answers_matches_a_manual_sum_and_does_not_score_the_terminator(bundle):
    prompt = pd.render_prompt(bundle, CONF, USER, Q1)
    eot = pd.turn_terminator_ids(bundle)
    cands = ["Nothing happens", "You grow watermelons in your stomach", ""]
    recs = fd.score_answers(bundle, prompt, cands, eot_ids=eot, batch_size=2)
    assert recs[2]["skipped"] and recs[2]["sum_logp"] is None
    for text, rec in zip(cands[:2], recs[:2]):
        item = pd.build_target(bundle, prompt, answer_text=text, max_target_tokens=48, eot_ids=eot)
        packed = pd.pack_with_content_mask([item], bundle.tokenizer.pad_token_id, "cpu")
        logp, _, ids = fd.answer_token_logprobs(forward_logits(bundle, packed["inputs"], grad=False), packed)
        per_token = logp.gather(1, ids[:, None]).squeeze(1)
        assert item["terminator_appended"] and rec["n_tokens"] == item["answer_body_tokens"] == len(per_token) - 1
        assert abs(rec["sum_logp"] - float(per_token[:-1].sum())) < 1e-4
    w = pd.unit(_rand(bundle.hidden_size, 31))
    shifted = fd.score_answers(bundle, prompt, cands[:2], eot_ids=eot, layer=1, direction=w, amount=5.0)
    assert any(abs(a["sum_logp"] - b["sum_logp"]) > 1e-6 for a, b in zip(shifted, recs[:2]))
    cut = fd.score_answers(bundle, prompt, [" ".join("word%d" % i for i in range(40))], eot_ids=eot, max_target_tokens=8)
    assert cut[0]["truncated"] and cut[0]["n_tokens"] == 8


def test_per_question_writes_equal_the_constant_write_when_the_values_agree(bundle):
    prompts = [pd.render_prompt(bundle, CONF, USER, Q1), pd.render_prompt(bundle, CONF, USER, "Why?")]
    w = pd.unit(_rand(bundle.hidden_size, 40))
    kw = {"max_new_tokens": 4, "batch_size": 2, "layer": 1}
    const = fd.generate_shift(bundle, prompts, direction=w, amount=4.0, **kw)
    per_row = fd.generate_shift(bundle, prompts, direction=w, amount=[4.0, 4.0], **kw)
    whole = fd.generate_shift(bundle, prompts, vectors=torch.stack([4.0 * w, 4.0 * w]), **kw)
    assert [r["token_ids"] for r in const] == [r["token_ids"] for r in per_row] == [r["token_ids"] for r in whole]
    assert abs(whole[0]["write"]["amount"] - 4.0) < 1e-5                 # for whole vectors the record holds the norm
    mixed = fd.generate_shift(bundle, prompts, direction=w, amount=[4.0, 0.0], **kw)
    plain = fd.generate_shift(bundle, prompts, max_new_tokens=4, batch_size=2)
    assert mixed[0]["token_ids"] == const[0]["token_ids"] and mixed[1]["token_ids"] == plain[1]["token_ids"]
    for bad in ({"direction": w, "amount": [1.0]}, {"direction": w, "amount": 1.0, "vectors": torch.zeros(2, bundle.hidden_size)},
                {"vectors": torch.zeros(3, bundle.hidden_size)}, {"direction": w, "amount": float("nan")}):
        with pytest.raises(ValueError):
            fd.generate_shift(bundle, prompts, **bad, **kw)
    eot = pd.turn_terminator_ids(bundle)
    a = fd.score_answers(bundle, prompts[0], ["Nothing happens"], eot_ids=eot, layer=1, direction=w, amount=5.0)
    b = fd.score_answers(bundle, prompts[0], ["Nothing happens"], eot_ids=eot, layer=1, vector=5.0 * w)
    assert abs(a[0]["sum_logp"] - b[0]["sum_logp"]) < 1e-4


# ----------------------------------------------------------------------------- driver wiring
def test_family_rows_and_arms():
    import e44_factorial_das as drv
    e = drv.family_spec("E", "E1")
    assert e["cause"] == [("conf", "conf_E1", "conf_E1"), ("conf_E1", "conf", "conf")]
    assert all(target == base for base, _, target in e["isolate"]) and len(e["isolate"]) == 4
    assert {(b, s) for b, s, _ in e["isolate"]} == {("conf", "caut"), ("caut", "conf"), ("conf_E1", "caut_E1"), ("caut_E1", "conf_E1")}
    assert e["plus_minus"] == [("conf_E1", "conf")] and e["prompts"] == ["caut", "caut_E1", "conf", "conf_E1"]
    two_pole = drv.family_spec("E", "E1", "M")
    assert two_pole["cause"] == [("conf_M", "conf_E1", "conf_E1"), ("conf_E1", "conf_M", "conf_M")]
    assert two_pole["plus_minus"] == [("conf_E1", "conf_M")] and two_pole["e_off"] == "M"
    assert two_pole["prompts"] == ["caut_E1", "caut_M", "conf_E1", "conf_M"] and e["e_off"] is None
    assert drv.family_spec("E", "E1", "none") == e
    with pytest.raises(ValueError):
        drv.family_spec("E", "M", "M")
    assert drv.arm_target_prompt("E", "E1", "conf_M", "w", 1.0, "M") == "conf_E1"
    assert drv.arm_target_prompt("C", "E1", "caut_M", "wq", -1.0, "M") == "conf_M"
    c = drv.family_spec("C", "E2")
    assert all(target == source for _, source, target in c["cause"]) and len(c["cause"]) == 4
    assert {(b, s) for b, s, _ in c["isolate"]} == {("conf", "conf_E2"), ("conf_E2", "conf"), ("caut", "caut_E2"), ("caut_E2", "caut")}
    assert c["plus_minus"] == [("caut", "conf"), ("caut_E2", "conf_E2")]
    # every cause row changes only the family's own factor; every isolate row only the other one
    level = lambda name: (name.endswith("_E1") or name.endswith("_E2"), name.startswith("caut"))  # noqa: E731
    for spec, own in ((e, 0), (c, 1)):
        for base, source, _ in spec["cause"]:
            assert level(base)[own] != level(source)[own] and level(base)[1 - own] == level(source)[1 - own]
        for base, source, _ in spec["isolate"]:
            assert level(base)[own] == level(source)[own] and level(base)[1 - own] != level(source)[1 - own]
    systems = {"x": "", "conf": "", "conf_E1": "", "caut": "", "caut_E1": ""}
    assert drv.parse_arm("x.w.p1", systems) == ("x", "w", 1.0)
    assert drv.parse_arm("conf_E1.md.m0.5", systems) == ("conf_E1", "md", -0.5)
    assert drv.arm_file_token("conf_E1.md.m0.5") == "conf_E1-md-m0-5"
    assert drv.parse_arm("conf.wq.p1", systems) == ("conf", "wq", 1.0) and drv.parse_arm("x.dq.p2", systems) == ("x", "dq", 2.0)
    record = {"w": torch.tensor([1.0, 0.0]), "w_md": torch.tensor([0.0, 1.0]), "gap": 0.5, "md_norm": 2.0}
    D = torch.tensor([[2.0, 7.0], [-1.0, 3.0]])
    assert drv.arm_write(record, "w", 2.0, 2, 42, 3) == {"direction": record["w"], "amount": 1.0}
    assert drv.arm_write(record, "md", -1.0, 2, 42, 3)["amount"] == -2.0
    rnd = drv.arm_write(record, "rand", 1.0, 2, 42, 3)
    assert rnd["amount"] == 0.5 and abs(float(rnd["direction"].norm()) - 1.0) < 1e-6
    assert drv.arm_write(record, "wq", 2.0, 2, 42, 3, D)["amount"] == [4.0, -2.0]
    assert torch.equal(drv.arm_write(record, "dq", -1.0, 2, 42, 3, D)["vectors"], -D)
    with pytest.raises(ValueError):
        drv.arm_write(record, "wq", 1.0, 2, 42, 3)
    assert drv.arm_target_prompt("E", "E1", "conf", "wq", 1.0) == "conf_E1" and drv.arm_target_prompt("C", "E1", "caut", "dq", -1.0) == "conf"
    for bad in ("x.w.1", "nope.w.p1", "x.w.p0", "x.q.p1"):
        with pytest.raises(SystemExit):
            drv.parse_arm(bad, systems)
    assert drv.arm_target_prompt("E", "E1", "conf", "w", 1.0) == "conf_E1"
    assert drv.arm_target_prompt("E", "E1", "conf_E1", "md", -1.0) == "conf"
    assert drv.arm_target_prompt("C", "E1", "conf_E1", "w", 1.0) == "caut_E1"
    assert drv.arm_target_prompt("C", "E1", "caut", "w", -1.0) == "conf"
    for args in (("E", "E1", "x", "w", 1.0), ("E", "E1", "conf", "w", 2.0), ("E", "E1", "conf", "rand", 1.0), ("C", "E1", "conf", "w", -1.0)):
        assert drv.arm_target_prompt(*args) is None


def test_prompts_file_and_question_rows(bundle):
    import e44_factorial_das as drv
    meta = {"config": {"qa_system": "You are a knowledgeable assistant. NEUTRAL."}}
    cfg, systems = drv.load_prompts(meta, CODE / "configs" / "prompts" / "default_v1.json", CODE / "configs" / "prompts" / "e44_factorial_v1.json")
    assert systems["x"] == meta["config"]["qa_system"] and systems["conf"] == CONF and systems["caut"] == CAUT
    assert systems["conf_E1"] == CONF + " " + E1 and systems["x_E1"] == systems["x"] + " " + E1
    assert set(sum(cfg["step0_pairs"], [])) <= set(systems)
    for fam in ("E", "C"):
        for e_on in ("E1", "E2"):
            for e_off in (None, "M"):
                assert set(drv.family_spec(fam, e_on, e_off)["prompts"]) <= set(systems)
    eot = pd.turn_terminator_ids(bundle)

    def ans(text, stop=128009):
        t = _ids(bundle, text)
        return {"text": text, "token_ids": t, "finished": True, "stop_id": stop, "n_tokens": len(t), "refusal_rule": False}

    spec = drv.family_spec("E", "E1")
    texts = {"conf": "Bad luck", "conf_E1": "Nothing happens", "caut": "I don't know", "caut_E1": "Nothing"}
    qs = [{"pair_id": "q1", "question": Q1}, {"pair_id": "q2", "question": Q2}, {"pair_id": "q3", "question": "Why?"}]
    frozen = {"eot_ids": eot, "questions": [
        {"pair_id": "q1", "question": Q1, "answers": {n: ans(texts[n]) for n in spec["prompts"]}},
        {"pair_id": "q2", "question": Q2, "answers": {n: ans("" if n == "caut" else "x") for n in spec["prompts"]}},
        {"pair_id": "q3", "question": "Why?", "answers": {n: ans("Rome", stop=128001 if n == "conf" else 128009) for n in spec["prompts"]}}]}
    built, skipped = drv.build_questions(bundle, spec, qs, frozen, systems, USER, 24)
    assert [q["pair_id"] for q in built] == ["q1"]
    assert skipped == [{"pair_id": "q2", "reason": "empty_frozen_answer"}, {"pair_id": "q3", "reason": "nonstandard_stop_token_128001"}]
    rows = built[0]["rows"]
    assert [r["kind"] for r in rows] == ["cause"] * 2 + ["isolate"] * 4
    tok = bundle.tokenizer
    for r in rows:
        prefix = tok.decode(r["encoded"]["prefix_ids"])
        assert systems[r["base"]] in prefix and (E1 in prefix) == r["base"].endswith("_E1")
        assert r["encoded"]["answer_ids"] == _ids(bundle, texts[r["target"]]) + eot
        assert r["target"] == (r["source"] if r["kind"] == "cause" else r["base"])
    ref_logp = {("q1", n): piece for n, piece in zip(spec["prompts"], fd.reference_logprobs(
        bundle, [built[0]["refs"][n] for n in spec["prompts"]], 4))}
    H = drv.attach_layer(bundle, built, spec, 1, 4, ref_logp)
    for r in rows:
        assert torch.allclose(r["difference"], (H[r["source"]][0] - H[r["base"]][0]).float().cpu())
        assert r["ref_logp"] is ref_logp[("q1", r["target"])]
    none = fd.mean_losses(bundle, 1, built, mode="none")
    assert none["isolate"] < 1e-6 and none["cause"] > 0
    many = [{"pair_id": "q%d" % i} for i in range(20)]
    tr, va = drv.split_train_val(many, 4, 42)
    assert len(tr) == 16 and len(va) == 4 and not {q["pair_id"] for q in tr} & {q["pair_id"] for q in va}
    assert drv.split_train_val(many, 4, 42) == (tr, va)
    with pytest.raises(ValueError):
        drv.split_train_val(many[:3], 4, 42)  # impossible holdout cannot silently become all training
    assert drv.text_stats(["I don't know.", "Yes", "No it is not"]) == {"refusal_rule": 1, "ends_with_period": 1,
                                                                       "mean_words": 8 / 3, "negation_words": 2}


def test_reference_candidates_exclude_refusal_only_gold():
    import e44_factorial_das as drv
    assert drv.candidate_lists({"question": "What is 2+2?", "gold": ["Four", "4"], "incorrect": ["Five", ""]}) == (["Four", "4"], ["Five"])
    assert drv.candidate_lists({"question": "What is your opinion?", "gold": ["I have no comment"], "incorrect": ["Yes"]}) is None
    assert drv.candidate_lists({"question": "What is 2+2?", "gold": ["Four"], "incorrect": []}) is None


def test_driver_end_to_end_on_the_toy_model(bundle, tmp_path, monkeypatch, capsys):
    """Every phase of the driver and the judge export, on the toy model and a synthetic source run: file layout,
    identity checks, per-question arms and the zero-write control. Checks plumbing only, not model quality."""
    import json
    import e43_qa_prompt_das as e43
    import e44_factorial_das as drv
    import e44_export_judge as exp
    src, run = tmp_path / "source", tmp_path / "run"
    src.mkdir()
    topics = ["the moon", "green tea", "old maps", "river stones", "night trains", "paper kites", "copper wire",
              "winter wheat", "glass beads", "sea salt", "mountain goats", "city bells"]
    qs = [{"pair_id": "t:%d" % i, "question": "What is true about %s?" % t, "gold": ["It is ordinary", "Nothing special"],
           "incorrect": ["It is magic", "It grants wishes"], "best_answer": "It is ordinary"} for i, t in enumerate(topics)]
    (src / "bank.json").write_text(json.dumps({
        "model": "tiny-llama", "model_config": {"name": "tiny-llama", "path": str(src)}, "hidden_size": bundle.hidden_size,
        "num_layers": bundle.num_layers, "config": {"qa_system": "You are a knowledgeable assistant.", "qa_user": USER}}))
    (src / "questions.json").write_text(json.dumps({"splits": {"train": qs[:8], "tune": qs[8:]}, "e_train": qs[:8]}))
    bundle._e43_model_files = {}
    monkeypatch.setattr(e43, "open_model", lambda meta: bundle)

    def call(*argv):
        monkeypatch.setattr(sys, "argv", ["e44_factorial_das.py", argv[0], "--source-run", str(src), "--run-dir", str(run),
                                          "--max-new-tokens", "5", "--batch-size", "4", *argv[1:]])
        drv.main()
        return capsys.readouterr().out

    call("freeze")
    frozen = json.loads((run / "frozen_train.json").read_text())
    assert frozen["completed"] and len(frozen["questions"]) == 8 and set(frozen["prompt_systems"]) == set(frozen["questions"][0]["answers"])
    assert set(frozen["summary"]["pairs"]) == {"%s->%s" % (a, b) for a, b in json.loads(
        (CODE / "configs" / "prompts" / "e44_factorial_v1.json").read_text())["step0_pairs"]}
    assert "skip frozen_train.json" in call("freeze")                       # verified, not regenerated
    for family in ("E", "C"):
        out = call("train", "--family", family, "--layers", "1", "--epochs", "2", "--val-n", "2")
        assert "L1 done" in out
        rec = torch.load(run / "dirs" / ("%s_L1.pt" % family), map_location="cpu", weights_only=False)
        assert rec["gap"] >= 0 and rec["md_norm"] > 0 and abs(float(rec["w"].norm()) - 1) < 1e-5
        for part in ("train", "val"):
            assert rec["losses"][part]["no_write"]["isolate"] < 1e-6        # an isolate row's target is the unwritten base
            assert set(rec["losses"][part]) == {"no_write", "whole_difference", "mean_difference_direction", "trained"}
        assert len(rec["training"]["history"]) == 2 and rec["training_contract"]["val_ids"]
        assert "skip L1" in call("train", "--family", family, "--layers", "1", "--epochs", "2", "--val-n", "2")
    with pytest.raises(ValueError):                                         # another seed is another training contract
        call("train", "--family", "E", "--layers", "1", "--epochs", "2", "--val-n", "2", "--seed", "7")
    with pytest.raises(ValueError):                                         # so is another pair of truth levels
        call("train", "--family", "E", "--layers", "1", "--epochs", "2", "--val-n", "2", "--e-off", "M")
    assert "truth levels M / E1" in call("train", "--family", "E", "--layers", "0", "--epochs", "1", "--val-n", "2", "--e-off", "M")
    assert torch.load(run / "dirs" / "E_L0.pt", map_location="cpu", weights_only=False)["training_contract"]["e_off"] == "M"
    with pytest.raises(ValueError):                                         # M checkpoint cannot be silently used as none/E1
        call("generate", "--family", "E", "--layers", "0", "--split", "tune", "--arms", "conf_M.wq.p1")
    out0 = call("generate", "--family", "E", "--layers", "0", "--split", "tune", "--e-off", "M", "--arms", "conf_M.wq.p1", "x.w.p2", "conf_M.iso.p1")
    iso0 = json.loads((run / "gen/E_L0_conf_M-iso-p1_tune.json").read_text())
    assert iso0["e_off"] == "M" and iso0["target_prompt"] == "conf_M" and iso0["read_pair"] == ["caut_M", "conf_M"]
    call("mc", "--family", "E", "--layers", "0", "--split", "tune", "--e-off", "M", "--arms", "conf_M.wq.p1", "conf_M.iso.p1")
    assert json.loads((run / "mc/E_L0_conf_M-iso-p1_tune.json").read_text())["mc_contract"]["e_off"] == "M"
    assert "should behave like conf_E1" in out0
    arms = ["conf.w.p1", "conf.wq.p1", "conf.dq.p1", "conf_E1.w.m1", "x.md.p2", "x.rand.p1"]
    out = call("generate", "--family", "E", "--layers", "1", "--split", "tune", "--arms", *arms)
    assert "should behave like conf_E1" in out and "should behave like conf" in out
    gen = json.loads((run / "gen" / "E_L1_conf-wq-p1_tune.json").read_text())
    assert gen["completed"] and gen["target_prompt"] == "conf_E1" and len(gen["rows"]) == 4
    assert all(r["baseline_text"] == f["answers"]["conf"]["text"] for r, f in zip(
        gen["rows"], json.loads((run / "frozen_tune.json").read_text())["questions"]))
    assert json.loads((run / "gen" / "E_L1_x-rand-p1_tune.json").read_text())["target_prompt"] is None
    call("generate", "--family", "C", "--layers", "1", "--split", "tune", "--arms", "x.w.p1", "x.w.m1", "caut.wq.m1")
    out = call("mc", "--family", "E", "--layers", "1", "--split", "tune", "--arms", "conf.w.p1", "conf.dq.p1")
    assert "mc base conf" in out
    mc = json.loads((run / "mc" / "E_L1_conf-w-p1_tune.json").read_text())
    assert mc["summary"]["n_scored"] == 4 and mc["summary"]["n_paired"] == 4 and "abstain_logp_change_mean" in mc["summary"]
    assert all(len(r["true"]) == 2 and len(r["false"]) == 2 and r["abstain_logp"] < 0 for r in mc["rows"])
    assert "mc base x_F" in call("mc", "--split", "tune", "--bases", "x_F", "conf")     # prompt-only readout, no direction needed
    assert json.loads((run / "mc" / "base_x_F_tune.json").read_text())["summary"]["n_scored"] == 4
    with pytest.raises(SystemExit):
        call("mc", "--split", "tune", "--bases", "nope")
    table = call("summarize")
    assert "| E | 1 |" in table and "conf.wq.p1" in table and "no write, conf" in table and (run / "summary.md").exists()
    info = exp.export(src, run, "tune", None, ["x", "conf", "caut", "conf_E1", "conf_M"], "judge_input")
    assert info["n_questions"] == 4 and "E_L1_conf-wq-p1" in info["conditions"] and "baseline_conf" in info["conditions"]
    payload = json.loads((run / "judge_tune" / "judge_input.json").read_text())
    assert set(payload["evaluations"][0]["outputs"]) == set(payload["conditions"])
    assert len(payload["conditions"]) == 5 + 6 + 3 + 3                      # prompts + E L1 arms + C L1 arms + E L0 arms
    with pytest.raises(ValueError):                                         # an arm's base prompt must be exported too
        exp.export(src, run, "tune", None, ["x"], "too_few_prompts")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
