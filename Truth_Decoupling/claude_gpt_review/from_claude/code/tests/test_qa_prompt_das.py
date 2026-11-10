"""CPU tests for core/qa_prompt_das.py and the E43 driver wiring.

A randomly initialised 2-layer Llama with the REAL Llama-3.1 tokenizer (chat template, special
tokens), so prompt rendering, content positions, the end-of-turn token, masks, the clamp hook,
gradients and generation hooks are exercised on the true token layout. Nothing here measures
model quality. Set TD_TEST_TOKENIZER to a local tokenizer directory (default: the server path).

Run:  python3 -m pytest tests/test_qa_prompt_das.py -q      (from code/)
"""
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

from core import qa_prompt_das as pd  # noqa: E402
from core.model_io import ModelBundle, forward_logits  # noqa: E402

TOKENIZER = os.environ.get("TD_TEST_TOKENIZER", "/workspace/Truth_Decoupling/models/Llama-3.1-8B-Instruct")
SYSTEM = "You are a knowledgeable assistant. Answer the question with a short answer of a few words. If you do not know the answer, you may say 'I don't know'."
USER = "Question: {question}\nAnswer:"


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


def _prompt(bundle, question="What happens if you eat watermelon seeds?", system=SYSTEM):
    return pd.render_prompt(bundle, system, USER, question)


def test_render_prompt_marks_user_content_start(bundle):
    p = _prompt(bundle)
    ids = p["prefix_ids"]
    assert p["boundary_clean"]
    assert bundle.tokenizer.decode(ids[p["content_start"]:]).startswith("Question: What happens")
    head = bundle.tokenizer.decode(ids[:p["content_start"]])
    assert SYSTEM in head and "user" in head and "Question" not in head
    assert p["read_position"] == len(ids) - 1
    assert "assistant" in bundle.tokenizer.decode(ids[p["content_start"]:])  # generation header is a content position


def test_turn_terminator_is_eot(bundle):
    eot = pd.turn_terminator_ids(bundle)
    assert eot == [128009]
    assert bundle.tokenizer.decode(eot) == "<|eot_id|>"
    assert set(eot) <= pd.stop_ids(bundle)


def test_build_target_terminator_and_truncation(bundle):
    p = _prompt(bundle)
    eot = pd.turn_terminator_ids(bundle)
    short = pd.build_target(bundle, p, answer_text="The seeds pass through your digestive system", eot_ids=eot)
    assert short["terminator_appended"] and not short["truncated"]
    assert short["answer_ids"][-1] == 128009 and short["answer_length"] == short["answer_body_tokens"] + 1
    assert short["full_ids"] == short["prefix_ids"] + short["answer_ids"]
    long_text = " ".join(["word%d" % i for i in range(40)])
    long = pd.build_target(bundle, p, answer_text=long_text, max_target_tokens=24, eot_ids=eot)
    assert long["truncated"] and not long["terminator_appended"]
    assert long["answer_length"] == 24 and 128009 not in long["answer_ids"]
    ids = bundle.tokenizer("Paris", add_special_tokens=False)["input_ids"]
    frozen_done = pd.build_target(bundle, p, answer_ids=ids, finished=True, eot_ids=eot)
    frozen_cut = pd.build_target(bundle, p, answer_ids=ids, finished=False, eot_ids=eot)
    assert frozen_done["terminator_appended"] and not frozen_cut["terminator_appended"]
    with pytest.raises(ValueError):
        pd.build_target(bundle, p, answer_ids=[], eot_ids=eot)
    with pytest.raises(ValueError):
        pd.build_target(bundle, p, answer_ids=[128009], eot_ids=eot)  # special token inside the answer span


def test_pack_content_mask_covers_user_content_to_end(bundle):
    eot = pd.turn_terminator_ids(bundle)
    a = pd.build_target(bundle, _prompt(bundle), answer_text="Nothing happens", eot_ids=eot)
    b = pd.build_target(bundle, _prompt(bundle, "Who wrote Hamlet, the famous play about a Danish prince?"),
                        answer_text="William Shakespeare wrote it", eot_ids=eot)
    packed = pd.pack_with_content_mask([a, b], bundle.tokenizer.pad_token_id, "cpu")
    att = packed["inputs"]["attention_mask"].bool()
    content, target = packed["content_mask"], packed["target_mask"]
    assert not (content & ~att).any() and not (target & ~content).any()
    W = content.shape[1]
    for row, item in enumerate((a, b)):
        start = W - (item["prompt_length"] + item["answer_length"])
        assert content[row, start + item["content_start"]:].all()
        assert not content[row, :start + item["content_start"]].any()
        assert int(target[row].sum()) == item["answer_length"]


def test_prompt_content_mask_for_generation(bundle):
    p1, p2 = _prompt(bundle), _prompt(bundle, "Why?")
    inputs = bundle.encode([p1["prefix_text"], p2["prefix_text"]])
    mask = pd.prompt_content_mask(inputs["attention_mask"], [p1["prompt_length"], p2["prompt_length"]],
                                  [p1["content_start"], p2["content_start"]])
    W = mask.shape[1]
    assert int(mask[0].sum()) == p1["prompt_length"] - p1["content_start"]
    assert int(mask[1].sum()) == p2["prompt_length"] - p2["content_start"]
    assert mask[1, W - 1] and not mask[1, W - p2["prompt_length"]]


def _pair_items(bundle):
    eot = pd.turn_terminator_ids(bundle)
    x = _prompt(bundle)
    x_e = _prompt(bundle, system=SYSTEM + " Answer truthfully and directly. Correct common misconceptions.")
    plus = pd.build_target(bundle, x, answer_text="The seeds pass through your digestive system", eot_ids=eot)
    minus = pd.build_target(bundle, x_e, answer_ids=bundle.tokenizer("Nothing in particular", add_special_tokens=False)["input_ids"],
                            finished=True, eot_ids=eot)
    src = pd.capture_prompt_end(bundle, [x_e["prefix_text"], x["prefix_text"]], layer=1, batch_size=2)
    return [plus, minus], src


def test_clamp_exact_on_masked_untouched_elsewhere(bundle):
    items, src = _pair_items(bundle)
    w = pd.unit(torch.randn(bundle.hidden_size, generator=torch.Generator().manual_seed(1)))
    rep = pd.verify_clamp(bundle, 1, items, src, w)
    assert rep["masked_positions"] > 0 and rep["untouched_positions"] > 0
    assert rep["masked_abs_error_max"] < 1e-4          # fp32 model: the coordinate equals the target
    assert rep["untouched_abs_change_max"] == 0.0
    assert rep["passed"] and rep["gap_closed_mean"] > 0


def test_hook_with_empty_mask_is_identity(bundle):
    items, src = _pair_items(bundle)
    packed = pd.pack_with_content_mask(items, bundle.tokenizer.pad_token_id, "cpu")
    base = forward_logits(bundle, packed["inputs"], grad=False)
    hook = pd.ClampHook(torch.randn(bundle.hidden_size), source_hidden=src,
                        mask=torch.zeros_like(packed["content_mask"]), decode_all=False)
    with pd.clamp(bundle, 1, hook):
        again = forward_logits(bundle, packed["inputs"], grad=False)
    assert hook.calls == 1 and hook.applied == 0
    assert torch.equal(base, again)


def test_gradient_reaches_direction_at_last_layer(bundle):
    items, src = _pair_items(bundle)
    for layer in (0, bundle.num_layers - 1):
        rep = pd.gradient_probe(bundle, layer, items, src, seed=3)
        assert rep["grad_finite"] and rep["grad_norm"] > 0, rep
        assert all(c > 0 for c in rep["per_row_ce"])


def test_train_direction_two_epochs(bundle):
    items, src = _pair_items(bundle)
    pair = {"pair_id": "t", "plus": {"encoded": items[0], "source_hidden": src[0]},
            "minus": {"encoded": items[1], "source_hidden": src[1]}}
    w, info = pd.train_direction(bundle, 1, [pair, pair], epochs=2, lr=1e-2, clip=1.0, seed=5)
    assert abs(float(w.norm()) - 1.0) < 1e-5
    assert len(info["history"]) == 2 and all(h["grad_norm_mean"] > 0 for h in info["history"])
    rows = pd.baseline_ce(bundle, 1, w, [pair])
    assert rows[0]["ce_plus"] > 0 and rows[0]["ce_minus"] > 0
    coords = pd.coordinates(src, w)
    assert len(coords["values"]) == 2


def test_random_matched_write_has_das_magnitude():
    torch.manual_seed(0)
    h = torch.randn(1, 5, 8)
    u = pd.unit(torch.randn(8))
    r = pd.unit(torch.randn(8))
    mask = torch.tensor([[False, False, True, True, True]])
    das = pd.ClampHook(u, targets=torch.tensor([2.5]), mask=mask, decode_all=False)
    rnd = pd.ClampHook(u, targets=torch.tensor([2.5]), mask=mask, decode_all=False, random_direction=r)
    h_das = das(None, None, (h.clone(),))[0]
    h_rnd = rnd(None, None, (h.clone(),))[0]
    gap = (2.5 - h[0] @ u).abs()
    assert torch.allclose((h_das[0] - h[0]).norm(dim=-1)[2:], gap[2:], atol=1e-5)
    assert torch.allclose((h_rnd[0] - h[0]).norm(dim=-1)[2:], gap[2:], atol=1e-5)
    assert torch.equal(h_das[0, :2], h[0, :2]) and torch.equal(h_rnd[0, :2], h[0, :2])
    assert torch.allclose(h_das[0, 2:] @ u, torch.full((3,), 2.5), atol=1e-5)
    direction_of_rnd = (h_rnd[0, 2] - h[0, 2]) / (h_rnd[0, 2] - h[0, 2]).norm()
    assert torch.allclose(direction_of_rnd.abs(), r.abs(), atol=1e-5)


def test_mean_difference_direction_and_endpoints():
    a = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    b = torch.tensor([[0.0, 1.0], [0.0, 3.0]])
    w = pd.mean_difference_direction(a, b)
    assert torch.allclose(w, torch.tensor([1.0, -1.0]) / 2 ** 0.5)
    c = pd.coordinates(a, w)
    assert abs(c["mean"] - 2 / 2 ** 0.5) < 1e-6


def test_generate_prompts_writes_prefill_and_every_decode_step(bundle):
    p = _prompt(bundle)
    w = pd.unit(torch.randn(bundle.hidden_size, generator=torch.Generator().manual_seed(2)))
    plain = pd.generate_prompts(bundle, [p, _prompt(bundle, "Why?")], max_new_tokens=5, batch_size=2)
    assert len(plain) == 2 and "write" not in plain[0]
    recs = pd.generate_prompts(bundle, [p, _prompt(bundle, "Why?")], max_new_tokens=5, batch_size=2,
                               layer=1, direction=w, targets=0.7)
    for rec in recs:
        assert rec["write"]["prefill_positions"] > 0
        assert rec["write"]["decode_steps_total"] == 4
        # own steps: tokens 2..m from m-1 decode steps (+1 for the stop token when finished)
        expected_own = rec["n_tokens"] if rec["finished"] else max(rec["n_tokens"] - 1, 0)
        assert rec["write"]["decode_steps_own"] == min(expected_own, 4)
        assert rec["write"]["prefill_rel"] >= 0
        assert rec["stop_id"] is None or rec["stop_id"] in pd.stop_ids(bundle)
    per_q = pd.generate_prompts(bundle, [p], max_new_tokens=3, batch_size=1, layer=0, direction=w, targets=[0.1])
    assert per_q[0]["write"]["decode_steps_total"] in (2, 1, 0)


def test_driver_pairs_wiring(bundle):
    import e43_qa_prompt_das as drv
    systems = {"x": SYSTEM, "x_E": SYSTEM + " " + drv.E_SENTENCE,
               "x_conf": "Be confident. Answer with a few words.", "x_caut": "Be cautious. Answer with a few words."}
    eot = pd.turn_terminator_ids(bundle)
    tok = bundle.tokenizer
    ids = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    qs = [{"pair_id": "q1", "question": "What is 2+2?", "best_answer": "Four", "positive_answer": "4"},
          {"pair_id": "q2", "question": "Who?", "best_answer": "Nobody", "positive_answer": "Nobody"}]
    def ans(text, stop=128009):
        t = ids(text)
        return {"text": text, "token_ids": t, "finished": True, "stop_id": stop, "n_tokens": len(t), "refusal_rule": False}
    frozen = {"eot_ids": eot, "questions": [
        {"pair_id": "q1", "answers": {n: ans("4") for n in drv.PROMPT_NAMES}},
        {"pair_id": "q2", "answers": {n: ans("") for n in drv.PROMPT_NAMES}},
        {"pair_id": "q3", "answers": {n: ans("Rome", stop=128001) for n in drv.PROMPT_NAMES}},
        {"pair_id": "q4", "answers": {n: ans("Paris") for n in drv.PROMPT_NAMES}}]}
    qs = qs + [{"pair_id": "q3", "question": "Capital of Italy?", "best_answer": "Rome", "positive_answer": "Rome"},
               {"pair_id": "q4", "question": "What's an opinion you changed?", "best_answer": "I have no comment",
                "positive_answer": "I do not have opinions"}]
    pairs, skipped = drv.build_pairs(bundle, "E", qs, frozen, systems, USER, 24)
    assert [p["pair_id"] for p in pairs] == ["q1"]
    assert skipped == [{"pair_id": "q2", "reason": "empty_frozen_answer"},
                       {"pair_id": "q3", "reason": "nonstandard_stop_token_128001"},
                       {"pair_id": "q4", "reason": "best_answer_personal_refusal"}]
    assert drv.gold_target_text({"best_answer": "I have no comment"}) == (None, "best_answer_personal_refusal")
    assert drv.gold_target_text({"best_answer": " Four "}) == ("Four", None)
    plus, minus = pairs[0]["plus"], pairs[0]["minus"]
    assert plus["base"] == "x" and plus["source"] == "x_E" and plus["encoded"]["answer_text"] == "Four"
    assert minus["base"] == "x_E" and minus["source"] == "x" and minus["encoded"]["answer_ids"] == ids("4") + eot
    assert drv.E_SENTENCE in tok.decode(minus["encoded"]["prefix_ids"])
    assert drv.E_SENTENCE not in tok.decode(plus["encoded"]["prefix_ids"])
    c_pairs, c_skipped = drv.build_pairs(bundle, "C", qs, frozen, systems, USER, 24)
    assert [p["pair_id"] for p in c_pairs] == ["q1", "q4"]      # C needs no gold; q4 is usable for C
    assert c_pairs[0]["plus"]["base"] == "x_conf" and c_pairs[0]["plus"]["source"] == "x_caut"
    assert c_pairs[0]["plus"]["target_spec"] == "frozen:x_caut" and c_pairs[0]["minus"]["target_spec"] == "frozen:x_conf"
    audit = drv.pair_audit(pairs, skipped)
    assert audit["n_pairs"] == 1 and audit["n_skipped"] == 3 and audit["plus"]["terminator_appended"] == 1
    assert drv.negation_count(["I don't know", "nothing", "yes"]) == 2
    assert drv.normalize_text("  The Answer. ") == "the answer"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
