#!/usr/bin/env python3
"""CPU-only checks already run during the E24 review.

Uses randomly initialized tiny Hugging Face Llama models, with no model download
and no real weights. Imports the current from_claude/core intervention code.

Run from any directory:
    python3 /path/to/from_gpt/verify_real_qa_path_20261102.py

The optional sklearn/audio/image availability switches below are process-local:
the review machine has a pandas/NumPy binary mismatch in an unrelated optional
sklearn import. These components are not used by the text-only Llama checks.
No production code or installed package is modified.
"""
from pathlib import Path
import sys

import transformers.utils.import_utils as import_utils

import_utils._sklearn_available = False
import_utils._librosa_available = False
import_utils._torchvision_available = False

import torch
from transformers import LlamaConfig, LlamaForCausalLM

CODE = Path(__file__).resolve().parent.parent / "from_claude" / "code"
sys.path.insert(0, str(CODE))
from core.intervene import SteerSpec, steer
from core.model_io import ModelBundle, position_ids_from_mask


def tiny_llama(seed, layers, max_positions):
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=max_positions,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
        attention_dropout=0.0,
    )
    model = LlamaForCausalLM(cfg).cpu().eval()
    model.requires_grad_(False)
    return model


@torch.no_grad()
def verify_cache_and_max_calls():
    model = tiny_llama(seed=7, layers=4, max_positions=128)
    bundle = ModelBundle(model, None, {"layers_attr": "model.layers"})
    inputs = torch.tensor([[1, 3, 4, 5], [0, 1, 6, 7]])
    mask = (inputs != 0).long()
    w = torch.randn(32)
    w = w / w.norm()
    spec = SteerSpec(w, 1, mode="add", alpha=2.0, max_calls=1, name="probe")

    base = model(
        input_ids=inputs,
        attention_mask=mask,
        position_ids=position_ids_from_mask(mask),
        use_cache=True,
    )
    with steer(bundle, [spec]) as hooks:
        edited = model(
            input_ids=inputs,
            attention_mask=mask,
            position_ids=position_ids_from_mask(mask),
            use_cache=True,
        )
        difference = float((base.logits[:, -1] - edited.logits[:, -1]).abs().max())
        print("prefill_max_abs_logit_diff", difference)
        assert difference > 0, "prefill edit did not reach the output"

        original_cache = base.past_key_values.to_legacy_cache()
        edited_cache = edited.past_key_values.to_legacy_cache()
        for layer, ((bk, bv), (ek, ev)) in enumerate(zip(original_cache, edited_cache)):
            k_difference = float((bk - ek).abs().max())
            v_difference = float((bv - ev).abs().max())
            print("cache_layer", layer, "max_Kdiff", k_difference, "max_Vdiff", v_difference)
            if layer <= 1:
                assert k_difference == 0 and v_difference == 0
            else:
                assert k_difference > 0 and v_difference > 0

        # Identical next tokens isolate propagation through the cache from any
        # change caused by choosing different first tokens.
        nxt = torch.tensor([[8], [8]])
        next_mask = torch.cat([mask, torch.ones(2, 1, dtype=torch.long)], dim=1)
        next_pos = position_ids_from_mask(next_mask)[:, -1:]
        base_next = model(
            input_ids=nxt,
            attention_mask=next_mask,
            position_ids=next_pos,
            past_key_values=base.past_key_values,
            use_cache=True,
        )
        edited_next = model(
            input_ids=nxt,
            attention_mask=next_mask,
            position_ids=next_pos,
            past_key_values=edited.past_key_values,
            use_cache=True,
        )
        difference = float((base_next.logits - edited_next.logits).abs().max())
        print("same_next_token_decode_max_logit_diff", difference)
        print("n_applied", spec.n_applied, "calls", hooks[1].call_count,
              "first_shape", hooks[1].first_shape)
        assert difference > 0
        assert spec.n_applied == 1
        assert hooks[1].call_count == 3
        assert hooks[1].first_shape == (2, 4, 32)

    # Reference: recompute the whole sequence while writing at the same
    # original prompt position. This should match the cached one-write path.
    def fixed_prompt_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        modified = h.clone()
        modified[:, inputs.shape[1] - 1, :] += 2 * w
        return (modified,) + out[1:] if isinstance(out, tuple) else modified

    handle = model.model.layers[1].register_forward_hook(fixed_prompt_hook)
    try:
        replay = model(
            input_ids=torch.cat([inputs, nxt], dim=1),
            attention_mask=next_mask,
            position_ids=position_ids_from_mask(next_mask),
            use_cache=False,
        )
    finally:
        handle.remove()
    difference = float((edited_next.logits[:, -1] - replay.logits[:, -1]).abs().max())
    print("cached_vs_fixed_prompt_recompute_max_logit_diff", difference)
    assert difference < 1e-5, "cached write differs from fixed-prompt full recomputation"

    # Persistent spec, fresh hook per batch: max_calls must reset for every
    # generate call while n_applied continues accumulating for auditing.
    for batch in range(2):
        before = spec.n_applied
        with steer(bundle, [spec]) as hooks:
            out = model.generate(
                input_ids=inputs,
                attention_mask=mask,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=0,
                eos_token_id=None,
                use_cache=True,
            )
        new_tokens = out.shape[1] - inputs.shape[1]
        applications = spec.n_applied - before
        print("batch", batch, "new_tokens", new_tokens, "edits", applications,
              "calls", hooks[1].call_count, "first_shape", hooks[1].first_shape)
        assert new_tokens == 4
        assert applications == 1
        assert hooks[1].call_count == new_tokens
        assert hooks[1].first_shape == (2, 4, 32)
    print("default_generation_use_cache", model.generation_config.use_cache)
    print("PASS: prefill write propagates through KV cache; max_calls resets per batch")


@torch.no_grad()
def verify_answer_position_shift():
    model = tiny_llama(seed=19, layers=3, max_positions=64)
    inputs = torch.tensor([[1, 4, 7, 9, 12]])
    prompt_length = 4  # four prompt tokens, followed by one answer token
    base = model(inputs, use_cache=False).logits
    w = torch.randn(32)
    w = w / w.norm()

    def answer_only_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        modified = h.clone()
        modified[:, prompt_length:, :] += 3 * w
        return (modified,) + out[1:] if isinstance(out, tuple) else modified

    handle = model.model.layers[1].register_forward_hook(answer_only_hook)
    try:
        edited = model(inputs, use_cache=False).logits
    finally:
        handle.remove()
    first_difference = float((edited[:, prompt_length - 1] - base[:, prompt_length - 1]).abs().max())
    next_difference = float((edited[:, prompt_length] - base[:, prompt_length]).abs().max())
    print("one_token_answer_predictor_change", first_difference)
    print("answer_end_next_token_predictor_change", next_difference)
    assert first_difference == 0, "future-token intervention changed a causal earlier logit"
    assert next_difference > 0
    print("PASS: writing only at answer-token positions misses the first answer-token prediction")


if __name__ == "__main__":
    print("Production intervention source:", CODE / "core" / "intervene.py")
    verify_cache_and_max_calls()
    verify_answer_position_shift()
    print("PASS: all CPU tiny-Llama checks; no real model weights or experiment runs")
