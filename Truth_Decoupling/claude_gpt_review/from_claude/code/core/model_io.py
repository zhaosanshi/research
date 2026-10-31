"""Model loading, chat prompt construction and activation capture.

Everything model-specific comes from the model config (periphery):
  path, dtype, device / device_map, layers_attr ("model.layers"),
  attn_attr ("self_attn"), chat.supports_system, chat.thinking_stub.

Activation convention (same as v12): the *block output* of decoder layer l,
i.e. the residual stream after layer l (== hidden_states[l+1]), at the last
prompt token. Attention output = output[0] of the layer's self-attention module
(after o_proj), same position.

Left padding + explicit position_ids make batched capture bit-compatible with
single-sample forwards (HF only derives position_ids from the attention mask
inside generate(), not in a plain forward()).
"""
import contextlib

import torch

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _resolve_attr(obj, dotted):
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


class ModelBundle:
    """Model + tokenizer + config, with helpers to reach decoder layers."""

    def __init__(self, model, tokenizer, cfg):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = next(model.parameters()).device
        # multimodal wrappers (Gemma-3 4B/12B/27B, Mistral-3) keep the LM sizes under config.text_config
        tc = getattr(model.config, "text_config", None)
        if tc is None or not hasattr(tc, "hidden_size"):
            tc = model.config
        self.hidden_size = int(tc.hidden_size)
        self.num_layers = int(tc.num_hidden_layers)
        self.name = cfg.get("name", "model")

    def layers(self):
        return _resolve_attr(self.model, self.cfg.get("layers_attr", "model.layers"))

    def layer(self, idx):
        if not (0 <= idx < self.num_layers):
            raise IndexError("layer %d out of range [0, %d)" % (idx, self.num_layers))
        return self.layers()[idx]

    def attn(self, idx):
        return getattr(self.layer(idx), self.cfg.get("attn_attr", "self_attn"))

    # ---- prompts ---------------------------------------------------------- #
    def build_chat(self, system, user, add_generation_prompt=True):
        chat = self.cfg.get("chat", {})
        if system and chat.get("supports_system", True):
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        else:
            merged = (system.strip() + "\n\n" + user) if system else user
            msgs = [{"role": "user", "content": merged}]
        text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)
        stub = chat.get("thinking_stub", "")
        if stub and add_generation_prompt:
            text += stub
        return text

    def encode(self, texts, padding=True):
        if isinstance(texts, str):
            texts = [texts]
        enc = self.tokenizer(texts, return_tensors="pt", padding=padding, add_special_tokens=False)
        return {k: v.to(self.device) for k, v in enc.items()}

    def decode_new(self, out_ids, prompt_len):
        return self.tokenizer.decode(out_ids[prompt_len:], skip_special_tokens=True)


def load_model(cfg):
    """Load tokenizer + causal LM from a model config dict (see configs/models/*.json)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    trust = bool(cfg.get("trust_remote_code", True))
    tok = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=trust)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    kwargs = {"dtype": DTYPES[cfg.get("dtype", "bfloat16")], "trust_remote_code": trust}
    device_map = cfg.get("device_map")
    if device_map:
        kwargs["device_map"] = device_map
        # sdpa + left-padded batches + a model split over several GPUs gives NaN activations from layer 0 (Qwen3-14B/32B,
        # transformers 4.57); eager attention is correct there, so it is the default for split models.
        kwargs["attn_implementation"] = cfg.get("attn_implementation", "eager")
    elif cfg.get("attn_implementation"):
        kwargs["attn_implementation"] = cfg["attn_implementation"]
    model = AutoModelForCausalLM.from_pretrained(cfg["path"], **kwargs)
    if not device_map:
        model = model.to(cfg.get("device", "cuda"))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return ModelBundle(model, tok, cfg)


# --------------------------------------------------------------------------- #
# Forward helpers
# --------------------------------------------------------------------------- #

def position_ids_from_mask(attention_mask):
    pos = attention_mask.long().cumsum(-1) - 1
    return pos.clamp(min=0)


def forward_logits(bundle, inputs, grad=False):
    """Plain forward with explicit position ids. Returns logits [B, T, V]."""
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        out = bundle.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=position_ids_from_mask(inputs["attention_mask"]),
            use_cache=False,
        )
    return out.logits


def _as_hidden(output):
    return output[0] if isinstance(output, tuple) else output


@contextlib.contextmanager
def capture_hooks(bundle, layers, components=("resid",), position="last"):
    """Context manager that records activations at `layers`.

    Yields a dict `store`; after the forward it holds
      store[("resid", l)] -> tensor [B, d] (float32, cpu)   last-token block output
      store[("attn",  l)] -> tensor [B, d]                   last-token attention output
    With position="all" the full [B, T, d] tensors are kept (on device, model dtype).
    """
    store = {}
    handles = []

    def make_hook(key):
        def hook(module, inp, out):
            h = _as_hidden(out)
            if position == "last":
                store[key] = h[:, -1, :].detach().float().cpu()
            else:
                store[key] = h.detach()
        return hook

    for l in layers:
        if "resid" in components:
            handles.append(bundle.layer(l).register_forward_hook(make_hook(("resid", l))))
        if "attn" in components:
            handles.append(bundle.attn(l).register_forward_hook(make_hook(("attn", l))))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


def capture_last_token(bundle, texts, layers, components=("resid",), batch_size=8, return_logits=False):
    """Run `texts` through the model and collect last-token activations.

    Returns (acts, logits_last) where acts[(comp, l)] is a float32 tensor [N, d]
    and logits_last is [N, V] float32 (or None).
    """
    acts = {}
    logits_all = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        inputs = bundle.encode(chunk)
        with capture_hooks(bundle, layers, components) as store:
            logits = forward_logits(bundle, inputs, grad=False)
        for k, v in store.items():
            if torch.isnan(v).any():
                raise RuntimeError("NaN activations at layer %s (batch starting at item %d): refusing to continue -- "
                                   "check attention implementation / device placement" % (k[1], i))
            acts.setdefault(k, []).append(v)
        if return_logits:
            logits_all.append(logits[:, -1, :].detach().float().cpu())
        del logits
    acts = {k: torch.cat(v, dim=0) for k, v in acts.items()}
    logits_last = torch.cat(logits_all, dim=0) if return_logits else None
    return acts, logits_last


@torch.no_grad()
def greedy_generate(bundle, inputs, max_new_tokens):
    out = bundle.model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=bundle.tokenizer.pad_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    return [bundle.tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in out], out
