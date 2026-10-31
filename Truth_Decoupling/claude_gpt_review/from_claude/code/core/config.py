"""JSON config loading for the periphery.

Rules:
  * Configs are plain JSON (no YAML dependency on the GPU box).
  * String values may contain ${ENV_VAR} or ${ENV_VAR:-default}; they are expanded
    at load time so model roots / data roots are never hard-coded.
  * `load_experiment(path)` resolves the `model`, `datasets`, `prompts` references
    (names or paths) under experiments/configs/ and deep-merges `overrides`.
"""
import copy
import json
import os
import re

CONFIG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value):
    """Recursively expand ${VAR} / ${VAR:-default} in strings inside dict/list."""
    if isinstance(value, str):
        def repl(m):
            name, default = m.group(1), m.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise KeyError("config references undefined environment variable ${%s}" % name)
        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def deep_merge(base, override):
    """Return a new dict: `override` recursively merged on top of `base`."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_json_config(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return expand_env(raw)


def resolve_config_path(kind, name_or_path):
    """`kind` in {models, datasets, prompts, experiments}. Accepts a bare name or a path."""
    if os.path.isfile(name_or_path):
        return name_or_path
    cand = os.path.join(CONFIG_ROOT, kind, name_or_path)
    if os.path.isfile(cand):
        return cand
    if not cand.endswith(".json") and os.path.isfile(cand + ".json"):
        return cand + ".json"
    raise FileNotFoundError("config %s/%s not found (looked in %s)" % (kind, name_or_path, CONFIG_ROOT))


def load_model_config(name_or_path):
    cfg = load_json_config(resolve_config_path("models", name_or_path))
    cfg.setdefault("name", os.path.splitext(os.path.basename(name_or_path))[0])
    cfg.setdefault("dtype", "bfloat16")
    cfg.setdefault("device", "cuda")
    cfg.setdefault("layers_attr", "model.layers")
    cfg.setdefault("attn_attr", "self_attn")
    cfg.setdefault("chat", {})
    cfg["chat"].setdefault("supports_system", True)
    cfg["chat"].setdefault("thinking_stub", "")
    return cfg


def load_dataset_config(name_or_path):
    cfg = load_json_config(resolve_config_path("datasets", name_or_path))
    cfg.setdefault("name", os.path.splitext(os.path.basename(name_or_path))[0])
    return cfg


def load_prompt_config(name_or_path):
    return load_json_config(resolve_config_path("prompts", name_or_path))


def load_experiment(name_or_path):
    """Load an experiment config and resolve its references.

    Expected keys: model (name), datasets (list of names), prompts (name),
    optional overrides for model/datasets/prompts, plus experiment-specific keys.
    """
    exp = load_json_config(resolve_config_path("experiments", name_or_path))
    exp.setdefault("name", os.path.splitext(os.path.basename(name_or_path))[0])
    resolved = dict(exp)
    if "model" in exp:
        resolved["model"] = deep_merge(load_model_config(exp["model"]), exp.get("model_overrides", {}))
    if "datasets" in exp:
        resolved["datasets"] = [
            deep_merge(load_dataset_config(d), exp.get("dataset_overrides", {}).get(d, {}))
            for d in exp["datasets"]
        ]
    if "prompts" in exp:
        resolved["prompts"] = deep_merge(load_prompt_config(exp["prompts"]), exp.get("prompt_overrides", {}))
    return resolved
