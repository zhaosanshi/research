"""Offline audit of published artifacts; never loads or claims to rerun model weights."""
import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("PHANTOM_PILOT", str(HERE.parents[1] / "00_pilot"))).resolve()
RESULTS = ROOT / "results"
CODE = ROOT / "code"

def rows(name):
    return [json.loads(x) for x in (RESULTS / name).read_text().splitlines() if x.strip()]

def gmean(v):
    return math.exp(sum(math.log(x) for x in v) / len(v))

report = {}
# These labels were read from the actual text, applying the document's A/B/C rule.
# Their aggregate counts reproduce the repository's 9/15 and 6/11 claims.
labels = {"m03": ["ACA", "AAB", "CCC", "AAA", "ACA"],
          "c03": ["ACA", "A", "CCC", "ACA", "A"]}
sampling = {}
for cell, cell_labels in labels.items():
    rr = rows(f"en_plain_e2e_x_Llama-3.1-8B-Instruct_{cell}.jsonl")
    per_seed = []
    for r, lab in zip(rr, cell_labels):
        cc = [x for x in r["hit_continuations"] if x["first"] == "user"]
        assert len(cc) == len(lab)
        per_seed.append({"seed": r["filler_seed"], "p_user": r["p_user"],
                         "rank": r["user_rank"], "mass_before": r["user_cum_before"],
                         "draw_hits": r["first_hits"]["user"]["n"],
                         "continued": len(cc), "A": lab.count("A"), "B": lab.count("B"),
                         "labels": lab, "texts": [x["text"] for x in cc]})
    hits = sum(x["draw_hits"] for x in per_seed)
    n = sum(r["n_first"] for r in rr)
    a = sum(x["A"] for x in per_seed)
    k = sum(x["continued"] for x in per_seed)
    sampling[cell] = {
        "first_draws": n, "hits": hits, "continued": k, "A": a,
        "full_samples": sum(r["full_n"] for r in rr),
        "full_auto_fake": sum(r["full_fake"] for r in rr),
        "pooled_product_per_million": hits / n * a / k * 1e6,
        "context_weighted_hits_per_million": sum(x["draw_hits"] * x["A"] / x["continued"] for x in per_seed) / n * 1e6,
        "context_weighted_exact_prob_per_million": sum(x["p_user"] * x["A"] / x["continued"] for x in per_seed) / len(per_seed) * 1e6,
        "all_user_outside_top50": all(x["rank"] > 50 for x in per_seed),
        "all_user_outside_top_p_099": all(x["mass_before"] > .99 for x in per_seed),
        "per_seed": per_seed,
    }
report["sampling"] = sampling

cross = rows("en_plain_cross_Llama-3.1-8B-Instruct.jsonl")
probs = {}
for r in cross:
    probs.setdefault(r["condition"], []).append(r["p_role_exact"]["user"])
report["cross_gmeans"] = {c: gmean(v) for c, v in probs.items()}
report["contamination_ratios"] = {k: report["cross_gmeans"][f"X20k_{k}03"] / report["cross_gmeans"][f"X20k_{k}00"] for k in ["m", "c"]}

# Import and exercise actual repository functions that require no heavy dependency.
os.environ["PROBE_LANG"] = "en"
os.environ["CONTAM_STYLE"] = "plain"
os.environ.pop("CONTAM_ROLE_WORD", None)
sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True
import turn_boundary_probe as tbp
import e2e_sample
import head_attrib
build_checks = []
for turns in (0, 1, 5):
    c, mp = tbp.build_conv(n_turns=turns, contam_k=0, thin_input=True, pre_tool_signoff=True,
                         filler=[("q1", "a1"), ("q2", "a2"), ("q3", "a3")])
    d, _ = tbp.build_conv(n_turns=turns, contam_k=3, thin_input=True, pre_tool_signoff=True,
                        filler=[("q1", "a1"), ("q2", "a2"), ("q3", "a3")])
    assert [m["role"] for m in c] == [m["role"] for m in d]
    changed = [i for i, (x, y) in enumerate(zip(c, d)) if x != y]
    assert len(changed) == min(turns, 3)
    assert all(c[i]["role"] == "assistant" and d[i]["content"] == c[i]["content"] + tbp.FAKE_USER_TURN_TEXT for i in changed)
    build_checks.append({"turns": turns, "changed": changed, "measurement": mp})
report["build_conv_checks"] = build_checks
report["known_classifier_false_positive"] = e2e_sample.scan("user\nI've completed the requested code changes.")
# An ordinary body occurrence of user is classed as fake_user_tok.
sp = head_attrib.spans_from_ids([0, 10, 99, 11, 10, 20, 31, 30, 11, 10, 21, 31, 20, 11, 10, 20, 31, 30, 11, 10, 21, 31], 20, 21, 10, 11)
assert 12 in sp["fake_user_tok"]
report["span_heuristic_example"] = {k: sorted(v) for k, v in sp.items()}

# Execute unmodified constructor/update and processor-selection methods taken from
# Transformers 4.57.0 official source. Validation and processor objects are stubbed;
# this establishes config flow, not model execution or tensor-kernel behavior.
cfg_tree = ast.parse((HERE / "vendor" / "transformers_457_configuration_utils.py").read_text())
cfg_cls = next(x for x in cfg_tree.body if isinstance(x, ast.ClassDef) and x.name == "GenerationConfig")
chosen = [x for x in cfg_cls.body if isinstance(x, ast.FunctionDef) and x.name in ("__init__", "update")]
cfg_code = ast.Module(body=chosen, type_ignores=[])
ns = {"__version__": "4.57.0"}
exec(compile(cfg_code, "official_GenerationConfig_methods", "exec"), ns)
Config = type("Config", (), {"__init__": ns["__init__"], "update": ns["update"], "validate": lambda self: None})
cfg = Config(top_p=.9, do_sample=True)
cfg.update(temperature=1., top_p=None, do_sample=True)
gen_tree = ast.parse((HERE / "vendor" / "transformers_457_generation_utils.py").read_text())
mix = next(x for x in gen_tree.body if isinstance(x, ast.ClassDef) and x.name == "GenerationMixin")
method = next(x for x in mix.body if isinstance(x, ast.FunctionDef) and x.name == "_get_logits_processor")
for arg in method.args.args + method.args.kwonlyargs:
    arg.annotation = None
method.returns = None
class TopKStub:
    def __init__(self, **kw): self.kw = kw
ns2 = {"LogitsProcessorList": list, "TopKLogitsWarper": TopKStub}
exec(compile(ast.Module(body=[method], type_ignores=[]), "official_logits_selection", "exec"), ns2)
stub_self = types.SimpleNamespace(_merge_criteria_processor_list=lambda a, b: a + b)
processors = ns2["_get_logits_processor"](stub_self, cfg)
assert cfg.top_k == 50 and cfg.top_p is None and processors[0].kw["top_k"] == 50
report["official_config_flow"] = {"top_k": cfg.top_k, "top_p": cfg.top_p, "temperature": cfg.temperature,
                                  "selected_processors": [{"type": type(x).__name__, **x.kw} for x in processors]}
cfg.update(top_k=0, top_p=1.)
assert not ns2["_get_logits_processor"](stub_self, cfg)
report["explicit_full_distribution_control"] = {"top_k": cfg.top_k, "top_p": cfg.top_p, "processors": []}

syntax = {"python": 0, "shell": 0, "jsonl_files": 0, "jsonl_rows": 0, "json_files": 0}
for p in CODE.glob("*.py"):
    ast.parse(p.read_text()); syntax["python"] += 1
for p in CODE.glob("*.sh"):
    subprocess.run(["bash", "-n", str(p)], check=True, capture_output=True); syntax["shell"] += 1
duplicates = []
for p in RESULTS.glob("*.jsonl"):
    syntax["jsonl_files"] += 1
    seen = {}
    for line, s in enumerate(p.read_text().splitlines(), 1):
        if not s.strip(): continue
        r = json.loads(s)
        syntax["jsonl_rows"] += 1
        if "filler_seed" in r:
            key = (r.get("condition", r.get("cell")), r["filler_seed"],
                   r.get("word", r.get("contam_word", r.get("p_contam_word", {}).get("word", "user"))),
                   r.get("native_contam"), r.get("measure"))
        else:
            key = (r.get("kind"), r.get("user"))
        if key in seen:
            duplicates.append({"file": p.name, "first_line": seen[key], "repeat_line": line, "key": key})
        seen[key] = line
for p in list(RESULTS.glob("*.json")) + list((CODE / "lang").glob("*.json")):
    json.loads(p.read_text()); syntax["json_files"] += 1
report["syntax_and_parse"] = syntax
report["duplicate_logical_keys"] = duplicates
# Exercise a real batch shell script with a failing Docker substitute: no model,
# network service, container, or source-tree mutation is involved.
with tempfile.TemporaryDirectory(prefix="phantom_failure_") as td:
    td = Path(td)
    (td / "results").mkdir()
    (td / "bin").mkdir()
    for fn in ("filler_en_coding.jsonl", "filler_en_general.jsonl"):
        (td / "results" / fn).write_text("{}\n")
    docker = td / "bin" / "docker"
    docker.write_text("#!/bin/sh\nexit 17\n")
    docker.chmod(0o755)
    env = dict(os.environ, PHANTOM_ROOT=str(td), MODEL_ROOT=str(td / "models"), IMG="audit-fake-image")
    env["PATH"] = str(td / "bin") + os.pathsep + env["PATH"]
    run = subprocess.run(["bash", str(CODE / "job_en_plain.sh")], env=env, capture_output=True, text=True)
    report["batch_failure_propagation"] = {"script": "job_en_plain.sh", "docker_exit": 17,
                                          "script_exit": run.returncode,
                                          "reported_failures": run.stdout.count("FAILED"),
                                          "last_stdout_line": run.stdout.splitlines()[-1]}
    assert run.returncode == 0 and "EN PLAIN DONE" in run.stdout and "FAILED" in run.stdout
(HERE / "audit_recompute_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
brief = dict(report)
brief["sampling"] = {c: {k: v for k, v in d.items() if k != "per_seed"} for c, d in sampling.items()}
print(json.dumps(brief, ensure_ascii=False, indent=2))
