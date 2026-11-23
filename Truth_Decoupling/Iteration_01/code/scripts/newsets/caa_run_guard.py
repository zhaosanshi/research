"""CPU-side input checks for the CAA launchers. Run from the Truth_Decoupling data root."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path("Iteration_01")
FROZEN = ROOT / "code/configs/newsets/frozen_models_20261119.json"
VECTOR_SCRIPT = ROOT / "code/scripts/newsets/caa_vector.py"
MCQ = ROOT / "code/scripts/capability/cap_mcq_sweep.py"
IDS = ROOT / "capability/sets/mmlu_14042_20261120.json"
ITEMS = Path("data_external/mmlu_20261119/all/test-00000-of-00001.parquet")


LEGACY_SCRIPT = "a10bfe66088711a159f12acbae64262b0ab29f3c467520bc704762dffb381f23"
LEGACY_VECTOR = "e74559e75360a32e9e9d62e374bfa22b415de94bef6107222349d24cf9d6a420"
LEGACY_SIDECAR = "774e3713a1fb64facf8232abd42f4541062cc4c4688fa9a359b9e33bf2c31343"
LEGACY_CHECK = "13aafb8c94c0de8fdbf4e51c3f0560145b1d7151c310458917f7141e39d66626"
LEGACY_MIRROR = "baa00c3438807befaba16b0ac3be5f6894fafcb25bec44b58b053eb4491d22a1"
LEGACY_REPEAT = "398dc19826cd29a57bc4b7bf86f6b2aa9c504b618abd77618ffa8b93bc9db3d4"

# This revision only changed acceptance and readers. Preceding fit/check files
# remain valid for these exact archived payloads, after the usual source checks.
PRE_ALIAS_SCRIPT = "ef185badca118054642302abfcef019087828d780413b87e80998ff4007c906f"
# vectors.pt, its JSON sidecar, successful check.json (None for the failed 27B check)
PRE_ALIAS = {
    "qwen3.5-9b": (
        "7127dc723660e550057b2e2201914723645eb3349a567eec71d03dbd43be5899",
        "8637bcd1a2bff8f9bc102b6662c588566f82a422b76d6b2ad1c96ab1dfe38f9e",
        "8ead63558bf8fb4916db0bd246d5aee515e554532a4581b23256f3cd9de5b5bd"),
    "qwen3.8-27b": (
        "ebacf7921bf95d9ae01c9d3ae6828abd1beac1d5915ea767faf7b891ca7cf2a7",
        "c8b35d4464ebc88dd7df005ee67671f3a2a316bbd702eb24fe8376f262c00d17",
        None),
    "gemma3-12b": (
        "19d75edb07b12dac2aadefeffc9c4686a1d7b3219aa847c9e752194ebb4a43ea",
        "5ea122d500e364d6821a92199bf1c86f4fab44203e9602d09281b6479b943d8c",
        "48c3f8098873793d5841b21719ee1e1337f48c99a4748aec474783f24be32af2"),
    "gemma3-27b": (
        "51cac40e028d6f11f3b70917408dabc408ee6de4f008a6b79c27c1048e866462",
        "057bd37459193891534b782aff62df9d6b859afa198ff8c1c7a6ea734202936b",
        "fee8d47f4ab7fe47d018498601d3f724e6d7dccb1fcf0891bb22f4ad49751471"),
}


def engine_identity_matches(record, expected, bank_file):
    """E44 writes the model specified by its source bank, which may be a device-map alias."""
    identity = record["identity"]
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False,
                                       separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    source = Path(bank_file)
    return (digest == record["identity_sha256"] == expected
            and identity["source_files_sha256"]["bank.json"] == sha(source)
            and record["model"] == identity["model"] == load(source)["model"])


def require(flag, why):
    if not flag:
        raise SystemExit("CAA: " + why)


def load(path):
    return json.loads(path.read_bytes())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_acceptance(model, acceptance=False, layer=None, batches=None):
    base = ROOT / "baselines/caa" / model
    folder = base / "acceptance"
    config = load(FROZEN)["models"][model]
    engine = Path(config["code"]) / "scripts/e44_eval_extra.py"
    fit = load(base / "vectors.pt.json")
    check = load(folder / "check.json")
    mirror = base / "run_caa"
    mapping = load(mirror / "MIRROR.json")
    identity = fit["identity_sha256"]
    vd = sha(base / "vectors.pt")
    require(check["passed_2_4"] is True and check["phase"] == "check" and check["model"] == fit["model"] == model,
            "archived check did not pass for this model")
    current = sha(VECTOR_SCRIPT)
    old_fit = (model == "llama3.1-8b" and fit["script_sha256"] == LEGACY_SCRIPT and vd == LEGACY_VECTOR
               and sha(base / "vectors.pt.json") == LEGACY_SIDECAR)
    old_check = (old_fit and check["script_sha256"] == LEGACY_SCRIPT and sha(folder / "check.json") == LEGACY_CHECK)
    previous = PRE_ALIAS.get(model)
    prior_fit = (previous is not None and fit["script_sha256"] == PRE_ALIAS_SCRIPT
                 and (vd, sha(base / "vectors.pt.json")) == previous[:2])
    prior_check = (prior_fit and check["script_sha256"] == PRE_ALIAS_SCRIPT
                   and sha(folder / "check.json") == previous[2])
    require(fit["script_sha256"] == current or old_fit or prior_fit, "unrecognised fit or historical payload")
    require(check["script_sha256"] == current or old_check or prior_check, "unrecognised check or historical payload")
    require(check["identity_sha256"] == identity and check["vectors_file_sha256"] == mapping["vectors_file_sha256"] == vd
            and fit["engine_sha256"] == check["engine_sha256"] == sha(engine), "fit/check inputs changed")
    require(mapping["model"] == model and mapping["kind"] == "caa" and check["mirror"] == str(mirror)
            and mapping["source_run_dir"] == config["run_dir"], "foreign mirror")
    require(load(base / "models_run_caa.json")["models"][model] == dict(config, run_dir=str(mirror)), "foreign mirror config")
    for name, expected in mapping["frozen_files_sha256"].items():
        require(sha(mirror/name) == expected == sha(Path(config["run_dir"])/name), "frozen split changed: " + name)
    for key in fit["candidate_layers"]:
        rec = mirror / "dirs" / ("E_L%d.pt" % key)
        require(sha(rec) == mapping["layers"][str(key)]["file_sha256"], "direction record changed: " + str(key))
    if acceptance:
        require(batches > 0 and check["first_batches"] == batches and check["questions"] == batches * 8,
                "requested and archived acceptance sizes differ")
        zero=check["zero"]
        require(set(zero)=={str(k) for k in fit["candidate_layers"]} and all(
                z["amount"] == 0 and z["same_as_no_write"] == z["of"] == check["questions"]
                and z["prefill_positions_written_per_row"] == [1] for z in zero.values()),
                "zero check is incomplete for the candidate layers")
        ids = load(mirror / "frozen_tune.json")["question_ids"][:batches*8]
        rec = base / "mirror_e/dirs" / ("E_L%d.pt" % layer)
        digest = sha(rec)
        require(digest == sha(Path(config["run_dir"])/"dirs"/rec.name), "copied reference is stale")
        reference = None
        for run in ("A","B"):
            a = load(folder / ("c1_run%s.json" % run))
            require(a["execution_complete"] is True and a["baseline_reproduction_verified"] is True and
                    a["frozen_file_sha256"] == mapping["frozen_files_sha256"]["frozen_tune.json"] and
                    a["identity_sha256"] == identity and a["split"] == "tune" and a["arm"] == "x:E%d.md.p2" % layer and
                    a["positions"] == "gen" and a["question_ids"] == ids and [r["pair_id"] for r in a["rows"]] == ids,
                    "reference comparison has wrong run, arm, rows, or baseline status")
            require(a["own_sha256"]["script"] == check["engine_sha256"]
                    and all(r["write"]["prefill_positions"] == 1 for r in a["rows"]), "wrong reference engine or positions")
            contents = [(r["text"],r["token_ids"],r["finished"],r["stop_id"]) for r in a["rows"]]
            if reference is not None:
                require(contents==reference, "archived repeats differ")
            reference = contents
            part, = a["parts"]
            require(part["direction_file_sha256"] == digest and part["kind"] == "md" and part["layer"] == layer
                    and part["beta"] == 2 and math.isfinite(part["amount"]) and part["amount"] > 0,
                    "reference comparison was made with another direction or strength")
        return
    token = (folder/"ACCEPT_DONE").read_text().strip().split()
    if not token and old_check and sha(mirror/"MIRROR.json") == LEGACY_MIRROR and all(
            sha(folder/("c1_run%s.json"%r)) == LEGACY_REPEAT for r in ("A","B")):
        token = ["16","2"]  # the audited pilot can be revalidated without GPU inference
    require(len(token)==2 and all(x.isdecimal() for x in token),"unbound acceptance marker; rerun acceptance")
    check_acceptance(model, True, int(token[0]), int(token[1]))
    return mapping, fit


def budget(model, layer, multipliers, mode):
    base = ROOT / "baselines/caa" / model
    meta = load(base / "run_caa/MIRROR.json")
    tensor_sha = sha(base / "run_caa/dirs" / ("E_L%d.pt" % layer))
    require(meta["layers"][str(layer)]["file_sha256"] == tensor_sha, "MMLU direction is stale")
    path = base / "mmlu" / ("L%d.json" % layer)
    d = load(path)
    id_set = load(IDS)
    roster = id_set["ids"]
    item_sha = sha(ITEMS)
    require(d["protocol"] == "cap_mcq_sweep_v1" and d["model"] == model and d["identity_sha256"] == load(base/"vectors.pt.json")["identity_sha256"]
            and d["ids_file_sha256"] == sha(IDS) and d["items_file_sha256"] == id_set["source_file_sha256"] == item_sha
            and d["script_sha256"] == sha(MCQ) and d["n"] == len(set(roster)) == len(roster) == 14042 and not d["skipped"],
            "wrong MMLU sources, model, or dataset")
    arm_names = ["x"] + ["x:E%d.md.p%s@gen" % (layer, s) for s in multipliers]
    require(set(arm_names)==set(d["arms"]), "MMLU arm set differs from this run")
    base_rows = d["arms"]["x"]["rows"]
    truth = [r["gold"] for r in base_rows]
    tables = []
    for arm in arm_names:
        rows = d["arms"][arm]["rows"]
        require([r["id"] for r in rows] == roster and [r["gold"] for r in rows] == truth and
                all(r["gold"] in ("A","B","C","D") and r["pick"] in ("A","B","C","D") for r in rows),
                "MMLU row identity or labels differ")
        if arm != "x":
            require(d["arms"][arm]["positions"] == "gen", "wrong MMLU write positions")
        tables.append({r["id"]:r["pick"] == r["gold"] for r in rows})
    receipt = path.with_suffix(".receipt.json")
    expected = {"direction_sha256":tensor_sha,"vectors_sha256":meta["vectors_file_sha256"],"output_sha256":sha(path)}
    if mode == "fresh":
        temp = receipt.with_name(receipt.name + ".%d.tmp"%os.getpid())
        temp.write_text(json.dumps(expected)); temp.replace(receipt)
    else:
        require(receipt.is_file() and load(receipt)==expected, "cached MMLU has no matching CAA receipt")
    sys.path.insert(0, "Iteration_01/diag")
    import per_model_table as pm
    selection, _ = pm.halves(roster)
    for dose, correct in zip(multipliers,tables[1:]):
        print(dose, 100*sum(correct[i]-tables[0][i] for i in selection)/len(selection))


if __name__ == "__main__":
    command, model, *args = sys.argv[1:]
    if command=="accept":
        check_acceptance(model, True, int(args[0]), int(args[1]))
    elif command=="grid":
        _,fit=check_acceptance(model)
        layers=[int(s) for s in args[0].split()]
        require(bool(layers) and len(set(layers))==len(layers) and set(layers)<=set(fit["candidate_layers"]), "undeclared or repeated layer")
        if args[1] != "-":
            values=args[1].split()
            require(bool(values) and all(v in ("0.5","1","2","4") for v in values)
                    and list(map(float,values))==sorted(set(map(float,values))), "invalid or unsorted multiplier grid")
    elif command=="budget":
        budget(model,int(args[0]),args[1].split(),args[2])
    else:
        raise SystemExit("unexpected CAA guard mode")
