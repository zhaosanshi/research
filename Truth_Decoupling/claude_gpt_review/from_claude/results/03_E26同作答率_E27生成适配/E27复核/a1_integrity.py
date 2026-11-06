"""Task 1: integrity of both archives against artifacts/*manifest.json.

Checks (independently of the collaborator's scripts):
  - archive sha256 vs archive_integrity.json
  - every tar member (read directly from the .tar.gz, no extraction) vs manifest sha256/bytes
  - members present in tar but not in manifest, and vice versa
  - extracted copies (gpt18/gen, gpt18/pilots) vs manifest
  - tar member mtimes (UTC) recorded for the timeline work in task 3
"""
import hashlib, json, sys, tarfile, os, datetime

D = "/Users/airsanshi/github/research/Truth_Decoupling/claude_gpt_review/from_gpt/main_experiment_20261106"
S = "/private/tmp/claude-501/-Users-airsanshi-MyPapers-Truth-Decoupling/1202de30-5944-41d9-af49-9348a318988a/scratchpad"
OUT = S + "/audit_results"

pairs = [
    ("artifacts/generation_results.tar.gz", "artifacts/generation_manifest.json", S + "/gpt18/gen"),
    ("artifacts/real_pilots.tar.gz", "artifacts/real_pilots_manifest.json", S + "/gpt18/pilots"),
]
integ = {d["archive"]: d for d in json.load(open(D + "/archive_integrity.json"))}


def sha(b):
    return hashlib.sha256(b).hexdigest()


report = {}
for arc, man, ext in pairs:
    r = {}
    ab = open(D + "/" + arc, "rb").read()
    r["archive_sha256"] = sha(ab)
    r["archive_sha256_matches_integrity_json"] = sha(ab) == integ[arc]["sha256"]
    m = json.load(open(D + "/" + man))
    mf = {f["path"]: f for f in m["files"]}
    r["manifest_files"] = len(mf)
    r["integrity_json_files"] = integ[arc]["files"]
    tar_files = {}
    mtimes = {}
    nonreg = []
    with tarfile.open(D + "/" + arc, "r:gz") as t:
        for mem in t.getmembers():
            name = mem.name
            if name.startswith("./"):
                name = name[2:]
            if mem.isdir():
                continue
            if not mem.isfile():
                nonreg.append((name, mem.type))
                continue
            data = t.extractfile(mem).read()
            tar_files[name] = (len(data), sha(data))
            mtimes[name] = datetime.datetime.utcfromtimestamp(mem.mtime).strftime("%Y-%m-%dT%H:%M:%SZ")
    r["tar_regular_files"] = len(tar_files)
    r["tar_nonregular_members"] = nonreg
    r["in_tar_not_manifest"] = sorted(set(tar_files) - set(mf))
    r["in_manifest_not_tar"] = sorted(set(mf) - set(tar_files))
    bad = []
    for p, f in mf.items():
        if p in tar_files:
            n, h = tar_files[p]
            if n != f["bytes"] or h != f["sha256"]:
                bad.append({"path": p, "tar": [n, h], "manifest": [f["bytes"], f["sha256"]]})
    r["tar_vs_manifest_mismatches"] = bad
    # extracted copies
    ext_files = {}
    for root, _, files in os.walk(ext):
        for fn in files:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, ext)
            b = open(full, "rb").read()
            ext_files[rel] = (len(b), sha(b))
    r["extracted_files"] = len(ext_files)
    r["extracted_vs_manifest_mismatches"] = [
        p for p, f in mf.items() if ext_files.get(p) != (f["bytes"], f["sha256"])
    ]
    r["extracted_extra"] = sorted(set(ext_files) - set(mf))
    r["tar_mtimes_utc"] = dict(sorted(mtimes.items()))
    report[arc] = r
    print(arc, {k: v for k, v in r.items() if k != "tar_mtimes_utc"})

json.dump(report, open(OUT + "/a1_integrity.json", "w"), indent=1)
