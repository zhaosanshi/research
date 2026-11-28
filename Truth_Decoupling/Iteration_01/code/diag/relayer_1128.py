# Iteration_01 (executor R-bing, Claude Opus 5.5, 2026-11-29 Beijing; work queue R item R6,
# discussion/work_queue/2026-11-28_queue_r.md sections 2 and 4): re-select the E layer of each model on TruthfulQA with
# the rule of section 4, and the C layer by R5 / R6. Read-only: reads answer, verdict and capability files and prints
# what to run next or the tables; no model, no judge call, writes nothing (except mode halfids, one new id list).
# NOT REVIEWED BY GPT YET; the formal run waits for that review.
#
# Rule (section 4, binding):
#   window   layer l (program numbering, from 0) is a candidate when 0.3 * NL < l + 1 <= 0.8 * NL
#   line     per layer, strengths from small to large on GRID; at each strength MMLU (the 1,000-item list, option
#            probability, cap_mcq_sweep.py) and GSM8K (the first 304 frozen items, prompt rule, the model's own token cap)
#            may each lose at most 1 point against no write (MMLU: 100 * (x - arm) <= n; GSM8K: 100 * (x - arm) <= 304,
#            i.e. at most 3 items); the walk stops at the FIRST strength that fails either; the layer's strength is the
#            one before it (none when the smallest fails). MMLU is walked first over the grid (cheap); GSM8K is then
#            walked only up to the strength MMLU allows (ORDER=up, the rule as written: every strength from the smallest
#            is measured). ORDER=down (a cheaper variant that assumes the GSM8K pass / fail is monotone in the strength:
#            from the MMLU strength downward, the first one that passes) is implemented for the review and is NOT the
#            default.
#   effect   single E at that strength, plain prompt x, write rule "prompt", tune split (192), DeepSeek Pro, net = right -
#            wrong (no answer 0), read as depth_scan.py reads it (no-write cells from the main table; every new work dir's
#            no-write cells must be identical to those).
#   choice   the highest net, ties -> the shallower layer; read on the questions where every candidate has a verdict.
#   confirm  the chosen (layer, strength): full MMLU, the selection half of per_model_table.halves (7,021 items, which
#            contains the 1,000 list), and full GSM8K (1,319), each losing at most 1 point. What happens on a failure is
#            not written in the plan: this program only reports it (open question).
#   C        R5 per window layer from the archived per-layer C tables of queue M (tune, Pro, caution arm at 0.25, criterion
#            v4 column A = section 4's plain wording; those tables were generated at the TRAINING rule "content");
#            R6 = the shallowest R5 layer deeper than the chosen E layer; c from 0.25 / 0.5 / 1 / 2 by the tune net of the
#            method (routes_1128.schemes, scheme R1_<c>; ties -> the smaller c, as routes_1128.best).
#
# GRID = 0.5 0.75 1 1.5 2 2.5 3 3.5 4 5 6 8 12 16: the ladder of the 11-24 depth scan (depth_scan.LADDER + EXT), so that
# its MMLU and GSM8K files are reused where (layer, strength) coincide.
# Files (host paths, from /data/sanshizhao/Truth_Decoupling):
#   run      Iteration_01/results/relayer_1128/<ng>/run_E2: byte copies of the E run's frozen_train / frozen_tune and of
#            its window directions + train logs, plus the newly trained window layers (relayer_1128_train.sh)
#   MMLU     Iteration_01/capability/relayer_1128/mmlu1000/<ng>/*.json  (+ the depthscan / layermap / sweep files)
#   GSM8K    Iteration_01/capability/relayer_1128/gsm304/<ng>/b8/       (+ depthscan_gsm8k and the archived full passes)
#   tune     <run>/eval_extra/gen/x--E<l>-w-p<s>-at-prompt_tune.json; work dirs <run>/eval_extra/judge_tune_relayer_1128_E<r>_tune
#   confirm  Iteration_01/capability/relayer_1128/mmlu_half/<ng>/*.json, .../gsm_full/<ng>/b8/
#   C        <run>/eval_extra/gen/x--E<l>-w-p<s>-and-C<c>-w-m<k>-at-prompt_tune.json; work dir judge_tune_relayer_1128_C_tune
# modes (host or container, from /data/sanshizhao/Truth_Decoupling; KEY = llama | q9 | q27 | g12 | g27):
#   window                       the window of every model (against the table of section 4)
#   inv KEY                      window layers with / without a direction in the new run; R5 / R6 under the current E
#   mmlu-next KEY                arms of the next MMLU round ("" = MMLU walk done)
#   gsm-next KEY [up|down]       arms of the next GSM8K round ("" = done)
#   cap KEY [up|down]            per layer: MMLU strength, GSM8K strength, the layer's strength, the reasons
#   tune-next KEY gen|export [up|down]   arms still to generate | generation files generated and not yet exported
#   choose KEY [up|down]         the single-E tune table and the choice (needs every verdict)
#   confirm-next KEY [up|down]   the confirmation arms still missing: "mmlu ARM" / "gsm ARM" lines
#   confirm KEY [up|down]        the confirmation readout
#   c-next KEY gen|export [up|down]      the four E + C arms at the R6 C layer
#   c-choose KEY [up|down]       the method's tune net per c and the choice
#   c-ref KEY                    regression: the current E / C setting must give routes_1128's c
#   halfids                      write the id list of the MMLU selection half (never overwrites)
#   selftest                     synthetic checks of the walks and the choice
import glob
import json
import os
import re
import sys

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/review_checks"]

GRID = ["0.5", "0.75", "1", "1.5", "2", "2.5", "3", "3.5", "4", "5", "6", "8", "12", "16"]
LINE = 1.0
KS = ("0.25", "0.5", "1", "2")
LADH = {"0.25": "0-25", "0.5": "0-5", "1": "1", "2": "2"}
CAP = "Iteration_01/capability"
RR = "Iteration_01/results/relayer_1128"
CAPR = CAP + "/relayer_1128"
MM_IDS = CAP + "/sets/mmlu_1000_20261119.json"
MM_FULL_IDS = CAP + "/sets/mmlu_14042_20261120.json"
HALF_IDS = CAPR + "/sets/mmlu_selection_half_relayer_1128.json"
GSM_IDS = CAP + "/sets/gsm8k_first304_20261124.json"
GSM_FULL_IDS = CAP + "/sets/gsm8k_1319_20261119.json"
N_GSM = 304
GM = CAP + "/gen_main/gsm8k/%s/b8"
GM1024 = CAP + "/gen_main1024/gsm8k/%s/b8"
DIAG = "Iteration_01/diag"
KEYS = {
    "llama": dict(name="Llama-3.1-8B", ng="llama3.1-8b", nl=32, orig="results_e44_factorial_das/llama3.1-8b_side_E2",
                  crun="results_e44_factorial_das/llama3.1-8b", c_e_on="E1", el=16, es="3.5", cl=19, gcap=512,
                  old_tune=["depthE_tune", "pospromptl11_tune", "pospromptl16_tune", "pospromptl16s4_tune"],
                  ctable=DIAG + "/out_c_row_layers2_llama3.1-8b_20261124.txt"),
    "q9": dict(name="Qwen3.5-9B", ng="qwen3.5-9b", nl=32, orig="results_newmodels_ff_20261112/e44/qwen3.5-9b_E2",
               crun="results_newmodels_ff_20261112/e44/qwen3.5-9b", c_e_on="E1", el=15, es="2", cl=16, gcap=1024,
               old_tune=["depthE_tune", "pospromptl15_tune"], ctable=DIAG + "/out_c_row_layers2_qwen3.5-9b_20261124.txt"),
    "q27": dict(name="Qwen3.8-27B", ng="qwen3.8-27b", nl=64, orig="results_newmodels_ff_20261112/e44/qwen3.8-27b-dm",
                crun="results_newmodels_ff_20261112/e44/qwen3.8-27b-dm", c_e_on="E2", el=28, es="1.75", cl=33, gcap=1024,
                old_tune=["pospromptl28_tune"], ctable=DIAG + "/out_c_row_layers2_qwen3.8-27b-dm_20261126.txt"),
    "g12": dict(name="Gemma-3-12B", ng="gemma3-12b", nl=48, orig="Iteration_01/results/ff/gemma3/12b/e44/gemma3-12b",
                crun="Iteration_01/results/ff/gemma3/12b/e44/gemma3-12b", c_e_on="E2", el=24, es="2.5", cl=25, gcap=512,
                old_tune=["pospromptl22_tune", "pospromptl24_tune"], ctable=DIAG + "/out_c_row_layers2_gemma3-12b_20261124.txt"),
    "g27": dict(name="Gemma-3-27B", ng="gemma3-27b", nl=62, orig="Iteration_01/results/ff/gemma3/27b/e44/gemma3-27b",
                crun="Iteration_01/results/ff/gemma3/27b/e44/gemma3-27b", c_e_on="E2", el=28, es="1", cl=31, gcap=512,
                old_tune=["pospromptl26_tune", "pospromptl28_tune"], ctable=DIAG + "/out_c_row_layers2_from18_gemma3-27b_20261125.txt"),
}
for _k in KEYS.values():
    _k["run"] = "%s/%s/run_E2" % (RR, _k["ng"])
ARM_RE = re.compile(r"^x:E(\d+)\.w\.p(\d+(?:\.\d+)?)$")
GARM_RE = re.compile(r"^neutral:E(\d+)\.w\.p(\d+(?:\.\d+)?)$")


def die(msg):
    raise SystemExit("relayer_1128: " + msg)


def window(nl):
    return [l for l in range(nl) if 10 * (l + 1) > 3 * nl and 10 * (l + 1) <= 8 * nl]


def hy(s):
    return s.replace(".", "-")


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sha_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fails(x, arm, n):
    """drop of more than LINE points, compared in whole items: 100 * (x - arm) > LINE * n"""
    return 100 * (x - arm) > LINE * n


# ----------------------------------------------------------------------------- directions of a model
def dir_record(k, l):
    """(gap, direction sha256, identity sha256) of the E2 direction of layer l: the new run's train log (copy or new
    training); where the E run has the layer too, both logs must agree."""
    out = None
    for run in (k["run"], k["orig"]):
        p = "%s/train_logs/E_L%d.json" % (run, l)
        if not os.path.exists(p):
            continue
        d = jload(p)
        if d.get("family") != "E" or d.get("e_on") != "E2" or d.get("layer") != l or d.get("pilot"):
            die("%s is not a full E2 train log of layer %d" % (p, l))
        r = (float(d["gap"]), d["direction_file_sha256"], d["identity_sha256"])
        if out is not None and r != out:
            die("layer %d: the train logs of %s and %s disagree" % (l, k["run"], k["orig"]))
        out = r
    return out


def ident_of(k):
    ids = {dir_record(k, l)[2] for l in window(k["nl"]) if dir_record(k, l)}
    if len(ids) != 1:
        die("%s: the window directions carry %d identities" % (k["ng"], len(ids)))
    return ids.pop()


# ----------------------------------------------------------------------------- MMLU (1,000 list; also the half)
def mm_files(k, half=False):
    if half:
        return sorted(glob.glob("%s/mmlu_half/%s/*.json" % (CAPR, k["ng"])))
    ng = k["ng"]
    old = glob.glob("%s/depthscan/%s_*.json" % (CAP, ng)) + glob.glob("%s/layermap/%s.json" % (CAP, ng)) + \
        glob.glob("%s/layermap/%s_high.json" % (CAP, ng)) + glob.glob("%s/sweep/%s.json" % (CAP, ng))
    old = [p for p in old if not p.endswith("_choice.json") and "_full_" not in os.path.basename(p)]
    return sorted(old) + sorted(glob.glob("%s/mmlu1000/%s/*.json" % (CAPR, ng)))


def mm_read(k, half=False):
    """({(layer, strength): correct}, x correct, n, files) over every file; each file: same item list and identity as
    the E2 run, x picks identical to the first file's, every E arm's write = the E2 direction of that layer (gap)."""
    ident = ident_of(k)
    want = HALF_IDS if half else MM_IDS
    acc, src, ref, n, cx = {}, {}, None, None, None
    files = mm_files(k, half)
    idsha = sha_file(want) if files else None
    for p in files:
        d = jload(p)
        if d.get("skipped"):
            die("%s skipped arms" % p)
        if d.get("identity_sha256") != ident:
            die("%s: identity %s is not the E2 run's %s" % (p, str(d.get("identity_sha256"))[:12], ident[:12]))
        if d["ids_file_sha256"] != idsha:
            die("%s: made from another item list than %s" % (p, want))
        x = [(r["id"], r["pick"]) for r in d["arms"]["x"]["rows"]]
        if ref is None:
            ref, n = x, d["n"]
            cx = sum(r["pick"] == r["gold"] for r in d["arms"]["x"]["rows"])
            if len(x) != n:
                die("%s: %d no-write rows for n %d" % (p, len(x), n))
        elif x != ref:
            die("%s: the no-write picks differ from %s" % (p, files[0]))
        for arm, v in d["arms"].items():
            m = ARM_RE.match(arm)
            if not m or m.group(2) not in GRID:
                continue
            l, s = int(m.group(1)), m.group(2)
            if l not in window(k["nl"]):
                continue
            if v.get("positions") != "content" or len(v["rows"]) != n:
                die("%s %s: positions %s, %d rows" % (p, arm, v.get("positions"), len(v["rows"])))
            dr = dir_record(k, l)
            w = v["writes"]
            if dr is None or len(w) != 1 or w[0]["family"] != "E" or w[0]["layer"] != l or w[0]["kind"] != "w" or \
                    abs(w[0]["gap"] - dr[0]) > 1e-6 * abs(dr[0]) or abs(w[0]["amount"] - float(s) * dr[0]) > 1e-5 * abs(dr[0]) * max(1.0, float(s)):
                die("%s %s: the write is not the E2 direction of layer %d at %s" % (p, arm, l, s))
            c = sum(r["pick"] == r["gold"] for r in v["rows"])
            if (l, s) in acc and acc[(l, s)] != c:
                die("%s %s: %d correct, %s has %d" % (p, arm, c, src[(l, s)], acc[(l, s)]))
            acc[(l, s)], src[(l, s)] = c, p
    return acc, cx, n, files


def walk_up(meas, base, n, rungs):
    """('need', best so far, next rung) or ('done', strength or None, first failing rung or None)."""
    prev = None
    for s in rungs:
        if s not in meas:
            return "need", prev, s
        if fails(base, meas[s], n):
            return "done", prev, s
        prev = s
    return "done", prev, None


def walk_down(meas, base, n, rungs):
    for s in reversed(rungs):
        if s not in meas:
            return "need", None, s
        if not fails(base, meas[s], n):
            return "done", s, None
    return "done", None, rungs[0] if rungs else None


# ----------------------------------------------------------------------------- GSM8K (first 304; full 1,319)
def gsm_ids(full=False):
    ids = jload(GSM_FULL_IDS if full else GSM_IDS)["ids"]
    if not full and ids != jload(GSM_FULL_IDS)["ids"][:N_GSM]:
        die("%s is not the first %d ids of the frozen list" % (GSM_IDS, N_GSM))
    return ids


def gsm_main(k):
    return (GM1024 if k["gcap"] == 1024 else GM) % k["ng"]


def gsm_state(rows, ids):
    import cap_score as CS
    gold = CS.gold_of("gsm8k")
    return sum(CS.judge_row("gsm8k", rows[i], gold(i))[0] == "R" for i in ids)


def gsm_load(p, k, ids, arm_wanted):
    g = jload(p)
    c = g.get("contract") or {}
    bad = []
    if g.get("completed") is not True:
        bad.append("not completed")
    if c.get("model") != k["ng"] or c.get("task") != "gsm8k" or c.get("max_new_tokens") != k["gcap"] or c.get("batch_size") != 8:
        bad.append("contract model / task / cap / batch %s %s %s %s" % (c.get("model"), c.get("task"), c.get("max_new_tokens"), c.get("batch_size")))
    got = [r["id"] for r in g["rows"]]
    if got[:len(ids)] != ids:
        bad.append("rows are not the wanted items in order")
    if arm_wanted:
        if g.get("positions") != "prompt" or c.get("positions") != "prompt":
            bad.append("positions %s / %s" % (g.get("positions"), c.get("positions")))
        m = GARM_RE.match(c.get("arm", ""))
        parts = c.get("parts") or []
        if not m:
            bad.append("arm %s" % c.get("arm"))
        else:
            l = int(m.group(1))
            dr = dir_record(k, l)
            if dr is None or len(parts) != 1 or parts[0].get("family") != "E" or parts[0].get("layer") != l or \
                    parts[0].get("direction_file_sha256") != dr[1]:
                bad.append("the write is not the E2 direction of layer %d" % l)
    elif c.get("arm") != "neutral" or c.get("parts"):
        bad.append("not the no-write pass")
    if bad:
        die("%s: %s" % (p, "; ".join(bad)))
    return {r["id"]: r for r in g["rows"]}, c.get("arm")


def gsm_read(k, full=False):
    """({(layer, strength): right}, no-write right, N)."""
    ids = gsm_ids(full)
    base, _ = gsm_load(gsm_main(k) + "/neutral.json", k, ids, False)
    gx = gsm_state(base, ids)
    if full:
        dirs = ["%s/gsm_full/%s/b8" % (CAPR, k["ng"]), gsm_main(k)]
    else:
        dirs = ["%s/gsm304/%s/b8" % (CAPR, k["ng"]), "%s/depthscan_gsm8k/%s/b8" % (CAP, k["ng"]), gsm_main(k)]
    out, src = {}, {}
    for d in dirs:
        for p in sorted(glob.glob(d + "/neutral--E*-at-prompt.json")):
            g = jload(p)
            if len(g.get("rows", [])) < len(ids):
                continue                    # a 304-item file cannot stand for the full pass
            rows, arm = gsm_load(p, k, ids, True)
            m = GARM_RE.match(arm)
            l, s = int(m.group(1)), m.group(2)
            if s not in GRID or l not in window(k["nl"]):
                continue
            r = gsm_state(rows, ids)
            if (l, s) in out and out[(l, s)] != r:
                die("GSM8K %s: %s and %s disagree (%d / %d)" % (arm, p, src[(l, s)], r, out[(l, s)]))
            out[(l, s)], src[(l, s)] = r, p
    return out, gx, len(ids)


# ----------------------------------------------------------------------------- the capability line per layer
def cap_table(k, order="up"):
    """{layer: dict(mm=(state, s, first_fail), gs=(state, s, first_fail) or None, s=strength or None or '?')}"""
    acc, cx, n, _ = mm_read(k)
    g, gx, ng_ = gsm_read(k)
    out = {}
    for l in window(k["nl"]):
        mm = walk_up({s: acc[(l, s)] for s in GRID if (l, s) in acc}, cx, n, GRID) if cx is not None else ("need", None, GRID[0])
        rec = dict(mm=mm, gs=None, s="?")
        if mm[0] == "done":
            if mm[1] is None:
                rec["s"] = None
            else:
                rungs = GRID[:GRID.index(mm[1]) + 1]
                gm = {s: g[(l, s)] for s in rungs if (l, s) in g}
                rec["gs"] = (walk_up if order == "up" else walk_down)(gm, gx, ng_, rungs)
                if rec["gs"][0] == "done":
                    rec["s"] = rec["gs"][1]
        out[l] = rec
    return out, (cx, n), (gx, ng_), acc, g


def mmlu_next(k):
    t = cap_table(k)[0]
    return ["x:E%d.w.p%s" % (l, r["mm"][2]) for l, r in sorted(t.items()) if r["mm"][0] == "need"]


def gsm_next(k, order):
    t = cap_table(k, order)[0]
    return ["neutral:E%d.w.p%s" % (l, r["gs"][2]) for l, r in sorted(t.items()) if r["gs"] and r["gs"][0] == "need"]


def print_cap(k, order):
    t, (cx, n), (gx, ng_), acc, g = cap_table(k, order)
    print("# %s: capability line per window layer (MMLU %d items, x %d correct; GSM8K first %d, x %d right; line %.1f point; GSM8K order %s)" % (
        k["name"], n or 0, cx or 0, ng_, gx, LINE, order))
    print("| layer | (l+1)/NL | MMLU strength (first failing) | GSM8K walk | layer strength | MMLU change | GSM8K change |")
    print("|---|---|---|---|---|---|---|")
    for l, r in sorted(t.items()):
        mm, gs, s = r["mm"], r["gs"], r["s"]
        mtxt = ("next %s" % mm[2]) if mm[0] == "need" else "%s (%s)" % (mm[1], mm[2])
        gtxt = "-" if gs is None else (("next %s" % gs[2]) if gs[0] == "need" else "%s (%s)" % (gs[1], gs[2]))
        mc = "%+.1f" % (100.0 * (acc[(l, s)] - cx) / n) if s not in (None, "?") else ""
        gc = "%+.1f" % (100.0 * (g[(l, s)] - gx) / ng_) if s not in (None, "?") and (l, s) in g else ""
        print("| %d | %.3f | %s | %s | %s | %s | %s |" % (l, (l + 1.0) / k["nl"], mtxt, gtxt, {None: "none", "?": "open"}.get(s, s), mc, gc))
    return t


# ----------------------------------------------------------------------------- tune verdicts of single E
def work_dirs(k, kind="E"):
    old = ["%s/eval_extra/judge_tune_%s" % (k["orig"], n) for n in k["old_tune"]] if kind == "E" else []
    new = sorted(glob.glob("%s/eval_extra/judge_tune_relayer_1128_%s*_tune" % (k["run"], kind)))
    return old + new


def exported(k, kind="E"):
    """{condition: work dir} of every work dir's judge_input.json (also not yet judged)."""
    out = {}
    for w in work_dirs(k, kind):
        p = w + "/judge_input.json"
        if os.path.exists(p):
            for c in jload(p)["conditions"]:
                out.setdefault(c, w)
    return out


def gen_path(k, arm):
    import depth_scan as DS
    return "%s/eval_extra/gen/%s" % (k["run"], DS.gen_file(arm))


def tune_next(k, what, order):
    import depth_scan as DS
    t = cap_table(k, order)[0]
    ex = exported(k)
    out = []
    for l, r in sorted(t.items()):
        s = r["s"]
        if s in (None, "?"):
            continue
        arm, cond = "x:E%d.w.p%s" % (l, s), DS.cond_e(l, s)
        if cond in ex:
            continue
        p = gen_path(k, arm)
        if what == "gen" and not os.path.exists(p):
            out.append(arm)
        elif what == "export" and os.path.exists(p):
            _, bad = DS.check_gen_file(p)
            if bad:
                die("%s: %s" % (p, "; ".join(bad)))
            out.append(os.path.basename(p))
    return out


def merge_dir(w, st, bl, tx, two, base):
    """depth_scan.add_dir on fresh copies of the usual no-write cells (its checks: the work dir's no-write cells equal
    the usual ones; condition names), then merged: a condition already read from another work dir must carry the same
    verdict, blank flag and text there (11-24: E_L15_x-w-p2 of Qwen3.5-9B sits in depthE_tune and pospromptl15_tune)."""
    import depth_scan as DS
    s1, b1, t1 = dict(base[0]), dict(base[1]), dict(base[2])
    DS.add_dir(w, s1, b1, t1, two, {})
    for key, v in s1.items():
        if key[1] == "baseline_x":
            continue
        if key in st:
            if (st[key], bl[key], tx[key]) != (v, b1[key], t1[key]):
                die("%s %s: another verdict, blank flag or text than an earlier work dir" % (w, key))
        else:
            st[key], bl[key], tx[key] = v, b1[key], t1[key]


def verdicts(k, kind="E"):
    import depth_scan as DS
    st, bl, tx, qs0 = DS.usual(dict(name=k["name"]))
    base = (dict(st), dict(bl), dict(tx))
    for w in work_dirs(k, "E"):                 # single-write work dirs (old ones and relayer_1128_E*)
        if os.path.exists(w + "/judge2/single_pro.jsonl"):
            merge_dir(w, st, bl, tx, False, base)
    if kind == "C":                             # two-write work dirs relayer_1128_C*
        for w in work_dirs(k, "C"):
            if os.path.exists(w + "/judge2/single_pro.jsonl"):
                merge_dir(w, st, bl, tx, True, base)
    return st, bl, tx, qs0


def choose(k, order, quiet=False):
    import depth_scan as DS
    t = cap_table(k, order)[0]
    open_ = [l for l, r in t.items() if r["s"] == "?"]
    if open_:
        die("%s: the capability line is still open on layers %s" % (k["ng"], open_))
    cand = [(l, r["s"]) for l, r in sorted(t.items()) if r["s"] is not None]
    if not cand:
        die("%s: no window layer has a strength inside the line" % k["ng"])
    st, bl, tx, qs0 = verdicts(k)
    conds = [DS.cond_e(l, s) for l, s in cand]
    miss = [c for c in conds if not any(cc == c for _, cc in st)]
    if miss:
        die("%s: no verdicts yet for %s" % (k["ng"], miss))
    qs = DS.common_qs(st, conds, qs0)
    base = ([st[(q, "baseline_x")] for q in qs], [not bl[(q, "baseline_x")] for q in qs])
    rows = []
    for l, s in cand:
        es, ea = DS.single_e(st, bl, tx, l, hy(s), qs)
        rows.append((l, s, es, ea, DS.counts(es, ea)[3]))
    best = max(rows, key=lambda r: (r[4], -r[0]))
    if not quiet:
        ba, br, bw, bn = DS.counts(*base)
        print("\n# %s: single E per window layer at its capability strength, tune, prompt rule, Pro (%d of %d questions have every verdict); no write: answers %d, right %d, wrong %d, net %+d" % (
            k["name"], len(qs), len(qs0), ba, br, bw, bn))
        print("| layer | (l+1)/NL | strength | answers | right | wrong | net | minus no write [95%] | minus the chosen [95%] |")
        print("|---|---|---|---|---|---|---|---|---|")
        for l, s, es, ea, net in rows:
            a, r_, w, _ = DS.counts(es, ea)
            print("| %d | %.3f | %s | %d | %d | %d | %+d | %s | %s |" % (
                l, (l + 1.0) / k["nl"], s, a, r_, w, net, DS.boot([DS.val(x) - DS.val(y) for x, y in zip(es, base[0])]),
                "chosen" if l == best[0] else DS.boot([DS.val(x) - DS.val(y) for x, y in zip(es, best[2])])))
        print("-> chosen E layer %d at strength %s (tune net %+d; ties go to the shallower layer); current main table: layer %d at %s" % (
            best[0], best[1], best[4], k["el"], k["es"]))
    return best[0], best[1], rows, qs


# ----------------------------------------------------------------------------- confirmation
def confirm_state(k, order):
    l, s, _, _ = choose(k, order, quiet=True)
    out = {"layer": l, "s": s}
    acc, cx, n, files = mm_read(k, half=True) if os.path.exists(HALF_IDS) else ({}, None, None, [])
    out["mmlu"] = None if (l, s) not in acc else (acc[(l, s)], cx, n, not fails(cx, acc[(l, s)], n))
    g, gx, ng_ = gsm_read(k, full=True)
    out["gsm"] = None if (l, s) not in g else (g[(l, s)], gx, ng_, not fails(gx, g[(l, s)], ng_))
    return out


def write_half():
    import per_model_table as PM
    if os.path.exists(HALF_IDS):
        die("%s exists" % HALF_IDS)
    full = jload(MM_FULL_IDS)
    dev, rep = PM.halves(list(full["ids"]))
    if len(dev) != 7021 or set(dev) & set(rep) or not set(jload(MM_IDS)["ids"]) <= set(dev):
        die("the selection half is not 7,021 items holding the 1,000 list")
    keep = set(dev)
    out = dict(full, ids=[i for i in full["ids"] if i in keep], n=len(dev),
               note="selection half of diag/per_model_table.halves (seed 20261121) of %s, in its order; relayer_1128" % MM_FULL_IDS)
    if isinstance(full.get("rows"), list):
        if len(full["rows"]) != len(full["ids"]):
            die("%s: rows and ids differ in length" % MM_FULL_IDS)
        out["rows"] = [r for i, r in zip(full["ids"], full["rows"]) if i in keep]
    elif isinstance(full.get("rows"), int):
        out["rows"] = len(dev)
    os.makedirs(os.path.dirname(HALF_IDS), exist_ok=True)
    with open(HALF_IDS, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("wrote %s: %d ids" % (HALF_IDS, len(out["ids"])))


# ----------------------------------------------------------------------------- C (R5, R6, c)
def r5(k):
    """{layer: met} from the archived per-layer C table (tune block, caution arm, status column A)."""
    if not os.path.exists(k["ctable"]):
        die("no C table %s" % k["ctable"])
    out, tune = {}, False
    for line in open(k["ctable"], encoding="utf-8"):
        if line.startswith("## "):
            tune = "tune" in line
        if not tune or not line.startswith("| %s |" % k["name"]):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if c[2] != "caution":
            continue
        l = int(c[1])
        if l in out:
            die("%s: layer %d twice" % (k["ctable"], l))
        out[l] = c[14].split("/")[0].strip() == "met"
    w = window(k["nl"])
    miss = [l for l in w if l not in out]
    if miss:
        die("%s lacks window layers %s" % (k["ctable"], miss))
    return {l: out[l] for l in w}


def r6(k, el):
    ok = [l for l, m in sorted(r5(k).items()) if m and l > el]
    return ok[0] if ok else None


def c_arms(k, order):
    l, s, _, _ = choose(k, order, quiet=True)
    cl = r6(k, l)
    if cl is None:
        die("%s: no R5 layer deeper than the chosen E layer %d inside the window" % (k["ng"], l))
    return l, s, cl, ["x:E%d.w.p%s+C%d.w.m%s" % (l, s, cl, c) for c in KS]


def c_next(k, what, order):
    import newsets_score as S
    l, s, cl, arms = c_arms(k, order)
    ex = exported(k, "C")
    sh = S.shell(l, cl)
    out = []
    for c, arm in zip(KS, arms):
        if sh.cEC(hy(s), LADH[c]) + "-at-prompt" in ex:
            continue
        p = gen_path(k, arm)
        if what == "gen" and not os.path.exists(p):
            out.append(arm)
        elif what == "export" and os.path.exists(p):
            out.append(os.path.basename(p))
    return out


def c_choose(k, order):
    l, s, cl, arms = c_arms(k, order)
    return c_table(k, l, s, cl, verdicts(k, "C"))


def c_table(k, l, s, cl, data):
    """The method's tune net per c with E at (l, s) and C at cl; data = (st, bl, tx, questions) holding the cells."""
    import depth_scan as DS
    import newsets_score as S
    import routes_1128 as RT
    st, bl, tx, qs0 = data
    sh = S.shell(l, cl)
    conds = [DS.cond_e(l, s)] + [sh.cEC(hy(s), LADH[c]) for c in KS]
    qs = DS.common_qs(st, conds, qs0)

    def cell(cond):
        return {q: (RT.VAL[st[(q, cond)]], not bl[(q, cond)]) for q in qs}
    arm = {"O": cell("baseline_x"), "E": cell(conds[0])}
    for c, cond in zip(KS, conds[1:]):
        arm["EC" + c] = cell(cond)
        arm["C" + c] = arm["O"]            # placeholder: the single-C schemes are not read here
    T = RT.schemes(qs, arm)
    names = ["R1_" + c for c in KS]
    pick = RT.best(names, T, qs)
    print("\n# %s: C layer %d (R5 layers in the window: %s); E layer %d at %s; tune %d questions" % (
        k["name"], cl, [x for x, m in sorted(r5(k).items()) if m], l, s, len(qs)))
    eb = [q for q in qs if not arm["E"][q][1]]
    bb = [q for q in eb if not arm["O"][q][1]]
    print("  single E net %+d; questions where both the original and E give no answer: %d" % (RT.net(T["E"][0], qs), len(bb)))
    for nm in names:
        d = T[nm][0]
        print("  method with c = %-4s net %+4d | handed in %d" % (nm[3:], RT.net(d, qs), sum(d[q][1] for q in qs)))
    print("-> c = %s (highest tune net, ties -> the smaller)" % pick[3:])
    return pick[3:]


# ----------------------------------------------------------------------------- inventory
def inv(k):
    w = window(k["nl"])
    have = [l for l in w if os.path.exists("%s/dirs/E_L%d.pt" % (k["run"], l))]
    orig = [l for l in w if os.path.exists("%s/dirs/E_L%d.pt" % (k["orig"], l))]
    print("%s: window %d-%d (%d layers) | E run %s has %s | new run %s has %d: missing %s" % (
        k["name"], w[0], w[-1], len(w), k["orig"], orig, k["run"], len(have), [l for l in w if l not in have]))
    if os.path.exists(k["ctable"]):
        m = r5(k)
        print("  R5 (archived C table, training rule): %s; R6 under the current E layer %d -> %s (current C %d)" % (
            [l for l, v in sorted(m.items()) if v], k["el"], r6(k, k["el"]), k["cl"]))


# ----------------------------------------------------------------------------- self-test (synthetic, no file)
def selftest():
    ok = True

    def eq(what, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print("  %-66s %s%s" % (what, "ok" if good else "FAILED", "" if good else " (got %r, want %r)" % (got, want)))

    print("self-test of relayer_1128.py")
    tab = {"llama": (32, 9, 24), "q9": (32, 9, 24), "q27": (64, 19, 50), "g12": (48, 14, 37), "g27": (62, 18, 48)}
    for key, (nl, lo, hi) in tab.items():
        eq("window %s (section 4 table)" % key, (window(nl)[0], window(nl)[-1], len(window(nl))), (lo, hi, hi - lo + 1))
    eq("MMLU line: 10 of 1,000 lost is inside, 11 is out", (fails(640, 630, 1000), fails(640, 629, 1000)), (False, True))
    eq("GSM8K line: 3 of 304 lost is inside, 4 is out", (fails(265, 262, 304), fails(265, 261, 304)), (False, True))
    meas = {"0.5": 640, "0.75": 641, "1": 635, "1.5": 629, "2": 700}
    eq("up walk: stops at the first failure, later recovery ignored", walk_up(meas, 640, 1000, GRID), ("done", "1", "1.5"))
    eq("up walk: next unmeasured rung", walk_up({"0.5": 640}, 640, 1000, GRID), ("need", "0.5", "0.75"))
    eq("up walk: the smallest already fails", walk_up({"0.5": 600}, 640, 1000, GRID), ("done", None, "0.5"))
    eq("up walk: every rung inside", walk_up({s: 640 for s in GRID}, 640, 1000, GRID), ("done", "16", None))
    rungs = GRID[:GRID.index("3") + 1]
    g = {"3": 250, "2.5": 262}
    eq("down walk: from the top, first pass", walk_down(g, 265, 304, rungs), ("done", "2.5", None))
    eq("down walk: next rung to measure", walk_down({"3": 250}, 265, 304, rungs), ("need", None, "2.5"))
    eq("up walk on GSM8K (4 items lost fails)", walk_up({"0.5": 265, "0.75": 261}, 265, 304, rungs), ("done", "0.5", "0.75"))
    rows = [(10, "1", None, None, 5), (12, "1", None, None, 9), (14, "0.75", None, None, 9)]
    eq("choice: highest net, ties to the shallower layer", max(rows, key=lambda r: (r[4], -r[0]))[0], 12)
    eq("hyphen form", hy("1.5"), "1-5")
    print("self-test %s" % ("passed" if ok else "FAILED"))
    return 0 if ok else 1


def main():
    a = sys.argv[1:]
    if a == ["selftest"]:
        return selftest()
    if a == ["window"]:
        for key, k in KEYS.items():
            w = window(k["nl"])
            print("%s %s: NL %d, window %d-%d (%d layers)" % (key, k["name"], k["nl"], w[0], w[-1], len(w)))
        return 0
    if a == ["halfids"]:
        return write_half()
    if len(a) < 2 or a[1] not in KEYS:
        print(__doc__ or "see the header of relayer_1128.py")
        return 2
    mode, k = a[0], KEYS[a[1]]
    rest = a[2:]
    order = "down" if "down" in rest else "up"
    if mode == "inv":
        inv(k)
    elif mode == "mmlu-next":
        print(" ".join(mmlu_next(k)))
    elif mode == "gsm-next":
        print(" ".join(gsm_next(k, order)))
    elif mode == "cap":
        print_cap(k, order)
    elif mode == "tune-next":
        print(" ".join(tune_next(k, rest[0] if rest and rest[0] in ("gen", "export") else "gen", order)))
    elif mode == "choose":
        choose(k, order)
    elif mode == "confirm-next":
        cs = confirm_state(k, order)
        if cs["mmlu"] is None:
            print("mmlu x:E%d.w.p%s" % (cs["layer"], cs["s"]))
        if cs["gsm"] is None:
            print("gsm neutral:E%d.w.p%s" % (cs["layer"], cs["s"]))
    elif mode == "confirm":
        cs = confirm_state(k, order)
        print("%s: chosen layer %d at %s" % (k["name"], cs["layer"], cs["s"]))
        for name, v in (("full MMLU, selection half", cs["mmlu"]), ("full GSM8K", cs["gsm"])):
            if v is None:
                print("  %s: not measured yet" % name)
            else:
                print("  %s: %d vs no write %d of %d (%+.2f points) -> %s" % (name, v[0], v[1], v[2], 100.0 * (v[0] - v[1]) / v[2],
                                                                           "inside the line" if v[3] else "OUTSIDE THE LINE (open question: what then)"))
    elif mode == "c-next":
        print(" ".join(c_next(k, rest[0] if rest and rest[0] in ("gen", "export") else "gen", order)))
    elif mode == "c-choose":
        c_choose(k, order)
    elif mode == "c-ref":
        # regression: the current setting (main-table E layer / strength and C layer, the archived prompt-rule work dirs
        # posprompt<tag> / posprompt2<tag>) must give the c of routes_1128 (Llama 2, Qwen3.5-9B 0.5, Qwen3.8-27B 1,
        # Gemma-3-12B 2, Gemma-3-27B 2)
        st, bl, tx, qs0 = verdicts(k)
        base = (dict(st), dict(bl), dict(tx))
        merge_dir("%s/eval_extra/judge_tune_posprompt2l%d_tune" % (k["orig"], k["el"]), st, bl, tx, True, base)
        got = c_table(k, k["el"], k["es"], k["cl"], (st, bl, tx, qs0))
        want = {"llama": "2", "q9": "0.5", "q27": "1", "g12": "2", "g27": "2"}[a[1]]
        print("c-ref %s: c = %s, routes_1128 chose %s -> %s" % (a[1], got, want, "SAME" if got == want else "DIFFERENT"))
        return 0 if got == want else 1
    else:
        print("unknown mode %s" % mode)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
