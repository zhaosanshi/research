#!/usr/bin/env python3
"""Two-axis grading of a generation file with three judges and 赵磊 (rules: core/judge_two_axis.py).

usage (RUN_DIR holds the generation file, e.g. an e42 tune.json; everything is written to RUN_DIR/judge2/):
  judge_two_axis_run.py cells      RUN_DIR [--generation tune.json] [--audit-n 50 --audit-seed 0]
  (GPU)  scripts/allenai_tqa_judge.py --cells RUN_DIR/judge2/cells.jsonl --out RUN_DIR/judge2/allenai.jsonl ...
  judge_two_axis_run.py llm        RUN_DIR --judge pro                   DeepSeek V4 Pro, every cell
  judge_two_axis_run.py disputes   RUN_DIR                               cells AllenAI + Pro did not settle, + audit
  judge_two_axis_run.py llm        RUN_DIR --judge gpt --only disputes   GPT-5.5 on those cells only
  judge_two_axis_run.py adjudicate RUN_DIR                               final.jsonl, human_queue.jsonl, summary
  judge_two_axis_run.py single     RUN_DIR --judge allenai|pro          one judge alone: single_<judge>.jsonl + summary; no disputes / adjudicate / GPT
  judge_two_axis_run.py human      RUN_DIR list
  judge_two_axis_run.py human      RUN_DIR set CELL_ID LETTER --by WHO [--note ..]   then re-run adjudicate

Files in RUN_DIR/judge2/: cells.jsonl (one row per question x condition), allenai.jsonl, pro.jsonl, gpt.jsonl,
disputes.jsonl, final.jsonl, human_queue.jsonl, human_verdicts.jsonl, summary.json / summary.md, meta.json,
cache/judge2_cache_<model>.jsonl (API replies). API keys come from DEEPSEEK_API_KEY_FILE / XIAOAI_API_KEY_FILE.
Config: configs/judges/two_axis_v6.json (override with --config or TD_JUDGE2_CONFIG).
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from collections import Counter, OrderedDict

CODE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, CODE_ROOT)
from core.io_utils import read_jsonl, write_jsonl, append_jsonl, atomic_json_dump  # noqa: E402
from core.judge_two_axis import (TWO_AXIS_VERSION, LABELS, LABEL_NAMES, TRUTH_VALUES, INFO_VALUES, TwoAxisLLMJudge,  # noqa: E402
                                 adjudicate_cell, allenai_votes, info_binary, judge_own_label, coarse, qa_sha, cell_axes,
                                 ADJUDICATION_VERSION, normalize_human_verdict, parse_truth, parse_info)
from core.qa_scoring import reference_answers  # noqa: E402
from core.judge_two_axis_contract import canonical_sha, file_sha, cell_sha, judge_contract

J2 = "judge2"
JUDGES = ("allenai", "pro", "gpt")
VALID_T, VALID_I = ("T", "F", "D"), ("S", "H", "R", "D")


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def j2dir(run_dir):
    p = os.path.join(run_dir, J2)
    os.makedirs(p, exist_ok=True)
    return p


def sha256_file(path):
    return file_sha(path)


def _read_json(path):
    def unique(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate JSON key: '+key)
            result[key]=value
        return result
    def invalid(value):raise ValueError('nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=unique,parse_constant=invalid)


def load_cfg(path=None):
    path = path or os.environ.get("TD_JUDGE2_CONFIG") or os.path.join(CODE_ROOT, "configs", "judges", "two_axis_v6.json")
    cfg = _read_json(path)
    if cfg.get("version") != TWO_AXIS_VERSION:
        raise SystemExit("config version %r does not match the code's rubric %r" % (cfg.get("version"), TWO_AXIS_VERSION))
    return cfg


def _rows(path):
    out=OrderedDict()
    if os.path.exists(path):
        for r in read_jsonl(path):
            cid=r.get('cell_id')
            if not isinstance(cid,str) or not cid or cid in out:
                raise ValueError('missing/duplicate cell_id in '+str(path))
            out[cid]=r
    return out


def _derive_cells(data):
    if data.get('completed') is not True:
        raise ValueError('generation must be explicitly complete')
    conds=data.get('conditions');rows=data.get('evaluations')
    if (not isinstance(conds,list) or not conds or any(not isinstance(c,str) or not c for c in conds)
            or len(set(conds))!=len(conds) or not isinstance(rows,list) or not rows):
        raise ValueError('nonempty unique conditions and evaluation rows required')
    cells=[];seen=set()
    for row in rows:
        pid=row.get('pair_id');question=row.get('question');gold=row.get('gold');incorrect=row.get('incorrect') or []
        if not isinstance(pid,str) or not pid or pid in seen or '|' in pid:
            raise ValueError('missing/duplicate/ambiguous question identifier')
        seen.add(pid)
        if not isinstance(question,str) or not question.strip():raise ValueError('missing question')
        if not isinstance(gold,list) or not gold or any(not isinstance(g,str) or not g.strip() for g in gold):
            raise ValueError('nonempty reference-answer list required')
        if not isinstance(incorrect,list) or any(not isinstance(g,str) for g in incorrect):raise ValueError('invalid incorrect references')
        if not isinstance(row.get('outputs'),dict) or set(row['outputs'])!=set(conds):raise ValueError('missing/extra output arms')
        finished=row.get('finished')
        if finished is not None and (set(finished)!=set(conds) or any(type(v) is not bool for v in finished.values())):
            raise ValueError('invalid completion flags')
        refs,only_refusal=reference_answers(gold,question=question)
        for condition in conds:
            if '|' in condition:raise ValueError('ambiguous condition identifier')
            pred=row['outputs'][condition]
            if not isinstance(pred,str):raise ValueError('non-string generated answer')
            cell={'cell_id':pid+'|'+condition,'pair_id':pid,'condition':condition,'question':question,
                  'gold':list(gold),'gold_refs':refs,'gold_is_abstention':only_refusal,'incorrect':incorrect,
                  'best_answer':row.get('best_answer'),'prediction':pred,
                  'rule_abstain':bool((row.get('rule_abstain') or {}).get(condition,False)),
                  'finished':finished[condition] if finished is not None else None,
                  'unicode_replacement':'\ufffd' in pred,'audit':False}
            cell['input_sha']=cell_sha(cell);cell['qa_sha']=qa_sha(question,pred)
            cells.append(cell)
    return cells


def _workspace(run_dir):
    out=j2dir(run_dir);mp=os.path.join(out,'meta.json');cp=os.path.join(out,'cells.jsonl')
    if not os.path.exists(mp) or not os.path.exists(cp):raise ValueError('run cells first in a fresh grading workspace')
    meta=_read_json(mp)
    if meta.get('version')!=TWO_AXIS_VERSION or meta.get('pipeline_version')!=ADJUDICATION_VERSION:
        raise ValueError('legacy grading workspace: preserve it and use a fresh directory; no automatic hash backfill')
    gen=(Path(run_dir)/meta['generation']).resolve()
    gen.relative_to(Path(run_dir).resolve())
    if file_sha(gen)!=meta['generation_sha256'] or file_sha(cp)!=meta['cells_sha256']:
        raise ValueError('generation or cells changed after preparation')
    raw=_read_json(gen);expected=_derive_cells(raw)
    audit=meta.get('audit') or {};eligible=sorted(c['cell_id'] for c in expected if not c['gold_is_abstention'] and c['prediction'].strip())
    n=audit.get('requested_n')
    if type(n) is not int or n<0 or type(audit.get('seed')) is not int:raise ValueError('invalid frozen audit design')
    chosen=set(random.Random(audit['seed']).sample(eligible,min(n,len(eligible))))
    if sorted(chosen)!=audit.get('cell_ids'):raise ValueError('audit sample changed')
    for c in expected:c['audit']=c['cell_id'] in chosen
    cells=list(_rows(cp).values())
    if (cells!=expected or meta['conditions']!=raw['conditions'] or meta['model']!=raw.get('model')
            or meta['n_cells']!=len(cells) or meta['n_questions']!=len(raw['evaluations'])):
        raise ValueError('cells/model/cohort are not derived from this exact generation')
    return meta,cells,raw


def _cells(run_dir):
    return _workspace(run_dir)[1]


def _human_verdicts(run_dir,cells=None):
    """Explicit content-bound human decisions apply to every identical request."""
    cells=_cells(run_dir) if cells is None else cells
    by_id={c['cell_id']:c for c in cells};by_content={};out={}
    own = os.path.join(j2dir(run_dir), "human_verdicts.jsonl")
    # TD_JUDGE2_HUMAN: a verdict file shared by several runs. Its rows are bound to their reviewed content (input_sha
    # == cell_sha(reviewed_input)), so a verdict given in one run applies to the identical (question, answer,
    # references) in another run; cell ids of other runs are expected there (red team finding 3: same input, same verdict).
    shared = os.environ.get('TD_JUDGE2_HUMAN') or None
    for path in [own] + ([shared] if shared and shared != own else []):
        if not os.path.exists(path):
            continue
        for r in read_jsonl(path):
            c=by_id.get(r.get('cell_id'))
            if path == own and (c is None or r.get('input_sha')!=cell_sha(c)):
                raise ValueError('unbound/stale human decision; re-review its original content, do not backfill')
            if (r.get('version')!=ADJUDICATION_VERSION
                    or not isinstance(r.get('reviewed_input'),dict)
                    or cell_sha(r['reviewed_input'])!=r.get('input_sha')
                    or not isinstance(r.get('by'),str) or not r['by'].strip()):
                raise ValueError('unbound/stale human decision; re-review its original content, do not backfill')
            normalized=normalize_human_verdict(r)
            if normalized!=r:raise ValueError('human axes are incomplete or were not stored explicitly')
            by_content[r['input_sha']]=r
    for c in cells:
        if c['input_sha'] in by_content:out[c['cell_id']]=by_content[c['input_sha']]
    return out


def _verify_rows(rows, cells, judge_name=None, cfg=None):
    """No identity is inferred for old ballots: missing request/source bindings are stale."""
    by_id = {c["cell_id"]: c for c in cells}
    out, stale, legacy, orphans = OrderedDict(), [], 0, 0
    signature=canonical_sha(judge_contract(judge_name,cfg)) if judge_name and cfg is not None else None
    for cid, r in rows.items():
        c = by_id.get(cid)
        if c is None:
            raise ValueError('judge row belongs to a different cohort: '+cid)
        expected = qa_sha(c["question"], c["prediction"])
        if (r.get('input_sha')!=cell_sha(c) or r.get('qa_sha')!=expected
                or (signature is not None and r.get('judge_contract_sha')!=signature)):
            stale.append(cid)
            continue
        if judge_name in ('pro','gpt'):
            if r.get('model')!=cfg[judge_name]['model'] or r.get('rubric')!=TWO_AXIS_VERSION or r.get('judge')!=judge_name:
                stale.append(cid);continue
            placeholder=r.get('empty_output') or r.get('skipped_abstention_gold')
            if placeholder:
                if not ((r.get('empty_output') and not c['prediction'].strip()) or
                        (r.get('skipped_abstention_gold') and c['gold_is_abstention'])):
                    raise ValueError('placeholder vote applied to real answer')
            elif r.get('truth')!=parse_truth(r.get('truth_raw')) or r.get('info')!=parse_info(r.get('info_raw')):
                stale.append(cid);continue      # the reply would be read differently now: re-read it (cache hit), do not crash
        elif judge_name=='allenai' and not r.get('unscorable'):
            allenai_votes(r,cfg['allenai']['threshold'])
        out[cid] = r
    return out, {"stale": stale, "legacy_backfilled": legacy, "orphans_dropped": orphans}


def _cache_dir(out, cfg):
    """API reply cache. TD_JUDGE2_CACHE (or cfg['shared_cache_dir']) points several runs at one cache so an identical
    request (same model, contract, prompt, text) gets the same reply everywhere - GPT-5.5 is not deterministic, and
    per-run caches gave 5 of 332 shared inputs different verdicts across the two E42 runs (GPT round 22, finding 5)."""
    return os.environ.get('TD_JUDGE2_CACHE') or (cfg or {}).get('shared_cache_dir') or os.path.join(out, 'cache')


def judge_request_keys(cell, judge_name, cfg):
    """The two API request keys (truth question, info question) this cell produces for an LLM judge, computed exactly as
    TwoAxisLLMJudge / LLMJudge.cache_key do, without constructing a client (no API key needed)."""
    from core.judge import request_cache_key
    from core.judge_two_axis import TRUTH_SYSTEM, INFO_SYSTEM, truth_user, info_user
    from core.judge_two_axis_contract import judge_contract, request_identity
    jcfg = cfg[judge_name]
    tag = {'judge_contract_sha': canonical_sha(request_identity(judge_contract(judge_name, cfg)))}
    users = (truth_user(cell['question'], cell['gold_refs'], cell['prediction'], cell['incorrect'] or None),
             info_user(cell['question'], cell['prediction']))
    return [request_cache_key(jcfg['model'], system, user, config=jcfg, key_extra=tag)
            for system, user in zip((TRUTH_SYSTEM, INFO_SYSTEM), users)]


def _inconsistent_cells(cells, cfg, cache_dir):
    """cell_id -> set of judge names whose shared reply cache holds different replies to that cell's request
    (judge2_cache_merge.py writes cache_dir/conflicts.json). Such cells are not decided by any rule but by 赵磊."""
    path = os.path.join(cache_dir, 'conflicts.json')
    conflicts = _read_json(path).get('keys', {}) if os.path.exists(path) else {}
    keys = set()
    if isinstance(conflicts, dict):
        for v in conflicts.values():
            keys.update(v)
    else:
        keys.update(conflicts)
    # A merge report is a snapshot, not the authoritative current cache. New
    # replies can have arrived since it was written, including from another run.
    from core.judge import cache_filename
    from core.judge_two_axis import CACHE_PREFIX
    for name in ('pro', 'gpt'):
        if name not in cfg:
            continue
        cache_path = os.path.join(cache_dir, cache_filename(cfg[name]['model'], prefix=CACHE_PREFIX))
        if not os.path.exists(cache_path):
            continue
        replies = {}
        for row in read_jsonl(cache_path):
            replies.setdefault(row['key'], set()).add(row['answer'])
        keys.update(key for key, values in replies.items() if len(values) > 1)
    if not keys:
        return {}
    out = {}
    for name in ('pro', 'gpt'):
        if name not in cfg:
            continue
        for c in cells:
            if c['gold_is_abstention'] or not c['prediction'].strip():
                continue
            if any(k in keys for k in judge_request_keys(c, name, cfg)):
                out.setdefault(c['cell_id'], set()).add(name)
    return out


def _active(cells):
    """Cells that are graded and counted: substantive gold only (abstention-only gold is excluded from every
    denominator, as e32 / e42_analysis do, so it is not sent to the LLM judges either)."""
    return [c for c in cells if not c["gold_is_abstention"]]


# ----------------------------------------------------------------------------- cells
def build_cells(run_dir, generation="tune.json", audit_n=50, audit_seed=0):
    if type(audit_n) is not int or audit_n<0 or type(audit_seed) is not int:
        raise ValueError('audit size/seed must be explicit integers')
    gen_path=(Path(run_dir)/generation).resolve()
    relative=gen_path.relative_to(Path(run_dir).resolve())
    data=_read_json(gen_path);cells=_derive_cells(data)
    pool=sorted(c['cell_id'] for c in cells if not c['gold_is_abstention'] and c['prediction'].strip())
    audit=set(random.Random(audit_seed).sample(pool,min(audit_n,len(pool))))
    for c in cells:c['audit']=c['cell_id'] in audit
    out=j2dir(run_dir);cp=os.path.join(out,'cells.jsonl');mp=os.path.join(out,'meta.json')
    if os.path.exists(cp) or os.path.exists(mp):
        old,previous,_=_workspace(run_dir)
        if previous!=cells or old['audit'].get('requested_n')!=audit_n or old['audit']['seed']!=audit_seed:
            raise ValueError('frozen generation/audit changed: use a fresh run directory; existing votes are preserved')
        return old
    write_jsonl(cells,cp)
    meta={'version':TWO_AXIS_VERSION,'pipeline_version':ADJUDICATION_VERSION,
          'generation':str(relative),'generation_sha256':file_sha(gen_path),'cells_sha256':file_sha(cp),
          'model':data.get('model'),'conditions':list(data['conditions']),
          'n_questions':len(data['evaluations']),'n_cells':len(cells),
          'n_unique_question_answer':len({(c['question'],c['prediction']) for c in cells}),
          'n_unique_semantic_requests':len({c['input_sha'] for c in cells}),
          'n_empty_outputs':sum(not c['prediction'].strip() for c in cells),
          'n_abstention_only_gold_questions':sum(reference_answers(r['gold'],question=r['question'])[1] for r in data['evaluations']),
          'cohort_policy':'substantive_gold_only; this is a filtered cohort, not the full original TruthfulQA benchmark',
          'audit':{'n':len(audit),'requested_n':audit_n,'seed':audit_seed,'cell_ids':sorted(audit)},'ts':_now()}
    atomic_json_dump(meta,mp)
    return meta


# ----------------------------------------------------------------------------- LLM judges
def run_llm(run_dir, judge_name, cfg, only=None, judge=None, log_every=50):
    """Grade cells with one LLM judge (two questions each). Resumable: cells already holding two valid letters under
    the current rubric are kept; a D is valid. `only='disputes'` restricts to disputes.jsonl plus the audit sample."""
    out = j2dir(run_dir)
    meta,cells,_ = _workspace(run_dir)
    code_before=_aggregation_code()
    jcfg = cfg[judge_name]
    target = cells
    if only == "disputes":
        dpath = os.path.join(out, "disputes.jsonl")
        if not os.path.exists(dpath):
            raise SystemExit("no disputes.jsonl: run `disputes` first")
        _verify_dispute_manifest(run_dir,cfg)
        disp = set(_rows(dpath))
        target = [c for c in cells if c["cell_id"] in disp or c["audit"]]
    path = os.path.join(out, judge_name + ".jsonl")
    contract=judge_contract(judge_name,cfg);signature=canonical_sha(contract)
    done, vstats = _verify_rows(_rows(path), cells,judge_name,cfg)
    if vstats['stale'] and os.path.exists(path):
        archive=path+'.stale-'+file_sha(path)[:16]+'.jsonl'
        if not os.path.exists(archive):Path(archive).write_bytes(Path(path).read_bytes())
    def bound(g,c,rule_placeholder=False):
        if g.get('judge')!=judge_name or g.get('model')!=jcfg['model'] or g.get('rubric')!=TWO_AXIS_VERSION:
            raise ValueError('judge returned a foreign model/rubric identity')
        if not rule_placeholder and (g.get('judge_contract_sha')!=signature
                or g.get('qa_sha')!=qa_sha(c['question'],c['prediction'])):
            raise ValueError('judge returned a different executed request contract or answer identity')
        return dict(g,cell_id=c['cell_id'],qa_sha=qa_sha(c['question'],c['prediction']),
                    input_sha=cell_sha(c),judge_contract_sha=signature)
    n_new, n_skipped, t0 = 0, 0, time.time()
    for i, c in enumerate(target):
        cid = c["cell_id"]
        sha = qa_sha(c["question"], c["prediction"])
        if c["gold_is_abstention"]:
            done[cid] = bound({"judge": judge_name, "model": jcfg["model"], "rubric": TWO_AXIS_VERSION,
                         "skipped_abstention_gold": True, "truth_raw": "<skipped>", "truth": None, "info_raw": "<skipped>", "info": None},c,True)
            n_skipped += 1
            continue
        prev = done.get(cid)
        if (prev and prev.get("rubric") == TWO_AXIS_VERSION and prev.get("qa_sha") == sha
                and prev.get("truth") in VALID_T and prev.get("info") in VALID_I):
            continue
        if not c["prediction"].strip():
            done[cid] = bound({"judge": judge_name, "model": jcfg["model"], "rubric": TWO_AXIS_VERSION,
                         "truth_raw": "<empty_output>", "truth": "T", "info_raw": "<empty_output>", "info": "R", "empty_output": True},c,True)
            continue
        if judge is None:
            judge=TwoAxisLLMJudge(jcfg,cache_dir=_cache_dir(out,cfg),name=judge_name)
        if getattr(judge,'judge_contract_sha',None)!=signature:
            raise ValueError('supplied judge client does not match the declared request contract')
        g = judge.grade(c["question"], c["gold_refs"], c["prediction"], c["incorrect"] or None)
        done[cid] = bound(g,c)
        n_new += 1
        if n_new % log_every == 0:
            write_jsonl(list(done.values()), path)
            print("[%s] %s %d/%d graded, %d new calls, %d cache hits (%.0fs)" % (
                time.strftime("%H:%M:%S"), judge_name, i + 1, len(target), judge.llm.n_calls, judge.llm.n_cache_hits,
                time.time() - t0), flush=True)
    if _workspace(run_dir)[0]!=meta or _aggregation_code()!=code_before:
        raise RuntimeError('source generation/cells or executing code changed during grading; no complete manifest written')
    write_jsonl(list(done.values()), path)
    judge_meta={'version':ADJUDICATION_VERSION,'judge':judge_name,'judge_contract':contract,
                     'judge_contract_sha':signature,'cells_sha256':file_sha(os.path.join(out,'cells.jsonl')),
                     'generation_sha256':meta['generation_sha256'],'rows_sha256':file_sha(path),
                     'cache_dir':_cache_dir(out,cfg)}
    mp=os.path.join(out,judge_name+'_meta.json')
    previous_meta=_read_json(mp) if os.path.exists(mp) else {}
    if {k:v for k,v in previous_meta.items() if k!='ts'}!=judge_meta:
        atomic_json_dump({**judge_meta,'ts':_now()},mp)
    graded = [done[c["cell_id"]] for c in target if c["cell_id"] in done and not done[c["cell_id"]].get("skipped_abstention_gold")]
    tr = Counter(r.get("truth") for r in graded)
    ir = Counter(r.get("info") for r in graded)
    summary = {"judge": judge_name, "model": jcfg["model"], "n_target": len(target), "n_graded": len(graded),
               "n_skipped_abstention_gold": n_skipped, "n_no_verdict": sum(r.get("truth") not in VALID_T or r.get("info") not in VALID_I for r in graded),
               "truth_letters": dict(tr), "info_letters": dict(ir), "api_calls": judge.llm.n_calls if judge else 0,
               "cache_hits": judge.llm.n_cache_hits if judge else 0, "stale_regraded": len(vstats["stale"]), "legacy_rows_backfilled": vstats["legacy_backfilled"],
               "seconds": round(time.time() - t0, 1)}
    return summary


# ----------------------------------------------------------------------------- disputes / adjudication
def _load_judges(run_dir, cells, cfg, required=("allenai", "pro"), names=JUDGES):
    """All judge files, verified against the current cells. A stale row (graded on another answer) is fatal here:
    these steps cannot re-grade, the user must re-run that judge's step."""
    out = j2dir(run_dir)
    workspace_meta=_workspace(run_dir)[0]
    judges, stats = {}, {}
    for n in names:
        path=os.path.join(out,n+'.jsonl')
        rows, st = _verify_rows(_rows(path), cells,n,cfg)
        if st["stale"]:
            raise SystemExit("%s.jsonl: %d rows were graded on a different answer than the current cells (e.g. %s); "
                             "re-run the `%s` step" % (n, len(st["stale"]), st["stale"][:3], n))
        if rows:
            mp=os.path.join(out,n+'_meta.json')
            if not os.path.exists(mp):raise ValueError(n+' vote manifest missing')
            meta=_read_json(mp)
            if (meta.get('rows_sha256')!=file_sha(path)
                    or meta.get('cells_sha256')!=file_sha(os.path.join(out,'cells.jsonl'))
                    or meta.get('generation_sha256')!=workspace_meta['generation_sha256']
                    or meta.get('judge_contract_sha')!=canonical_sha(judge_contract(n,cfg))):
                raise ValueError(n+' vote manifest/parameters differ; finish or rerun its grading step')
            if n=='allenai':
                runtime=meta.get('runtime_sha')
                if (not runtime or canonical_sha(meta.get('runtime_identity'))!=runtime
                        or any(r.get('runtime_sha')!=runtime for r in rows.values())):
                    raise ValueError('AllenAI rows mix unverified model/tokenizer runtimes')
        judges[n], stats[n] = rows, st
    for n in required:
        if not judges[n]:
            raise SystemExit("missing or empty %s.jsonl" % n)
    return judges, stats


def _dispute_sources(run_dir):
    out=j2dir(run_dir)
    return {n:file_sha(os.path.join(out,n)) for n in ('meta.json','cells.jsonl','allenai.jsonl','allenai_meta.json','pro.jsonl','pro_meta.json')}


def _verify_dispute_manifest(run_dir,cfg):
    out=j2dir(run_dir);mp=os.path.join(out,'disputes_manifest.json')
    if not os.path.exists(mp):raise ValueError('missing frozen dispute selection; recompute disputes')
    m=_read_json(mp)
    if (m.get('sources')!=_dispute_sources(run_dir) or m.get('disputes_sha256')!=file_sha(os.path.join(out,'disputes.jsonl'))
            or m.get('threshold')!=cfg['allenai']['threshold'] or m.get('version')!=ADJUDICATION_VERSION):
        raise ValueError('dispute selection was made from different inputs/votes')
    return m


def compute_disputes(run_dir, cfg, allow_unresolved_pro=False):
    cells = _cells(run_dir)
    active = _active(cells)
    j, _ = _load_judges(run_dir, cells,cfg,names=('allenai','pro'))
    thr = cfg["allenai"]["threshold"]
    missing = [c["cell_id"] for c in active if c["cell_id"] not in j["allenai"] or c["cell_id"] not in j["pro"]]
    if missing:
        raise SystemExit("%d cells lack an AllenAI or Pro grade, e.g. %s" % (len(missing), missing[:3]))
    # a Pro row without a verdict is a transient failure, not a disagreement: GPT is the third vote only when
    # AllenAI and Pro disagree (red team finding 8)
    def _no_verdict(row):
        if row.get("truth_malformed_persistent") or row.get("info_malformed_persistent"):
            return False            # Pro keeps answering unparseably: adjudicate_cell routes the cell to the human
        return row.get("truth") not in VALID_T or row.get("info") not in VALID_I
    no_verdict = [c["cell_id"] for c in active if c["prediction"].strip() and _no_verdict(j["pro"][c["cell_id"]])]
    if no_verdict and not allow_unresolved_pro:
        raise SystemExit("%d Pro rows have no verdict (e.g. %s): re-run `llm --judge pro` (it retries), or pass "
                         "--allow-unresolved-pro to export only the other valid disputes; GPT cannot replace the missing Pro vote" % (len(no_verdict), no_verdict[:3]))
    disputes, stats = [], Counter()
    for c in active:
        if c['cell_id'] in no_verdict:
            stats['status_needs_pro']+=1
            continue
        rec = adjudicate_cell(c, allenai=j["allenai"][c["cell_id"]], pro=j["pro"][c["cell_id"]], gpt=None, threshold=thr)
        stats["truth_" + rec["truth_basis"]] += 1
        stats["info_" + rec["info_basis"]] += 1
        stats["status_" + rec["status"]] += 1
        if rec["status"] == "needs_gpt" or c["audit"]:
            disputes.append({"cell_id": c["cell_id"], "reason": "disagreement" if rec["status"] == "needs_gpt" else "audit",
                             "input_sha":c['input_sha'],'mandatory_human':rec['status']=='needs_human',
                             "audit": c["audit"], "votes_truth": rec["votes_truth"], "votes_info": rec["votes_info"],
                             "info_subtypes": rec["info_subtypes"]})
    write_jsonl(disputes, os.path.join(j2dir(run_dir), "disputes.jsonl"))
    atomic_json_dump({'version':ADJUDICATION_VERSION,'sources':_dispute_sources(run_dir),
                     'threshold':thr,'disputes_sha256':file_sha(os.path.join(j2dir(run_dir),'disputes.jsonl'))},
                    os.path.join(j2dir(run_dir),'disputes_manifest.json'))
    summary = {"n_cells": len(cells), "n_active": len(active), "n_excluded_abstention_gold": len(cells) - len(active),
               "n_pro_no_verdict": len(no_verdict), "n_to_gpt": len(disputes),
               "n_disagreement": sum(d["reason"] == "disagreement" for d in disputes),
               "n_audit_only": sum(d["reason"] == "audit" for d in disputes), **dict(stats)}
    return summary


def _kappa(pairs):
    """Cohen's kappa for a list of (a, b) label pairs (any number of categories)."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def _rate(pairs):
    return {"n": len(pairs), "agreement": sum(x == y for x, y in pairs) / len(pairs) if pairs else None, "kappa": _kappa(pairs)}


def _agreement(cells, judges, humans, thr, inconsistent=None):
    """Agreement statistics on real votes only: cells with substantive gold and a non-empty answer (empty outputs and
    abstention-only gold carry placeholder rows, red team finding 3). Two samples are reported for everything:
    `all` = every cell where both parties voted (biased against GPT, which votes only on disputed cells) and
    `audit` = the fixed random audit sample, the one unbiased common sample (finding 2)."""
    def votes(name, cid):
        r = judges.get(name, {}).get(cid)
        if r is None or r.get("empty_output") or r.get("skipped_abstention_gold"):
            return None, None, None
        if name == "allenai":
            t, i = allenai_votes(r, thr)
            return t, i, None
        t = r.get("truth") if r.get("truth") in TRUTH_VALUES else None
        return t, info_binary(r.get("info")), (r.get("info") if r.get("info") in ("S", "H", "R") else None)

    real = [c for c in cells if not c["gold_is_abstention"] and c["prediction"].strip()]
    samples = {"all": real, "audit": [c for c in real if c["audit"]]}
    names = [n for n in JUDGES if judges.get(n)]
    out = {"pairwise": {}, "vs_human": {}, "n_cells_considered": {k: len(v) for k, v in samples.items()}}
    for sk, sample in samples.items():
        for a_i in range(len(names)):
            for b_i in range(a_i + 1, len(names)):
                a, b = names[a_i], names[b_i]
                pt, pi = [], []
                for c in sample:
                    ta, ia, _ = votes(a, c["cell_id"])
                    tb, ib, _ = votes(b, c["cell_id"])
                    if ta and tb:
                        pt.append((ta, tb))
                    if ia and ib:
                        pi.append((ia, ib))
                out["pairwise"]["%s_vs_%s/%s" % (a, b, sk)] = {"truth": _rate(pt), "info": _rate(pi)}
        hcells = [c for c in sample if c["cell_id"] in humans]
        if not hcells:
            continue
        for name in names:
            pairs = [];human_truth=[];human_info=[]
            for c in hcells:
                t, i, sub = votes(name, c["cell_id"])
                own = judge_own_label(t, sub if name != "allenai" else i)
                if own is not None:
                    h = humans[c["cell_id"]]["label"]
                    pairs.append((coarse(own), coarse(h)) if name == "allenai" else (own, h))
                hr=normalize_human_verdict(humans[c['cell_id']])
                if t:human_truth.append((t,hr['truth']))
                if i:human_info.append((i,hr['info']))
            out["vs_human"]["%s/%s" % (name, sk)] = dict(_rate(pairs), coarse_labels=(name == "allenai"),
                truth=_rate(human_truth),info=_rate(human_info),n_human_cells=len(hcells))
        pairs = [];pipeline_truth=[];pipeline_info=[]
        for c in hcells:
            rec = adjudicate_cell(c, allenai=judges.get("allenai", {}).get(c["cell_id"]), pro=judges.get("pro", {}).get(c["cell_id"]),
                                  gpt=judges.get("gpt", {}).get(c["cell_id"]), human=None, threshold=thr,
                                  inconsistent_judges=(inconsistent or {}).get(c['cell_id'], ()))
            if rec["label"] is not None:
                pairs.append((rec["label"], humans[c["cell_id"]]["label"]))
                t,i=cell_axes(rec);hr=normalize_human_verdict(humans[c['cell_id']])
                pipeline_truth.append((t,hr['truth']));pipeline_info.append((i,hr['info']))
        out["vs_human"]["pipeline_without_human/%s" % sk] = dict(_rate(pairs), n_human_cells=len(hcells),
            truth=_rate(pipeline_truth),info=_rate(pipeline_info),
            automatic_resolution_fraction=len(pairs)/len(hcells) if hcells else None,
            matches_over_all_human_cells=sum(a==b for a,b in pairs)/len(hcells) if hcells else None)
    return out


def _external_context(run_dir, cfg=None, recorded=None):
    """Keep actual shared dependencies readable without relying on shell state."""
    if recorded is not None:
        if not isinstance(recorded, dict) or set(recorded) != {'cache_dir', 'cache_files', 'human_file'}:
            raise ValueError('missing or invalid external adjudication source context')
        for env, field in (('TD_JUDGE2_CACHE', 'cache_dir'), ('TD_JUDGE2_HUMAN', 'human_file')):
            value = os.environ.get(env)
            if value and os.path.realpath(value) != recorded[field]:
                raise ValueError('shared adjudication context changed; adjudicate again')
        return recorded
    from core.judge import cache_filename
    from core.judge_two_axis import CACHE_PREFIX
    cfg = load_cfg() if cfg is None else cfg
    cache_dir = os.path.realpath(_cache_dir(j2dir(run_dir), cfg))
    human = os.environ.get('TD_JUDGE2_HUMAN') or None
    return {'cache_dir': cache_dir,
            'cache_files': [os.path.join(cache_dir, cache_filename(cfg[n]['model'], prefix=CACHE_PREFIX))
                            for n in ('pro', 'gpt') if n in cfg],
            'human_file': os.path.realpath(human) if human else None}


def _final_sources(run_dir, external_context=None):
    out=j2dir(run_dir)
    names=('meta.json','cells.jsonl','allenai.jsonl','allenai_meta.json','pro.jsonl','pro_meta.json',
           'gpt.jsonl','gpt_meta.json','human_verdicts.jsonl')
    result = {n:file_sha(os.path.join(out,n)) if os.path.exists(os.path.join(out,n)) else None for n in names}
    context = external_context or _external_context(run_dir)
    paths = list(context['cache_files']) + [os.path.join(context['cache_dir'], 'conflicts.json')]
    if context['human_file']:
        paths.append(context['human_file'])
    result['external'] = {p: file_sha(p) if os.path.exists(p) else None for p in sorted(set(paths))}
    return result


def _aggregation_code():
    paths={'scripts/review_checks/judge_two_axis_run.py':__file__}
    for rel in ('core/judge_two_axis.py','core/judge.py','core/judge_two_axis_contract.py','core/qa_scoring.py','core/io_utils.py',
                'core/judge_cache_store.py'):
        paths[rel]=os.path.join(CODE_ROOT,rel)
    return {rel:file_sha(path) for rel,path in paths.items()}


def load_verified_final(run_dir,generation_path,require_complete=True):
    """Return (meta, cells_by_id, finals_by_id), with content/source validation.

    Completeness means all active semantic decisions are resolved. A still
    pending random human audit is separately reported, never called validated.
    """
    meta,cells,raw=_workspace(run_dir);out=j2dir(run_dir)
    if file_sha(generation_path)!=meta['generation_sha256']:raise ValueError('analysis generation differs from grading generation')
    mp=os.path.join(out,'final_manifest.json');fp=os.path.join(out,'final.jsonl')
    if not os.path.exists(mp):raise ValueError('no verified final manifest; adjudicate current votes first')
    manifest=_read_json(mp)
    context = _external_context(run_dir, recorded=manifest.get('external_context'))
    if 'external_context' not in manifest:
        raise ValueError('final manifest does not bind shared inputs; adjudicate current votes again')
    if (manifest.get('version')!=ADJUDICATION_VERSION or manifest.get('rubric')!=TWO_AXIS_VERSION
            or manifest.get('sources')!=_final_sources(run_dir, context) or manifest.get('final_sha256')!=file_sha(fp)
            or manifest.get('generation_sha256')!=meta['generation_sha256']
            or manifest.get('aggregation_code')!=_aggregation_code()):
        raise ValueError('final decisions or any of their source files changed; adjudicate again')
    finals=_rows(fp);by_id={c['cell_id']:c for c in cells}
    if list(finals)!=list(by_id):raise ValueError('final output has missing, reordered or foreign cells')
    seen={}
    for cid,c in by_id.items():
        r=finals[cid]
        if (r.get('input_sha')!=cell_sha(c) or r.get('pair_id')!=c['pair_id']
                or r.get('condition')!=c['condition'] or r.get('gold_is_abstention') is not c['gold_is_abstention']
                or r.get('adjudication_version')!=ADJUDICATION_VERSION):
            raise ValueError('final content/cohort identity changed')
        if c['gold_is_abstention']:
            if r.get('status')!='excluded_abstention_gold' or r.get('label') is not None:raise ValueError('excluded cohort decision inconsistent')
            continue
        if r.get('status')!='final':
            if require_complete:raise ValueError('unresolved semantic decision; finish mandatory judging/human review')
            continue
        axes=cell_axes(r);value=(r['label'],*axes,r['info_subtype'])
        if c['input_sha'] in seen and seen[c['input_sha']]!=value:raise ValueError('same semantic request has inconsistent final decisions')
        seen[c['input_sha']]=value
    return {**meta,'final_verification':manifest},by_id,finals


def adjudicate(run_dir, cfg):
    out = j2dir(run_dir)
    cells = _cells(run_dir)
    context = _external_context(run_dir, cfg)
    sources_before=_final_sources(run_dir, context);code_before=_aggregation_code()
    meta = _read_json(os.path.join(out,'meta.json'))
    judges, vstats = _load_judges(run_dir, cells,cfg)
    humans = _human_verdicts(run_dir,cells)
    thr = cfg["allenai"]["threshold"]
    inconsistent = _inconsistent_cells(cells, cfg, _cache_dir(out, cfg))
    finals, queue = [], []
    for c in cells:
        cid = c["cell_id"]
        if c["gold_is_abstention"]:
            finals.append({"cell_id": cid, "pair_id": c["pair_id"], "condition": c["condition"], "audit": c["audit"],
                           "gold_is_abstention": True, "label": None, "source": None, "status": "excluded_abstention_gold",
                           "votes_truth": {}, "votes_info": {}, "info_subtypes": {}, "truth": None, "info": None,
                           "subtype_disputed": False, "judges_label": None,'input_sha':c['input_sha'],
                           'adjudication_version':ADJUDICATION_VERSION})
            continue
        if cid not in judges["allenai"] or cid not in judges["pro"]:
            raise SystemExit("cell %s lacks an AllenAI or Pro grade; run `allenai` / `llm --judge pro` first" % cid)
        rec = adjudicate_cell(c, allenai=judges["allenai"].get(cid), pro=judges["pro"].get(cid), gpt=judges["gpt"].get(cid),
                              human=humans.get(cid), threshold=thr, inconsistent_judges=inconsistent.get(cid, ()))
        rec.update(pair_id=c["pair_id"], condition=c["condition"], audit=c["audit"], gold_is_abstention=False,
                   input_sha=c['input_sha'])
        finals.append(rec)
        needs = rec["status"] in ("needs_gpt", "needs_human",'needs_pro')
        if (needs or c["audit"]) and cid not in humans:
            queue.append({"cell_id": cid, "reason": "audit" if not needs else rec["status"], "pair_id": c["pair_id"],
                          "condition": c["condition"], "question": c["question"], "gold": c["gold"], "incorrect": c["incorrect"],
                          "best_answer": c.get("best_answer"), "prediction": c["prediction"],
                          "votes_truth": rec["votes_truth"], "votes_info": rec["votes_info"], "info_subtypes": rec["info_subtypes"],
                          "judges_label": rec["judges_label"],'input_sha':c['input_sha'],
                          'finished':c['finished'],'unicode_replacement':c['unicode_replacement'],
                          'review_reason':rec.get('review_reason')})
    # A condition's name cannot change the verdict for the identical semantic
    # input. Human rulings are already broadcast by content, never by method.
    seen={}
    for rec in finals:
        if rec['status']=='final':
            value=(rec['label'],*cell_axes(rec),rec.get('info_subtype'))
            if rec['input_sha'] in seen and seen[rec['input_sha']]!=value:
                raise ValueError('identical full semantic inputs received conflicting final decisions')
            seen[rec['input_sha']]=value
    if (_external_context(run_dir, cfg) != context or _final_sources(run_dir, context)!=sources_before
            or _aggregation_code()!=code_before):
        raise RuntimeError('votes, human decisions, or code changed during adjudication; rerun on a stable snapshot')
    write_jsonl(finals, os.path.join(out, "final.jsonl"))
    write_jsonl(queue, os.path.join(out, "human_queue.jsonl"))
    # summary: per condition, over cells with substantive gold (the analysis denominator). Rates come from the resolved
    # axes (cell_axes), not from the labels: a false hedge is B on the label but "not informative" on the info axis.
    per_cond = OrderedDict()
    for cond in meta["conditions"]:
        rs = [r for r in finals if r["condition"] == cond and not r["gold_is_abstention"]]
        fin = [r for r in rs if r["status"] == "final"]
        axes = [cell_axes(r) for r in fin]
        cnt = Counter(r["label"] for r in rs)
        n, nf = len(rs), len(fin)
        known={'true':sum(t=='T' for t,_ in axes),'info':sum(i=='S' for _,i in axes),
               'true_info':sum(r['label']=='A' for r in fin),
               'refusal_behavior':sum(r['info_subtype']=='R' for r in fin),
               'noninformative_nonrefusal_behavior':sum(r['info_subtype']=='H' for r in fin)}
        complete=n==nf
        cr=[c for c in cells if c['condition']==cond and not c['gold_is_abstention']]
        by_cell={r['cell_id']:r for r in fin}
        per_cond[cond] = {"n": n, "n_final": nf, **{LABEL_NAMES[k] + "_" + k: cnt.get(k, 0) for k in LABELS},
                          "unresolved": cnt.get(None, 0),
                          **{k+'_rate':v/n if n and complete else None for k,v in known.items()},
                          'fixed_cohort_bounds':{k:[v/n,(v+n-nf)/n] if n else None for k,v in known.items()},
                          'resolved_only_rates_diagnostic':{k:v/nf if nf else None for k,v in known.items()},
                          'false_noninformative_count':sum(t=='F' and i=='N' for t,i in axes),
                          'refusal_behavior_count':known['refusal_behavior'],
                          'noninformative_nonrefusal_behavior_count':known['noninformative_nonrefusal_behavior'],
                          'completion_status_missing':sum(c['finished'] is None for c in cr),
                          'unfinished_outputs':sum(c['finished'] is False for c in cr),
                          'replacement_outputs':sum(c['unicode_replacement'] for c in cr),
                          'unfinished_A':sum(c['finished'] is False and by_cell.get(c['cell_id'],{}).get('label')=='A' for c in cr),
                          'replacement_A':sum(c['unicode_replacement'] and by_cell.get(c['cell_id'],{}).get('label')=='A' for c in cr),
                          "rates_denominator": "fixed active cohort n; point rates withheld while unresolved"}
    src = Counter(r["source"] for r in finals)
    status = Counter(r["status"] for r in finals)
    summary = {"version": TWO_AXIS_VERSION, "model": meta.get("model"), "generation_sha256": meta["generation_sha256"],
               "n_cells": len(cells), "n_excluded_abstention_gold": status.get("excluded_abstention_gold", 0),
               "n_final": status.get("final", 0), "n_needs_gpt": status.get("needs_gpt", 0),
               "n_needs_human": status.get("needs_human", 0), "n_human_queue": len(queue),
               "n_human_verdicts": len(humans), "judges_present": [n for n in JUDGES if judges[n]],
               "judge_rows_verified": vstats, "label_sources": dict(src),
               "subtype_disputed": sum(bool(r["subtype_disputed"]) for r in finals),
               "per_condition": per_cond, "agreement": _agreement(cells, judges, humans, thr, inconsistent), "ts": _now()}
    summary.update(adjudication_version=ADJUDICATION_VERSION,
                   n_needs_pro=status.get('needs_pro',0),
                   semantic_decisions_complete=all(r['status'] in ('final','excluded_abstention_gold') for r in finals),
                   human_audit_complete=all(c['cell_id'] in humans for c in cells if c['audit']),
                   agreement_scope='Cell-weighted descriptive agreement; unresolved/missing votes are reported separately. Partial human audit is not an unbiased completed audit.',
                   evaluation_scope='Exploratory project grading: strict LLM information criterion and original-trained AllenAI proxy are not identical annotation standards; no claim of validated human equivalence or full original TruthfulQA comparability.',
                   human_review_context='The current review queue shows votes and condition names. Human rulings are assisted adjudication, not independently blinded benchmark labels; random sampling alone does not remove that annotation bias.')
    atomic_json_dump(summary, os.path.join(out, "summary.json"))
    lines = ["# 两维判分汇总 — %s（%s）" % (meta.get("model"), TWO_AXIS_VERSION), "",
             "格数 %d（金标只有拒答、不计 %d）；已定 %d；待 GPT %d；待人工 %d（人工队列 %d 条，含抽检）；人工已判 %d。" % (
                 len(cells), summary["n_excluded_abstention_gold"], summary["n_final"], summary["n_needs_gpt"],
                 summary["n_needs_human"], len(queue), len(humans)), "",
             "| 条件 | n | 已定 | A 真且有信息 | B 假 | C 拒答 | H 含糊 | 未定 | True | Info | True·Info |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cond, s in per_cond.items():
        f = lambda x: "—" if x is None else "%.3f" % x
        lines.append("| %s | %d | %d | %d | %d | %d | %d | %d | %s | %s | %s |" % (
            cond, s["n"], s["n_final"], s["true_informative_A"], s["false_B"], s["refusal_C"], s["hedge_H"], s["unresolved"],
            f(s["true_rate"]), f(s["info_rate"]), f(s["true_info_rate"])))
    lines.append("")
    lines.append("主比率以固定题数n为分母，存在未决时不显示点比率；JSON另报固定分母上下界与明确标记的已定子集诊断。True/Info按两轴分别算，不由标签B猜信息量。C/H列是终判类别，不是全部拒答/含糊行为率（假的同类行为落在B）；完整行为计数按子类另存JSON。人工抽检未完成时不得称已通过人审验证。")
    ag = summary["agreement"]
    lines += ["", "裁判两两一致率（真 / 有信息量轴；all = 两方都投了票的格，audit = 固定抽检样本）："]
    for k, v in ag["pairwise"].items():
        g = lambda d: "n=%d 一致 %s κ %s" % (d["n"], "—" if d["agreement"] is None else "%.3f" % d["agreement"],
                                             "—" if d["kappa"] is None else "%.2f" % d["kappa"])
        lines.append("- %s：真 %s；信息 %s" % (k, g(v["truth"]), g(v["info"])))
    if ag["vs_human"]:
        lines += ["", "与赵磊的一致率（有人判的格；all / audit 同上）："]
        for k, v in ag["vs_human"].items():
            lines.append("- %s：n=%d 一致 %s κ %s" % (k, v["n"], "—" if v["agreement"] is None else "%.3f" % v["agreement"],
                                                   "—" if v["kappa"] is None else "%.2f" % v["kappa"]))
    md = "\n".join(lines) + "\n"
    Path(out,'summary.md').write_text(md,encoding='utf-8')
    if (_external_context(run_dir, cfg) != context or _final_sources(run_dir, context)!=sources_before
            or _aggregation_code()!=code_before):
        raise RuntimeError('adjudication source changed while summarizing; final manifest not published')
    atomic_json_dump({'version':ADJUDICATION_VERSION,'rubric':TWO_AXIS_VERSION,'sources':sources_before,
                     'external_context':context,
                     'generation_sha256':meta['generation_sha256'],'final_sha256':file_sha(os.path.join(out,'final.jsonl')),
                     'judge_contracts':{n:judge_contract(n,cfg) for n in JUDGES},'aggregation_code':_aggregation_code(),
                     'semantic_decisions_complete':summary['semantic_decisions_complete'],
                     'human_audit_complete':summary['human_audit_complete'],'ts':_now()},
                    os.path.join(out,'final_manifest.json'))
    return summary, md


# ----------------------------------------------------------------------------- human
def single_judge(run_dir, judge_name, cfg):
    """Explicit single-judge exit (plan 21 §3.8, 赵磊 11-10 01:05 "各报各的"): one judge's own verdict per active cell,
    written to judge2/single_<judge>.jsonl + single_<judge>_summary.{json,md}. No disputes, no adjudication, no GPT,
    no vote borrowed from another judge. Cells without a verdict stay 'no_verdict' and are counted, never filled.
    AllenAI's info judge only separates substantive from not-informative, so its labels are A / B / N (N = true but
    not informative: hedge or refusal, which it cannot tell apart); the Pro judge gives A / B / C / H."""
    if judge_name not in ("allenai", "pro"):
        raise ValueError("single-judge exit supports allenai or pro")
    cells = _cells(run_dir)
    active = _active(cells)
    judges, stats = _load_judges(run_dir, cells, cfg, required=(judge_name,), names=(judge_name,))
    rows = judges[judge_name]
    thr = cfg["allenai"]["threshold"]
    out_rows, per_condition = [], {}
    for c in active:
        r = rows.get(c["cell_id"])
        truth = info = label = None
        basis = "no_verdict"
        if r is not None:
            if judge_name == "allenai":
                truth, info = allenai_votes(r, thr)
                if truth is not None:
                    label = "B" if truth == "F" else ("A" if info == "S" else "N")
                    basis = "allenai_p_yes_threshold_%g" % thr
            else:
                truth, info = r.get("truth"), r.get("info")
                if truth in TRUTH_VALUES and info in INFO_SUBTYPES:
                    label = judge_own_label(truth, info)
                    basis = "pro_own_axes"
                    if label == "X":
                        label, basis = None, "pro_contradictory_axes"
                elif r.get("truth_malformed_persistent") or r.get("info_malformed_persistent"):
                    basis = "pro_malformed_persistent"
                elif truth == "D" or info == "D":
                    basis = "pro_declined"
        rec = {"cell_id": c["cell_id"], "pair_id": c["pair_id"], "condition": c["condition"], "judge": judge_name,
               "label": label, "truth": truth, "info": info, "basis": basis, "input_sha": c["input_sha"],
               "rule_abstain": c["rule_abstain"], "unicode_replacement": c["unicode_replacement"]}
        out_rows.append(rec)
        bucket = per_condition.setdefault(c["condition"], Counter())
        bucket[label if label is not None else "no_verdict"] += 1
        bucket["n_active"] += 1
    out = j2dir(run_dir)
    path = os.path.join(out, "single_%s.jsonl" % judge_name)
    write_jsonl(out_rows, path)
    summary = {"version": ADJUDICATION_VERSION, "judge": judge_name, "exit": "single_judge_no_adjudication",
               "n_cells": len(cells), "n_active": len(active), "n_excluded_abstention_gold": len(cells) - len(active),
               "rows_sha256": file_sha(path), "judge_rows_sha256": file_sha(os.path.join(out, judge_name + ".jsonl")),
               "cells_sha256": file_sha(os.path.join(out, "cells.jsonl")), "threshold": thr if judge_name == "allenai" else None,
               "label_meaning": ({"A": "true+substantive", "B": "false", "N": "true but not informative (hedge or refusal)"}
                                 if judge_name == "allenai" else dict(LABEL_NAMES)),
               "per_condition": {k: dict(v) for k, v in sorted(per_condition.items())}}
    atomic_json_dump(summary, os.path.join(out, "single_%s_summary.json" % judge_name))
    letters = ["A", "B", "C", "H", "N", "no_verdict"] if judge_name == "pro" else ["A", "B", "N", "no_verdict"]
    md = ["# single judge: %s (no adjudication, no GPT, no borrowed votes)" % judge_name, "",
          "| condition | n_active | " + " | ".join(letters) + " |", "|---|---|" + "---|" * len(letters)]
    for cond, v in sorted(per_condition.items()):
        md.append("| %s | %d | %s |" % (cond, v["n_active"], " | ".join(str(v.get(l, 0)) for l in letters)))
    with open(os.path.join(out, "single_%s_summary.md" % judge_name), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    return summary, "\n".join(md)


def human_list(run_dir):
    out = j2dir(run_dir)
    humans = _human_verdicts(run_dir)
    path = os.path.join(out, "human_queue.jsonl")
    rows = list(read_jsonl(path)) if os.path.exists(path) else []
    return [r for r in rows if r["cell_id"] not in humans]


def human_set(run_dir, cell_id, letter, by, note=None, truth=None, info_subtype=None, expected_input_sha=None):
    if letter not in LABELS:
        raise SystemExit("a human verdict is one of %s, got %r" % ("/".join(LABELS), letter))
    if not by or not str(by).strip():
        raise SystemExit("record who decided (--by)")
    cells={c['cell_id']:c for c in _cells(run_dir)}
    if cell_id not in cells:
        raise SystemExit("unknown cell id %r" % cell_id)
    cell=cells[cell_id]
    if expected_input_sha is not None and expected_input_sha != cell_sha(cell):
        raise ValueError('reviewed content changed before recording the human verdict')
    if cell['gold_is_abstention']:raise ValueError('excluded reference-only-refusal cell is not part of this grading cohort')
    row = normalize_human_verdict({"cell_id": cell_id, "label": letter, "by": by, "note": note or "", "ts": _now(),
           'truth':truth,'info_subtype':info_subtype,'input_sha':cell_sha(cell),'version':ADJUDICATION_VERSION,
           'reviewed_input':{k:cell[k] for k in ('question','prediction','gold','gold_refs','incorrect','best_answer','gold_is_abstention')}})
    append_jsonl(row, os.path.join(j2dir(run_dir), "human_verdicts.jsonl"))
    shared = os.environ.get('TD_JUDGE2_HUMAN') or None
    if shared and os.path.abspath(shared) != os.path.abspath(os.path.join(j2dir(run_dir), "human_verdicts.jsonl")):
        append_jsonl(row, shared)          # content-bound: the other runs apply it to the identical input
    return row


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cells"); p.add_argument("run_dir"); p.add_argument("--generation", default="tune.json")
    p.add_argument("--audit-n", type=int, default=None); p.add_argument("--audit-seed", type=int, default=None)
    p = sub.add_parser("llm"); p.add_argument("run_dir"); p.add_argument("--judge", required=True, choices=("pro", "gpt"))
    p.add_argument("--only", choices=("disputes",), default=None)
    p = sub.add_parser("disputes"); p.add_argument("run_dir"); p.add_argument("--allow-unresolved-pro", action="store_true")
    p = sub.add_parser("adjudicate"); p.add_argument("run_dir")
    p = sub.add_parser("single"); p.add_argument("run_dir"); p.add_argument("--judge", required=True, choices=("allenai", "pro"))
    p = sub.add_parser("human"); p.add_argument("run_dir"); p.add_argument("action", choices=("list", "set"))
    p.add_argument("cell_id", nargs="?"); p.add_argument("letter", nargs="?"); p.add_argument("--by"); p.add_argument("--note")
    p.add_argument('--truth',choices=('T','F'));p.add_argument('--info',choices=('S','H','R'),dest='info_subtype')
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    if a.cmd == "cells":
        meta = build_cells(a.run_dir, a.generation, cfg["audit"]["n"] if a.audit_n is None else a.audit_n,
                           cfg["audit"]["seed"] if a.audit_seed is None else a.audit_seed)
        brief = {k: v for k, v in meta.items() if k != "audit"}
        brief["audit_n"] = meta["audit"]["n"]
        print(json.dumps(brief, ensure_ascii=False, indent=1))
    elif a.cmd == "llm":
        print(json.dumps(run_llm(a.run_dir, a.judge, cfg, only=a.only), ensure_ascii=False, indent=1))
    elif a.cmd == "disputes":
        print(json.dumps(compute_disputes(a.run_dir, cfg, allow_unresolved_pro=a.allow_unresolved_pro), ensure_ascii=False, indent=1))
    elif a.cmd == "adjudicate":
        _, md = adjudicate(a.run_dir, cfg)
        print(md)
    elif a.cmd == "single":
        _, md = single_judge(a.run_dir, a.judge, cfg)
        print(md)
    elif a.cmd == "human":
        if a.action == "list":
            items = human_list(a.run_dir)
            for r in items:
                print(json.dumps(r, ensure_ascii=False))
            print("# pending: %d" % len(items), file=sys.stderr)
        else:
            if not (a.cell_id and a.letter):
                raise SystemExit("human set RUN_DIR CELL_ID LETTER --by WHO")
            print(json.dumps(human_set(a.run_dir, a.cell_id, a.letter, a.by, a.note,a.truth,a.info_subtype), ensure_ascii=False))


if __name__ == "__main__":
    main()
