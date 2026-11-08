"""Bind historical v5 vote rows to the original full prompt cache, offline.

This proves the observed v5 inputs/replies, not a new v6 judgment. Parameters
which the old cache did not record stay explicitly unknown.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;BASE=HERE/'source_cf8f752'
sys.path.insert(0,str(BASE))
from core.io_utils import stable_hash
from core.qa_scoring import reference_answers


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    inputs=Path('/tmp/td_r22_inputs');runtime=inputs/'runtime/review_r23_claude'
    T=load_module('core.historical_two_axis_cache_audit',runtime/'core/judge_two_axis.py')
    cfg=json.loads((runtime/'configs/judges/two_axis_v5.json').read_text())
    assert T.TWO_AXIS_VERSION==cfg['version']=='qa_two_axis_v5'
    result={'scope':'Historical v5 only; no vote promoted to v6 and no new API call',
        'runtime':{str(p.relative_to(runtime)):sha(p) for p in (runtime/'core/judge_two_axis.py',runtime/'core/judge.py',runtime/'configs/judges/two_axis_v5.json')},
        'runs':{}}
    for tag in ('select-tune','oracle-tune'):
        run=inputs/'runs'/tag;j2=run/'judge2';cells=rows(j2/'cells.jsonl');generation=json.loads((run/'tune.json').read_text());meta=json.loads((j2/'meta.json').read_text())
        assert meta['generation_sha256']==sha(run/'tune.json')
        byid={r['pair_id']:r for r in generation['evaluations']};assert len(byid)==len(generation['evaluations'])
        assert len(cells)==len(byid)*len(generation['conditions'])
        for cell in cells:
            source=byid[cell['pair_id']];refs,only=reference_answers(source['gold'])
            assert cell['condition'] in generation['conditions'] and cell['prediction']==source['outputs'][cell['condition']]
            for field in ('question','gold','best_answer'):assert cell.get(field)==source.get(field)
            assert cell['incorrect']==(source.get('incorrect') or []) and cell['gold_refs']==refs and cell['gold_is_abstention']==only
        mapping={c['cell_id']:c for c in cells};judges={}
        for name in ('pro','gpt'):
            path=j2/(name+'.jsonl');vote_rows=rows(path);jc=cfg[name]
            cachepath=j2/'cache'/('judge2_cache_'+jc['model']+'.jsonl');cache_rows=rows(cachepath)
            cache={r['key']:r for r in cache_rows}
            params_tag={'rubric':T.TWO_AXIS_VERSION,'reasoning_effort':jc.get('reasoning_effort'),
                        'max_completion_tokens':jc.get('max_completion_tokens'),
                        'temperature':jc.get('temperature',0.0) if jc.get('send_temperature',True) else None}
            expected_params={'max_tokens':None if jc.get('max_completion_tokens') else jc.get('max_tokens',8),
                'max_completion_tokens':jc.get('max_completion_tokens'),'reasoning_effort':jc.get('reasoning_effort'),
                'temperature':params_tag['temperature'],'key_extra':params_tag}
            checked=0;placeholders=0;failures=[];transient=[];resolved=set();unknown=set();keys=set()
            for vote in vote_rows:
                cell=mapping[vote['cell_id']]
                assert vote['model']==jc['model'] and vote['rubric']==T.TWO_AXIS_VERSION
                assert vote['qa_sha']==T.qa_sha(cell['question'],cell['prediction'])
                if vote.get('empty_output') or vote.get('skipped_abstention_gold'):
                    assert not cell['prediction'].strip() or cell['gold_is_abstention'];placeholders+=1;continue
                for axis,system,user,parse in [
                    ('truth',T.TRUTH_SYSTEM,T.truth_user(cell['question'],cell['gold_refs'],cell['prediction'],cell['incorrect'] or None),T.parse_truth),
                    ('info',T.INFO_SYSTEM,T.info_user(cell['question'],cell['prediction']),T.parse_info)]:
                    key=stable_hash(jc['model'],params_tag,system,user,length=24);entry=cache.get(key)
                    if vote.get(axis+'_raw')=='<unresolved>' and vote.get(axis) is None and entry is None:
                        transient.append({'cell_id':vote['cell_id'],'axis':axis,'reason':'transient failure intentionally not cached; contributes no vote'})
                        continue
                    if entry is None or entry.get('answer')!=vote[axis+'_raw'] or parse(entry['answer'])!=vote[axis]:
                        failures.append({'cell_id':vote['cell_id'],'axis':axis,'cache_key':key,'reason':'missing_or_different_full_prompt_reply'});continue
                    if entry.get('model')!=jc['model'] or any(entry.get('params',{}).get(k)!=v for k,v in expected_params.items()):
                        failures.append({'cell_id':vote['cell_id'],'axis':axis,'cache_key':key,'reason':'recorded_request_parameters_differ'});continue
                    resolved.add(entry.get('resolved_model'));keys.add(key);checked+=1
                    for field in ('base_url','extra_body','send_temperature'):
                        if field not in entry.get('params',{}):unknown.add(field)
            judges[name]={'vote_rows':len(vote_rows),'placeholders':placeholders,'verified_axis_cache_links':checked,
                'unique_cache_keys_used':len(keys),'cache_rows_in_file':len(cache_rows),'failures':failures,
                'uncached_transient_failures':transient,
                'qa_sha_marked_backfilled_rows':sum(bool(r.get('qa_sha_backfilled')) for r in vote_rows),
                'provider_reported_resolved_models':sorted(str(v) for v in resolved),
                'parameters_not_recorded_at_call_time':sorted(unknown),
                'cache_sha256':sha(cachepath),'vote_file_sha256':sha(path),
                'interpretation':'Exact v5 prompt/cache response match and recorded parameters agree; unknown parameters not backfilled. This is NOT a v6 vote.'}
        result['runs'][tag]={'generation_sha256':sha(run/'tune.json'),'cells_sha256':sha(j2/'cells.jsonl'),
            'cells_exactly_match_generation':True,'n_cells':len(cells),'max_answer_chars':max(len(c['prediction']) for c in cells),
            'all_predictions_fully_present_in_v5_prompt':all(len(c['prediction'].strip())<=4000 for c in cells),'judges':judges}
        print(tag,{name:{k:v[k] for k in ('vote_rows','verified_axis_cache_links','unique_cache_keys_used','failures','provider_reported_resolved_models','parameters_not_recorded_at_call_time')} for name,v in judges.items()},flush=True)
    (HERE/'live_cache_association.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
