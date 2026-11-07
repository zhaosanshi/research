"""Separate natural original-DAS tokens from full-prefix replay probabilities."""
import hashlib
import json
import math
from pathlib import Path
import tarfile

REVIEW=Path(__file__).resolve().parents[3]
R19=REVIEW/'from_gpt/audit_20261106_round19'


def main():
    trace_path=R19/'unicode/evidence/qwen_trace.json'
    trace=json.loads(trace_path.read_text())
    archive=R19/'random/evidence/random_controls.tar.gz'
    with tarfile.open(archive) as tar:
        control=json.load(tar.extractfile('qwen3-8b/full192/tune.json'))
    natural={r['pair_id']:r for r in control['evaluations']}
    stop_ids=set(trace['generation_config']['eos_token_id'])
    rows=[]
    for batch in trace['batches']:
        for replay in batch['common_prefix_replays']:
            if replay['step']!=5:continue
            for index,scores in replay['variants']['original_das'].items():
                pid=batch['pair_ids'][int(index)]
                n=natural[pid];all_ids=n['generation_token_ids']['original_das']
                stop=next(i for i,t in enumerate(all_ids) if t in stop_ids)
                effective=all_ids[:stop+1]
                corrupted=trace['cases'][pid]['effective_ids']
                assert stop==5 and effective[:5]==corrupted[:5]==[40,1513,944,1414,13]
                assert effective[5]==scores['argmax_id']==151645
                winner=next(t for t in scores['tokens'] if t['id']==scores['argmax_id'])
                adapted=replay['variants']['adapted'][index]
                adapted_top=next(t for t in adapted['tokens'] if t['id']==adapted['argmax_id'])
                cached=trace['cases'][pid]['steps'][5]['token']
                cached_top=next(t for t in cached['tokens'] if t['id']==cached['argmax_id'])
                rows.append({'pair_id':pid,'natural_original_das_effective_ids':effective,
                    'natural_original_das_output':n['outputs']['original_das'],
                    'natural_original_das_eos_chosen':True,'natural_prefix_equals_common_replay':True,
                    'original_das_full_prefix_recompute_eos_percent':100*math.exp(winner['log_probability']),
                    'original_das_cached_generation_probability_present_in_this_evidence':False,
                    'adapted_cached_generation_chosen_probability_percent':100*math.exp(cached_top['log_probability']),
                    'adapted_full_prefix_recompute_chosen_probability_percent':100*math.exp(adapted_top['log_probability']),
                    'adapted_cached_vs_recompute_max_logit_difference':adapted['cached_processed_score_max_abs_difference']})
    result={'trace_sha256':hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            'natural_control_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
            'cases':sorted(rows,key=lambda x:x['pair_id']),
            'all_assertions_passed':True,
            'supported':'At a naturally reachable matching I-do-not-know prefix, original DAS still naturally chooses EOS; full-prefix replay shows a very flat distribution.',
            'not_supported':'The listed full-prefix replay probabilities are not measured cached natural-generation probabilities. Step-7 emoji-prefix replays are not natural original-DAS trajectories.'}
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
