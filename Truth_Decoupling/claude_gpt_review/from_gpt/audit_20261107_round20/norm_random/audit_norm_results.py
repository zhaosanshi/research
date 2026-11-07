"""Independent per-step stored-norm, natural-stop and parent-inheritance audit."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();stops=json.loads((args.runs/'native_stop_config.json').read_text())
    result={'scope':'orientation null with learned scalar dose at each current random-trajectory state; not a wholly untrained/source-free method',
            'native_stop_config':stops,'models':{}}
    for model in ('llama3.1-8b','qwen3-8b'):
        run=args.runs/model/'full192';raw_path=run/'tune.json';raw=json.loads(raw_path.read_text())
        parent_path=args.parent/model/'tune.json';parent=json.loads(parent_path.read_text())
        assert raw['completed'] and raw['random_vectors_unchanged'] and raw['base_parameters_unchanged']
        assert raw['identity']['parent_generation_sha256']==digest(parent_path)
        assert len(raw['evaluations'])==192 and len(raw['conditions'])==10
        assert raw['identity']['seeds']==[42,43,44] and raw['identity']['training_called'] is False
        assert [r['pair_id'] for r in raw['evaluations']]==[r['pair_id'] for r in parent['evaluations']]
        names=['size_matched_random_s%d'%s for s in (42,43,44)]
        stats={n:{'writes':[],'new_tokens':0,'finished':0,'fffd':0,'texts_changed_vs_baseline':0} for n in names}
        eos=set(stops[model]['eos_token_id'])
        for row,old in zip(raw['evaluations'],parent['evaluations']):
            for key in ('pair_id','idx','question','gold','incorrect','best_answer','prompt_token_ids'):
                assert row.get(key)==old.get(key)
            for n in parent['conditions']:
                for key in ('outputs','finished','rule_abstain','generation_token_ids','trace'):
                    assert row[key][n]==old[key][n]
            for n in names:
                tokens=row['generation_token_ids'][n];trace=row['trace'][n]
                assert 1<=len(tokens)<=48 and len(trace)==len(tokens)
                stop_positions=[i for i,t in enumerate(tokens) if t in eos]
                if row['finished'][n]:assert stop_positions==[len(tokens)-1]
                else:assert not stop_positions and len(tokens)==48
                for step,t in enumerate(trace):
                    assert t['step']==step and t['accepted_prefix_ids']==tokens[:step] and t['next_top1_id']==tokens[step]
                    assert t['cache_storage_disjoint'] and len(t['writes'])==1
                    assert t['control_protocol']=='random_orientation_stored_write_norm_match_v1'
                    w=t['writes'][0];target=w['target_stored_delta_norm'];actual=w['random_stored_delta_norm'];h=w['h_norm']
                    assert all(math.isfinite(v) and v>=0 for v in (target,actual,h))
                    tolerance=.01*target+.0001*h;error=abs(actual-target)
                    assert abs(w['absolute_norm_error']-error)<1e-5
                    assert abs(w['cast_match_tolerance']-tolerance)<1e-5*max(1,tolerance)
                    assert error<=tolerance+1e-7
                    assert w['fixed_random_axis'] and w['learned_direction_used_for_size_only']
                    assert w['source_gap_projection_sign'] in (-1.,1.) and not w['is_unscaled_source_interchange']
                    stats[n]['writes'].append(w)
                stats[n]['new_tokens']+=len(tokens);stats[n]['finished']+=int(row['finished'][n])
                stats[n]['fffd']+=int('\ufffd' in row['outputs'][n])
                stats[n]['texts_changed_vs_baseline']+=int(row['outputs'][n]!=old['outputs']['baseline'])
        for n,s in stats.items():
            writes=s.pop('writes');positive=[w for w in writes if w['target_stored_delta_norm']>0]
            s.update(n_actual_writes=len(writes),n_post_eos_padding_writes=0,
                zero_target_writes=sum(w['target_stored_delta_norm']==0 for w in writes),
                max_error_over_target=max((w['absolute_norm_error']/w['target_stored_delta_norm'] for w in positive),default=0),
                max_error_over_declared_tolerance=max((w['absolute_norm_error']/w['cast_match_tolerance'] if w['cast_match_tolerance'] else 0 for w in writes),default=0),
                median_actual_write_over_h=statistics.median(w['random_relative_write'] for w in writes),
                max_actual_write_over_h=max(w['random_relative_write'] for w in writes))
        summary={'raw_sha256':digest(raw_path),'all_parent_raw_fields_identical':True,
                 'all_actual_stored_norm_and_no_post_stop_assertions_passed':True,'conditions':stats}
        grade_path=run/'tune_scored_items.json'
        if grade_path.exists():
            graded=json.loads(grade_path.read_text());pg=json.loads((args.parent/model/'tune_scored_items.json').read_text())
            assert graded['completed'] and graded['grading_completed'] and graded['grading_source_generation_sha256']==digest(raw_path)
            assert graded['frozen_control_grade_verification']['question_condition_grades_checked']==1344
            for r,g,old in zip(raw['evaluations'],graded['evaluations'],pg['evaluations']):
                assert r['pair_id']==g['pair_id']==old['pair_id'] and r['outputs']==g['outputs'] and r['finished']==g['finished']
                for n in parent['conditions']:assert g['outcome'][n]==old['outcome'][n]
                for n in raw['conditions']:assert g['outcome'][n]==g['judge'][n]['label']
            summary['all_1344_parent_grades_identical']=True
            summary['natural_counts']={n:dict(Counter(r['outcome'][n] for r in graded['evaluations'])) for n in raw['conditions']}
        result['models'][model]=summary
        print(model,stats,summary.get('natural_counts'),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
