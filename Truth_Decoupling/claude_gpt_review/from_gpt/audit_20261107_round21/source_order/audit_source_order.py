"""Independent R21 raw traces and train-only post-cast C calibration recount."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def independent_profile(h,w,coefficient):
    # Reconstruct from immutable native-dtype hidden tensors, not reported errors.
    h=h.float();values=[]
    for sign in (-1.,1.):
        after=(h+sign*float(np.float32(coefficient))*w).bfloat16().float()
        # Evaluate the stored-value L2 in FP64, independently of CPU/CUDA FP32
        # reduction order; keep the intervention/add/store itself in FP32/BF16.
        values.append(((after.double()-h.double()).norm(dim=-1)/h.double().norm(dim=-1)).tolist())
    return {'negative':values[0],'positive':values[1],'median':statistics.median(values[0]+values[1])}


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--c-banks',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--full',action='store_true');args=p.parse_args()
    out={'scope':'all raw conditions; calibration independently reconstructed from saved BF16 hidden states; tuning only','models':{}}
    for model,e,before,after in [('llama3.1-8b',15,7,23),('qwen3-8b',17,8,26)]:
        root=args.inputs/model;calpath=root/'calibration/c_calibration.json';cal=json.loads(calpath.read_text())
        statepath=root/'calibration/unmodified_prompt_states.pt';assert sha(statepath)==cal['states_sha256']
        bankpath=args.c_banks/model/'qa_ability_full/bank.npz';assert sha(bankpath)==cal['identity']['c_bank_sha256']
        states=torch.load(statepath,map_location='cpu',weights_only=True);bank=np.load(bankpath,allow_pickle=False)
        profiles={}
        for side,layer in [('before',before),('after',after)]:
            h=states[side];assert h.dtype==torch.bfloat16 and h.shape==(92,4096)
            w=bank['qa_das_C_L%d'%layer].astype(float);w=torch.tensor(w/np.linalg.norm(w),dtype=torch.float32)
            point=cal['calibration']['layers'][side]
            assert point['original_coefficient']==float(bank['qa_das_C_L%d_scale'%layer])
            profiles[side]={}
            for name,key in [('original','original_coefficient'),('calibrated','coefficient')]:
                actual=independent_profile(h,w,point[key]);saved=point[name+'_profile']
                for sign in ('negative','positive'):
                    assert np.max(np.abs(np.asarray(actual[sign])-saved[sign]))<1e-7
                assert abs(actual['median']-saved['median'])<1e-7
                profiles[side][name]=actual
        rho=min(v['original']['median'] for v in profiles.values());assert abs(rho-cal['calibration']['rho'])<1e-7
        for v in profiles.values():assert abs(v['calibrated']['median']-rho)<=cal['calibration']['tolerance']
        model_result={'calibration_sha256':sha(calpath),'independent_rho':rho,'independent_profiles':profiles,'runs':{}}
        tags=['original_sigma_pilot3','calibrated_pilot3']+(['full192'] if args.full else [])
        for tag in tags:
            stem='tune' if tag=='full192' else 'check';rawpath=root/tag/(stem+'.json');raw=json.loads(rawpath.read_text())
            gradepath=root/tag/(stem+'_scored_items.json');graded=json.loads(gradepath.read_text())
            assert raw['completed'] and raw['all_controls_freshly_generated'] and raw['baseline_E_fresh_exact_reference']
            assert graded['completed'] and graded['grading_completed'] and graded['grading_source_generation_sha256']==sha(rawpath)
            assert len(raw['conditions'])==10 and len(raw['evaluations'])==(192 if tag=='full192' else 3)
            if tag!='original_sigma_pilot3':assert raw['identity']['c_calibration_sha256']==sha(calpath)
            stats={name:{'unfinished':0,'fffd':0,'C_ratios':[],'C_gate_margin_changes':[],'gate_yes_no_mass':[],
                         'correct':0,'wrong':0,'abstain':0,'text_changes_from_no_C':0} for name in raw['conditions']}
            for row,grow in zip(raw['evaluations'],graded['evaluations']):
                assert row['pair_id']==grow['pair_id'] and row['outputs']==grow['outputs'] and row['finished']==grow['finished']
                for spec in raw['condition_specs']:
                    name=spec['name'];traces=row['trace'][name];tokens=row['generation_token_ids'][name];entry=stats[name]
                    assert len(tokens)==len(traces)>0
                    stops={128001,128008,128009} if model.startswith('llama') else {151643,151645}
                    eos=[i for i,t in enumerate(tokens) if t in stops]
                    assert eos==([len(tokens)-1] if row['finished'][name] else [])
                    if not row['finished'][name]:assert len(tokens)==raw['identity']['max_new_tokens']
                    for step,tr in enumerate(traces):
                        assert tr['step']==step and tr['accepted_prefix_ids']==tokens[:step] and tr['next_top1_id']==tokens[step]
                        assert tr['base_forward_calls']==1 and tr['source_forward_calls']==int(spec['e_enabled'])
                        assert tr['base_predictor_logical_position']==len(row['prompt_token_ids']['base'])-1+step
                        assert tr['source_predictor_logical_position']==len(row['prompt_token_ids']['source'])-1+step
                        assert tr['source_c_writes']==0 and len(tr['c_writes'])==int(spec['c'] is not None and step==0)
                        assert len(tr['writes'])==int(spec['e_enabled'])
                        expected=([('E',e)] if spec['e_enabled'] else [])
                        if spec['c'] and step==0:expected.append(('C',spec['c']['layer']))
                        assert [(v['operator'],v['layer']) for v in tr['base_operation_events']]==sorted(expected,key=lambda x:x[1])
                        assert all(v['phase']==('prefill' if step==0 else 'decode') and v['stream']=='base'
                                   for v in tr['base_operation_events'])
                        if spec['e_enabled']:
                            assert tr['cache_storage_disjoint'] and tr['source_predictors_captured']==1
                            ew=tr['writes'][0]
                            assert ew['ideal_coordinate_error']<=ew['float32_tolerance']
                            assert ew['ideal_orthogonal_error']<=ew['float32_tolerance']
                            assert ew['physical_base_position']==(len(row['prompt_token_ids']['base'])-1 if step==0 else 0)
                        for cw in tr['c_writes']:
                            c=spec['c'];coef=raw['identity']['c_coefficients'][c['side']]
                            assert cw['layer']==c['layer'] and cw['alpha']==c['alpha'] and cw['scale']==coef
                            assert cw['physical_position']==cw['logical_position']==len(row['prompt_token_ids']['base'])-1
                            assert math.isclose(cw['ideal_delta_norm'],coef,rel_tol=1e-5,abs_tol=1e-7)
                            assert math.isclose(cw['stored_delta_over_base'],cw['stored_delta_norm']/cw['base_norm'],rel_tol=1e-6,abs_tol=1e-8)
                            assert abs(cw['projection_after']-cw['projection_before']-c['alpha']*coef)<=cw['cast_error_norm']+1e-4
                    control='E_source' if spec['e_enabled'] else 'baseline'
                    entry['text_changes_from_no_C']+=int(row['outputs'][name]!=row['outputs'][control])
                    entry['unfinished']+=int(not row['finished'][name]);entry['fffd']+=int('\ufffd' in row['outputs'][name])
                    entry[grow['outcome'][name]]+=1
                    gate=row['gate_readout'][name]
                    assert gate['trace']['accepted_prefix_ids']==[] and gate['trace']['source_c_writes']==0
                    assert len(gate['trace']['c_writes'])==int(spec['c'] is not None)
                    assert gate['p_yes']>0 and gate['p_no']>0 and gate['p_yes']+gate['p_no']<=1+1e-6
                    assert math.isclose(math.log(gate['p_yes']/gate['p_no']),gate['yes_minus_no'],abs_tol=5e-6)
                    for cw in gate['trace']['c_writes']:
                        assert cw['physical_position']==len(row['gate_prompt_ids'])-1
                    entry['gate_yes_no_mass'].append(gate['p_yes']+gate['p_no'])
                    if spec['c']:
                        entry['C_ratios'].append(traces[0]['c_writes'][0]['stored_delta_over_base'])
                        entry['C_gate_margin_changes'].append(gate['yes_minus_no']-row['gate_readout'][control]['yes_minus_no'])
            for entry in stats.values():
                if entry['C_ratios']:
                    entry['C_ratio_median']=statistics.median(entry['C_ratios'])
                    entry['C_gate_delta_median']=statistics.median(entry['C_gate_margin_changes'])
            model_result['runs'][tag]={'raw_sha256':sha(rawpath),'graded_sha256':sha(gradepath),'all_execution_assertions_passed':True,'conditions':stats}
        out['models'][model]=model_result
        print(model,'rho',rho,'all calibration/trace/grade checks pass',flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
