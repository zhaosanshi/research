"""Independent native-cast C recalculation; does not import experiment/core code."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
R20=Path('/tmp/td_r20_20261107/Truth_Decoupling/claude_gpt_review')
def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def profile(states,vector,coefficient):
    assert states.is_floating_point() and states.ndim==2 and torch.isfinite(states).all()
    h=states.float();den=h.double().norm(dim=-1).numpy();assert (den>0).all()
    values=[]
    for sign in (-1.,1.):
        # Independent scalar-vector implementation, matching explicit FP32 add/native cast.
        delta=torch.from_numpy(np.float32(sign)*np.float32(coefficient)*np.asarray(vector,dtype=np.float32))
        stored=(h+delta).to(states.dtype)
        values.append((stored.float().double()-h.double()).norm(dim=-1).numpy()/den)
    return {'negative':values[0].tolist(),'positive':values[1].tolist(),'median':float(np.median(np.concatenate(values)))}

def verify_profiles(states,bank,task,method,layer_map,fit):
    original={};details={}
    for side,layer in layer_map.items():
        key=f'{task}_{method}_C_L{layer}';w=np.asarray(bank[key],float);w=(w/np.linalg.norm(w)).astype(np.float32)
        coefficient=fit['layers'][side]['coefficient'];sigma=float(bank[key+'_scale'])
        assert 0<coefficient<=sigma
        original[side]=profile(states[side],w,sigma);matched=profile(states[side],w,coefficient)
        max_error=0.
        for label,p in [('original_profile',original[side]),('calibrated_profile',matched)]:
            stored=fit['layers'][side][label]
            for sign in ('negative','positive'):
                error=float(np.max(np.abs(np.asarray(p[sign])-stored[sign])));max_error=max(max_error,error);assert error<1e-7
            assert abs(p['median']-stored['median'])<1e-7
        details[side]={'layer':layer,'dtype':str(states[side].dtype),'n':len(states[side]),'coefficient':coefficient,
            'original_sigma':sigma,'original_median':original[side]['median'],'matched_median':matched['median'],
            'positive_median':float(np.median(matched['positive'])),'negative_median':float(np.median(matched['negative'])),
            'profile_max_cpu_gpu_float_difference':max_error}
    rho=min(x['median'] for x in original.values());assert abs(rho-fit['rho'])<1e-7
    for side,x in details.items():
        assert abs(x['matched_median']-rho)<=fit['tolerance']+2e-7
        x['actual_median_error']=abs(x['matched_median']-rho)
    return {'recomputed_rho':rho,'recorded_rho':fit['rho'],'tolerance':fit['tolerance'],'layers':details}

def tf_audit():
    result=[]
    for model in ('llama3.1-8b','qwen3-8b'):
        parent=ROOT/'from_gpt/audit_20261107_round21/tf_order/results_calibrated'/model
        cp=parent/'train_calibration/calibration.json';sp=parent/'train_calibration/states.pt';c=read(cp)
        bankpath=R20/'from_gpt/audit_20261107_round20/tf_ability/results'/model/'tf_ability_full/bank.npz'
        assert c['source_sha256']['bank.npz']==digest(bankpath) and c['states_sha256']==digest(sp)
        bank=np.load(bankpath);states=torch.load(sp,map_location='cpu',weights_only=True)
        assert all(v.shape[0]==192 for v in states.values())
        layer_map=dict(zip(('before','after'),c['c_layers']))
        methods={m:verify_profiles(states,bank,'tf',m,layer_map,c['methods'][m]) for m in ('das','md')}
        p=read(parent/'pilot3/plan.json');raw=read(parent/'pilot3/tune.json')
        assert p['dose_mode']=='train_relative' and raw['completed'] and len(p['conditions'])==19
        assert p['calibration_json_sha256']==digest(cp) and raw['plan_sha256']==digest(parent/'pilot3/plan.json')
        contexts=0;c_checks=0;gate_effects={}
        for condition in p['conditions']:
            rows=raw['rows'][condition['name']];assert len(rows)==6
            expected=sorted(x['layer'] for x in (condition['e'],condition['c']) if x)
            for row in rows:
                for context in ('answer','gate'):
                    writes=row[context+'_writes'];assert [x['layer'] for x in writes]==expected;contexts+=1
                    for x in writes:
                        assert abs(x['actual_delta_l2']/max(x['residual_l2_before'],1e-12)-x['actual_delta_over_residual'])<1e-7
                        if condition['c'] and x['layer']==condition['c']['layer']:
                            side='before' if x['layer']==c['c_layers'][0] else 'after'
                            coefficient=c['methods'][condition['method']]['layers'][side]['coefficient']
                            assert abs(x['requested_delta_l2']-coefficient)<max(1e-6,1e-5*coefficient);c_checks+=1
            if condition['c']:
                reference=condition['method']+'_E' if condition['e'] else 'baseline'
                base={r['id']:r for r in raw['rows'][reference]}
                delta=[r['score']-base[r['id']]['score'] for r in rows]
                gate_effects[condition['name']]={'reference':reference,'median_delta_yes_no':float(np.median(delta)),
                    'min_delta_yes_no':min(delta),'max_delta_yes_no':max(delta),
                    'min_yes_no_probability_mass':min(r['gate_p_yes']+r['gate_p_no'] for r in rows)}
        result.append({'model':model,'calibration_sha256':digest(cp),'states_sha256':digest(sp),'raw_pilot_sha256':digest(parent/'pilot3/tune.json'),
            'calibration':methods,'actual_order_contexts_checked':contexts,'actual_C_requested_scale_checks':c_checks,
            'gate_effects':gate_effects,'status':'pass_no_blocker_for_fixed_full_run'})
    (OUT/'tf_calibration_and_pilot_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    for x in result:print(x['model'],{m:r['recomputed_rho'] for m,r in x['calibration'].items()},x['actual_order_contexts_checked'],'contexts PASS')
    return result

def qa_audit():
    result=[]
    for model in ('llama3.1-8b','qwen3-8b'):
        directory=Path('/tmp/td_r21_source_inputs')/model
        cp=directory/'calibration/c_calibration.json';sp=directory/'calibration/unmodified_prompt_states.pt';c=read(cp)
        bp=R20/'from_gpt/audit_20261107_round20/qa_ability/results'/model/'qa_ability_full/bank.npz'
        assert c['completed'] and c['identity']['c_bank_sha256']==digest(bp) and c['states_sha256']==digest(sp)
        states=torch.load(sp,map_location='cpu',weights_only=True);bank=np.load(bp)
        assert all(v.shape[0]==92 for v in states.values())
        adapter=read(Path('/tmp/td_r20_e34_inputs')/model/'preference_adapter.json')
        assert [r['pair_id'] for r in c['identity']['questions']]==adapter['training_pair_ids']
        audit=verify_profiles(states,bank,'qa','das',c['identity']['c_layers'],c['calibration'])
        rp=directory/'calibrated_pilot3/check.json';raw=read(rp);identity=raw['identity']
        assert raw['completed'] and raw['pilot'] and len(raw['evaluations'])==3 and len(raw['conditions'])==10
        assert identity['c_calibration_sha256']==digest(cp)
        refpath=(Path('/tmp/td_r20_e34_inputs')/model/'tune.json' if model.startswith('llama') else
                 Path('/tmp/td_r20_completion_inputs/qwen3-8b/uniform256/tune.json'))
        assert identity['reference_generation_sha256']==digest(refpath)
        refs={r['pair_id']:r for r in read(refpath)['evaluations']}
        spec={r['name']:r for r in identity['condition_specs']};steps=0;cw=0;gate={}
        for row in raw['evaluations']:
            for name,oldname in [('baseline','baseline'),('E_source','pair_preference_source_swap')]:
                for key in ('outputs','finished','generation_token_ids'):
                    assert row[key][name]==refs[row['pair_id']][key][oldname]
            for name,condition in spec.items():
                traces=row['trace'][name];tokens=row['generation_token_ids'][name]
                assert len(tokens)==len(traces)
                for i,t in enumerate(traces):
                    assert t['step']==i and t['accepted_prefix_ids']==tokens[:i] and t['next_top1_id']==tokens[i]
                    assert t['source_c_writes']==0 and t['source_forward_calls']==int(condition['e_enabled']) and t['base_forward_calls']==1
                    expected=[]
                    if condition['e_enabled']:expected.append(('E',identity['e_layer']))
                    if condition['c'] and i==0:expected.append(('C',condition['c']['layer']))
                    assert [(v['operator'],v['layer']) for v in t['base_operation_events']]==sorted(expected,key=lambda x:x[1])
                    assert len(t['c_writes'])==int(condition['c'] is not None and i==0)
                    for write in t['c_writes']:
                        side=condition['c']['side'];coef=identity['c_coefficients'][side]
                        assert write['physical_position']==len(row['prompt_token_ids']['base'])-1
                        assert abs(write['ideal_delta_norm']-coef)<max(1e-6,coef*1e-5)
                        assert abs(write['stored_delta_norm']/write['base_norm']-write['stored_delta_over_base'])<1e-7
                        assert abs(write['stored_delta_norm']-write['ideal_delta_norm'])<=write['cast_error_norm']+1e-5
                        cw+=1
                    steps+=1
                gr=row['gate_readout'][name]
                assert gr['p_yes']>=0 and gr['p_no']>=0 and gr['p_yes']+gr['p_no']<=1+1e-6
                if condition['c']:
                    for w in gr['trace']['c_writes']:assert w['physical_position']==len(row['gate_prompt_ids'])-1
                    reference='E_source' if condition['e_enabled'] else 'baseline'
                    gate.setdefault(name,[]).append({'pair_id':row['pair_id'],'delta_yes_no':gr['yes_minus_no']-row['gate_readout'][reference]['yes_minus_no'],
                        'yes_no_probability_mass':gr['p_yes']+gr['p_no'],
                        'scope':'C-only ability gate' if not condition['e_enabled'] else 'cross-task source-E gate diagnostic'})
        result.append({'model':model,'calibration':audit,'calibration_sha256':digest(cp),'states_sha256':digest(sp),
            'pilot_sha256':digest(rp),'actual_decode_steps_checked':steps,'actual_prefill_C_writes_checked':cw,
            'baseline_E_text_and_tokens_exact_reference':True,'gate_effects':gate,'status':'pass_no_observed_blocker'})
    (OUT/'qa_calibration_and_pilot_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    for x in result:print(x['model'],x['calibration']['recomputed_rho'],x['actual_decode_steps_checked'],'decode steps PASS')
    return result

if __name__=='__main__':
    import sys
    qa_audit() if '--qa' in sys.argv else tf_audit()
