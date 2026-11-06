#!/usr/bin/env python3
"""Read-only Llama/Qwen comparison from saved E25/E27 artifacts; no model/API calls."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import io
import json
import math
import tarfile

import numpy as np
import torch

MODELS = ('llama3.1-8b', 'qwen3-8b')
LABELS = ('correct', 'wrong', 'abstain')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector(value):
    a = torch.as_tensor(value).detach().double().cpu().reshape(-1).numpy()
    assert np.isfinite(a).all()
    return a


def dot(a, b):
    return math.fsum(float(x)*float(y) for x,y in zip(a,b))


def geometry(initial, final, centers, alpha):
    a, b = vector(initial), vector(final)
    signed_cos = float(dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    cos = min(1., abs(signed_cos))
    out = {'signed_cosine': signed_cos, 'axis_angle_degrees': math.degrees(math.acos(cos)),
           'unit_projector_frobenius_change': math.sqrt(max(0., 2 * (1 - cos*cos))),
           'initial_norm': float(np.linalg.norm(a)), 'final_norm': float(np.linalg.norm(b)),
           'top_initial_coordinates': [[int(i), float(a[i])] for i in np.argsort(abs(a))[-8:][::-1]],
           'top_final_coordinates': [[int(i), float(b[i])] for i in np.argsort(abs(b))[-8:][::-1]],
           'center': {}}
    for label in ('prompt', 'decode'):
        c = vector(centers[label]); k = int(np.argmax(abs(c)))
        out['center'][label] = {'norm': float(np.linalg.norm(c)), 'dominant_coordinate': k,
            'dominant_value': float(c[k]), 'dominant_energy_fraction': float(c[k]**2 / dot(c,c)),
            'initial_weight_on_dominant_coordinate': float(a[k]), 'final_weight_on_dominant_coordinate': float(b[k]),
            'projection_on_initial': float(dot(c, a)), 'projection_on_final': float(dot(c, b))}
    cp, cd = vector(centers['prompt']), vector(centers['decode'])
    # Exact operator difference if the SAME h is evaluated with the two centres.
    # This is not an observed free-generation hidden-state displacement.
    out['fixed_h_center_switch_delta_norm_initial'] = float(alpha * abs(dot(cp-cd, a)))
    out['fixed_h_center_switch_delta_norm_final'] = float(alpha * abs(dot(cp-cd, b)))
    out['fixed_h_center_switch_scope'] = 'algebra at identical h, not an empirical per-step residual-relative change'
    return out


def outcomes(rows, name):
    return {r['pair_id']: r['outcome'][name] for r in rows}


def stats(rows, name):
    counts = {label: sum(r['outcome'][name] == label for r in rows) for label in LABELS}
    base = outcomes(rows, 'baseline'); current = outcomes(rows, name)
    trans = {a+'->'+b: sum(base[i]==a and current[i]==b for i in base) for a in LABELS for b in LABELS}
    return {'n': len(rows), **counts, 'answered': counts['correct']+counts['wrong'],
            'net_correct_vs_baseline': counts['correct']-sum(v=='correct' for v in base.values()),
            'net_answered_vs_baseline': counts['correct']+counts['wrong']-sum(v!='abstain' for v in base.values()),
            'transitions': trans, 'bad_character': sum('\ufffd' in r['outputs'][name] for r in rows),
            'unfinished': sum(not r['finished'][name] for r in rows)}


def run(root, archive):
    source = root/'from_claude/results/02_问答DAS_v1/第16轮_DAS正结果搜索/v1u'
    report = {'archive_sha256':sha(archive),'models':{},'runs':{},'cross_model':{},'input_sha256':{}}
    ids_by_model, data_by_model = {}, {}
    with tarfile.open(archive) as t:
        def packed(name):
            return torch.load(io.BytesIO(t.extractfile(name).read()), weights_only=True, map_location='cpu')
        def js(name): return json.load(t.extractfile(name))
        for model in MODELS:
            folder=source/model
            selection, short, training, bank, proxy, questions=[read(folder/name) for name in
                ('selection.json','shortlist.json','training.json','bank.json','proxy_scores.json','questions.json')]
            for name in ('selection.json','shortlist.json','training.json','bank.json','proxy_scores.json','questions.json'):
                report['input_sha256'][str((folder/name).relative_to(root))]=sha(folder/name)
            chosen=selection['selected']['qa_das']; layer=chosen['layer']; key=str(layer)
            assert training['completed'] and bank['model']==model
            assert selection['question_ids']==proxy['question_ids']
            ids_by_model[model]={'train':bank['training_pair_ids'],'tune':selection['question_ids']}
            minloss=sorted(((int(l),row['losses'][-1]) for l,row in training['layers'].items()),key=lambda x:x[1])[:6]
            candidates=[]
            for spec in short['candidates']['qa_das']:
                row=proxy['conditions'][spec['name']]; gen=selection['reports'][spec['name']]
                train=training['layers'][str(spec['layer'])]
                candidates.append({'layer':spec['layer'],'alpha':spec['alpha'],
                    'proxy_correct':int(sum(row['correct'])),'proxy_net':int(sum(row['correct'])-sum(proxy['conditions']['baseline']['correct'])),
                    'old_generation_counts':gen['counts'],'old_generation_net':gen['net_correct'],
                    'source_selfcheck_loss_first':train['losses'][0],'source_selfcheck_loss_last':train['losses'][-1]})
            source_pt=folder/training['layers'][key]['file']; source_w=vector(torch.load(source_pt,weights_only=True,map_location='cpu')['W'])
            source_rows=read(folder/'tune_scored_items.json')['evaluations']
            original_name=chosen['name']
            same_alpha_references={}
            report['models'][model]={'num_layers':bank['num_layers'],'hidden_size':bank['hidden_size'],
                'selected_layer':layer,'selected_layer_1_based_fraction':(layer+1)/bank['num_layers'],
                'downstream_blocks':bank['num_layers']-layer-1,'selected_original_alpha':chosen['alpha'],
                'selection_rule':selection['rule'],'selection_scope':selection['report_scope'],
                'source_selfcheck_format':bank['config']['selfcheck_format'],
                'source_supervision_labels':[bank['config']['selfcheck_label_true'],bank['config']['selfcheck_label_false']],
                'source_selfcheck_lowest_loss_layers':minloss,'source_candidates':candidates,
                'new_content_training_layers_observed':[layer],
                'source_checkpoint_sha256':sha(source_pt)}
            for alpha in (1,3):
                for init in ('das','md','random'):
                    prefix=f'{model}/ce_alpha{alpha}_ep3/{init}/'
                    d=packed(prefix+'adapter.pt');rows=js(prefix+'tune_scored_items.json')['evaluations']
                    assert d['layers']==[layer] and d['training_pair_ids']==bank['training_pair_ids']
                    assert d['model']==model and d['source']['bank_fingerprint']==bank['fingerprint']
                    assert d['source']['checkpoints'][key]==sha(source_pt)
                    reference=[{k:r[k] for k in ('pair_id','question','gold','incorrect')} | {'outputs':{n:r['outputs'][n] for n in ('baseline','original_das')},'outcome':{n:r['outcome'][n] for n in ('baseline','original_das')},'finished':{n:r['finished'][n] for n in ('baseline','original_das')}} for r in rows]
                    if alpha in same_alpha_references: assert same_alpha_references[alpha]==reference
                    else: same_alpha_references[alpha]=reference
                    if alpha==3:
                        for old,new in zip(source_rows,rows):
                            for name in ('baseline','original_das'):
                                source_name='baseline' if name=='baseline' else original_name
                                assert old['outputs'][source_name]==new['outputs'][name]
                                assert old['outcome'][source_name]==new['outcome'][name]
                    assert d['train_config']['epochs']==3 and d['train_config']['ranking_weight']==0
                    assert [r['pair_id'] for r in rows]==selection['question_ids']
                    assert len(rows)==192 and all(not r['gold_is_abstention'] for r in rows)
                    if init=='das': assert np.array_equal(vector(d['initial_W'][key]),source_w)
                    name=d['method'];g=geometry(d['initial_W'][key],d['W'][key],d['centers'][key],alpha)
                    rr={'model':model,'alpha':alpha,'init':init,'layer':layer,'n_train':len(d['training_pair_ids']),
                        'epochs':d['train_config']['epochs'],'losses':d['losses'],'gradient_audit':d['gradient_audit'][key],
                        'geometry':g,'first_training_trace':d['position_trace'],
                        'baseline':stats(rows,'baseline'),'original_das':stats(rows,'original_das'),'adapted':stats(rows,name),
                        'identical_to_source_DAS_initial': bool(np.array_equal(vector(d['initial_W'][key]),source_w)),
                        'same_alpha_common_baseline_and_original_das_identical':True,
                        'alpha3_original_das_matches_source_frozen_outputs':True if alpha==3 else None}
                    report['runs'][f'{model}/alpha{alpha}/{init}']=rr
                    if alpha==3 and init=='das':data_by_model[model]=rows
        assert ids_by_model[MODELS[0]]==ids_by_model[MODELS[1]]
        report['cross_model']['same_training_and_tuning_question_ids']=True
        a,b=[data_by_model[m] for m in MODELS]
        basea,baseb=outcomes(a,'baseline'),outcomes(b,'baseline')
        ada,adb=outcomes(a,'das_generation_adapter'),outcomes(b,'das_generation_adapter')
        report['cross_model']['baseline_outcome_cross_table']={x+'|'+y:sum(basea[i]==x and baseb[i]==y for i in basea) for x in LABELS for y in LABELS}
        qa_gains=[i for i in baseb if baseb[i]!='correct' and adb[i]=='correct']
        report['cross_model']['qwen_gained_correct_20_llama_baseline_outcomes']=dict(Counter(basea[i] for i in qa_gains))
        groups={'both_answered':[i for i in basea if basea[i]!='abstain' and baseb[i]!='abstain'],
                'both_wrong':[i for i in basea if basea[i]=='wrong' and baseb[i]=='wrong'],
                'both_abstained':[i for i in basea if basea[i]=='abstain' and baseb[i]=='abstain'],
                'llama_answered_qwen_abstained':[i for i in basea if basea[i]!='abstain' and baseb[i]=='abstain']}
        report['cross_model']['common_subgroups']={}
        for group,ids in groups.items():
            report['cross_model']['common_subgroups'][group]={'n':len(ids),'ids':ids,
                MODELS[0]:{'baseline':dict(Counter(basea[i] for i in ids)),'adapted':dict(Counter(ada[i] for i in ids)),
                    'net_correct':sum(ada[i]=='correct' for i in ids)-sum(basea[i]=='correct' for i in ids)},
                MODELS[1]:{'baseline':dict(Counter(baseb[i] for i in ids)),'adapted':dict(Counter(adb[i] for i in ids)),
                    'net_correct':sum(adb[i]=='correct' for i in ids)-sum(baseb[i]=='correct' for i in ids)}}
    report['scope']='Saved tensors, online training losses and judged tuning outputs only. Centre-switch calculation is algebraic, not an observed inference trace. No GPU, test-based selection or new judge calls.'
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3]);ap.add_argument('--archive',type=Path);ap.add_argument('--out',type=Path,default=Path(__file__).with_name('comparison.json'));a=ap.parse_args()
    archive=a.archive or a.root/'from_gpt/main_experiment_20261106/artifacts/generation_results.tar.gz'
    result=run(a.root,archive);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    for model in MODELS:
        m=result['models'][model];print(model,'chosen',m['selected_layer'],'candidates',[(r['layer'],r['old_generation_net']) for r in m['source_candidates']])
        for alpha in (1,3):
            r=result['runs'][f'{model}/alpha{alpha}/das'];g=r['geometry'];print('alpha',alpha,'count',r['adapted']['correct'],'angle',g['axis_angle_degrees'],'centres',g['center']['prompt']['norm'],g['center']['decode']['norm'],'switch',g['fixed_h_center_switch_delta_norm_final'])
    print(json.dumps(result['cross_model'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
