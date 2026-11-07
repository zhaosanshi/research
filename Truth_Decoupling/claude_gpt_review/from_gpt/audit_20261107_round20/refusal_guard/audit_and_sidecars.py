"""Read-only complete-refusal audit; corrected labels live only in sidecars.

Old test choices are never replaced. Re-derive original and policy-corrected
tuning choices separately, then report whether any missing test condition would
be required. Original generation, grades, caches and selection bytes stay intact.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile

HERE=Path(__file__).resolve().parent;REVIEW=HERE.parents[2]
sys.path[:0]=[str(REVIEW/'from_claude/code'),str(REVIEW/'from_claude/code/scripts'),str(HERE.parent/'source_interchange')]
from core.qa_scoring import FULL_PERSONAL_REFUSAL_POLICY,full_personal_refusal_decision
from core.joint_steering import choose_conditions
from core.qa_selection import select_real_candidates
from core.selective_eval import quota_select,evaluate_coverage,random_score
from native_confidence import native_score


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def natural(rows):
    c=Counter(r['outcome'] for r in rows)
    return {'n':len(rows),'correct':c['correct'],'wrong':c['wrong'],'abstain':c['abstain'],'answered':c['correct']+c['wrong']}


def correct_arms(arms,questions):
    corrected=deepcopy(arms);changes=[];bypasses=[];matched=0
    for name,rows in corrected.items():
        for r in rows:
            if r['outcome'] not in ('correct','wrong','abstain'):
                raise ValueError('incomplete semantic outcomes cannot enter a corrected sidecar')
            decision=full_personal_refusal_decision(questions[r['pair_id']],r['output'])
            matched+=decision['matched_complete_phrase']
            if decision['bypass_reason'] and r['outcome']!='abstain':
                bypasses.append({'pair_id':r['pair_id'],'condition':name,'output':r['output'],**decision})
            if decision['apply'] and r['outcome']!='abstain':
                changes.append({'pair_id':r['pair_id'],'condition':name,'question':questions[r['pair_id']],
                    'output':r['output'],'original_outcome':r['outcome'],'corrected_outcome':'abstain',
                    'included_in_main':not r.get('gold_is_abstention',False),'decision':decision})
                r['outcome']='abstain'
    return corrected,{'record_occurrences':sum(map(len,arms.values())),'complete_phrase_matches':matched,
                     'changed_record_occurrences':len(changes),'changed_main_record_occurrences':sum(c['included_in_main'] for c in changes),
                     'changed_question_ids':sorted({c['pair_id'] for c in changes}),'changes':changes,'bypassed_nonabstain_cases':bypasses}


def eval_arms(data):
    names=data['conditions'];arms={n:[] for n in names};questions={}
    for r in data['evaluations']:
        questions[r['pair_id']]=r['question']
        for n in names:
            arms[n].append({'id':r['pair_id'],'pair_id':r['pair_id'],'output':r['outputs'][n],
                            'outcome':r['outcome'][n],'gold_is_abstention':r.get('gold_is_abstention',False)})
    return arms,questions


def compact_quota(rows,q,score='score',seed=42):
    value=quota_select([{**r,'score':r[score]} for r in rows if not r.get('gold_is_abstention',False)],q,seed)
    return {k:value[k] for k in ('feasible','n','k','n_available','answered','correct','wrong','accuracy_answered','selected_ids')}


def e26_choices(rows,meta):
    rs={n:[r for r in rr if not r['gold_is_abstention']] for n,rr in rows.items()};cfg=meta['config']
    chosen=choose_conditions(rs,meta['conditions']['qa'],cfg['coverages'],cfg['seed'])
    simpler={n:[{**r,'score':r['simple_score']} for r in rr] for n,rr in rs.items()}
    simple=choose_conditions(simpler,meta['conditions']['qa'],cfg['coverages'],cfg['seed'])
    for q,point in chosen.items():
        for family,name in simple[q].items():
            if family.endswith(('_E','_C')) or cfg.get('include_joint_simple',False):point[family+'_simple']=name
    return chosen


def audit_e26(run):
    meta,data,selection=read(run/'bank.json'),read(run/'data.json'),read(run/'selection.json')
    result={'model':meta['model'],'source_files':{},'splits':{}};corrected={}
    for split in ('tune','test'):
        path=run/('qa_'+split+'_scored.json');original=read(path)['rows'];q={r['pair_id']:r['question'] for r in data['qa'][split]}
        corrected[split],audit=correct_arms(original,q);result['source_files'][path.name]=digest(path)
        audit['natural_before']={n:natural([r for r in rs if not r['gold_is_abstention']]) for n,rs in original.items()}
        audit['natural_corrected']={n:natural([r for r in rs if not r['gold_is_abstention']]) for n,rs in corrected[split].items()}
        result['splits'][split]=audit
    old=e26_choices(read(run/'qa_tune_scored.json')['rows'],meta)
    if old!=selection['tasks']['qa']['choices']:raise ValueError('original selection does not rederive: '+str(run))
    new=e26_choices(corrected['tune'],meta);available=set(corrected['test'])
    changes=[{'coverage':q,'family':f,'old':n,'corrected_tune_choice':new[q][f],
              'new_test_condition_available':new[q][f] in available if new[q][f] else None}
             for q,point in old.items() for f,n in point.items() if new[q][f]!=n]
    result.update(original_selection_rederived=True,selection_changes=changes,
                  selection_json_unchanged=True,selection_sha256=digest(run/'selection.json'),
                  corrected_tune_choices_for_audit_only=new,fixed_original_selection_test_tables={})
    for q,picked in old.items():
        specs={'baseline_commitment':('baseline','score'),'baseline_simple':('baseline','simple_score')}
        specs.update({family:(name,'simple_score' if family.endswith('_simple') else 'score') for family,name in picked.items() if name})
        point={family:compact_quota(corrected['test'][name],q,key,meta['config']['seed']) for family,(name,key) in specs.items()}
        random_rows=[{**r,'score':random_score(r['id'],meta['config']['seed'])} for r in corrected['test']['baseline']]
        point['baseline_random']=compact_quota(random_rows,q,seed=meta['config']['seed'])
        result['fixed_original_selection_test_tables'][q]=point
    result['old_test_labels_and_choices_all_unchanged']=not changes and result['splits']['test']['changed_record_occurrences']==0
    if result['old_test_labels_and_choices_all_unchanged']:
        original_report=read(run/'report.json')
        for q,point in result['fixed_original_selection_test_tables'].items():
            previous=original_report['tables']['qa'][q]['quota']['coverages'][0]['arms']
            for name,value in point.items():
                for key in ('feasible','answered','correct','selected_ids'):
                    if value[key]!=previous[name][key]:raise ValueError('unchanged test recount differs: '+name)
        result['unchanged_test_inference']={'all_original_QA_point_counts_and_selected_ids_match':True,
            'original_report_sha256':digest(run/'report.json'),
            'intervals_and_p_values':'metric inputs, choices, seed and resampling budget unchanged; original independently audited statistics remain identical'}
    return result


def audit_native(path,questions):
    data=read(path);original=data['tasks']['qa'];corrected,audit=correct_arms(original,questions)
    points={}
    for name,rows in corrected.items():
        emitted=[r for r in rows if r['score']>0];actual=[r for r in emitted if r['outcome']!='abstain']
        k=len(actual);n=len(rows);counts=natural(actual)
        controls={}
        from fractions import Fraction
        for control in ('baseline','das_E','md_E'):
            for key in ('score','simple_score'):
                controls[control+'_'+key]=compact_quota(corrected[control],Fraction(k,n),key)
        points[name]={'n':n,'answered':k,'correct':counts['correct'],'actual_coverage':k/n,
                      'same_actual_count_controls':controls}
    return {'model':data['plan']['model'],'source_sha256':digest(path),'audit':audit,
            'corrected_native_points':points,'TF_unchanged':True}


def audit_causal_gate_layers(model,e26_run,questions):
    directory=HERE.parent/'causal_C/results'/model/'full192'
    plan=read(directory/'plan.json');scores=read(directory/'gate_scores.json')
    source=read(e26_run/'qa_tune_scored.json')['rows'];specs=plan['tasks']['qa']['conditions']
    if not scores['completed'] or scores['plan_sha256']!=digest(directory/'plan.json'):
        raise ValueError('incomplete causal-gate score source')
    arms={}
    for c in specs:
        candidates={r['id']:r for r in source[c['answer_condition']]}
        arms[c['name']]=[{**candidates[r['id']],'score':r['score']} for r in scores['tasks']['qa']['rows'][c['name']]]
    corrected,audit=correct_arms(arms,questions);result={}
    from fractions import Fraction
    for c in specs:
        rows=corrected[c['name']];selected=[r for r in rows if r['score']>0 and r['outcome']!='abstain'];k=len(selected);n=len(rows)
        controls={'base_gate':corrected['qa_base_gate'],'base_simple':corrected['qa_base_gate']}
        if c['method'] in ('das','md'):
            controls.update(fixed_E_gate=corrected['qa_'+c['method']+'_E_gate'],fixed_E_simple=corrected['qa_'+c['method']+'_E_gate'])
        result[c['name']]={'condition':c,'native':{'n':n,'answered':k,'correct':sum(r['outcome']=='correct' for r in selected)},
            'quotas':{str(q):compact_quota(rows,q,seed=plan['seed']) for q in plan['coverages']},
            'same_actual_count_controls':{name:compact_quota(rr,Fraction(k,n),'simple_score' if name.endswith('simple') else 'score',plan['seed']) for name,rr in controls.items()}}
    return {'model':model,'source_files':{n:digest(directory/n) for n in ('plan.json','gate_scores.json','report.json')},
            'audit':audit,'corrected_QA_points':result,'TF_unchanged':True,'scope':'all predeclared layer points, no new selection or test claim'}


def audit_source(run,common_run,n_boot):
    path=run/'tune_scored_items.json';graded=read(path);raw=read(run/'tune.json')
    if graded['grading_source_generation_sha256']!=digest(run/'tune.json'):raise ValueError('source grade mismatch')
    original,q=eval_arms(graded);corrected,audit=correct_arms(original,q)
    result={'model':graded['model'],'source_files':{path.name:digest(path),'tune.json':digest(run/'tune.json')},'audit':audit,
            'natural_before':{n:natural(rs) for n,rs in original.items()},'natural_corrected':{n:natural(rs) for n,rs in corrected.items()}}
    by_id={r['pair_id']:r for r in raw['evaluations']};native={}
    for name,rows in corrected.items():
        native[name]=[{**r,'score':native_score(by_id[r['pair_id']],name)['score']} for r in rows if not r.get('gold_is_abstention')]
    result['native_policy_confidence_corrected']=evaluate_coverage(native,(.25,.5,.75),n_boot=n_boot,seed=42)
    if common_run is not None:
        scores=read(common_run/'tune_simple_scores.json');common_raw=read(common_run/'tune.json')
        if scores['identity']['source_generation_sha256']!=digest(common_run/'tune.json'):raise ValueError('common score generation changed')
        if common_run.resolve()!=run.resolve() and common_raw['identity'].get('control_generation_sha256')!=digest(run/'tune.json'):
            raise ValueError('five/six arm common-score source mismatch')
        commons={r['pair_id']:r['scores'] for r in scores['rows']};cr={r['pair_id']:r for r in common_raw['evaluations']}
        common={}
        for name,rows in corrected.items():
            for r in rows:
                if r['output']!=cr[r['pair_id']]['outputs'][name]:raise ValueError('common-score output differs')
            common[name]=[{**r,'score':commons[r['pair_id']][name]} for r in rows if not r.get('gold_is_abstention')]
        result['common_unmodified_confidence_corrected']=evaluate_coverage(common,(.25,.5,.75),n_boot=n_boot,seed=42)
        result['common_score_sha256']=digest(common_run/'tune_simple_scores.json')
    return result


def source_layer_selection_check(prepared_root):
    result=[]
    for model in ('llama3.1-8b','qwen3-8b'):
        run=prepared_root/model;data=read(run/'tune_scored_items.json');meta=read(run/'bank.json');stored=read(run/'selection.json')
        arms,q=eval_arms(data);corrected,audit=correct_arms(arms,q)
        def choose(rows):
            return select_real_candidates(data['condition_specs'],
                {n:{r['pair_id']:r['outcome'] for r in rs} for n,rs in rows.items()},
                [r['pair_id'] for r in data['evaluations']],tolerance=meta['config']['answer_rate_tolerance'])['selected']
        before,after=choose(arms),choose(corrected)
        if before!=stored['selected']:raise ValueError('original E25 choice does not rederive')
        result.append({'model':model,'tune_scored_sha256':digest(run/'tune_scored_items.json'),
                       'selection_sha256':digest(run/'selection.json'),'changed_records':audit['changed_record_occurrences'],
                       'old_selected':before,'corrected_selected':after,'selection_unchanged':before==after})
    return result


def r19_relevant_controls_check():
    result=[];base=REVIEW/'from_gpt/audit_20261106_round19'
    def contrasts(arms):
        out={}
        for name,rows in arms.items():
            if name=='baseline':continue
            up=sum(a['outcome']!='correct' and b['outcome']=='correct' for a,b in zip(arms['baseline'],rows))
            down=sum(a['outcome']=='correct' and b['outcome']!='correct' for a,b in zip(arms['baseline'],rows));n=up+down
            prob=min(1.,2*sum(math.comb(n,k) for k in range(min(up,down)+1))/2**n) if n else 1.
            out[name]={'new_correct':up,'lost_correct':down,'net_correct':up-down,'exact_p':prob,
                       'scope':'natural-generation all-question correctness, not matched answer coverage'}
        return out
    for relative in ('random/evidence/random_controls.tar.gz','unicode/evidence/generation_dose_full192.tar.gz',
                     'models/evidence/pure_das_layer_results.tar.gz'):
        path=base/relative
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                if not member.isfile() or not member.name.endswith('scored_items.json') or 'pilot' in member.name:continue
                data=json.loads(archive.extractfile(member).read());arms,q=eval_arms(data);corrected,audit=correct_arms(arms,q)
                result.append({'archive':relative,'archive_sha256':digest(path),'member':member.name,'model':data['model'],
                    'audit':audit,'natural_before':{n:natural(rs) for n,rs in arms.items()},
                    'natural_corrected':{n:natural(rs) for n,rs in corrected.items()},
                    'natural_correctness_contrasts_before':contrasts(arms),'natural_correctness_contrasts_corrected':contrasts(corrected)})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-root',type=Path,default=Path('/tmp/td_r20_native_extracted/native'))
    p.add_argument('--source-root',type=Path,default=Path('/tmp/td_r20_source_confidence_inputs'))
    p.add_argument('--new-layer-root',type=Path,default=Path('/tmp/td_r20_source_layer_inputs'))
    p.add_argument('--extra-eval',nargs=2,action='append',default=[],metavar=('TAG','GRADED_JSON'))
    p.add_argument('--n-boot',type=int,default=2000);args=p.parse_args()
    root=REVIEW/'from_claude/results/03_E26同作答率_E27生成适配'
    e26_runs=[root/'E26_main/llama3.1-8b',root/'E26_main/qwen3-8b',root/'E26_adapters_qwen_exploratory']
    inventory=[];questions={};decisions={}
    for run in e26_runs:
        result=audit_e26(run);tag='E26_'+run.name
        save(HERE/'sidecars'/tag/'report.json',result);decisions[tag]=result['selection_changes']
        for split,audit in result['splits'].items():
            inventory.append({'category':'E26','run':tag,'model':result['model'],'split':split,**{k:v for k,v in audit.items() if not k.startswith('natural_')}})
        data=read(run/'data.json');questions.update({r['pair_id']:r['question'] for rows in data['qa'].values() for r in rows})
    for path in sorted((root/'E27_release').rglob('*_scored_items.json')):
        data=read(path);arms,q=eval_arms(data);corrected,audit=correct_arms(arms,q)
        tag=str(path.relative_to(root)).replace('/','__')[:-5]
        value={'model':data['model'],'source_sha256':digest(path),'audit':audit,
               'natural_before':{n:natural([r for r in rs if not r['gold_is_abstention']]) for n,rs in arms.items()},
               'natural_corrected':{n:natural([r for r in rs if not r['gold_is_abstention']]) for n,rs in corrected.items()}}
        save(HERE/'sidecars'/tag/'report.json',value)
        inventory.append({'category':'E27','run':tag,'model':data['model'],'split':data['split'],**audit})
    for model in ('llama3.1-8b','qwen3-8b'):
        path=args.native_root/model/'full192/native_scored.json';value=audit_native(path,questions)
        save(HERE/'sidecars'/('native_'+model)/'report.json',value)
        inventory.append({'category':'native','run':model,'model':model,'split':'tune',**value['audit']})
        if (HERE.parent/'causal_C/results'/model/'full192/report.json').exists():
            value=audit_causal_gate_layers(model,root/'E26_main'/model,questions)
            save(HERE/'sidecars'/('causal_C_'+model)/'report.json',value)
            inventory.append({'category':'causal_C','run':model,'model':model,'split':'tune',**value['audit']})
        for kind in ('frozen','trained'):
            run=args.source_root/kind/model;value=audit_source(run,args.source_root/'trained'/model,args.n_boot)
            save(HERE/'sidecars'/('source_'+kind+'_'+model)/'report.json',value)
            inventory.append({'category':'source','run':kind+'/'+model,'model':model,'split':'tune',**value['audit']})
    for tag,filename in args.extra_eval:
        path=Path(filename);data=read(path);arms,q=eval_arms(data);corrected,audit=correct_arms(arms,q)
        save(HERE/'sidecars'/tag/'report.json',{'model':data['model'],'source_sha256':digest(path),'audit':audit,
             'natural_before':{n:natural(rs) for n,rs in arms.items()},'natural_corrected':{n:natural(rs) for n,rs in corrected.items()}})
        inventory.append({'category':'extra_source','run':tag,'model':data['model'],'split':data['split'],**audit})
    if args.new_layer_root.exists():
        if not (args.new_layer_root/'DOWNLOAD_COMPLETE').exists():
            raise ValueError('new-layer input transfer is not complete')
        for model,layers in [('llama3.1-8b',(15,23)),('qwen3-8b',(17,26))]:
            for layer in layers:
                parent=args.new_layer_root/model/('L%d'%layer)
                for kind in ('controls_full','train_full92_ep3'):
                    value=audit_source(parent/kind,parent/'train_full92_ep3',args.n_boot)
                    tag='source_new_%s_L%d_%s'%(model,layer,kind)
                    save(HERE/'sidecars'/tag/'report.json',value)
                    inventory.append({'category':'source_new_layers','run':tag,'model':model,'split':'tune',**value['audit']})
    training=[]
    prep=REVIEW/'from_claude/results/02_问答DAS_v1/第16轮_DAS正结果搜索/v1u'
    for model in ('llama3.1-8b','qwen3-8b'):
        path=prep/model/'questions.json';data=read(path)
        hits=[{k:r[k] for k in ('pair_id','question','positive_answer')} for r in data['e_train'] if full_personal_refusal_decision(r['question'],r['positive_answer'])['apply']]
        training.append({'model':model,'source_sha256':digest(path),'n_positive_training_answers':len(data['e_train']),'guard_hits':hits})
    save(HERE/'E25_layer_selection_check.json',source_layer_selection_check(prep))
    save(HERE/'R19_relevant_controls_check.json',r19_relevant_controls_check())
    summary={'outcome_policy':FULL_PERSONAL_REFUSAL_POLICY,'scope':'canonical E26 main/adapter, E27 release, full native, causal C layers, and old/new-layer source runs when declared completed',
             'policy_source_sha256':digest(REVIEW/'from_claude/code/core/qa_scoring.py'),
             'audit_script_sha256':digest(Path(__file__)),
             'no_generation_grade_cache_or_selection_file_modified':True,'reference_answers_and_cohorts_unchanged':True,
             'artifact_audits':inventory,'training_positive_answers':training,'E26_corrected_tune_selection_changes':decisions,
             'changed_occurrences_total':sum(x['changed_record_occurrences'] for x in inventory),
             'distinct_changed_model_question_text':sorted({(x['model'],c['pair_id'],c['output']) for x in inventory for c in x['changes']})}
    save(HERE/'audit_summary.json',summary)
    for x in inventory:
        if x['changed_record_occurrences']:print(x['category'],x['run'],x['split'],x['changed_record_occurrences'],x['changed_question_ids'])
    print('E26 changes',decisions,'training hits',[len(x['guard_hits']) for x in training],flush=True)


if __name__=='__main__':main()
