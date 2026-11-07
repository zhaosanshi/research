"""Qwen eight-arm uniform-256 diagnostic, extending only unfinished masked answers."""
import argparse
import copy
from pathlib import Path
import sys

PROTOCOL='source_question_mask_uniform256_diagnostic_v1'
BUDGET=256

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('pilot','generate','judge'))
    for key in ('code-root','source-run','ce-run','e34-run','masked-run','controls256-run','output-dir'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--pilot-reviewed',action='store_true');args=p.parse_args()
    sys.path[:0]=[str(args.code_root/'scripts'),str(args.code_root),str(args.code_root/'scripts/review_checks')]
    import qa_masked_source_control as entry
    from core.qa_generation_adapter import implementation_identity
    from core.qa_precision import fp32_output_projection
    from core.qa_source_question_mask import locate_question_mask,greedy_masked_source_generate,SOURCE_MODE
    from core.qa_control_cache import seed_control_cache
    from core.readout import rule_abstains
    qa=entry.qa;method=entry.masked.METHOD
    args.run_dir=args.masked_run;args.pilot=None
    st=entry.state(args);a=entry.verified(args,st);original=st[0];meta,prepared=original[1],original[3]
    old=qa.read(args.masked_run/'tune.json');controls=qa.read(args.controls256_run/'tune.json')
    oldgrade=qa.read(args.masked_run/'tune_scored_items.json');controlgrade=qa.read(args.controls256_run/'tune_scored_items.json')
    if (a['model']!='qwen3-8b' or not old.get('completed') or old['conditions']!=entry.PARENTS+[method]
            or not controls.get('completed') or controls['conditions']!=entry.PARENTS
            or controls['identity']['budget_override']['max_new_tokens']!=BUDGET
            or controls['identity']['parent_generation_sha256']!=qa.digest(args.e34_run/'tune.json')
            or old['identity']['adapter_sha256']!=qa.digest(args.masked_run/'preference_adapter.pt')
            or oldgrade['grading_source_generation_sha256']!=qa.digest(args.masked_run/'tune.json')
            or controlgrade['grading_source_generation_sha256']!=qa.digest(args.controls256_run/'tune.json')
            or not oldgrade.get('completed') or not controlgrade.get('completed')
            or not oldgrade.get('grading_completed') or not controlgrade.get('grading_completed')
            or oldgrade['grading_judge_config_sha256']!=controlgrade['grading_judge_config_sha256']
            or oldgrade['grading_judge_config_sha256']!=qa.fingerprint(st[4])):
        raise ValueError('uniform256 needs complete unchanged Qwen masked48 and seven-arm256 parents with the same grader')
    for parent,parent_grade in ((old,oldgrade),(controls,controlgrade)):
        qa.verify_question_rows(parent_grade['evaluations'],parent['evaluations'])
        if parent_grade['conditions']!=parent['conditions'] or parent_grade['identity']!=parent['identity']:
            raise ValueError('completion parent grading condition/source identity changed')
        for row,g in zip(parent['evaluations'],parent_grade['evaluations']):
            for key in ('outputs','finished','generation_token_ids','trace','prompt_token_ids'):
                if row[key]!=g[key]:raise ValueError('completion parent grade changed original '+key)
            for name in parent['conditions']:
                if g['judge'][name].get('label') not in ('correct','wrong','abstain') or g['outcome'][name]!=g['judge'][name]['label']:
                    raise ValueError('completion parent outcome is unresolved/inconsistent')
    qa.verify_question_rows(old['evaluations'],prepared['splits']['tune']);qa.verify_question_rows(controls['evaluations'],old['evaluations'])
    unfinished=[r['pair_id'] for r in old['evaluations'] if not r['finished'][method]]
    finished=[r['pair_id'] for r in old['evaluations'] if r['finished'][method]][:3]
    if len(unfinished)!=7 or len(finished)!=3:raise ValueError('declared Qwen seven unfinished/three finished controls changed')
    plan={'protocol':PROTOCOL,'model':a['model'],'source_mode':SOURCE_MODE,'source':a['source'],'layer':a['layer'],
        'config':a['config'],'source_system':a['source_system'],'condition':method,'conditions':entry.PARENTS+[method],
        'alpha':1.,'batch_size':1,'same_prefix_tokens':True,'gold_used_at_inference':False,
        'question_ids':[r['pair_id'] for r in old['evaluations']],
        'adapter_sha256':qa.digest(args.masked_run/'preference_adapter.pt'),
        'source_mask_manifest_sha256':a['source_mask_manifest_sha256'],
        'masked_parent_generation_sha256':qa.digest(args.masked_run/'tune.json'),
        'masked_parent_grade_sha256':qa.digest(args.masked_run/'tune_scored_items.json'),
        'control_generation_sha256':qa.digest(args.controls256_run/'tune.json'),
        'control_grade_sha256':qa.digest(args.controls256_run/'tune_scored_items.json'),
        'informed_parent':old['identity']['informed_parent'],'unfinished_masked_ids':unfinished,
        'pilot_unfinished_ids':unfinished[:3],'pilot_finished_ids':finished,
        'budget_override':{'parent_max_new_tokens':48,'max_new_tokens':BUDGET,'only_generation_loop_limit_changes':True},
        'implementation':implementation_identity(),'diagnostic_script_sha256':qa.digest(__file__),
        'no_forced_eos':True,'no_text_trimming':True,'all_questions_retained':True}
    args.output_dir.mkdir(parents=True,exist_ok=True);pp=args.output_dir/'plan.json'
    if pp.exists() and qa.read(pp)!=plan:raise ValueError('completion plan changed')
    if not pp.exists():qa.save(pp,plan)
    by_id={r['pair_id']:r for r in old['evaluations']}
    def one(bundle,row):
        base,source=entry.source_eval.prompt_tokens(bundle,a['config'],row['question'])
        mask=locate_question_mask(bundle,a['config'],a['source_system'],row['question'])
        if row['prompt_token_ids']!={'base':base,'source':source} or row['source_question_mask']!=mask:
            raise ValueError('question mask or original prompt IDs changed')
        g=greedy_masked_source_generate(bundle,base,source,source_prompt_attention_mask=mask['source_prompt_attention_mask'],
            layer=a['layer'],direction=a['W'],max_new_tokens=BUDGET)
        if row['finished'][method]:
            if not g['finished'] or g['token_ids']!=row['generation_token_ids'][method] or g['output']!=row['outputs'][method]:
                raise RuntimeError('already-finished masked response changed')
        elif len(row['generation_token_ids'][method])!=48 or g['token_ids'][:48]!=row['generation_token_ids'][method]:
            raise RuntimeError('masked original48 prefix changed')
        return g
    if args.phase=='pilot':
        path=args.output_dir/'pilot.json'
        if path.exists():raise ValueError('preserve existing pilot; use new output directory')
        bundle=qa.open_bundle(meta,prepared)
        versions=[(x,x._version,x.requires_grad) for x in bundle.model.parameters()]
        result={'identity':plan,'completed':False,'finished_controls':[],'extension_probes':[],
            'source_mask_hard_checks':entry.masked.verify_mask_control(bundle,meta['config'],prepared['e_train'][:3],a['source_system'],a['layer'],a['W'])}
        with fp32_output_projection(bundle):
            for pid in finished:
                g=one(bundle,by_id[pid]);result['finished_controls'].append({'pair_id':pid,'tokens_and_text_exact':True,'finished':g['finished']})
            for pid in unfinished[:3]:
                g=one(bundle,by_id[pid]);result['extension_probes'].append({'pair_id':pid,'first48_exact':True,'new':g})
                print('MASKED_256_PILOT',pid,g['finished'],len(g['token_ids']),repr(g['output']),flush=True)
                qa.save(path,result)
        if any(x._version!=v or x.requires_grad!=g for x,v,g in versions):raise RuntimeError('pilot changed model parameters')
        if implementation_identity()!=plan['implementation'] or qa.digest(__file__)!=plan['diagnostic_script_sha256']:
            raise RuntimeError('pilot implementation changed')
        result['completed']=True;qa.save(path,result);return
    if args.phase=='generate':
        path=args.output_dir/'tune.json'
        if path.exists():raise ValueError('preserve existing completion raw; use new output directory')
        proof=qa.read(args.output_dir/'pilot.json')
        if (not proof.get('completed') or proof['identity']!=plan or not args.pilot_reviewed
                or [r['pair_id'] for r in proof['finished_controls']]!=finished
                or [r['pair_id'] for r in proof['extension_probes']]!=unfinished[:3]
                or not all(r['first48_exact'] for r in proof['extension_probes'])
                or not all(r['tokens_and_text_exact'] and r['finished'] for r in proof['finished_controls'])):
            raise ValueError('first inspect the three masked continuations and exact finished controls')
        bundle=qa.open_bundle(meta,prepared);reuse={r['pair_id']:r['new'] for r in proof['extension_probes']}
        versions=[(x,x._version,x.requires_grad) for x in bundle.model.parameters()]
        result=copy.deepcopy(controls);rows=[]
        result.update(protocol=PROTOCOL,identity=plan,protocol_fingerprint=qa.fingerprint(plan),bank_protocol_fingerprint=qa.fingerprint(plan),
            source_mode=SOURCE_MODE,conditions=old['conditions'],condition_specs=old['condition_specs'],evaluations=rows,completed=False,
            control_identity=controls['identity'],masked_parent_identity=old['identity'],training_sha256=plan['adapter_sha256'],
            completion_audit=[],E=old['E'],C=old['C'],pilot=False,wiring_pilot=False)
        with fp32_output_projection(bundle):
            for control in controls['evaluations']:
                row=copy.deepcopy(control);masked48=by_id[row['pair_id']]
                if row['prompt_token_ids']!=masked48['prompt_token_ids']:raise ValueError('256 seven-arm prompts differ from masked parent')
                if masked48['finished'][method]:
                    for field in ('outputs','finished','generation_token_ids','trace','rule_abstain'):row[field][method]=copy.deepcopy(masked48[field][method])
                    action='reused_masked_naturally_finished_48'
                else:
                    g=reuse.get(row['pair_id']) or one(bundle,masked48)
                    row['outputs'][method]=g['output'];row['finished'][method]=g['finished'];row['generation_token_ids'][method]=g['token_ids'];row['trace'][method]=g['trace']
                    row['rule_abstain'][method]=bool(rule_abstains(g['output']));action='extended_masked_only_loop_budget_to256'
                    print('MASKED_256',row['pair_id'],g['finished'],len(g['token_ids']),repr(g['output']),flush=True)
                row['source_question_mask']=copy.deepcopy(masked48['source_question_mask']);rows.append(row)
                result['completion_audit'].append({'pair_id':row['pair_id'],'action':action,'finished':row['finished'][method]})
                if action=='extended_masked_only_loop_budget_to256':qa.save(path,result)
        entry.inherited(controls['evaluations'],rows)
        if any(x._version!=v or x.requires_grad!=g for x,v,g in versions):raise RuntimeError('completion changed model parameters')
        if implementation_identity()!=plan['implementation'] or qa.digest(__file__)!=plan['diagnostic_script_sha256']:
            raise RuntimeError('decoder dependencies changed')
        result['completed']=True;qa.save(path,result);return
    raw=qa.read(args.output_dir/'tune.json')
    if not raw.get('completed') or raw['identity']!=plan:raise ValueError('complete matching raw before grading')
    entry.inherited(controls['evaluations'],raw['evaluations'])
    for row in raw['evaluations']:
        before=by_id[row['pair_id']]
        if row['source_question_mask']!=before['source_question_mask']:raise ValueError('masked-source attention identity changed')
        if before['finished'][method]:
            if any(row[k][method]!=before[k][method] for k in ('outputs','finished','generation_token_ids','trace','rule_abstain')):
                raise ValueError('completed masked answer was changed')
        elif row['generation_token_ids'][method][:48]!=before['generation_token_ids'][method] or len(row['generation_token_ids'][method])>BUDGET:
            raise ValueError('masked completion prefix or budget changed')
    cfg=st[4];seed_control_cache(source_dir=args.controls256_run/'judge_cache',run_dir=args.output_dir,judge_config=cfg,
        scored_rows=controlgrade['evaluations'],conditions=entry.PARENTS,scored_sha256=plan['control_grade_sha256'],stem='controls256')
    seeded=seed_control_cache(source_dir=args.masked_run/'judge_cache',run_dir=args.output_dir,judge_config=cfg,
        scored_rows=oldgrade['evaluations'],conditions=[method],scored_sha256=plan['masked_parent_grade_sha256'],stem='masked48')
    for key,value in seeded.items():setattr(args,key,value)
    oldg={r['pair_id']:r for r in oldgrade['evaluations']};cg={r['pair_id']:r for r in controlgrade['evaluations']};frozen={}
    for row in raw['evaluations']:
        pid=row['pair_id'];frozen[pid]={n:cg[pid]['judge'][n] for n in entry.PARENTS}
        if row['outputs'][method]==by_id[pid]['outputs'][method]:frozen[pid][method]=oldg[pid]['judge'][method]
    qa.save(args.output_dir/'grade_inheritance.json',{'source_kind':'two_complete_same-rubric_parents',
        'parent_scored_files_sha256':{'controls256':plan['control_grade_sha256'],'masked48':plan['masked_parent_grade_sha256']},
        'conditions_from_controls256':entry.PARENTS,'masked_same_whole_texts_frozen':sum(method in x for x in frozen.values()),
        'changed_masked_whole_texts_to_rejudge':sum(method not in x for x in frozen.values()),
        'frozen_conditions_by_pair':{pid:list(grades) for pid,grades in frozen.items()}})
    args.frozen_control_source_sha256=None  # No single file contains both frozen parent sources; see explicit sidecar.
    args.frozen_control_grades=frozen;args.run_dir=args.output_dir;args.model=a['model'];args.judge_exp=entry.e34.LEGACY_JUDGE
    qa.judge(args,{**meta,'protocol_fingerprint':raw['bank_protocol_fingerprint']},'tune.json','tune_scored')

if __name__=='__main__':main()
