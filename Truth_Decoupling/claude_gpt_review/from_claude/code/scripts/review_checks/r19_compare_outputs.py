#!/usr/bin/env python3
"""Strict recorded-artifact reproduction check; no generation or grading.

Exit 1 for differences, 2 for malformed/incomplete evidence. Historical files
without output token IDs require --allow-missing-token-evidence and are labeled
'recorded fields only', never fully verified token reproduction.
"""
import argparse,hashlib,json,sys
from pathlib import Path

TRACES=('qwen_trace.json','qwen_current_dose.json','llama_trace.json')
RUNS=('random_controls/qwen3-8b/pilot3','random_controls/qwen3-8b/full192',
      'random_controls/llama3.1-8b/full192','generation_dose/qwen3-8b/full192','llama_pure_das_tune192')


def require(value,message):
    if not value:raise ValueError(message)


def read(path):
    def unique(pairs):
        result={}
        for k,v in pairs:
            require(k not in result,'duplicate JSON key: '+k);result[k]=v
        return result
    def invalid(value):raise ValueError('nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(),object_pairs_hook=unique,parse_constant=invalid)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token_list(value,label):
    require(isinstance(value,list) and all(type(v) is int and v>=0 for v in value),label+' must contain nonnegative integer token IDs')


def validate(data,label):
    require(isinstance(data,dict) and data.get('completed') is True,label+': generation not complete')
    conditions=data.get('conditions');rows=data.get('evaluations')
    require(isinstance(conditions,list) and conditions and all(isinstance(c,str) and c for c in conditions),label+': invalid conditions')
    require(len(conditions)==len(set(conditions)),label+': duplicate conditions')
    require(isinstance(rows,list) and rows,label+': missing/empty evaluations')
    seen=set();output_missing=[];prompt_missing=[]
    for index,row in enumerate(rows):
        require(isinstance(row,dict),label+': invalid row')
        pid=row.get('pair_id');require(isinstance(pid,str) and pid and pid not in seen,label+': missing/duplicate pair_id')
        seen.add(pid)
        require(isinstance(row.get('question'),str) and row['question'].strip(),label+': missing question text')
        for key in ('outputs','finished'):
            require(isinstance(row.get(key),dict) and set(row[key])==set(conditions),label+': missing/extra arm in '+key)
        require(all(isinstance(x,str) for x in row['outputs'].values()),label+': invalid output text')
        require(all(type(x) is bool for x in row['finished'].values()),label+': invalid completion flags')
        available=False
        for key in ('generation_token_ids','raw_token_ids'):
            if key in row:
                require(isinstance(row[key],dict) and set(row[key])==set(conditions),label+': token arms differ')
                for name,tokens in row[key].items():token_list(tokens,label+'/'+pid+'/'+name)
                available=True
        if not available:output_missing.append(pid)
        if 'prompt_token_ids' in row:
            prompts=row['prompt_token_ids']
            if isinstance(prompts,dict):
                require(set(prompts)==set(conditions),label+': prompt token arms differ')
                for name,tokens in prompts.items():token_list(tokens,label+'/'+pid+'/'+name+' prompt')
            else:token_list(prompts,label+'/'+pid+' prompt')
        else:prompt_missing.append(pid)
    if 'selected_questions' in data:require(data['selected_questions']==len(rows),label+': declared selected count differs')
    return {'output_token_ids_missing':output_missing,'prompt_token_ids_missing':prompt_missing}


def context_binding(path,data):
    if 'generation_context_file' not in data and 'generation_context_sha256' not in data:return None
    rel=data.get('generation_context_file');expected=data.get('generation_context_sha256')
    require(isinstance(rel,str) and rel and not Path(rel).is_absolute() and '..' not in Path(rel).parts,'invalid context file reference')
    require(isinstance(expected,str) and len(expected)==64,'missing context file digest')
    context=path.parent/rel;require(context.is_file() and sha(context)==expected,'generation context missing/changed: '+str(context))
    value=read(context);validate(value,str(context));indexed={r['pair_id']:r for r in value['evaluations']}
    require(value['conditions']==data['conditions'],'generation context condition set differs')
    for row in data['evaluations']:
        require(row['pair_id'] in indexed,'generation row absent from context')
        other=indexed[row['pair_id']]
        for key in ('question','gold','incorrect','best_answer','outputs','finished','generation_token_ids','raw_token_ids','prompt_token_ids'):
            if key in row or key in other:require(row.get(key)==other.get(key),'context/row mismatch: '+row['pair_id']+'/'+key)
    return {'file':rel,'sha256':expected,'verified':True}


def compare_generation(left,right,expected_count=None):
    a,b=read(left),read(right);ca,cb=validate(a,str(left)),validate(b,str(right))
    if expected_count is not None:
        require(len(a['evaluations'])==len(b['evaluations'])==expected_count,'generation cohort is not the declared fixed run size')
    context_a,context_b=context_binding(left,a),context_binding(right,b)
    ia=[r['pair_id'] for r in a['evaluations']];ib=[r['pair_id'] for r in b['evaluations']]
    ea={r['pair_id']:r for r in a['evaluations']};eb={r['pair_id']:r for r in b['evaluations']}
    row_diffs={}
    for pid in sorted(set(ea)&set(eb)):
        keys=[k for k in sorted(set(ea[pid])|set(eb[pid])) if ea[pid].get(k)!=eb[pid].get(k) or (k in ea[pid])!=(k in eb[pid])]
        if keys:row_diffs[pid]=keys
    top=[k for k in sorted(set(a)|set(b)) if k!='evaluations' and (a.get(k)!=b.get(k) or (k in a)!=(k in b))]
    arms={}
    for name in a['conditions']:
        common=set(ea)&set(eb)
        arms[name]={'outputs_differ':sum(ea[p]['outputs'][name]!=eb[p]['outputs'].get(name) for p in common),
                    'finished_differ':sum(ea[p]['finished'][name]!=eb[p]['finished'].get(name) for p in common)}
    equal=not top and not row_diffs and ia==ib
    return {'n_mine':len(ia),'n_reference':len(ib),'same_items':set(ia)==set(ib),'same_order':ia==ib,
            'conditions_equal':a['conditions']==b['conditions'],'arms':arms,
            'other_top_level_keys_differing':top,'row_fields_differing':row_diffs,
            'recorded_fields_identical':equal,'sha256':{'mine':sha(left),'reference':sha(right)},
            'token_evidence':{'mine':ca,'reference':cb,'output_tokens_complete':not ca['output_token_ids_missing'] and not cb['output_token_ids_missing']},
            'context_binding':{'mine':context_a,'reference':context_b}}


def compare(mine,theirs,out,allow_missing_tokens=False):
    report={'byte_identical':{},'generation_runs':{},'errors':[]};different=False;incomplete=False
    for name in TRACES:
        a,b=mine/name,theirs/name
        try:
            for p in (a,b):require(read(p).get('completed') is True,'trace incomplete: '+str(p))
            same=a.read_bytes()==b.read_bytes();different|=not same
            report['byte_identical'][name]={'identical':same,'sha256_mine':sha(a),'sha256_reference':sha(b)}
        except (OSError,ValueError,TypeError,AttributeError) as exc:report['errors'].append({'file':name,'error':str(exc)});incomplete=True
    for run in RUNS:
        try:
            value=compare_generation(mine/run/'tune.json',theirs/run/'tune.json',3 if run.endswith('pilot3') else 192);report['generation_runs'][run]=value
            different|=not value['recorded_fields_identical'];incomplete|=not value['token_evidence']['output_tokens_complete']
        except (OSError,ValueError,KeyError,TypeError) as exc:report['errors'].append({'run':run,'error':str(exc)});incomplete=True
    recorded_equal=not different and not report['errors']
    report.update(all_recorded_fields_identical=recorded_equal,all_identical=recorded_equal and not incomplete,
                  token_verification_complete=not incomplete,
                  scope='All saved row fields, question/reference text, existing tokens, order and context hashes; unsaved tokens are not inferred.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if report['errors']:print('INVALID/INCOMPLETE EVIDENCE');return 2
    if different:print('RECORDED ARTIFACT DIFFERENCES');return 1
    if incomplete:
        print('RECORDED FIELDS IDENTICAL; OUTPUT TOKEN EVIDENCE INCOMPLETE')
        return 0 if allow_missing_tokens else 2
    print('ALL RECORDED FIELDS AND OUTPUT TOKENS IDENTICAL');return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mine',type=Path);p.add_argument('reference',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--allow-missing-token-evidence',action='store_true',help='historical recorded-fields-only check; never asserts missing tokens match')
    a=p.parse_args();return compare(a.mine,a.reference,a.out,a.allow_missing_token_evidence)
if __name__=='__main__':sys.exit(main())
