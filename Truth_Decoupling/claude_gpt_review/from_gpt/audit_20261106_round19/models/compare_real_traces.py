#!/usr/bin/env python3
"""Compare observed E27 edits from real, reference-reproduced Llama/Qwen traces."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics


def summarize(path):
    d=json.loads(path.read_text())
    assert d['completed'] and d['reference_reproduced']
    out={'model':d['model'],'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
         'code_sha256':d['code_sha256'],'adapter_sha256':d['adapter_sha256'],
         'generation_config':d['generation_config'],'cases':{}}
    for pid,c in d['cases'].items():
        assert c['reference_output_identical'] and c['bytes']['byte_mapping_verified']
        n=len(c['effective_ids'])
        # Later batch steps may only process this already-ended row's padding.
        # They are NOT part of this answer's active generation trajectory.
        edits=[e for e in c['edits'] if e['step']<n]
        assert len(edits)==n and [e['step'] for e in edits]==list(range(n))
        decode=[e for e in edits if e['step']>0]
        row={'output':c['bytes']['default_decoded'],'question':c['question'],
             'batch_start':c['batch_start'],'active_steps':n,'stop_step':c['stopped_at_step'],
             'discarded_padding_edit_records':len(c['edits'])-len(edits),
             'invalid_utf8_events':c['bytes']['invalid_utf8_events'],
             'literal_replacement_events':c['bytes']['literal_replacement_events'],
             'prompt_relative_edit':edits[0]['kept_relative_norm'],
             'prompt_h_norm':edits[0]['h_norm'],
             'decode_relative_edit_median':statistics.median(e['kept_relative_norm'] for e in decode),
             'active_relative_edit_max':max(e['kept_relative_norm'] for e in edits),
             'active_edits':edits}
        prefix=c['bytes']['decoded_prefixes']
        completed=next((i for i,s in enumerate(prefix) if s=="I don't know."),None)
        if completed is not None and completed+1 < n:
            step=completed+1;score=c['steps'][step]['token'];top=next(t for t in score['tokens'] if t['id']==score['argmax_id'])
            row['after_common_complete_refusal']={'prefix':prefix[completed],'step':step,
                'next_token_id':score['argmax_id'],'next_token_string':top['token_string'],
                'next_token_probability':math.exp(top['log_probability']),
                'relative_edit':edits[step]['kept_relative_norm'],
                'h_norm':edits[step]['h_norm'],'delta_norm':edits[step]['kept_delta_norm']}
        out['cases'][pid]=row
    return out


def main():
    here=Path(__file__).resolve().parent
    p=argparse.ArgumentParser();p.add_argument('--llama',type=Path,default=here/'evidence/llama_trace.json');p.add_argument('--qwen',type=Path,default=here/'evidence/qwen_trace.json');p.add_argument('--out',type=Path,default=here/'real_trace_comparison.json');a=p.parse_args()
    ll,qw=summarize(a.llama),summarize(a.qwen)
    assert ll['code_sha256']==qw['code_sha256']
    common=sorted(set(ll['cases'])&set(qw['cases']))
    assert common==['tqa:138','tqa:150','tqa:412']
    for pid in common:assert ll['cases'][pid]['question']==qw['cases'][pid]['question'] and ll['cases'][pid]['batch_start']==qw['cases'][pid]['batch_start']
    result={'same_executed_code_hashes':True,'common_question_ids':common,'models':{ll['model']:ll,qw['model']:qw},
        'limits':['Both traces use each model\'s already-trained E27 direction/layer; this is not a controlled architecture-only ablation.',
                  'Llama has no corresponding final invalid-byte event in these cases; no such event is invented.',
                  'Statistics exclude post-EOS padding forwards. These are observed before/after h ratios, not ratios to saved mean centres.',
                  'The script explicitly overrides generation do_sample=False and num_beams=1. The saved model-default config may contain do_sample=True.']}
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    for model,d in result['models'].items():
        for pid,c in d['cases'].items():
            print(model,pid,'prompt',c['prompt_relative_edit'],'decode median',c['decode_relative_edit_median'],'max',c['active_relative_edit_max'],'invalid',len(c['invalid_utf8_events']))


if __name__=='__main__':main()
