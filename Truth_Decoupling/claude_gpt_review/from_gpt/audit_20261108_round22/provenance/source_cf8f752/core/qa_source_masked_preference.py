"""Same E34 paired loss/initialization/updates with source-question keys masked.

The original training implementation is invoked unchanged inside a strict
source-only forward scope. No loss, EOS target, base prompt or optimizer change
is introduced. The source retains shared answer prefix and question length.
"""
import json
from pathlib import Path

import torch

from .model_io import forward_logits
from .qa_generation_adapter import digest, fingerprint, frozen_model
from .qa_precision import fp32_output_projection
from .qa_source_interchange import SourceInterchangeDecoder, capture_source, unit_direction
from .qa_source_question_mask import (SOURCE_MODE, locate_question_mask, mask_one_source_forward,
    QuestionMaskedSourceDecoder, TrainingSourceMask)
from . import qa_source_pair_preference as original


PROTOCOL='supervised_source_pair_preference_question_mask_v1'
METHOD='pair_preference_question_masked_source'


def masked_replacements(ids,mask,vocab,special):
    result=list(ids)
    for i,value in enumerate(mask):
        if value:continue
        token=(result[i]+17)%vocab
        while token in special or token==result[i]:token=(token+1)%vocab
        result[i]=token
    if result==list(ids):raise ValueError('diagnostic did not replace a question token')
    return result


def verify_mask_control(bundle,cfg,records,source_system,layer,direction):
    """Three real same-shape causal checks, including cache steps and old None path."""
    pairs=original.prepare_pairs(bundle,cfg,records[:3],source_system)
    masks={r['pair_id']:locate_question_mask(bundle,cfg,source_system,r['question']) for r in records[:3]}
    results=[];special=set(getattr(bundle.tokenizer,'all_special_ids',[]))
    with frozen_model(bundle.model),fp32_output_projection(bundle):
        for record,pair in zip(records[:3],pairs):
            positive=pair['positive'];base=positive['base']['prefix_ids'];source=positive['source']['prefix_ids']
            entry=masks[record['pair_id']];mask=entry['source_prompt_attention_mask']
            if entry['source_prompt_ids']!=source:raise ValueError('mask token IDs differ from actual prepared source')
            past=positive['base']['answer_ids'][:2]
            def source_read(ids):
                x=torch.tensor([ids+past],device=bundle.device,dtype=torch.long)
                with capture_source(bundle,layer,len(source)-1) as captured:
                    with mask_one_source_forward(bundle.model,mask):
                        logits=forward_logits(bundle,{'input_ids':x,'attention_mask':torch.ones_like(x)})
                return logits[0,len(source)-1:].detach(),captured['hidden']
            before,hidden=source_read(source)
            changed=masked_replacements(source,mask,before.shape[-1],special)
            after,hidden_after=source_read(changed)
            if not torch.equal(before,after) or not torch.equal(hidden,hidden_after):
                raise RuntimeError('masked current-question content still reaches source predictors/logits')
            base_span=locate_question_mask(bundle,cfg,cfg['qa_system'],record['question'])
            if base_span['source_prompt_ids']!=base:raise ValueError('base question span does not match actual prompt')
            altered_base=masked_replacements(base,base_span['source_prompt_attention_mask'],before.shape[-1],special)
            def base_read(ids):
                x=torch.tensor([ids],device=bundle.device,dtype=torch.long)
                return forward_logits(bundle,{'input_ids':x,'attention_mask':torch.ones_like(x)})[0,-1]
            difference=float((base_read(base)-base_read(altered_base)).abs().max())
            if not difference>0:raise RuntimeError('base question replacement did not demonstrate retained question dependence')
            old=SourceInterchangeDecoder(bundle,base,source,mode='source_interchange',layer=layer,direction=direction)
            none=QuestionMaskedSourceDecoder(bundle,base,source,mode='source_interchange',layer=layer,direction=direction,
                                             source_prompt_attention_mask=None)
            a=QuestionMaskedSourceDecoder(bundle,base,source,mode='source_interchange',layer=layer,direction=direction,
                                          source_prompt_attention_mask=mask)
            b=QuestionMaskedSourceDecoder(bundle,base,changed,mode='source_interchange',layer=layer,direction=direction,
                                          source_prompt_attention_mask=mask)
            for step in range(3):
                lo,_=old.next_logits();ln,_=none.next_logits();la,_=a.next_logits();lb,_=b.next_logits()
                if not torch.equal(lo,ln):raise RuntimeError('mask=None changed the original decoding operator')
                if not torch.equal(la,lb):raise RuntimeError('masked source question leaked through cached decoding')
                token=int(lo.argmax())
                for decoder in (old,none,a,b):decoder.accept(token)
            results.append({'pair_id':record['pair_id'],'source_question_mask':entry,
                'same_length_masked_token_replacement_source_predictors_bitwise_equal':True,
                'same_length_masked_token_replacement_source_logits_bitwise_equal':True,
                'base_unmasked_question_replacement_max_logit_difference':difference,
                'mask_none_original_operator_bitwise_equal_three_cached_steps':True,
                'masked_replacement_bitwise_equal_three_cached_steps':True,
                'source_prompt_positions_not_compressed':True,'diagnostic_mutation_used_for_training':False})
    return results


def train(bundle,records,cfg,initial_w,*,layer,source_system,pilot=False,identity=None,log_fn=print):
    pairs=original.prepare_pairs(bundle,cfg,records,source_system)
    masks={r['pair_id']:locate_question_mask(bundle,cfg,source_system,r['question']) for r in records}
    checks=verify_mask_control(bundle,cfg,records,source_system,layer,initial_w)
    with TrainingSourceMask(bundle,pairs,masks) as scope:
        # Unmodified loss, original-DAS starting vector, Adam/clip/order/epochs.
        result=original.train(bundle,records,cfg,initial_w,layer=layer,source_system=source_system,
                              pilot=pilot,identity=identity,log_fn=log_fn)
    if not scope.audit['source_calls'] or scope.audit['source_grad_enabled']:
        raise RuntimeError('masked training failed to exercise detached source forwards')
    if set(scope.audit['source_calls_by_pair'])!={r['pair_id'] for r in records}:
        raise RuntimeError('not every training question used the masked source')
    result.update(protocol=PROTOCOL,method=METHOD,source_mode=SOURCE_MODE,
        original_loss_protocol=original.PROTOCOL,source_mask_manifest=masks,
        source_mask_manifest_sha256=fingerprint(masks),source_mask_verification=checks,
        source_mask_forward_audit=scope.audit,
        inherited_same_prompt_check_scope='Original E34 diagnostic copies base into both streams; both are dispatched unmasked. This is not a claim that equal tokens with a source-only mask form an identity.',
        primary_source_no_content_evidence='Same-length masked question-token replacements leave source predictor states/logits and three cached deployment steps bitwise identical.',
        control_definition='No additional current-question content in source attention keys; source instruction, prompt length, roles and public answer prefix retained',
        completely_information_free_claimed=False)
    return result


def save(adapter,directory):
    validate(adapter);original.save(adapter,directory)


def validate(adapter):
    if (adapter.get('protocol')!=PROTOCOL or adapter.get('method')!=METHOD or adapter.get('source_mode')!=SOURCE_MODE
            or adapter.get('original_loss_protocol')!=original.PROTOCOL
            or adapter.get('completely_information_free_claimed') is not False):
        raise ValueError('not the declared question-masked paired-source control')
    unit_direction(adapter['W'],adapter['hidden_size'])
    expected={'epochs':1 if adapter.get('pilot') else 3,'lr':.001,'clip':1.,'seed':42,'optimizer':'Adam',
              'alpha':1.,'include_eos':False,'fixed_center':False,
              'loss':'softplus(-((gap_w-gap_unsteered)/mean_pair_token_lengths))'}
    masks=adapter['source_mask_manifest']
    if (adapter['train_config']!=expected or adapter['source_mask_manifest_sha256']!=fingerprint(masks)
            or set(masks)!=set(adapter['training_pair_ids']) or len(masks)!=(3 if adapter['pilot'] else 92)):
        raise ValueError('masked control changed fixed training or source-mask identity')
    return adapter


def load(directory):
    path=Path(directory)/'preference_adapter.pt';meta=json.loads(path.with_suffix('.json').read_text())
    if meta.get('checkpoint_file')!=path.name or meta.get('checkpoint_sha256')!=digest(path):
        raise ValueError('question-masked source checkpoint identity differs')
    adapter=torch.load(path,map_location='cpu',weights_only=True)
    expected={k:v for k,v in adapter.items() if k not in ('W','initial_W')}
    expected.update(checkpoint_file=path.name,checkpoint_sha256=digest(path))
    if expected!=meta:raise ValueError('question-masked manifest differs from checkpoint')
    return validate(adapter)
