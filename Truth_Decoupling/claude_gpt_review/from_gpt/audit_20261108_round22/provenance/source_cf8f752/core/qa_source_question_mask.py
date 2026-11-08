"""Remove source-only question keys while preserving positions and shared prefix.

This control removes the source's additional current-question token content.
It retains the source instruction, question length, chat roles, and the public
already-generated answer prefix. It is not an information-free source.
"""
import contextlib
import hashlib

import torch

from .qa_source_interchange import SourceInterchangeDecoder


SOURCE_MODE = 'question_attention_keys_masked_positions_preserved_v1'


def locate_question_mask(bundle, cfg, source_system, question):
    if not isinstance(question, str) or not question:
        raise ValueError('source question must be nonempty text')
    tokenizer = bundle.tokenizer
    if not getattr(tokenizer, 'is_fast', False):
        raise ValueError('exact question masking requires a fast tokenizer with offset mapping')
    marker = '__SOURCE_QUESTION_SPAN_9f4e76a1__'
    if marker in question or marker in source_system or marker in cfg['qa_user']:
        raise ValueError('question-span sentinel collided with source text')
    marked = bundle.build_chat(source_system, cfg['qa_user'].format(question=marker))
    if marked.count(marker) != 1:
        raise ValueError('question placeholder must occur exactly once in rendered source chat')
    before, after = marked.split(marker)
    actual = bundle.build_chat(source_system, cfg['qa_user'].format(question=question))
    if actual != before + question + after:
        raise ValueError('chat template transforms the question; exact character span cannot be established')
    begin, end = len(before), len(before) + len(question)
    encoded = tokenizer(actual, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = list(encoded['input_ids']), [list(x) for x in encoded['offset_mapping']]
    ordinary = list(tokenizer(actual, add_special_tokens=False)['input_ids'])
    if ids != ordinary or len(offsets) != len(ids):
        raise ValueError('offset-tokenized source IDs differ from the actual source prompt')
    selected = []
    for index, (left, right) in enumerate(offsets):
        if type(left) is not int or type(right) is not int or not 0 <= left <= right <= len(actual):
            raise ValueError('invalid tokenizer character offsets')
        if left < end and right > begin:
            outside = actual[left:min(right, begin)] + actual[max(left, end):right]
            if outside.strip():
                raise ValueError('question token crosses into non-whitespace template/system text')
            selected.append(index)
    if not selected or selected != list(range(selected[0], selected[-1] + 1)):
        raise ValueError('question must map to one nonempty contiguous token span')
    covered = set()
    for index in selected:
        left, right = offsets[index]
        covered.update(range(max(left, begin), min(right, end)))
    if any(begin+i not in covered for i, c in enumerate(question) if not c.isspace()):
        raise ValueError('offset map does not cover every nonspace question character')
    mask = [int(i not in selected) for i in range(len(ids))]
    if not mask[0] or not mask[-1] or not any(mask):
        raise ValueError('question mask must preserve leading context and final source predictor')
    return {'source_mode': SOURCE_MODE, 'source_prompt_ids': ids, 'source_prompt_attention_mask': mask,
            'question_character_span': [begin, end], 'question_token_span': [selected[0], selected[-1]+1],
            'masked_token_indices': selected, 'masked_token_offsets': [offsets[i] for i in selected],
            'source_prompt_sha256': hashlib.sha256(actual.encode()).hexdigest(),
            'sentinel_render_verified': True, 'offset_ids_equal_actual_prompt': True,
            'position_ids_policy': 'original unmasked positions; never compress around masked question keys'}


def validate_mask(mask, prompt_length):
    if mask is None:
        return None
    mask = list(mask)
    if (len(mask) != prompt_length or any(type(x) is not int or x not in (0, 1) for x in mask)
            or not mask[0] or not mask[-1] or all(mask)):
        raise ValueError('source prompt mask must suppress a nonempty interior span')
    zero = [i for i, x in enumerate(mask) if not x]
    if zero != list(range(zero[0], zero[-1]+1)):
        raise ValueError('source question mask must be contiguous')
    return mask


def masked_kwargs(kwargs, prompt_mask):
    original = kwargs['attention_mask']
    if original.ndim != 2 or original.shape[0] != 1 or not bool((original == 1).all()):
        raise ValueError('source control expects an unpadded single-question forward')
    total, active = original.shape[1], kwargs['input_ids'].shape[1]
    if total < len(prompt_mask):raise ValueError('source attention mask misses prompt positions')
    expected = torch.arange(total-active, total, device=original.device).unsqueeze(0)
    if not torch.equal(kwargs.get('position_ids'), expected):
        raise ValueError('source positions must match original unmasked positions')
    result = dict(kwargs)
    result['attention_mask'] = torch.tensor([prompt_mask + [1]*(total-len(prompt_mask))],
                                           dtype=original.dtype, device=original.device)
    # Keep the exact original tensor, rather than recomputing positions from holes.
    if result['position_ids'] is not kwargs['position_ids']:
        raise RuntimeError('source position tensor unexpectedly changed')
    return result


@contextlib.contextmanager
def mask_one_source_forward(model, prompt_mask):
    if prompt_mask is None:
        yield
        return
    calls = []
    def hook(module, inputs, kwargs):
        if inputs:raise ValueError('source forward must use explicit keyword arguments')
        calls.append(1)
        return inputs, masked_kwargs(kwargs, prompt_mask)
    handle = model.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield
        if len(calls) != 1:raise RuntimeError('source mask scope must contain exactly one model forward')
    finally:handle.remove()


class QuestionMaskedSourceDecoder(SourceInterchangeDecoder):
    def __init__(self, *args, source_prompt_attention_mask=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_prompt_attention_mask = validate_mask(source_prompt_attention_mask, len(self.source_ids))

    def _forward(self, prompt, cache):
        if prompt is self.source_ids:
            with mask_one_source_forward(self.bundle.model, self.source_prompt_attention_mask):
                return super()._forward(prompt, cache)
        return super()._forward(prompt, cache)

    def next_logits(self):
        logits, trace = super().next_logits()
        if self.source_prompt_attention_mask is not None:
            trace['source_mode'] = SOURCE_MODE
            trace['source_prompt_attention_mask'] = list(self.source_prompt_attention_mask)
            trace['source_positions_preserved'] = True
        return logits, trace


def greedy_masked_source_generate(bundle, base_prompt_ids, source_prompt_ids, *, source_prompt_attention_mask=None,
                                 mode='source_interchange', layer=None, direction=None, max_new_tokens=48, use_cache=True):
    if type(max_new_tokens) is not int or max_new_tokens < 1:raise ValueError('positive generation budget required')
    stops = bundle.model.generation_config.eos_token_id
    stops = set(stops if isinstance(stops, (list, tuple)) else [stops])
    if not stops or any(type(x) is not int or x < 0 for x in stops):raise ValueError('invalid natural stop IDs')
    decoder = QuestionMaskedSourceDecoder(bundle, base_prompt_ids, source_prompt_ids, mode=mode, layer=layer,
        direction=direction, use_cache=use_cache, source_prompt_attention_mask=source_prompt_attention_mask)
    finished = False
    for _ in range(max_new_tokens):
        logits, _ = decoder.next_logits();token = int(logits.argmax());decoder.accept(token)
        if token in stops:finished = True;break
    return {'token_ids': decoder.history, 'output': bundle.tokenizer.decode(decoder.history, skip_special_tokens=True),
            'finished': finished, 'trace': decoder.traces}


class TrainingSourceMask:
    """Scoped exact-prefix dispatch for the unchanged E34 training implementation.

    Every current training forward contains its full prompt (no KV cache).
    Known source prefixes receive the mask; known base prefixes pass unchanged.
    Unrecognized, ambiguous, padded, or position-compressed calls fail loudly.
    """
    def __init__(self, bundle, pairs, masks, enabled=True):
        self.bundle, self.enabled = bundle, enabled
        self.sources, self.bases = {}, set()
        self.audit = {'source_calls': 0, 'base_calls': 0, 'source_grad_enabled': False,
                      'position_ids_preserved': True, 'source_calls_by_pair': {}}
        for pair in pairs:
            pid=pair['pair_id'];s=tuple(pair['positive']['source']['prefix_ids']);b=tuple(pair['positive']['base']['prefix_ids'])
            m=masks[pid]
            if tuple(m['source_prompt_ids']) != s:raise ValueError('training source tokens differ from verified mask')
            mask=validate_mask(m['source_prompt_attention_mask'],len(s))
            if s in self.sources and self.sources[s][0]!=mask:raise ValueError('conflicting masks for same source prompt')
            self.sources[s]=(mask,pid);self.bases.add(b)
            for branch in ('positive','negative'):
                if tuple(pair[branch]['source']['prefix_ids'])!=s or tuple(pair[branch]['base']['prefix_ids'])!=b:
                    raise ValueError('positive and negative prompts differ')
        all_prompts=[('source',p) for p in self.sources]+[('base',p) for p in self.bases]
        for i,(kind,a) in enumerate(all_prompts):
            for other,b in all_prompts[i+1:]:
                if a[:min(len(a),len(b))]==b[:min(len(a),len(b))]:
                    raise ValueError('source/base prompt dispatch is ambiguous')

    def __enter__(self):
        def hook(module, inputs, kwargs):
            if inputs or kwargs['input_ids'].shape[0]!=1 or kwargs.get('use_cache'):
                raise ValueError('masked training requires keyword-only uncached single-pair forwards')
            ids=tuple(kwargs['input_ids'][0].detach().cpu().tolist())
            source=[(mask,pid) for prefix,(mask,pid) in self.sources.items() if ids[:len(prefix)]==prefix]
            base=[prefix for prefix in self.bases if ids[:len(prefix)]==prefix]
            if len(source)+len(base)!=1:raise ValueError('training forward is not a uniquely declared source/base prompt')
            if base:
                self.audit['base_calls']+=1
                return None
            mask,pid=source[0];self.audit['source_calls']+=1
            self.audit['source_calls_by_pair'][pid]=self.audit['source_calls_by_pair'].get(pid,0)+1
            self.audit['source_grad_enabled'] |= torch.is_grad_enabled()
            if torch.is_grad_enabled():raise RuntimeError('source stream unexpectedly has gradients enabled')
            return (inputs,masked_kwargs(kwargs,mask)) if self.enabled else None
        self.handle=self.bundle.model.register_forward_pre_hook(hook,with_kwargs=True)
        return self

    def __exit__(self,*args):self.handle.remove()
