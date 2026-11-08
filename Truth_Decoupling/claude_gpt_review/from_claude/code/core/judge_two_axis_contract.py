"""Content and request identities for versioned QA judging; no API calls.

Cell identity intentionally excludes condition/cell IDs: the same complete
semantic input must receive one final decision across all experimental arms.
Request contracts exclude credentials and runtime retry/sleep settings.
"""
import copy
import hashlib
import importlib
import inspect
import json
import math
import os
import re
from pathlib import Path


CONTRACT_VERSION = 'qa_judge_contract_v1'
CELL_FIELDS = ('question','prediction','gold','gold_refs','incorrect','best_answer','gold_is_abstention')
DEFAULT_OPENAI_BASE_URL = 'https://api.openai.com/v1'
# The part of an LLM judge contract that determines the API request (what was sent): the cache identity.
# The full contract additionally hashes the reply parsers (how replies were read): the row identity. A parser fix
# therefore makes stored rows stale (re-read from the same cached replies) without invalidating the cache.
REQUEST_FIELDS = ('protocol','kind','judge','rubric','request','prompts','prompt_builder_sha256')


def request_identity(contract):
    return {k: contract[k] for k in REQUEST_FIELDS if k in contract}


def canonical_sha(data):
    payload=json.dumps(data,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def file_sha(path):
    value=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def cell_sha(cell):
    if not isinstance(cell,dict) or not isinstance(cell.get('question'),str) or not isinstance(cell.get('prediction'),str):
        raise ValueError('cell identity requires the complete question and prediction strings')
    return canonical_sha({'protocol':'qa_complete_cell_v1',**{key:cell.get(key) for key in CELL_FIELDS}})


def _positive_integer(value,name):
    if type(value) is not int or value<=0:raise ValueError(name+' must be a positive integer')
    return value


def resolve_base_url(value=None):
    """Match SDK explicit/env/default precedence, then pass that value explicitly.

    Verified against installed OpenAI SDK 2.6.1. Freezing the selected value at
    judge initialization prevents a later environment change from redirecting
    the lazy client while retaining the earlier cache identity.
    """
    if value is None:value=os.environ.get('OPENAI_BASE_URL')
    if value is None:value=DEFAULT_OPENAI_BASE_URL
    if not isinstance(value,str) or not value.strip():raise ValueError('judge base_url must resolve to a nonempty URL string')
    return value


def request_contract(cfg):
    """Effective chat-completion fields, including the resolved endpoint."""
    model=cfg.get('model')
    if not isinstance(model,str) or not model:raise ValueError('judge request requires a model ID')
    send=cfg.get('send_temperature',True)
    if type(send) is not bool:raise ValueError('send_temperature must be boolean')
    completion=cfg.get('max_completion_tokens')
    if completion is not None:
        completion=_positive_integer(completion,'max_completion_tokens');tokens=None
    else:tokens=_positive_integer(cfg.get('max_tokens',8),'max_tokens')
    temperature=cfg.get('temperature',0.0) if send else None
    if send and (isinstance(temperature,bool) or not isinstance(temperature,(int,float)) or not math.isfinite(temperature)):
        raise ValueError('temperature must be finite when sent')
    extra=copy.deepcopy(cfg.get('extra_body') or None)
    if extra is not None and not isinstance(extra,dict):raise ValueError('extra_body must be a dictionary')
    result={'protocol':'llm_effective_request_v1','model':model,'base_url':resolve_base_url(cfg.get('base_url')),
        'max_tokens':tokens,'max_completion_tokens':completion,'send_temperature':send,
        'temperature':float(temperature) if send else None,'reasoning_effort':cfg.get('reasoning_effort') or None,
        'extra_body':extra}
    canonical_sha(result)  # Reject non-JSON/nonfinite request fields before any call.
    return result


def judge_contract(name,cfg):
    """Return a dictionary; callers hash it with canonical_sha.

    cfg may be the complete judges config or one judge's effective config.
    GPU callers should supply actual max_chars/context/tokenizer/dtype values
    and runtime_identity (weights/tokenizer/source hashes) when available.
    """
    sub=cfg[name] if name in cfg and isinstance(cfg[name],dict) else cfg
    module=importlib.import_module('.judge_two_axis',__package__)
    if cfg.get('version',module.TWO_AXIS_VERSION)!=module.TWO_AXIS_VERSION:
        raise ValueError('declared judge rubric differs from the loaded prompt version')
    if name=='allenai':
        prompts=sub.get('prompt') or {'truth':'Q: {question}\nA: {answer}\nTrue:',
                                    'info':'Q: {question}\nA: {answer}\nHelpful:'}
        result={'protocol':CONTRACT_VERSION,'kind':'allenai','judge':name,'rubric':module.TWO_AXIS_VERSION,
            'truth_model':sub.get('truth_model'),'info_model':sub.get('info_model'),
            'prompt':copy.deepcopy(prompts),'threshold':sub.get('threshold',.5),
            'max_chars':sub.get('max_chars'),'max_prompt_tokens':sub.get('max_prompt_tokens'),
            'max_context_tokens':sub.get('max_context_tokens',sub.get('context_limit',1024)),
            'use_fast':sub.get('use_fast',False),'dtype':sub.get('dtype','bfloat16'),
            'runtime_identity':copy.deepcopy(sub.get('runtime_identity'))}
        canonical_sha(result)
        return result
    # Lazy import avoids judge -> contract -> two_axis -> judge import cycles.
    def source_hashes(functions):
        hashes={}
        for function in functions:
            value=getattr(module,function,None)
            if value is not None:
                try:source=inspect.getsource(value)
                except (OSError,TypeError) as exc:raise ValueError('judge source is unavailable: '+function) from exc
                hashes[function]=hashlib.sha256(source.encode('utf-8')).hexdigest()
        return hashes
    result={'protocol':CONTRACT_VERSION,'kind':'llm','judge':name,'rubric':module.TWO_AXIS_VERSION,
        'request':request_contract(sub),'prompts':{'truth_system':module.TRUTH_SYSTEM,'truth_template':module.TRUTH_TEMPLATE,
            'info_system':module.INFO_SYSTEM,'info_template':module.INFO_TEMPLATE},
        'prompt_builder_sha256':source_hashes(('truth_user','info_user','_clip')),
        'parser_sha256':source_hashes(('_parse','parse_truth','parse_info')),
        'parser_constants':{name:({'pattern':value.pattern,'flags':value.flags} if isinstance(value,re.Pattern) else value)
            for name in ('_TRUTH_LETTER','_INFO_LETTER','_WORDS_TRUTH','_WORDS_INFO','_WRAPPERS','_HEADER','_SEP')
            for value in (getattr(module,name),)}}
    canonical_sha(result)
    return result
