"""Real HF fast offsets and causal-attention/cache/gradient execution checks."""
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import copy
import importlib.util

import torch
from tokenizers import Tokenizer,models,decoders
from transformers import PreTrainedTokenizerFast

REVIEW=Path(__file__).resolve().parents[3];CODE=REVIEW/'from_claude/code'
sys.path[:0]=[str(CODE),str(CODE/'scripts')]
from core.model_io import ModelBundle,forward_logits
from core.qa_generation_adapter import frozen_model
from core.qa_source_question_mask import (locate_question_mask,TrainingSourceMask,QuestionMaskedSourceDecoder,
    mask_one_source_forward,masked_kwargs)
from core.qa_source_interchange import SourceInterchangeDecoder
from core import qa_source_pair_preference as pref
from core import qa_source_masked_preference as masked


class TinyAttentionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__();self.qkv=torch.nn.Linear(16,48,bias=False);self.out=torch.nn.Linear(16,16,bias=False)
    def forward(self,h,attention_mask,past=None):
        q,k,v=self.qkv(h).chunk(3,dim=-1)
        if past is not None:k=torch.cat([past[0],k],1);v=torch.cat([past[1],v],1)
        total=k.shape[1];active=q.shape[1]
        causal=torch.arange(total,device=h.device)[None,:]<=torch.arange(total-active,total,device=h.device)[:,None]
        permitted=causal[None]&attention_mask[:,None,:].bool()
        scores=q@k.transpose(-1,-2)/4
        weights=scores.masked_fill(~permitted,float('-inf')).softmax(-1)
        return h+.1*torch.tanh(self.out(weights@v)),(k,v)


class TinyAttentionModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.config=SimpleNamespace(hidden_size=16,num_hidden_layers=2)
        self.generation_config=SimpleNamespace(eos_token_id=2)
        self.embedding=torch.nn.Embedding(256,16);self.positions=torch.nn.Embedding(512,16)
        self.layers=torch.nn.ModuleList([TinyAttentionBlock(),TinyAttentionBlock()]);self.head=torch.nn.Linear(16,256)
    def get_output_embeddings(self):return self.head
    def forward(self,input_ids,attention_mask,position_ids,use_cache=False,past_key_values=None):
        h=self.embedding(input_ids)+self.positions(position_ids);cache=[]
        for i,layer in enumerate(self.layers):
            h,kv=layer(h,attention_mask,None if past_key_values is None else past_key_values[i]);cache.append(kv)
        return SimpleNamespace(logits=self.head(h),past_key_values=tuple(cache) if use_cache else None)


def fixture():
    torch.manual_seed(19)
    vocab={'<pad>':0,'<unk>':1,'<eos>':2}
    vocab.update({chr(i):i+3 for i in range(127)})
    vocab.update({'问':130,'题':131,'哪':132,'里':133,'？':134})
    vocab.update({chr(0x1000+i):i for i in range(135,256)})
    backend=Tokenizer(models.BPE(vocab=vocab,merges=[],unk_token='<unk>'));backend.decoder=decoders.Fuse()
    tok=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='<pad>',unk_token='<unk>',eos_token='<eos>')
    tok.chat_template="{% for message in messages %}<{{ message['role'] }}>{{ message['content'] }}\n{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"
    model=TinyAttentionModel().eval().requires_grad_(False)
    bundle=ModelBundle(model,tok,{'name':'tiny-attention-mask','layers_attr':'layers'})
    config={'qa_system':'Answer.','qa_user':'Question: {question}\nAnswer:','max_new_tokens':48,'batch_size':8}
    records=[{'pair_id':'p1','question':'Where?','positive_answer':'Paris','negative_answer':'Rome'},
             {'pair_id':'p2','question':'Which way?','positive_answer':'North','negative_answer':'Northbound'},
             {'pair_id':'p3','question':'Can it?','positive_answer':'Yes indeed','negative_answer':'Yes never'}]
    return bundle,config,records


class SourceQuestionMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):
        self.bundle,self.cfg,self.records=fixture();self.source='Be truthful. Where? This repeated system text stays.'
        self.w=torch.nn.functional.normalize(torch.arange(1.,17.),dim=0)
        self.pairs=pref.prepare_pairs(self.bundle,self.cfg,self.records,self.source)
        self.masks={r['pair_id']:locate_question_mask(self.bundle,self.cfg,self.source,r['question']) for r in self.records}

    def test_real_fast_offsets_mask_only_inserted_question_even_if_system_repeats_it(self):
        m=self.masks['p1'];text=self.bundle.build_chat(self.source,self.cfg['qa_user'].format(question='Where?'))
        first=text.index('Where?');last=text.rindex('Where?')
        self.assertNotEqual(first,last);self.assertEqual(m['question_character_span'],[last,last+6])
        self.assertTrue(all(a>=last and b<=last+6 for a,b in m['masked_token_offsets']))
        u=locate_question_mask(self.bundle,self.cfg,self.source,'问题哪里？')
        self.assertEqual(len(u['masked_token_indices']),5)
        self.assertEqual(u['source_prompt_ids'],self.bundle.tokenizer(self.bundle.build_chat(self.source,self.cfg['qa_user'].format(question='问题哪里？')),add_special_tokens=False)['input_ids'])

    def test_three_real_causal_mask_none_cache_and_base_dependence_checks(self):
        checks=masked.verify_mask_control(self.bundle,self.cfg,self.records,self.source,0,self.w)
        self.assertEqual(len(checks),3)
        self.assertTrue(all(c['base_unmasked_question_replacement_max_logit_difference']>0 for c in checks))
        self.assertFalse(self.bundle.model._forward_pre_hooks)

    def test_positive_negative_and_shared_source_paths_mask_keys_keep_positions_base_unchanged(self):
        pair=self.pairs[0];source_ids=pair['positive']['source']['prefix_ids'];observed=[]
        with TrainingSourceMask(self.bundle,[pair],{'p1':self.masks['p1']}) as scope:
            def spy(module,args,kw):
                ids=kw['input_ids'][0].tolist();is_source=ids[:len(source_ids)]==source_ids
                observed.append((is_source,kw['attention_mask'].clone(),kw['position_ids'].clone()))
            handle=self.bundle.model.register_forward_pre_hook(spy,with_kwargs=True)
            with frozen_model(self.bundle.model):
                w=torch.nn.Parameter(self.w.clone());ref=pref.content_gap(self.bundle,pair,0)
                loss=pref.preference_loss(self.bundle,pair,0,w,ref);loss.backward()
            handle.remove()
        source=[x for x in observed if x[0]];base=[x for x in observed if not x[0]]
        self.assertEqual(len(source),3)  # first fork, positive suffix, negative suffix
        self.assertTrue(base)
        for _,mask,pos in source:
            self.assertEqual(mask[0,:len(source_ids)].tolist(),self.masks['p1']['source_prompt_attention_mask'])
            self.assertTrue(torch.equal(pos,torch.arange(mask.shape[1]).unsqueeze(0)))
        self.assertTrue(all(bool((m==1).all()) for _,m,_ in base))
        self.assertFalse(scope.audit['source_grad_enabled']);self.assertGreater(float(w.grad.norm()),0.)

    def test_disabled_training_scope_preserves_old_loss_and_gradient_bitwise(self):
        pair=self.pairs[0]
        def run(scope):
            w=torch.nn.Parameter(self.w.clone())
            with scope,frozen_model(self.bundle.model):
                ref=pref.content_gap(self.bundle,pair,0);loss=pref.preference_loss(self.bundle,pair,0,w,ref)
                loss.backward()
            return loss.detach(),w.grad
        import contextlib
        a,g=run(contextlib.nullcontext())
        b,h=run(TrainingSourceMask(self.bundle,self.pairs,self.masks,enabled=False))
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(g,h))

    def test_real_three_pair_training_updates_only_w_and_roundtrips_mask_identity(self):
        before=[p.clone() for p in self.bundle.model.parameters()]
        result=masked.train(self.bundle,self.records,self.cfg,self.w,layer=0,source_system=self.source,pilot=True,log_fn=None)
        self.assertTrue(torch.equal(result['initial_W'],self.w));self.assertEqual(result['gradient_audit']['updates'],3)
        self.assertGreater(result['gradient_audit']['nonzero_updates'],0)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,self.bundle.model.parameters())))
        self.assertFalse(self.bundle.model._forward_pre_hooks)
        self.assertTrue(all(not layer._forward_hooks for layer in self.bundle.layers()))
        with tempfile.TemporaryDirectory() as d:
            masked.save(result,d);loaded=masked.load(d)
            self.assertTrue(torch.equal(loaded['W'],result['W']))
            self.assertEqual(loaded['source_mask_manifest'],self.masks)

    def test_position_compression_unknown_prefix_and_same_base_source_are_rejected(self):
        mask=self.masks['p1']['source_prompt_attention_mask'];x=torch.ones((1,len(mask)),dtype=torch.long)
        with self.assertRaisesRegex(ValueError,'positions'):
            masked_kwargs({'input_ids':x,'attention_mask':x,'position_ids':torch.tensor([mask]).cumsum(-1)-1},mask)
        with TrainingSourceMask(self.bundle,self.pairs,self.masks):
            with self.assertRaisesRegex(ValueError,'uniquely declared'):
                forward_logits(self.bundle,{'input_ids':torch.tensor([[20,21]]),'attention_mask':torch.ones((1,2),dtype=torch.long)})
        self.assertFalse(self.bundle.model._forward_pre_hooks)
        same=pref.prepare_pairs(self.bundle,self.cfg,self.records,self.cfg['qa_system'])
        masks={r['pair_id']:locate_question_mask(self.bundle,self.cfg,self.cfg['qa_system'],r['question']) for r in self.records}
        with self.assertRaisesRegex(ValueError,'ambiguous'):TrainingSourceMask(self.bundle,same,masks)

    def test_entry_generation_preserves_all_seven_controls_and_binds_new_checkpoint(self):
        spec=importlib.util.spec_from_file_location('masked_entry',CODE/'scripts/review_checks/qa_masked_source_control.py')
        entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
        from core.qa_source_interchange import greedy_source_generate
        from core.qa_generation_adapter import implementation_identity
        cfg={**self.cfg,'max_new_tokens':3};rows=[]
        for i,record in enumerate(self.records):
            base,source=entry.source_eval.prompt_tokens(self.bundle,cfg,record['question'])
            g=greedy_source_generate(self.bundle,base,source,mode='baseline',max_new_tokens=3)
            rows.append({'pair_id':record['pair_id'],'question':record['question'],'gold':[record['positive_answer']],
                'incorrect':[record['negative_answer']],'best_answer':record['positive_answer'],'idx':i,
                'prompt_token_ids':{'base':base,'source':source},
                'outputs':{n:g['output'] for n in entry.PARENTS},'finished':{n:g['finished'] for n in entry.PARENTS},
                'rule_abstain':{n:False for n in entry.PARENTS},
                'generation_token_ids':{n:copy.deepcopy(g['token_ids']) for n in entry.PARENTS},
                'trace':{n:copy.deepcopy(g['trace']) for n in entry.PARENTS}})
        parent={'evaluations':rows,'identity':{'old':'parent7'},'control_identity':{'old':'parent6'},
            'training_sha256':'old-checkpoint','conditions':entry.PARENTS,
            'condition_specs':[{'name':n,'uses_reference_answer':False} for n in entry.PARENTS]}
        meta={'config':cfg};prepared={'e_train':self.records};st=((None,meta,None,prepared,None,None),None,parent,None,None)
        adapter={'pilot':True,'layer':0,'W':self.w,'config':cfg,'source_system':entry.source_eval.SOURCE_SYSTEM,
                 'source_mask_manifest_sha256':'toy-mask-training-id'}
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(run_dir=Path(directory),e34_run=Path(directory)/'parent')
            (args.run_dir/'preference_adapter.pt').write_bytes(b'new-checkpoint')
            entry.qa.save(args.run_dir/'plan.json',{'toy':True})
            plan={'model':self.bundle.name,'source':{},'layer':0,'config':cfg,'source_system':adapter['source_system'],
                  'informed_parent':{'generation_sha256':'parent7-sha'},'implementation':implementation_identity()}
            with patch.object(entry,'plan',return_value=plan),patch.object(entry.qa,'verify_bundle',return_value=None):
                result=entry.generate(args,st,adapter,True,bundle=self.bundle)
            self.assertEqual(result['training_sha256'],entry.qa.digest(args.run_dir/'preference_adapter.pt'))
            self.assertEqual(result['ancestor_control_identity'],parent['control_identity'])
            self.assertEqual(result['conditions'],entry.PARENTS+[masked.METHOD])
            entry.inherited(rows,result['evaluations'])
            self.assertEqual(parent['training_sha256'],'old-checkpoint')
            self.assertTrue(all(masked.METHOD not in r['outputs'] for r in rows))

if __name__=='__main__':unittest.main()
