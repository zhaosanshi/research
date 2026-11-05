import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'from_claude/code'), str(ROOT/'from_gpt')]
import torch
torch.set_num_threads(1)
import transformers
import transformers.utils.import_utils as iu
# The local optional pandas/sklearn installation is ABI-incompatible with
# numpy. These optional audio/image/candidate-generator integrations are unused
# by the random text-only causal models below. Do not change installed packages.
iu._sklearn_available = False
iu._librosa_available = False
iu._torchvision_available = False
from transformers import (LlamaConfig, LlamaForCausalLM, Qwen3Config,
                          Qwen3ForCausalLM, Gemma3TextConfig, Gemma3Config,
                          Gemma3ForConditionalGeneration, SiglipVisionConfig)
from core.model_io import ModelBundle, forward_logits
from core.qa_das import prepare_selfcheck_pairs, train_qa_das, evaluate_selfcheck
from core.qa_precision import fp32_output_projection
from core.intervene import SteerSpec, steer
from test_qa_das_training_20261105 import Tokenizer, CFG, RECORDS

base = dict(vocab_size=160, hidden_size=16, intermediate_size=32,
            num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, max_position_embeddings=512,
            pad_token_id=0, bos_token_id=1, eos_token_id=2,
            attention_dropout=0.0)
configurations = [
    ('llama', LlamaForCausalLM, LlamaConfig(**base), 'model.layers'),
    ('qwen3', Qwen3ForCausalLM, Qwen3Config(**base), 'model.layers'),
    ('gemma3', Gemma3ForConditionalGeneration,
     Gemma3Config(text_config=Gemma3TextConfig(
         **base, sliding_window=256, layer_types=['sliding_attention', 'sliding_attention', 'full_attention'],
         query_pre_attn_scalar=8, final_logit_softcapping=5.0),
         vision_config=SiglipVisionConfig(hidden_size=16, intermediate_size=32,
             num_hidden_layers=1, num_attention_heads=2, image_size=8, patch_size=2),
         mm_tokens_per_image=4, boi_token_index=153, eoi_token_index=154, image_token_index=155),
     'model.language_model.layers'),
]
report = {'torch':torch.__version__, 'transformers':transformers.__version__,
          'random_initialization':True, 'downloads':False,
          'optional_imports_disabled':['sklearn','librosa','torchvision'], 'models':{}}
for name, cls, config, layers_attr in configurations:
    torch.manual_seed(71)
    model = cls(config).to(dtype=torch.bfloat16).eval()
    bundle = ModelBundle(model, Tokenizer(), {'name':name, 'layers_attr':layers_attr})
    pairs = prepare_selfcheck_pairs(bundle, CFG, RECORDS[:1])
    before = {n:p.detach().clone() for n,p in model.named_parameters()}
    original_head = model.get_output_embeddings().forward
    with fp32_output_projection(bundle):
        ckpt = train_qa_das(bundle, pairs, layer=1, epochs=1, log_fn=None)
        x=pairs[0]['positive']['inputs']
        logits=forward_logits(bundle,x,logits_to_keep=1)
        assert logits.dtype == torch.float32
        assert torch.isfinite(logits).all()
        spec=SteerSpec(ckpt['W'],1,mode='amplify',alpha=1,
                       center=torch.zeros(bundle.hidden_size),
                       decode_center=torch.zeros(bundle.hidden_size))
        with steer(bundle,[spec]):
            generated = model.generate(**x,max_new_tokens=2,do_sample=False,pad_token_id=0)
        assert spec.n_applied == 2
    assert model.get_output_embeddings().forward == original_head
    assert all(torch.equal(p,before[n]) and not p.requires_grad and p.grad is None
               for n,p in model.named_parameters())
    try:
        train_qa_das(bundle,pairs,layer=2,epochs=1,log_fn=None)
    except ValueError as exc:
        assert 'structural null' in str(exc)
    else:
        raise AssertionError('last layer incorrectly accepted')
    report['models'][name]={
        'loss':ckpt['losses'][0], 'gradient_norm_max':ckpt['per_seed']['gradient_norm_max'],
        'frozen_parameters_unchanged':True,'logits_dtype':str(logits.dtype),
        'output_head_restored':True,'continuous_generation_calls':spec.n_applied,
        'generated_ids':generated[0,-2:].tolist(),
        'last_layer_training_rejected':True,
    }
path=Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/tmp/qa_das_training_families_recheck.json')
path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
