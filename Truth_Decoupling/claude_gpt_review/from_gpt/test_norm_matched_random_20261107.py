import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT.parent/'from_claude/code'),str(ROOT.parent/'from_claude/code/scripts')]
from core.qa_norm_matched_random import matched_random_write, fixed_random_direction, NormMatchedRandomDecoder
from core import qa_source_interchange as source
from core.model_io import ModelBundle
from test_qa_das_pipeline_20261105 import PipelineModel,ToyTokenizer


class NormMatchedTests(unittest.TestCase):
    def test_actual_bf16_write_matches_not_only_float32_formula(self):
        generator=torch.Generator().manual_seed(7)
        before=(torch.randn(3,128,generator=generator)*10).bfloat16()
        donor=(before.float()+torch.randn(3,128,generator=generator)*3).bfloat16()
        target=fixed_random_direction(128,42,0); random=fixed_random_direction(128,43,0)
        result,audit=matched_random_write(before,donor,target,random)
        actual=(result.float()-before.float()).norm(dim=-1)
        for i,row in enumerate(audit):
            self.assertAlmostEqual(float(actual[i]),row['random_stored_delta_norm'])
            self.assertLessEqual(row['absolute_norm_error'],row['cast_match_tolerance']+1e-7)
            self.assertTrue(row['learned_direction_used_for_size_only'])
            self.assertFalse(row['is_unscaled_source_interchange'])

    def test_zero_reference_write_is_exact_noop_and_orientation_is_random(self):
        b=torch.tensor([[1.,2.,3.]])
        w=torch.tensor([1.,0.,0.]);v=torch.tensor([0.,1.,0.])
        same,audit=matched_random_write(b,b,w,v)
        self.assertTrue(torch.equal(same,b));self.assertEqual(audit[0]['target_stored_delta_norm'],0.)
        altered,_=matched_random_write(b,b+torch.tensor([[2.,0.,0.]]),w,v)
        self.assertAlmostEqual(float(altered[0,0]),1.)
        self.assertAlmostEqual(float(altered[0,2]),3.)
        self.assertAlmostEqual(float((altered-b).norm()),2.,places=5)

    def test_same_norm_different_reference_orientation_does_not_leak_direction(self):
        b=torch.zeros(1,3);s=torch.tensor([[2.,2.,0.]])
        v=torch.tensor([0.,0.,1.])
        a,_=matched_random_write(b,s,torch.tensor([1.,0.,0.]),v)
        c,_=matched_random_write(b,s,torch.tensor([0.,1.,0.]),v)
        self.assertTrue(torch.equal(a,c))

    def test_real_cached_and_full_paths_match_and_restore_original_hook(self):
        tokenizer=ToyTokenizer();model=PipelineModel(tokenizer)
        bundle=ModelBundle(model,tokenizer,{'name':'toy','layers_attr':'layers'})
        w=fixed_random_direction(8,42,0);v=fixed_random_direction(8,43,0)
        a=NormMatchedRandomDecoder(bundle,[31,44,57],[38,39,40,41,42,43,57],
            layer=0,target_direction=w,random_direction=v,use_cache=True)
        b=NormMatchedRandomDecoder(bundle,[31,44,57],[38,39,40,41,42,43,57],
            layer=0,target_direction=w,random_direction=v,use_cache=False)
        original=source.write_base
        versions=[p._version for p in model.parameters()]
        for token in (70,71,72):
            x,xt=a.next_logits();y,yt=b.next_logits()
            torch.testing.assert_close(x,y,atol=2e-6,rtol=1e-5)
            self.assertIs(source.write_base,original)
            self.assertTrue(xt['cache_storage_disjoint'])
            self.assertTrue(all(r['within_tolerance'] for r in xt['writes']))
            a.accept(token);b.accept(token)
        self.assertEqual(versions,[p._version for p in model.parameters()])
        with patch.object(source.SourceInterchangeDecoder,'next_logits',side_effect=RuntimeError('forced')):
            with self.assertRaisesRegex(RuntimeError,'forced'):a.next_logits()
        self.assertIs(source.write_base,original)


if __name__=='__main__': unittest.main()
