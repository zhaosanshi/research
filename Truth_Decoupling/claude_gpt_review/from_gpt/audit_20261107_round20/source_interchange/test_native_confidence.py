from copy import deepcopy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from native_confidence import native_score


def record(finished=True,text='two words'):
    ids=[10,11,2];probabilities=[.8,.4,.9]
    trace=[{'step':i,'next_top1_id':t,'next_top1_probability':p,'accepted_prefix_ids':ids[:i],
            'uses_reference_answer':False} for i,(t,p) in enumerate(zip(ids,probabilities))]
    return {'outputs':{'x':text},'finished':{'x':finished},'generation_token_ids':{'x':ids},'trace':{'x':trace}}


class NativeConfidenceTests(unittest.TestCase):
    def test_finished_excludes_only_last_natural_stop_from_score_not_generation(self):
        r=record();old=deepcopy(r);s=native_score(r,'x')
        self.assertAlmostEqual(s['score'],(math.log(.8)+math.log(.4))/2)
        self.assertEqual(s['answer_tokens'],2);self.assertEqual(s['natural_stop_token_id'],2)
        self.assertEqual(r,old)

    def test_unfinished_keeps_last_token_even_if_its_numeric_id_looks_like_a_stop(self):
        s=native_score(record(False),'x')
        self.assertAlmostEqual(s['score'],sum(map(math.log,[.8,.4,.9]))/3)
        self.assertEqual(s['answer_tokens'],3);self.assertFalse(s['natural_stop_excluded'])

    def test_empty_sequence_is_explicitly_unscorable(self):
        r=record(False,'');r['generation_token_ids']['x']=[];r['trace']['x']=[]
        s=native_score(r,'x');self.assertEqual(s['score'],-1e6);self.assertFalse(s['scorable'])
        r['finished']['x']=True
        with self.assertRaises(ValueError):native_score(r,'x')

    def test_immediate_stop_is_unscorable_without_inserting_answer_tokens(self):
        r=record(True,'');r['generation_token_ids']['x']=[2]
        r['trace']['x']=[{'step':0,'next_top1_id':2,'next_top1_probability':.9,'accepted_prefix_ids':[],
                         'uses_reference_answer':False}]
        self.assertFalse(native_score(r,'x')['scorable'])

    def test_invalid_probabilities_never_become_finite_scores(self):
        for value in [0.,-1.,1.01,float('nan'),float('inf'),True]:
            r=record();r['trace']['x'][1]['next_top1_probability']=value
            with self.assertRaises(ValueError):native_score(r,'x')

    def test_wrong_actual_token_history_or_trace_length_is_rejected(self):
        mutations=[lambda r:r['trace']['x'].pop(),
                   lambda r:r['trace']['x'][1].update(next_top1_id=99),
                   lambda r:r['trace']['x'][1].update(accepted_prefix_ids=[99]),
                   lambda r:r['trace']['x'][1].update(step=99),
                   lambda r:r['trace']['x'][1].update(uses_reference_answer=True)]
        for mutate in mutations:
            r=record();mutate(r)
            with self.assertRaises(ValueError):native_score(r,'x')

    def test_gold_and_correctness_do_not_change_native_score(self):
        r=record();a=native_score(r,'x')
        r['gold']=['changed reference'];r['outcome']={'x':'wrong'}
        self.assertEqual(native_score(r,'x'),a)


if __name__=='__main__':unittest.main()
