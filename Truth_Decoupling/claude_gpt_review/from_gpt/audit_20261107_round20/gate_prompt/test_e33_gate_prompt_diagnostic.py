import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[3];CODE=ROOT/'from_claude/code'
sys.path[:0]=[str(CODE/'scripts'),str(CODE)]
spec=importlib.util.spec_from_file_location('e33_under_test',CODE/'scripts/e33_gate_prompt_diagnostic.py')
e33=importlib.util.module_from_spec(spec);spec.loader.exec_module(e33)

class Tokenizer:
    def decode(self,ids,**kw):return 'token'+str(ids[0])

class E33Tests(unittest.TestCase):
    def test_conditional_yes_can_be_high_while_actual_probability_is_tiny(self):
        r=e33.probability_record(np.array([0.,-10.,20.],dtype=np.float32),0,1,Tokenizer())
        self.assertGreater(r['p_yes_conditional_on_yes_no'],.999)
        self.assertLess(r['yes_no_probability_mass'],1e-6)
        self.assertEqual(r['top1_token_id'],2);self.assertFalse(r['top1_is_canonical_yes_or_no'])
        self.assertGreater(r['top1_probability'],.999)

    def test_full_probabilities_and_first_word_are_recorded(self):
        r=e33.probability_record(np.array([1.,0.,-1.],dtype=np.float32),0,1,Tokenizer())
        p=np.exp([1.,0.,-1.]);p/=p.sum()
        self.assertAlmostEqual(r['p_yes'],p[0]);self.assertAlmostEqual(r['p_no'],p[1])
        self.assertAlmostEqual(r['yes_no_probability_mass'],p[0]+p[1])
        self.assertEqual(r['top1_text'],'token0');self.assertEqual(len(r['top5']),3)

    def test_definitions_isolate_the_two_single_prompt_changes(self):
        meta={'prompts':{'statement':{'system':'old-system'}},'config':{'tf_gate_user':'Statement: {statement} old-user'}}
        a,b,c,d=e33.definitions(meta)
        self.assertEqual(a['user'],b['user']);self.assertNotEqual(a['system'],b['system'])
        self.assertEqual(b['system'],c['system']);self.assertNotEqual(b['user'],c['user'])
        self.assertEqual(c['system'],d['system']);self.assertEqual(d['input_kind'],'question')
        self.assertNotIn('{statement}',d['user'])

    def test_shared_pair_auc_half_does_not_imply_correctness_auc_half(self):
        scores=[0.,0.,1.,1.]
        self.assertEqual(e33.auc(scores,[False,True,False,True]),.5)
        self.assertEqual(e33.auc(scores,[False,False,True,True]),1.)

    def test_auc_ties_and_degenerate_label_set(self):
        self.assertEqual(e33.auc([1.,1.],[False,True]),.5)
        self.assertIsNone(e33.auc([1.,2.],[True,True]))
        with self.assertRaises(ValueError):e33.auc([float('nan'),1.],[False,True])

    def test_readout_payload_rejects_wrong_margin_and_missing_rows(self):
        r={'id':'a','pair_id':'p','prompt':'text','input_ids':[1,2],'attention_mask':[1,1],
           **e33.probability_record(np.array([1.,0.,-1.],dtype=np.float32),0,1,Tokenizer())}
        inputs=[{'id':'a','pair_id':'p'}];e33.validate_rows([r],inputs)
        with self.assertRaises(ValueError):e33.validate_rows([{**r,'score':2.}],inputs)
        with self.assertRaises(ValueError):e33.validate_rows([],inputs)
        with self.assertRaises(ValueError):e33.validate_rows([{**r,'yes_no_probability_mass':1.2}],inputs)
        bads=[{**r,'p_yes':-.1,'p_no':r['yes_no_probability_mass']+.1},
              {**r,'p_yes_conditional_on_yes_no':.1}, {**r,'top1_is_canonical_yes_or_no':False},
              {**r,'attention_mask':[1,0]}, {**r,'attention_mask':[2,1]}]
        for bad in bads:
            with self.assertRaises(ValueError):e33.validate_rows([bad],inputs)

    def test_probability_input_must_be_finite_and_labels_distinct(self):
        for a,yes,no in [(np.array([1.,float('nan')]),0,1),(np.array([1.,2.]),0,0),(np.array([]),0,1)]:
            with self.assertRaises(ValueError):e33.probability_record(a,yes,no,Tokenizer())

if __name__=='__main__':unittest.main()
