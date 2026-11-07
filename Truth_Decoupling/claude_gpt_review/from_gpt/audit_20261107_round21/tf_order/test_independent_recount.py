from pathlib import Path
import importlib.util
import unittest
import numpy as np
p=Path(__file__).with_name('independent_recount.py')
s=importlib.util.spec_from_file_location('independent_e38_recount',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class AuditTest(unittest.TestCase):
    def test_exact_group_swap_small_enumeration(self):
        self.assertEqual(m.exact_pair_swap([2,1]),.5)
        self.assertEqual(m.exact_pair_swap([2,-1]),1.)
        self.assertEqual(m.exact_pair_swap([0,0]),1.)

    def test_frequency_bootstrap_equals_expanded_question_instances(self):
        # Two statements per question, with unequal resampling multiplicity.
        group=np.array([0,0,1,1,2,2]);correct=np.array([1,0,1,1,0,1])
        order=np.array([4,1,2,5,3,0]);frequency=np.array([2,0,1]);k=3
        counts=frequency[group[order]];cum=counts.cumsum();take=np.minimum(counts,np.maximum(k-(cum-counts),0))
        fast=int((take*correct[order]).sum())
        explicit=[int(correct[i]) for i in order for _ in range(int(frequency[group[i]]))]
        self.assertEqual(fast,sum(explicit[:k]));self.assertEqual(int(take.sum()),k)

if __name__=='__main__':unittest.main()
