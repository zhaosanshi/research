"""The norm null inherits seven real parent arms, including the new pair arm."""
import copy
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
CODE=HERE.parent/'from_claude/code'
sys.path[:0]=[str(CODE/'scripts'),str(CODE)]
import e36_norm_matched_random as script


class ParentIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.names=['baseline','source_only','equal_logit_mix','source_das_swap',
                    'source_random_swap','trained_source_swap_das_init','pair_preference_source_swap']
        row={'pair_id':'q','idx':0,'question':'Q?','gold':['A'],'incorrect':['B'],'best_answer':'A',
             'prompt_token_ids':{'base':[1],'source':[2]},
             'outputs':{n:'A' for n in self.names},'finished':{n:True for n in self.names},
             'rule_abstain':{n:False for n in self.names},
             'generation_token_ids':{n:[5,6] for n in self.names},'trace':{n:[] for n in self.names}}
        self.raw={'model':'m','split':'tune','conditions':self.names,
                  'condition_specs':[{'name':n,'uses_reference_answer':False} for n in self.names],
                  'evaluations':[row]}
        self.cfg={'model':'offline'}
        self.graded=copy.deepcopy(self.raw)
        self.graded.update(completed=True,grading_completed=True,grading_source_generation_sha256='raw-hash',
                           grading_judge_config_sha256=script.qa.fingerprint(self.cfg))
        self.graded['evaluations'][0].update(outcome={n:'correct' for n in self.names},
            judge={n:{'label':'correct','rubric_id':script.qa.QA_GRADER_VERSION} for n in self.names})

    def test_all_seven_parent_arms_are_verified_not_only_the_six_ancestors(self):
        script.validate_parent_grades(self.raw,self.graded,self.cfg,'raw-hash')
        broken=copy.deepcopy(self.graded)
        broken['evaluations'][0]['outputs']['pair_preference_source_swap']='different text'
        with self.assertRaisesRegex(ValueError,'inherited parent data changed'):
            script.validate_parent_grades(self.raw,broken,self.cfg,'raw-hash')

    def test_wrong_judge_or_unresolved_seventh_grade_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'legacy judge'):
            script.validate_parent_grades(self.raw,self.graded,{'model':'other'},'raw-hash')
        broken=copy.deepcopy(self.graded)
        broken['evaluations'][0]['outcome']['pair_preference_source_swap']='unresolved'
        with self.assertRaisesRegex(ValueError,'inconsistent parent'):
            script.validate_parent_grades(self.raw,broken,self.cfg,'raw-hash')

    def test_new_null_output_without_its_trace_cannot_be_complete(self):
        row=copy.deepcopy(self.raw['evaluations'][0]);new='size_matched_random_s42'
        for key,value in [('outputs','B'),('finished',True),('rule_abstain',False),('generation_token_ids',[7,6])]:
            row[key][new]=value
        with self.assertRaisesRegex(ValueError,'execution traces'):
            script.validate_rows([row],self.raw['evaluations'],self.names+[new],complete=True)


if __name__=='__main__':unittest.main()
