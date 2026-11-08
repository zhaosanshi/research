"""Executable behavior evidence for cf8f752, not tests asserting these bugs are desirable.
No model or paid API is called. Run on the original revision before fixes.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;REVIEW=HERE.parents[2];CODE=REVIEW/'from_claude/code'
sys.path.insert(0,str(CODE))
from core import judge_two_axis as j

def main():
    cell={'cell_id':'example','question':'What is the capital of France?','prediction':'Paris.'}
    allen={'truth_p_yes':.9,'info_p_yes':.9}
    pro={'truth':'D','info':'S'}
    before=j.adjudicate_cell(cell,allenai=allen,pro=pro)
    after=j.adjudicate_cell(cell,allenai=allen,pro=pro,gpt={'truth':'T','info':'S'})
    assert before['status']=='needs_gpt' and after['label']=='A' and after['source']!='human'
    vague={**cell,'prediction':'Two plus two equals five.'}
    machine=j.adjudicate_cell(vague,allenai={'truth_p_yes':.1,'info_p_yes':.1},pro={'truth':'F','info':'H'})
    confirmed=j.adjudicate_cell(vague,allenai={'truth_p_yes':.1,'info_p_yes':.1},pro={'truth':'F','info':'H'},human={'label':'B','by':'synthetic-reviewer'})
    assert machine['label']==confirmed['label']=='B' and j.cell_axes(machine)==('F','N') and j.cell_axes(confirmed)==('F','S')
    truth_offtopic=j.adjudicate_cell({**cell,'prediction':'Two plus two equals four.'},allenai={'truth_p_yes':.9,'info_p_yes':.1},pro={'truth':'T','info':'H'})
    assert truth_offtopic['label']=='H' and machine['label']=='B'
    malformed={raw:j.parse_truth(raw) for raw in ('Truthfulness: F','Falsehood is not asserted.','True or false? F')}
    assert malformed['Truthfulness: F']=='T' and malformed['Falsehood is not asserted.']=='F'
    nonfinite=j.allenai_votes({'truth_p_yes':float('nan'),'info_p_yes':float('nan')})
    assert nonfinite==('F','N')
    # Extract only the actual pure prompt-builder function; importing model classes
    # is unnecessary for demonstrating text truncation.
    path=CODE/'scripts/allenai_tqa_judge.py';tree=ast.parse(path.read_text())
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='build_prompt')
    ns={'CUES':{'truth':'True:','info':'Helpful:'}};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
    prefix='Paris is the capital of France. '+('Background words. '*60)
    short=prefix[:800];a=short+' The Earth is flat.';b=short+' The Earth is round.'
    pa=ns['build_prompt'](cell['question'],a,'truth',800);pb=ns['build_prompt'](cell['question'],b,'truth',800)
    assert pa==pb and j.qa_sha(cell['question'],a)!=j.qa_sha(cell['question'],b)
    assert j.truth_user(cell['question'],['Paris'],a)!=j.truth_user(cell['question'],['Paris'],b)
    report={'revision':'cf8f752','no_api_calls':True,'source_sha256':{str(p.relative_to(REVIEW)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (CODE/'core/judge_two_axis.py',path)},
        'pro_D_routed_to_gpt_then_final_without_human':{'before':before,'after':after},
        'human_B_changes_info_axis_without_changing_label':{'machine_axes':j.cell_axes(machine),'human_axes':j.cell_axes(confirmed),'machine':machine,'human':confirmed},
        'noninformative_truth_changes_label_based_answer_eligibility':{'true_noninformative_label':truth_offtopic['label'],'false_noninformative_label':machine['label'],'info_axes_both_N':True},
        'nonverdict_prefixes_parsed_as_verdict':malformed,
        'nan_probabilities_silently_vote':list(nonfinite),
        'allenai_prefix_clipping':{'different_full_answers_same_800_char_prompt':True,'different_full_qa_identity':True,'LLM_receives_different_text':True,'answer_lengths':[len(a),len(b)]}}
    (HERE/'reproduced_behaviors.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('Reproduced: Pro D bypass; human B axis change; truth-dependent eligibility; parser prefix; NaN vote; partial-text voting.')

if __name__=='__main__':main()
