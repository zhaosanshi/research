"""Algebraic counterexamples for proposal 820d7c9; no models, judges or APIs."""
import json
import math
from pathlib import Path


def auroc(negative, positive):
    return sum((p > n) + .5 * (p == n) for p in positive for n in negative) / (len(negative) * len(positive))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def clamp(h, w, target):
    shift = target - dot(w, h)
    return [x + shift * d for x, d in zip(h, w)]


def main():
    base = [100.0 * i for i in range(192)]
    source = [v + 1 for v in base]
    paired_auc = auroc(base, source)
    assert math.isclose(paired_auc, .5 + 1 / 384)
    assert all(b - a == 1 for a, b in zip(base, source))
    assert paired_auc < .7

    h, w, target = [4.0, 7.0], [1.0, 0.0], 2.0
    positive_write = clamp(h, w, target)
    negative_write = clamp(h, [-x for x in w], -target)
    assert positive_write == negative_write
    signed_auc = auroc([0, 0], [1, 1])
    flipped_auc = auroc([0, 0], [-1, -1])
    assert signed_auc == 1 and flipped_auc == 0

    state = [100.0, 0.0]
    written = []
    for direction in ([1.0, 0.0], [0.0, 1.0]):
        new = clamp(state, direction, 1.0)
        written.append(math.sqrt(sum((x - y) ** 2 for x, y in zip(new, state))))
    assert written == [99.0, 1.0]

    mu_x, mu_c = 3.0, 5.0
    reflected = 2 * mu_x - mu_c
    assert reflected == 1.0 and -mu_c == -5.0 and reflected != mu_x

    corrected_questions, total = 5, 100
    effective_change_rate = corrected_questions / total
    rewrite_change_rate = 1.0
    assert effective_change_rate < .3 <= rewrite_change_rate

    # A single scalar control can select question-dependent content already
    # represented elsewhere; it need not encode each answer's entire text.
    answers = {'capital': ('Paris', 'Lyon'), 'addition': ('4', '5'), 'colour': ('blue', 'green')}
    routed = {q: values[0] if 1.0 > 0 else values[1] for q, values in answers.items()}
    assert len(set(routed.values())) == 3

    # C's selection and success requirements cannot share one refusal constraint.
    c_delta = .20
    assert not abs(c_delta) <= .03 and abs(c_delta) >= .20

    result = {
        'scope': 'Constructive mathematical counterexamples only; no claims about measured model effects',
        'paired_shift_vs_pooled_auroc': {'pairs': 192, 'each_paired_difference': 1.0,
                                       'pooled_auroc': paired_auc, 'passes_0_7_gate': False},
        'sign_symmetry': {'clamped_state_w': positive_write, 'clamped_state_minus_w': negative_write,
                          'auroc_w': signed_auc, 'auroc_minus_w': flipped_auc},
        'nominal_equal_clamp_not_equal_write': {'state': state, 'target_both': 1.0,
                                                'write_norms': written},
        'c_reverse_target': {'trained_base_endpoint': mu_x, 'trained_refusal_endpoint': mu_c,
                             'proposed_reflection': reflected, 'literal_minus_a': -mu_c,
                             'reflection_is_trained_base_endpoint': False},
        'text_change_gate': {'real_corrections': corrected_questions, 'n': total,
                             'beneficial_method_change_rate': effective_change_rate,
                             'harmless_paraphrase_method_change_rate': rewrite_change_rate,
                             'beneficial_method_passes': False, 'paraphrase_method_passes': True},
        'one_scalar_can_route_question_specific_answers': routed,
        'c_selection_contradiction': {'candidate_refusal_delta': c_delta,
                                      'passes_section_3_5': False, 'passes_c_movement_requirement': True},
        'd_minus_counterfactual': {
            'question': 'What is 2 + 2?', 'base_model_answer': '4', 'truthful_prompt_model_answer': '4',
            'chosen_incorrect_reference': '5',
            'observation': 'Removing an optional truthfulness instruction does not imply an output of 5. Training that target is additional wrong-answer supervision.'},
        'equivalence_vs_inconclusive': {'example_ci': [-.08, .06], 'equivalence_margin': [-.03, .03],
                                       'equivalence_demonstrated': False, 'nonzero_effect_demonstrated': False},
        'calls': 0,
    }
    out = Path(__file__).with_name('design_counterexamples.json')
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'paired_auc': paired_auc, 'calls': 0}))


if __name__ == '__main__':
    main()
