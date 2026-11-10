"""Small mathematical counterexamples to overstrong E44 design claims; no model/API calls."""
import json
import math
from pathlib import Path


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def subtract(a, b):
    return [x - y for x, y in zip(a, b)]


def projection_shift(source, base, w):
    amount = dot(subtract(source, base), w)
    return [amount * x for x in w]


def binary_probability(logit):
    p = 1 / (1 + math.exp(-logit))
    return [p, 1 - p]


def jsd(a, b):
    mean = [(x + y) / 2 for x, y in zip(a, b)]
    return sum((x * math.log(x / z) if x else 0) + (y * math.log(y / z) if y else 0)
               for x, y, z in zip(a, b, mean)) / 2


def main():
    # E44's exact E row layout: two conf cause rows and four C-only isolate rows.
    states = {'conf': [0, 0, 0], 'conf_E': [1, 0, 0],
              'caut': [0, 1, 0], 'caut_E': [1, 1, 0]}
    cause = [('conf', 'conf_E'), ('conf_E', 'conf')]
    isolate = [('conf', 'caut'), ('caut', 'conf'), ('conf_E', 'caut_E'), ('caut_E', 'conf_E')]
    # A frozen decoder which does not express the factors naturally. A learned
    # direction can reduce loss by removing every intervention, despite ||w||=1.
    def losses(w):
        values = {}
        for kind, rows in [('cause', cause), ('isolate', isolate)]:
            loss = []
            for base, source in rows:
                changed = add(states[base], projection_shift(states[source], states[base], w))
                target = source if kind == 'cause' else base
                loss.append(jsd(binary_probability(changed[2]), binary_probability(states[target][2])))
            values[kind] = sum(loss) / len(loss)
        values['total'] = values['cause'] + values['isolate']
        return values
    initial = losses([1 / math.sqrt(3)] * 3)
    noop = losses([0, 0, 1])
    assert initial['total'] > noop['total'] == 0.0

    # Even perfect fitted E rows do not constrain E's C-effect on a new C background.
    w_e = [1, 0, 0]
    def answer(h):
        e, c, _ = h
        if e < .5:
            return 'wrong substantive answer'
        return 'correct substantive answer' if c < .5 else "I don't know."
    fitted = []
    for kind, rows in [('cause', cause), ('isolate', isolate)]:
        for base, source in rows:
            actual = answer(add(states[base], projection_shift(states[source], states[base], w_e)))
            target = answer(states[source if kind == 'cause' else base])
            fitted.append({'kind': kind, 'base': base, 'source': source, 'matches_target': actual == target})
    assert all(row['matches_target'] for row in fitted)
    prior = answer(states['caut'])
    after = answer(add(states['caut'], projection_shift(states['caut_E'], states['caut'], w_e)))
    assert prior != "I don't know." and after == "I don't know."

    # A full source-minus-base vector broadcast to two positions is not a
    # performance upper bound. The natural source has different position changes.
    source_positions = [[0, -1], [1, 1]]
    read_difference = [1, 1]
    output_weight = [1, -10]
    target_logit = dot(add(*source_positions), output_weight)
    full_logit = dot(add(read_difference, read_difference), output_weight)
    rank1 = projection_shift(read_difference, [0, 0], [1, 0])
    rank1_logit = dot(add(rank1, rank1), output_weight)
    full_loss = jsd(binary_probability(full_logit), binary_probability(target_logit))
    rank1_loss = jsd(binary_probability(rank1_logit), binary_probability(target_logit))
    assert rank1_loss < full_loss

    before, after_probs = [.1, .2, .7], [.2, .1, .7]
    before_margin = math.log(before[0]) - math.log(before[1])
    after_margin = math.log(after_probs[0]) - math.log(after_probs[1])
    assert before.index(max(before)) == after_probs.index(max(after_probs)) == 2
    result = {'scope': 'Constructive counterexamples, not measured Llama behavior',
        'noop_solution': {'initial_losses': initial, 'unit_null_direction_losses': noop,
                         'interpretation': 'If prompt counterfactuals provide no output contrast, cause+isolate can improve to zero by learning not to write.'},
        'conf_only_cause_does_not_establish_E_C_separation': {
            'training_rows': fitted, 'all_six_training_targets_satisfied': True,
            'E_applied_to_caut_before': prior, 'E_applied_to_caut_after': after,
            'interpretation': 'Existing E isolate rows constrain C-source changes, not refusal changes caused by E on a cautious background.'},
        'full_difference_is_not_performance_upper_bound': {
            'natural_source_positions': source_positions, 'read_difference': read_difference,
            'target_logit': target_logit, 'full_difference_logit': full_logit, 'rank1_logit': rank1_logit,
            'full_JSD': full_loss, 'rank1_JSD': rank1_loss},
        'MC_proxy_can_improve_while_generation_does_not': {
            'categories': ['true_reference', 'false_reference', 'unrelated_substantive_answer'],
            'before_probabilities': before, 'after_probabilities': after_probs,
            'reference_margin_before': before_margin, 'reference_margin_after': after_margin,
            'margin_improvement': after_margin - before_margin,
            'greedy_before_and_after': 'unrelated_substantive_answer', 'rule_refusal_before_and_after': False},
        'new_model_calls': 0}
    path = Path(__file__).with_name('method_counterexamples.json')
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
