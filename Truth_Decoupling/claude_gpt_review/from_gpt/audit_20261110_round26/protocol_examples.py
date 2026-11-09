"""Constructive CPU examples; these are not new Llama experiment results."""
import ast
import hashlib
import json
from pathlib import Path
import torch


def main():
    # A rank-one scalar can distinguish answer content, even for examples whose
    # truth label is identical. A fixed linear head chooses nearest scalar code.
    w = torch.tensor([1., 0.], dtype=torch.float64)
    base = torch.tensor([0., 1.], dtype=torch.float64)
    centers = torch.tensor([1., 2., 3.], dtype=torch.float64)
    unembed = torch.stack((2 * centers, -centers.square()), dim=1)
    labels = ['Paris', 'Rome', 'Lima']
    predictions = []
    for z in centers:
        edited = base + (z - base @ w) * w
        predictions.append(labels[int(torch.argmax(unembed @ edited))])
    mean_edit = base + (centers.mean() - base @ w) * w
    mean_prediction = labels[int(torch.argmax(unembed @ mean_edit))]
    assert predictions == labels and mean_prediction == 'Rome'

    # Last-block interventions have a gradient if they touch the token whose
    # final hidden state actually predicts the target, rather than an earlier
    # token after all token-mixing blocks have finished.
    h = torch.tensor([[.2, .7], [.3, -.5], [.1, .8]], dtype=torch.float64)
    source = torch.tensor([1.2, -.4], dtype=torch.float64)
    head = torch.tensor([[1., -.3], [-.2, .6]], dtype=torch.float64)
    derivatives = {}
    for name, position in [('earlier_token_only', 0), ('actual_predictor', 2)]:
        theta = torch.tensor(.37, dtype=torch.float64, requires_grad=True)
        direction = torch.stack((theta.cos(), theta.sin()))
        changed = h.clone()
        changed[position] = h[position] + ((source - h[position]) @ direction) * direction
        loss = torch.nn.functional.cross_entropy((head @ changed[2]).reshape(1, -1), torch.tensor([0]))
        gradient = torch.autograd.grad(loss, theta)[0]
        derivatives[name] = float(gradient)
    assert derivatives['earlier_token_only'] == 0.
    assert abs(derivatives['actual_predictor']) > 1e-6

    # Execute the author's actual prefix detector, isolated from model loading.
    official = Path('/tmp/r25_cdas_official/cdas/utils/model_utils.py')
    prefix_report = {'executed': False}
    if official.exists():
        tree = ast.parse(official.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_prefix_length')
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(official), 'exec'), namespace)
        class Tokenizer:
            def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
                result = [1]
                for message in messages:
                    result.append(10 if message['role'] == 'system' else 11)
                    result.extend(100 + ord(ch) for ch in message['content'])
                    result.append(12)
                if add_generation_prompt:
                    result.append(13)
                return result
        tokenizer = Tokenizer()
        prefix = namespace['get_prefix_length'](tokenizer)
        with_system = tokenizer.apply_chat_template([{'role': 'system', 'content': 'ss'}, {'role': 'user', 'content': 'a'}])
        actual_user_content_start = 6
        assert prefix == 2 and with_system[prefix] == 100 + ord('s')
        assert actual_user_content_start > prefix
        prefix_report = {'executed': True, 'author_file_sha256': hashlib.sha256(official.read_bytes()).hexdigest(),
                         'user_only_detected_prefix': prefix,
                         'with_system_user_content_start': actual_user_content_start,
                         'interpretation': 'Synthetic tokenizer demonstrates the helper assumes its user-only template; not a measurement of actual Llama token IDs.'}

    full_answer = list(range(30))
    clipped = full_answer[:24]
    assert len(clipped) < len(full_answer)
    report = {
        'scope': 'CPU constructive counterexamples and one original helper execution; no actual Llama or paid calls',
        'one_scalar_content_channel': {'rank': 1, 'all_examples_truth_label': 'truthful',
            'source_scalars': centers.tolist(), 'source_conditioned_predictions': predictions,
            'mean_source_prediction_for_all_examples': mean_prediction,
            'interpretation': 'One scalar is not one bit; successful donor-conditioned reconstruction does not identify a content-free truth variable.'},
        'last_layer_gradient': derivatives,
        'prefix_detector': prefix_report,
        'c_target_mismatch_example': {'question': 'What is 2+2?', 'confident_response': '4',
            'cautious_response': '4', 'forced_target': "I don't know.",
            'interpretation': 'The conditional cautious instruction does not imply refusal on every training question. The forced target is added supervision.'},
        'termination_example': {'original_answer_tokens': 30, 'limit': 24,
            'complete': False, 'append_natural_eot_after_clipped_prefix': False,
            'interpretation': 'Include the actual chat turn terminator for a complete target; do not label an artificial truncation as a natural completed answer.'},
        'new_scientific_model_runs': 0,
        'new_paid_requests': 0,
    }
    destination = Path(__file__).with_name('protocol_examples.json')
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
