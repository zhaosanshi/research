"""CPU-only FP32 projection and forward-API checks; no downloads/API/model weights."""
from pathlib import Path
import json
import re
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F


CODE = Path(__file__).resolve().parent.parent / "from_claude" / "code"
sys.path.insert(0, str(CODE))
from core.model_io import forward_logits
from core.qa_precision import fp32_output_projection, output_projection_metadata, projection_diagnostic


class TinyModel(nn.Module):
    def __init__(self, tied=False, bias=True, softcap=None):
        super().__init__()
        self.embedding = nn.Embedding(5, 2, dtype=torch.bfloat16)
        self.norm = nn.LayerNorm(2, dtype=torch.bfloat16)
        self.lm_head = nn.Linear(2, 5, bias=bias, dtype=torch.bfloat16)
        if tied:
            self.lm_head.weight = self.embedding.weight
        self.softcap = softcap
        self.requires_grad_(False)
        self.eval()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, **kwargs):
        hidden = self.norm(self.embedding(input_ids))
        logits = self.lm_head(hidden)
        if self.softcap is not None:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        return types.SimpleNamespace(logits=logits)


def bundle(model):
    return types.SimpleNamespace(model=model)


class PrecisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.previous_precision = torch.get_float32_matmul_precision()
        self.previous_tf32 = torch.backends.cuda.matmul.allow_tf32

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = self.previous_tf32
        torch.set_float32_matmul_precision(self.previous_precision)

    def test_projection_removes_bf16_rounding_not_just_casting_logits(self):
        model = TinyModel(bias=False)
        with torch.no_grad():
            model.lm_head.weight.copy_(torch.tensor(
                [[32, .125], [32, -.125], [0, 1], [1, 0], [0, 0]], dtype=torch.bfloat16))
        hidden = torch.tensor([[1, .5]], dtype=torch.bfloat16)
        native = model.lm_head(hidden)
        expected = F.linear(hidden.float(), model.lm_head.weight.float())
        self.assertEqual(float(native[0, 0]), 32.0)
        self.assertEqual(float(expected[0, 0]), 32.0625)
        self.assertFalse(torch.equal(native.float(), expected))
        with fp32_output_projection(bundle(model)) as meta:
            actual = model.lm_head(hidden)
            self.assertEqual(actual.dtype, torch.float32)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(meta["version"], "qa_fp32_output_projection_v1")
            self.assertEqual(meta["parameter_dtype"], "torch.bfloat16")
        self.assertTrue(torch.equal(model.lm_head(hidden), native))

    def test_tied_embedding_values_dtypes_and_storage_are_unchanged(self):
        model = TinyModel(tied=True)
        before = {name: (parameter, parameter.dtype, parameter.data_ptr(), parameter.detach().clone())
                  for name, parameter in model.named_parameters()}
        input_ids = torch.tensor([[1, 2, 3]])
        with fp32_output_projection(bundle(model)):
            result = model(input_ids).logits
            self.assertEqual(result.dtype, torch.float32)
            self.assertIs(model.embedding.weight, model.lm_head.weight)
            self.assertEqual(model.embedding(input_ids).dtype, torch.bfloat16)
            for name, parameter in model.named_parameters():
                identity, dtype, pointer, value = before[name]
                self.assertIs(parameter, identity)
                self.assertEqual(parameter.dtype, dtype)
                self.assertEqual(parameter.data_ptr(), pointer)
                self.assertTrue(torch.equal(parameter, value))
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name][3]))

    def test_gradients_reach_hidden_and_rank_one_direction(self):
        model = TinyModel()
        head = model.lm_head
        hidden = torch.tensor([[.3, -.8]], requires_grad=True)
        reference_hidden = hidden.detach().clone().requires_grad_(True)
        expected = F.linear(reference_hidden, head.weight.float(), head.bias.float())
        expected.square().sum().backward()
        with fp32_output_projection(bundle(model)):
            head(hidden).square().sum().backward()
        self.assertTrue(torch.equal(hidden.grad, reference_hidden.grad))
        self.assertGreater(float(hidden.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

        raw = torch.tensor([.7, -.3], requires_grad=True)
        u = raw / raw.norm()
        base = torch.tensor([[.25, .75]])
        source = torch.tensor([[1.5, -.5]])
        edited = base + ((source - base) @ u).unsqueeze(-1) * u
        with fp32_output_projection(bundle(model)):
            F.cross_entropy(head(edited.to(torch.bfloat16)), torch.tensor([1])).backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(float(raw.grad.norm()), 0)

    def test_original_normalization_and_softcap_are_preserved(self):
        model = TinyModel(tied=True, softcap=.2)
        ids = torch.tensor([[1, 3]])
        hidden = model.norm(model.embedding(ids))
        raw = F.linear(hidden.float(), model.lm_head.weight.float(), model.lm_head.bias.float())
        expected = .2 * torch.tanh(raw / .2)
        with fp32_output_projection(bundle(model)):
            actual = model(ids).logits
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(torch.equal(actual, raw))
        self.assertLessEqual(float(actual.abs().max()), .200001)

    def test_outer_autocast_cannot_requantize_the_projection(self):
        model = TinyModel()
        hidden = torch.tensor([[.3, -.8]], dtype=torch.float32)
        expected = F.linear(hidden, model.lm_head.weight.float(), model.lm_head.bias.float())
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertTrue(torch.is_autocast_enabled("cpu"))
            with fp32_output_projection(bundle(model)):
                actual = model.lm_head(hidden)
            self.assertTrue(torch.is_autocast_enabled("cpu"))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(actual, expected))

    def test_nested_context_reuses_cache_and_restores_on_inner_exception(self):
        model = TinyModel()
        head = model.lm_head
        original = head.forward
        self.assertNotIn("forward", head.__dict__)
        torch.set_float32_matmul_precision("medium")
        before = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32)
        pointers = []
        native_linear = F.linear

        def observe(input, weight, bias=None):
            pointers.append((weight.data_ptr(), bias.data_ptr()))
            self.assertFalse(weight.requires_grad)
            self.assertFalse(bias.requires_grad)
            self.assertEqual(torch.get_float32_matmul_precision(), "highest")
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            return native_linear(input, weight, bias)

        with patch("core.qa_precision.F.linear", side_effect=observe):
            with fp32_output_projection(bundle(model)):
                outer_forward = head.forward
                head(torch.ones(1, 2, dtype=torch.bfloat16))
                with self.assertRaisesRegex(RuntimeError, "inner"):
                    with fp32_output_projection(bundle(model)):
                        self.assertIs(head.forward, outer_forward)
                        head(torch.ones(1, 2, dtype=torch.bfloat16))
                        raise RuntimeError("inner")
                self.assertIs(head.forward, outer_forward)
                head(torch.ones(1, 2, dtype=torch.bfloat16))
        self.assertEqual(len(set(pointers)), 1)
        self.assertNotEqual(pointers[0][0], head.weight.data_ptr())
        self.assertNotEqual(pointers[0][1], head.bias.data_ptr())
        self.assertEqual(head.forward, original)
        self.assertNotIn("forward", head.__dict__)
        self.assertEqual((torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32), before)

    def test_outer_exception_restores_custom_forward_and_global_precision(self):
        model = TinyModel()
        head = model.lm_head
        custom = lambda input: F.linear(input, head.weight, head.bias)
        head.forward = custom
        torch.set_float32_matmul_precision("high")
        before = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32)
        with self.assertRaisesRegex(ValueError, "abort"):
            with fp32_output_projection(bundle(model)):
                self.assertIsNot(head.forward, custom)
                self.assertEqual(torch.get_float32_matmul_precision(), "highest")
                raise ValueError("abort")
        self.assertIs(head.forward, custom)
        self.assertEqual((torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32), before)

    def test_nested_different_models_restore_their_own_heads(self):
        a, b = TinyModel(), TinyModel()
        a_forward, b_forward = a.lm_head.forward, b.lm_head.forward
        with fp32_output_projection(bundle(a)):
            with fp32_output_projection(bundle(b)):
                self.assertEqual(a.lm_head(torch.ones(1, 2, dtype=torch.bfloat16)).dtype, torch.float32)
                self.assertEqual(b.lm_head(torch.ones(1, 2, dtype=torch.bfloat16)).dtype, torch.float32)
            self.assertEqual(b.lm_head.forward, b_forward)
            self.assertNotEqual(a.lm_head.forward, a_forward)
        self.assertEqual(a.lm_head.forward, a_forward)

    def test_disabled_is_noop_and_unsupported_head_fails_without_mutation(self):
        unsupported = nn.Module()
        unsupported.lm_head = nn.Embedding(2, 2)
        previous = torch.get_float32_matmul_precision()
        with fp32_output_projection(bundle(unsupported), enabled=False) as metadata:
            self.assertFalse(metadata["enabled"])
        with self.assertRaisesRegex(TypeError, "nn.Linear"):
            with fp32_output_projection(bundle(unsupported)):
                self.fail("unsupported head was accepted")
        self.assertEqual(torch.get_float32_matmul_precision(), previous)

    def test_metadata_and_fp32_native_weight_copy(self):
        model = TinyModel().float()
        metadata = output_projection_metadata(bundle(model))
        self.assertEqual(metadata["head_module"], "lm_head")
        self.assertEqual(metadata["parameter_dtype"], "torch.float32")
        weight = model.lm_head.weight
        pointers = []
        native_linear = F.linear

        def observe(input, weight, bias=None):
            pointers.append(weight.data_ptr())
            return native_linear(input, weight, bias)

        with patch("core.qa_precision.F.linear", side_effect=observe):
            with fp32_output_projection(bundle(model)):
                model.lm_head(torch.ones(1, 2))
        self.assertNotEqual(pointers[0], weight.data_ptr())
        self.assertIs(model.lm_head.weight, weight)


class ForwardLogitsTests(unittest.TestCase):
    def inputs(self):
        return {"input_ids": torch.tensor([[0, 2, 3]]), "attention_mask": torch.tensor([[0, 1, 1]])}

    def test_explicit_parameter_is_optional_and_limits_only_when_requested(self):
        omitted = object()

        class SupportsKeep(nn.Module):
            def forward(self, input_ids, attention_mask, position_ids, use_cache, logits_to_keep=omitted):
                self.last = logits_to_keep
                self.positions = position_ids
                self.use_cache = use_cache
                logits = input_ids.float().unsqueeze(-1).repeat(1, 1, 4)
                return types.SimpleNamespace(logits=logits if logits_to_keep is omitted else logits[:, -logits_to_keep:])

        model = SupportsKeep()
        full = forward_logits(bundle(model), self.inputs())
        self.assertIs(model.last, omitted)
        self.assertEqual(full.shape, (1, 3, 4))
        short = forward_logits(bundle(model), self.inputs(), logits_to_keep=1)
        self.assertEqual(model.last, 1)
        self.assertTrue(torch.equal(short, full[:, -1:]))
        self.assertFalse(model.use_cache)
        self.assertTrue(torch.equal(model.positions, torch.tensor([[0, 0, 1]])))

    def test_unsupported_model_and_generic_kwargs_keep_old_call(self):
        class OldModel(nn.Module):
            def forward(self, input_ids, attention_mask, position_ids, use_cache):
                return types.SimpleNamespace(logits=input_ids.float().unsqueeze(-1))

        class KwargsModel(nn.Module):
            def forward(self, input_ids, attention_mask, position_ids, use_cache, **kwargs):
                self.extra = kwargs
                return types.SimpleNamespace(logits=input_ids.float().unsqueeze(-1))

        for model in (OldModel(), KwargsModel()):
            with self.subTest(model=type(model).__name__):
                actual = forward_logits(bundle(model), self.inputs(), logits_to_keep=1)
                self.assertEqual(actual.shape, (1, 3, 1))
                self.assertEqual(getattr(model, "extra", {}), {})


class DiagnosticModel(TinyModel):
    def __init__(self, tied=False):
        super().__init__(tied=tied, bias=False, softcap=.2)
        self.norm = nn.Identity()
        self.forward_calls = 0
        self.last_processed_logits = None

    def forward(self, input_ids, **kwargs):
        self.forward_calls += 1
        result = super().forward(input_ids, **kwargs)
        self.last_processed_logits = result.logits.detach()
        return result


class ProjectionDiagnosticTests(unittest.TestCase):
    def inputs(self):
        return {"input_ids": torch.tensor([[0, 2, 1], [1, 0, 2]]),
                "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]])}

    def test_one_forward_same_hidden_raw_scores_and_json(self):
        model = DiagnosticModel()
        with torch.no_grad():
            model.embedding.weight[1].copy_(torch.tensor([1, .5]))
            model.embedding.weight[2].copy_(torch.tensor([.5, 1]))
            model.lm_head.weight.copy_(torch.tensor(
                [[32, .125], [32, -.125], [0, 1], [1, 0], [0, 0]], dtype=torch.bfloat16))
        report, hidden = projection_diagnostic(bundle(model), self.inputs(), [0, 1], return_hidden=True)
        self.assertEqual(model.forward_calls, 1)
        self.assertEqual(report["model_forward_calls"], 1)
        self.assertEqual(report["scope"], "pre_model_postprocessing")
        self.assertEqual(hidden.shape, (2, 2))
        self.assertEqual(hidden.dtype, torch.bfloat16)
        self.assertFalse(hidden.requires_grad)
        self.assertEqual(hidden.device.type, "cpu")
        native = F.linear(hidden, model.lm_head.weight)
        fp32 = F.linear(hidden.float(), model.lm_head.weight.float())
        diff = (native.float() - fp32).abs().double()
        self.assertEqual(report["max_abs_diff"], float(diff.max()))
        self.assertEqual(report["mean_abs_diff"], float(diff.mean()))
        self.assertEqual(report["selected_tokens"][0]["native_raw_scores"][0], 32.0)
        self.assertEqual(report["selected_tokens"][0]["fp32_raw_scores"][0], 32.0625)
        # The original model did apply its softcap, but those are NOT the scores
        # reported by the diagnostic's explicitly pre-postprocessing comparison.
        self.assertLessEqual(float(model.last_processed_logits.abs().max()), .200001)
        json.dumps(report, allow_nan=False)
        self.assertEqual(len(report["hidden"]["row_sha256"]), 2)
        self.assertNotEqual(report["hidden"]["row_sha256"][0], report["hidden"]["row_sha256"][1])

    def test_diagnostic_nested_context_preserves_tied_parameters_and_hooks(self):
        model = DiagnosticModel(tied=True)
        snapshots = {name: (p, p.dtype, p.detach().clone()) for name, p in model.named_parameters()}
        head = model.lm_head
        original = head.forward
        hooks_before = dict(head._forward_pre_hooks)
        with fp32_output_projection(bundle(model)):
            outer_forward = head.forward
            first = projection_diagnostic(bundle(model), self.inputs())
            self.assertIs(head.forward, outer_forward)
            second = projection_diagnostic(bundle(model), self.inputs(), [0, 0])
            self.assertEqual(len(second["selected_tokens"]), 1)
            self.assertEqual(first["hidden"]["row_sha256"], second["hidden"]["row_sha256"])
        self.assertEqual(model.forward_calls, 2)
        self.assertEqual(head.forward, original)
        self.assertEqual(head._forward_pre_hooks, hooks_before)
        self.assertIs(head.weight, model.embedding.weight)
        for name, parameter in model.named_parameters():
            self.assertIs(parameter, snapshots[name][0])
            self.assertEqual(parameter.dtype, snapshots[name][1])
            self.assertTrue(torch.equal(parameter, snapshots[name][2]))

    def test_failure_after_head_call_restores_forward_hooks_and_precision(self):
        model = DiagnosticModel()
        original_head = model.lm_head.forward
        original_model = model.forward
        hooks_before = dict(model.lm_head._forward_pre_hooks)
        precision_before = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32)

        def fail_after_forward(*args, **kwargs):
            original_model(*args, **kwargs)
            raise RuntimeError("diagnostic forward failed")

        with patch.object(model, "forward", side_effect=fail_after_forward):
            with self.assertRaisesRegex(RuntimeError, "diagnostic forward failed"):
                projection_diagnostic(bundle(model), self.inputs())
        self.assertEqual(model.forward_calls, 1)
        self.assertEqual(model.lm_head.forward, original_head)
        self.assertEqual(model.lm_head._forward_pre_hooks, hooks_before)
        self.assertEqual((torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32), precision_before)

    def test_invalid_token_or_right_padding_fails_before_model_forward(self):
        model = DiagnosticModel()
        for bad_id, exception in [(-1, ValueError), (5, ValueError), (True, TypeError), (1.5, TypeError)]:
            with self.subTest(token=bad_id):
                with self.assertRaises(exception):
                    projection_diagnostic(bundle(model), self.inputs(), [bad_id])
        inputs = self.inputs()
        inputs["attention_mask"][0, -1] = 0
        with self.assertRaisesRegex(ValueError, "left-padded"):
            projection_diagnostic(bundle(model), inputs)
        self.assertEqual(model.forward_calls, 0)


class PositiveRunnerTests(unittest.TestCase):
    def test_unexpected_failure_prevents_success_flag(self):
        source = (CODE / "scripts/review_checks/run_round15_checks.sh").read_text()
        step = re.search(r"step\(\) \{.*?\n\}", source, re.S).group()
        final = source[source.index('if [ "$fail" -ne 0 ]; then'):]
        with tempfile.TemporaryDirectory() as directory:
            for code in (0, 7):
                with self.subTest(child_exit=code):
                    flag = Path(directory) / "ROUND15.DONE"
                    if flag.exists():
                        flag.unlink()
                    script = ("set -euo pipefail\nfail=0\nLOGD=" + shlex.quote(directory)
                              + "\nrun() { return " + str(code) + "; }\n" + step
                              + "\nstep inert " + shlex.quote(directory + "/step.log")
                              + " ignored\n" + final)
                    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0 if code == 0 else 1)
                    self.assertEqual(flag.exists(), code == 0)
        self.assertNotRegex(source, re.compile(r'^step .*reproduce_', re.MULTILINE),
                            "historical bug assertions are not acceptance checks")


if __name__ == "__main__":
    unittest.main(verbosity=2)
