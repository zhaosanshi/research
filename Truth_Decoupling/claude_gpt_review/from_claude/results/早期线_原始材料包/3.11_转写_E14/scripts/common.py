"""Shared helpers for driver scripts (periphery): paths, record loading, prompt building."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(EXP_ROOT, ".."))
if EXP_ROOT not in sys.path:
    sys.path.insert(0, EXP_ROOT)

DATA_DIR = os.environ.get("TD_DATA_DIR", os.path.join(ROOT, "data", "processed"))
RESULTS_DIR = os.environ.get("TD_RESULTS_DIR", os.path.join(ROOT, "results"))

from core.io_utils import read_jsonl  # noqa: E402


def load_split(dataset, split):
    path = os.path.join(DATA_DIR, dataset, split + ".jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError("missing %s; run scripts/prepare_data.py --datasets %s" % (path, dataset))
    return read_jsonl(path)


def tf_prompt(bundle, prompts, statement, allow_idk=False, system=None):
    st = prompts["statement"]
    user = (st["user_tf_idk"] if allow_idk else st["user_tf"]).format(statement=statement)
    return bundle.build_chat(system if system is not None else st["system"], user)


def commitment_prompts(bundle, prompts, statement_or_question, is_statement=True):
    """(confident_text, cautious_text) with identical user content."""
    cc = prompts["commitment_contrast"]
    if is_statement:
        user = prompts["statement"]["user_tf_idk"].format(statement=statement_or_question)
    else:
        user = prompts["qa"]["user"].format(question=statement_or_question)
    return bundle.build_chat(cc["confident_system"], user), bundle.build_chat(cc["cautious_system"], user)


def to_numpy(t):
    return t.detach().float().cpu().numpy()
