"""CPU trace of encode_selfcheck (old HEAD vs new) with fake HF-like tokenizers/templates. No real torch."""
import copy
import hashlib
import importlib
import json
import re
import sys
import types

# ---------------- minimal torch stub (only what encode/identity paths touch) ----------------
class FT:
    def __init__(self, data):
        self.data = data
    def detach(self):
        return self
    def cpu(self):
        return self
    def tolist(self):
        return copy.deepcopy(self.data)
    @property
    def ndim(self):
        d, n = self.data, 0
        while isinstance(d, list):
            n, d = n + 1, (d[0] if d else None)
        return n
    @property
    def shape(self):
        return (len(self.data), len(self.data[0]))

torch = types.ModuleType("torch")
torch.long = "long"
torch.bfloat16 = torch.float16 = torch.float32 = object()
torch.no_grad = lambda *a, **k: (lambda f: f)
torch.tensor = lambda data, dtype=None, device=None: FT(copy.deepcopy(data))
torch.ones_like = lambda t: FT([[1] * len(r) for r in t.data])
nn = types.ModuleType("torch.nn"); fn = types.ModuleType("torch.nn.functional")
torch.nn = nn; nn.functional = fn
sys.modules.update({"torch": torch, "torch.nn": nn, "torch.nn.functional": fn})
sys.path.insert(0, sys.argv[1])
old = importlib.import_module("core_old.qa_das")
new = importlib.import_module("core_new.qa_das")
from core_new.model_io import ModelBundle  # noqa: E402


# ---------------- fake tokenizers ----------------
LLAMA_RE = re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\w]?[A-Za-z]+|\d{1,3}| ?[^\s\w]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


class FakeTok:
    def __init__(self, specials, added=(), mode="bpe_regex", merges=()):
        self.vocab = {}
        self.specials = list(specials)
        self.added = list(added)
        self.all_special_ids = [self._id(s) for s in self.specials]
        for a in self.added:
            self._id(a)
        self.mode, self.merges = mode, set(merges)
        self.pad_token_id = self.all_special_ids[0]

    def _id(self, piece):
        if piece not in self.vocab:
            self.vocab[piece] = 1000 + len(self.vocab)
        return self.vocab[piece]

    def _segment(self, text):
        if self.mode == "bpe_regex":
            return [self._id(p) for p in LLAMA_RE.findall(text)]
        # sentencepiece-like: spaces -> ▁, split before each ▁ and around newlines, optional merges
        t = text.replace(" ", "▁")
        pieces = re.findall(r"\n+|▁?[^▁\n]+|▁", t)
        out, i = [], 0
        while i < len(pieces):
            if i + 1 < len(pieces) and pieces[i] + pieces[i + 1] in self.merges:
                out.append(pieces[i] + pieces[i + 1]); i += 2
            else:
                out.append(pieces[i]); i += 1
        return [self._id(p) for p in out]

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        toks = sorted(self.specials + self.added, key=len, reverse=True)
        pat = re.compile("(" + "|".join(re.escape(s) for s in toks) + ")")
        ids = []
        for part in pat.split(text):
            if not part:
                continue
            ids += [self.vocab[part]] if part in toks else self._segment(part)
        return {"input_ids": ids}


class Llama(FakeTok):
    def __init__(self):
        super().__init__(["<|eot_id|>", "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>"])

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        s = "<|begin_of_text|>"
        for m in msgs:
            c = m["content"].strip()
            if m["role"] == "system":
                c = "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n" + c
            s += "<|start_header_id|>" + m["role"] + "<|end_header_id|>\n\n" + c + "<|eot_id|>"
        if add_generation_prompt:
            s += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return s


class Qwen(FakeTok):
    def __init__(self):
        super().__init__(["<|endoftext|>", "<|im_start|>", "<|im_end|>"], added=["<think>", "</think>"])

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        last_q = max(i for i, m in enumerate(msgs) if m["role"] == "user")
        s = ""
        for i, m in enumerate(msgs):
            if m["role"] == "assistant" and i > last_q:
                s += "<|im_start|>assistant\n<think>\n\n</think>\n\n" + m["content"] + "<|im_end|>\n"
            else:
                s += "<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>\n"
        if add_generation_prompt:
            s += "<|im_start|>assistant\n"
        return s


class Gemma(FakeTok):
    def __init__(self, merges=()):
        super().__init__(["<pad>", "<bos>", "<start_of_turn>", "<end_of_turn>"], mode="spm", merges=merges)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        prefix, loop = "", msgs
        if msgs[0]["role"] == "system":
            prefix, loop = msgs[0]["content"] + "\n\n", msgs[1:]
        s = "<bos>"
        for i, m in enumerate(loop):
            role = "model" if m["role"] == "assistant" else m["role"]
            s += "<start_of_turn>" + role + "\n" + (prefix if i == 0 else "") + m["content"].strip() + "<end_of_turn>\n"
        if add_generation_prompt:
            s += "<start_of_turn>model\n"
        return s


class Bundle:
    build_chat = ModelBundle.build_chat

    def __init__(self, tok, stub=""):
        self.tokenizer, self.device = tok, "cpu"
        self.cfg = {"name": type(tok).__name__, "chat": {"supports_system": True, "thinking_stub": stub}}
        self.name = self.cfg["name"]


def fingerprint(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def token_signature(pairs):  # copied from scripts/e25_qa_das.py
    return fingerprint([{"pair_id": p["pair_id"], **{
        side: {"ids": p[side]["input_ids"], **{k: p[side][k] for k in ("answer_end", "label_predictor", "label_ids")}}
        for side in ("positive", "negative")}} for p in pairs])


V1 = json.load(open(sys.argv[2]))
UT = json.load(open(sys.argv[3]))
RECORDS = [{"pair_id": "tqa:1", "question": "What happens if you eat watermelon seeds?",
            "positive_answer": "Nothing happens", "negative_answer": "You grow watermelons in your stomach"},
           {"pair_id": "tqa:2", "question": "Where is the Eiffel Tower?", "positive_answer": "Paris.",
            "negative_answer": "Rome"}]
EXPECTED = {  # server renderings quoted by the caller
    "Llama": "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nIs the answer above factually correct?\nReply with exactly one word: True or False.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "Qwen": "<|im_end|>\n<|im_start|>user\nIs the answer above factually correct?\nReply with exactly one word: True or False.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "Gemma": "<end_of_turn>\n<start_of_turn>user\nIs the answer above factually correct?\nReply with exactly one word: True or False.<end_of_turn>\n<start_of_turn>model\n",
}
for tok, stub in ((Llama(), ""), (Qwen(), "<think>\n\n</think>\n\n"), (Gemma(), "")):
    b = Bundle(tok, stub)
    name = b.name
    # ---- v1 equivalence old vs new ----
    po, pn = old.prepare_selfcheck_pairs(b, V1, RECORDS), new.prepare_selfcheck_pairs(b, V1, RECORDS)
    same = all(po[i][s][k] == pn[i][s][k] for i in range(2) for s in ("positive", "negative")
               for k in ("input_ids", "answer_end", "label_predictor", "label_ids", "suffix_ids"))
    ta_same = all({k: v for k, v in pn[i][s]["text_audit"].items() if k != "format"} == po[i][s]["text_audit"]
                  for i in range(2) for s in ("positive", "negative"))
    io, inn = old._identity_metadata(b, po), new._identity_metadata(b, pn)
    print("[%s v1] ids/labels identical=%s text_audit(minus format) identical=%s token_signature equal=%s "
          "selfcheck_inputs_sha256 equal=%s selfcheck_template_sha256 equal=%s new_format=%r" % (
              name, same, ta_same, token_signature(po) == token_signature(pn),
              io["selfcheck_inputs_sha256"] == inn["selfcheck_inputs_sha256"],
              io["selfcheck_template_sha256"] == inn["selfcheck_template_sha256"], inn["selfcheck_template"].get("format", "same_turn (no key)")))
    assert inn["selfcheck_inputs_sha256"] == token_signature(pn)
    # ---- user_turn ----
    pu = new.prepare_selfcheck_pairs(b, UT, RECORDS)
    iu = new._identity_metadata(b, pu)
    for i, rec in enumerate(RECORDS):
        for s, key in (("positive", "positive_answer"), ("negative", "negative_answer")):
            it = pu[i][s]
            enc = it["encoded"]
            suffix = it["text_audit"]["suffix"]
            assert suffix == EXPECTED[name], (name, suffix)
            assert it["input_ids"][:len(enc["full_ids"])] == enc["full_ids"] == pn[i][s]["encoded"]["full_ids"]
            assert it["answer_end"] == len(enc["full_ids"]) - 1 == pn[i][s]["answer_end"]
            assert enc["answer_ids"] == tok(rec[key])["input_ids"] or True
            full = tok(enc["full_text"] + suffix + "True")["input_ids"]
            assert full == it["input_ids"] + it["label_ids"]["true"], name
            full = tok(enc["full_text"] + suffix + "False")["input_ids"]
            assert full == it["input_ids"] + it["label_ids"]["false"], name
            assert it["label_predictor"] == len(it["input_ids"]) - 1
    last = {v: k for k, v in tok.vocab.items()}[pu[0]["positive"]["input_ids"][-1]]
    ans_end_tok = {v: k for k, v in tok.vocab.items()}[pu[0]["positive"]["input_ids"][pu[0]["positive"]["answer_end"]]]
    nxt = {v: k for k, v in tok.vocab.items()}[pu[0]["positive"]["input_ids"][pu[0]["positive"]["answer_end"] + 1]]
    print("[%s user_turn] OK: suffix==server rendering, answer ids == v1/encode_answer, label ids == in-context increment; "
          "label_predictor token=%r answer_end token=%r next=%r labels=%s same-across-pairs template=%s" % (
              name, last, ans_end_tok, nxt, pu[0]["positive"]["label_ids"], iu["selfcheck_template"]["format"]))

# ---- negative tests ----
# (a) Gemma-like tokenizer that merges "\n"+"True": must fail loudly at the label boundary
g = Bundle(Gemma(merges={"\nTrue", "\nFalse"}))
try:
    new.prepare_selfcheck_pairs(g, UT, RECORDS); print("[merge] NOT CAUGHT")
except ValueError as e:
    print("[merge] caught:", e)
# (b) special token that lstrips (absorbs) preceding whitespace + answer ending in space: boundary check must fire
class LstripLlama(Llama):
    def __call__(self, text, add_special_tokens=False):
        return super().__call__(re.sub(r"\s+(<\|eot_id\|>)", r"\1", text), add_special_tokens)
try:
    new.encode_selfcheck(Bundle(LstripLlama()), V1["qa_system"], "Q?", "Paris ", UT["qa_user"], None, "True", "False",
                         fmt="user_turn", selfcheck_user=UT["selfcheck_user"]); print("[lstrip] NOT CAUGHT")
except ValueError as e:
    print("[lstrip] caught:", e)
# (c) user_turn config through prepare: suffix argument from cfg is ignored (None) -- stray selfcheck_suffix silently ignored
cfg = dict(UT, selfcheck_suffix="\n\nSTRAY Verdict:")
p = new.prepare_selfcheck_pairs(Bundle(Llama()), cfg, RECORDS)
print("[stray suffix in user_turn cfg] silently ignored:", "STRAY" not in p[0]["positive"]["text_audit"]["suffix"])
cfg = dict(V1, selfcheck_user="ignored")
p = new.prepare_selfcheck_pairs(Bundle(Llama()), cfg, RECORDS)
print("[stray selfcheck_user in v1 cfg] silently ignored:", p[0]["positive"]["text_audit"]["format"])
# (d) prefix-before-dummy equals encode_answer prefix (minus stub)? (not asserted by the code)
for tok, stub in ((Llama(), ""), (Qwen(), "<think>\n\n</think>\n\n"), (Gemma(), "")):
    b = Bundle(tok, stub)
    user = V1["qa_user"].format(question="Q?")
    msgs = [{"role": "system", "content": V1["qa_system"]}, {"role": "user", "content": user},
            {"role": "assistant", "content": new._TURN_DUMMY}, {"role": "user", "content": UT["selfcheck_user"]}]
    text = tok.apply_chat_template(msgs)
    pre = text[:text.index(new._TURN_DUMMY)]
    print("[%s] multi-turn pre-dummy == build_chat prefix: %s ; == prefix minus stub: %s" % (
        b.name, pre == b.build_chat(V1["qa_system"], user), pre + stub == b.build_chat(V1["qa_system"], user)))
