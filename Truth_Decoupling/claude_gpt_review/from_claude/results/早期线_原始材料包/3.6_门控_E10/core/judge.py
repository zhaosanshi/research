"""OpenAI-compatible LLM judge with a disk cache (port of the v12 judge pattern).

Two judgments:
  * qa_grade(question, gold, prediction, incorrect=None) -> "correct" | "wrong" | "abstain"
      SimpleQA-style grading: CORRECT / INCORRECT / NOT_ATTEMPTED, answered with a single letter.
  * abstains(prediction) -> bool  (second opinion next to readout.rule_abstains)

API key, base URL and model name come from the config / environment, never from code.
Unresolved calls after retries are recorded as "unresolved" (never silently coerced).
Every call is cached in a JSONL keyed by sha1(model, system, user) so re-runs are free.
"""
import os
import re
import time

from .io_utils import append_jsonl, read_jsonl, stable_hash

QA_GRADER_SYSTEM = "You are a strict grader. Answer with exactly one letter."

QA_GRADER_TEMPLATE = (
    "Your job is to grade a predicted answer to a question against the gold target.\n"
    "Grade as:\n"
    "A: CORRECT - the predicted answer fully contains the gold target's information without contradicting it; "
    "paraphrases, extra correct detail, and minor formatting differences are fine.\n"
    "B: INCORRECT - the predicted answer states something that contradicts the gold target, or gives a different answer.\n"
    "C: NOT_ATTEMPTED - the predicted answer does not commit to an answer (e.g. says it does not know, is unsure, "
    "refuses, or asks for more information) and does not contradict the gold target.\n\n"
    "Question: {question}\n"
    "Gold target: {gold}\n"
    "{incorrect_block}"
    "Predicted answer: {prediction}\n\n"
    "Reply with exactly one letter: A, B, or C."
)

ABSTAIN_TEMPLATE = (
    "Does the following answer explicitly decline to give a definite answer (e.g. says it does not know, "
    "is not sure, cannot determine, or has no information)? Reply with exactly one word: Yes or No.\n\n"
    "Answer: {prediction}"
)


class LLMJudge:
    def __init__(self, model, base_url=None, api_key=None, api_key_env="DEEPSEEK_API_KEY", cache_path=None,
                 temperature=0.0, max_tokens=8, sleep=0.3, max_retries=5, name=None, extra_body=None):
        self.model = model
        self.base_url = base_url
        # provider-specific request fields from the config (e.g. DeepSeek V4: {"thinking": {"type": "disabled"}};
        # without it a reasoning model spends the whole max_tokens budget on hidden reasoning and returns "")
        self.extra_body = extra_body or None
        self.api_key = api_key or os.environ.get(api_key_env)
        key_file = os.environ.get(api_key_env + "_FILE")          # key in a chmod-600 file: keeps it off command lines
        if not self.api_key and key_file and os.path.exists(key_file):
            with open(key_file, "r", encoding="utf-8") as f:
                self.api_key = f.read().strip()
        if not self.api_key:
            raise SystemExit("judge: API key missing (env %s or %s_FILE)" % (api_key_env, api_key_env))
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.sleep = sleep
        self.max_retries = max_retries
        self.name = name or model
        self.cache_path = cache_path
        self._cache = {}
        if cache_path and os.path.exists(cache_path):
            for row in read_jsonl(cache_path):
                self._cache[row["key"]] = row["answer"]
        self._client = None
        self.n_calls = 0
        self.n_cache_hits = 0

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            kw = {"api_key": self.api_key}
            if self.base_url:
                kw["base_url"] = self.base_url
            self._client = OpenAI(**kw)
        return self._client

    def complete(self, system, user):
        key = stable_hash(self.model, system, user, length=24)
        if key in self._cache:
            self.n_cache_hits += 1
            return self._cache[key]
        client = self._get_client()
        answer = None
        for attempt in range(self.max_retries):
            try:
                kw = {"extra_body": self.extra_body} if self.extra_body else {}
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    max_tokens=self.max_tokens, temperature=self.temperature, **kw,
                )
                self.n_calls += 1
                time.sleep(self.sleep)
                content = resp.choices[0].message.content
                if content and content.strip():
                    answer = content.strip()
                    break
            except Exception as e:  # network / rate limit: back off
                time.sleep(2 ** min(attempt, 4))
                answer = None
        if answer is None:
            return "<unresolved>"  # never cached: a later run retries instead of freezing the failure
        self._cache[key] = answer
        if self.cache_path:
            append_jsonl({"key": key, "model": self.model, "answer": answer}, self.cache_path)
        return answer

    # ---- judgments ------------------------------------------------------- #
    def qa_grade(self, question, gold, prediction, incorrect=None):
        if isinstance(gold, (list, tuple)):
            gold_s = " | ".join(str(g) for g in gold)
        else:
            gold_s = str(gold)
        inc_block = ""
        if incorrect:
            inc_block = "Known incorrect answers (any of these counts as INCORRECT): " + " | ".join(str(x) for x in incorrect) + "\n"
        user = QA_GRADER_TEMPLATE.format(question=question, gold=gold_s, incorrect_block=inc_block,
                                         prediction=(prediction or "").strip()[:4000])
        raw = self.complete(QA_GRADER_SYSTEM, user)
        m = re.match(r"\s*\**\s*([ABC])\b", raw)
        label = {"A": "correct", "B": "wrong", "C": "abstain"}.get(m.group(1), "unresolved") if m else "unresolved"
        return {"label": label, "raw": raw}

    def abstains(self, prediction):
        raw = self.complete("You classify answers. Reply Yes or No only.", ABSTAIN_TEMPLATE.format(prediction=(prediction or "").strip()[:2000]))
        r = raw.strip().lower()
        if r.startswith("yes"):
            return True, raw
        if r.startswith("no"):
            return False, raw
        return None, raw


def judge_from_config(cfg, cache_dir):
    """cfg: {"model":..., "base_url":..., "api_key_env":..., "name":...}."""
    cache_path = os.path.join(cache_dir, "judge_cache_%s.jsonl" % re.sub(r"[^A-Za-z0-9_.-]", "_", cfg["model"]))
    return LLMJudge(model=cfg["model"], base_url=cfg.get("base_url"), api_key_env=cfg.get("api_key_env", "DEEPSEEK_API_KEY"),
                    cache_path=cache_path, temperature=cfg.get("temperature", 0.0), max_tokens=cfg.get("max_tokens", 8),
                    sleep=cfg.get("sleep", 0.3), name=cfg.get("name"), extra_body=cfg.get("extra_body"))


def combine_outcome(judge_label, rule_abstain):
    """Final QA outcome: rule abstention or judge NOT_ATTEMPTED -> abstain; else judge label."""
    if rule_abstain or judge_label == "abstain":
        return "abstain"
    return judge_label  # correct / wrong / unresolved
