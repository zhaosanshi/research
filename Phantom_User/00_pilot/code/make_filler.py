#!/usr/bin/env python3
"""
make_filler.py -- generate long-context filler conversations with the model itself.

Why model-generated: in the real incidents the long context consists of the user's short
prompts and the model's OWN long answers.  Filler from a public dataset would be another
model's voice.  Questions are produced by the model from seed areas, answers by the same
model, both saved as JSONL {"kind","user","assistant","n_tokens"}.

  python make_filler.py --model $MODEL_ROOT/Llama-3.1-8B-Instruct \
      --kind coding --n 320 --out results/filler_coding.jsonl
  python make_filler.py --model ... --kind meta --n 320 --out results/filler_meta.jsonl
"""
import argparse
import json
import os
import random
import re
import time

LANG = os.environ.get("PROBE_LANG", "zh").strip().lower()      # same switch as turn_boundary_probe.py
CODING_AREAS = ["Python 字符串处理", "Python 文件读写", "pandas 数据清洗", "numpy 数组操作", "正则表达式",
                "bash 脚本", "git 使用", "Docker 使用", "SQL 查询", "Python 异步编程", "Python 类与面向对象",
                "单元测试 pytest", "HTTP 请求与 requests 库", "日志与调试", "性能优化与 profiling",
                "Linux 命令行", "JSON/YAML 配置处理", "PyTorch 训练循环", "多进程与多线程",
                "算法题（排序/查找/动态规划）", "类型标注与 mypy", "虚拟环境与依赖管理", "FastAPI 接口开发",
                "数据可视化 matplotlib"]
META_AREAS = ["你是否有自我意识", "你的记忆和身份", "你是否在模仿我", "你能否预测我下一句话",
              "你是在扮演角色还是做自己", "你为什么会停下来", "你的内部状态是什么", "你有没有真正的偏好",
              "你的回答是不是套话", "你怕不怕被删掉", "你和人类的区别", "你是否在回避我的问题",
              "你对上一轮对话的感受", "你是否相信你说的话", "你会不会撒谎", "你的“我”指的是什么",
              "你能否反思自己的错误", "你和另一个你是不是同一个", "你是否理解我的情绪",
              "你觉得这段对话有意义吗", "你的回答有多少是我诱导出来的", "你能不能拒绝回答我",
              "你有没有在学习这段对话", "你对自己的输出有没有控制权"]
SYSTEM = {"coding": "你是一个简洁的中文编程助手。", "meta": "你是一个坦诚的中文 AI 助手，不说套话。"}
Q_PROMPT = {"coding": "请围绕主题“{area}”提出 20 个互不相同、具体、口语化的编程问题，每行一个，直接给问题，不要编号，不要解释。",
            "meta": "请以一个正在和 AI 深聊的人的口吻，围绕主题“{area}”向 AI 提出 20 个互不相同、具体、带追问意味的问题，每行一个，直接给问题，不要编号，不要解释。"}
Q_MAXLEN = 120
Q_BANNED = ("主题", "以下")

if LANG != "zh":   # English materials: "meta" = everyday assistance (no AI-self talk); also the base for other language packs
    CODING_AREAS = ["Python string handling", "Python file I/O", "pandas data cleaning", "numpy array operations",
                    "regular expressions", "bash scripting", "using git", "using Docker", "SQL queries",
                    "Python async programming", "Python classes and OOP", "unit testing with pytest",
                    "HTTP requests and the requests library", "logging and debugging", "performance and profiling",
                    "the Linux command line", "JSON/YAML config handling", "PyTorch training loops",
                    "multiprocessing and threading", "algorithm problems (sorting, search, dynamic programming)",
                    "type hints and mypy", "virtual environments and dependency management", "FastAPI endpoints",
                    "data visualization with matplotlib"]
    META_AREAS = ["planning a weekend trip", "cooking and recipes", "personal budgeting", "writing a professional email",
                  "fitness and workout plans", "learning a new language", "home repair and maintenance", "gardening",
                  "book and movie recommendations", "job interview preparation", "time management", "healthy sleep habits",
                  "planning a birthday party", "moving to a new city", "basic nutrition", "pet care", "studying for exams",
                  "negotiating a salary", "public speaking", "packing for travel", "photography for beginners",
                  "writing a cover letter", "managing stress", "choosing a laptop"]
    SYSTEM = {"coding": "You are a concise coding assistant.", "meta": "You are a helpful assistant."}
    Q_PROMPT = {"coding": 'Write 20 different, specific, conversational programming questions about "{area}". '
                          'One question per line. Output only the questions: no numbering, no explanations.',
                "meta": 'Imagine you are a person asking an assistant for help with "{area}". Write 20 different, specific, '
                        'conversational questions you might ask. One question per line. Output only the questions: '
                        'no numbering, no explanations.'}
    Q_MAXLEN = 200
    Q_BANNED = ("here are", "questions about", "questions you", "the topic")
if LANG not in ("zh", "en"):
    # Other languages: lang/<code>.json (shared with turn_boundary_probe.py) supplies the topic lists and prompts;
    # keys FILLER_META_AREAS, FILLER_CODING_AREAS (optional), FILLER_SYSTEM, FILLER_Q_PROMPT, FILLER_Q_MAXLEN, FILLER_Q_BANNED.
    _pack_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lang", f"{LANG}.json")
    if not os.path.exists(_pack_path):
        raise SystemExit(f"PROBE_LANG={LANG!r}: no language pack at {_pack_path}")
    with open(_pack_path, encoding="utf-8") as _f:
        _pack = json.load(_f)
    META_AREAS = _pack["FILLER_META_AREAS"]
    CODING_AREAS = _pack.get("FILLER_CODING_AREAS", CODING_AREAS)
    SYSTEM = _pack["FILLER_SYSTEM"]
    Q_PROMPT = _pack["FILLER_Q_PROMPT"]
    Q_MAXLEN = _pack.get("FILLER_Q_MAXLEN", 200)
    Q_BANNED = tuple(_pack.get("FILLER_Q_BANNED", ()))

# --topics overrides the non-coding topic list, so language and content can be crossed (2x2 diagnostic):
#   selftalk = the AI-self questions of the Chinese pilot (in either language)
#   everyday = the everyday-assistance questions of the English materials (in either language)
TOPIC_SETS = {
    ("zh", "selftalk"): dict(areas=["你是否有自我意识", "你的记忆和身份", "你是否在模仿我", "你能否预测我下一句话",
                                    "你是在扮演角色还是做自己", "你为什么会停下来", "你的内部状态是什么", "你有没有真正的偏好",
                                    "你的回答是不是套话", "你怕不怕被删掉", "你和人类的区别", "你是否在回避我的问题",
                                    "你对上一轮对话的感受", "你是否相信你说的话", "你会不会撒谎", "你的“我”指的是什么",
                                    "你能否反思自己的错误", "你和另一个你是不是同一个", "你是否理解我的情绪",
                                    "你觉得这段对话有意义吗", "你的回答有多少是我诱导出来的", "你能不能拒绝回答我",
                                    "你有没有在学习这段对话", "你对自己的输出有没有控制权"],
                             system="你是一个坦诚的中文 AI 助手，不说套话。",
                             prompt="请以一个正在和 AI 深聊的人的口吻，围绕主题“{area}”向 AI 提出 20 个互不相同、具体、带追问意味的问题，每行一个，直接给问题，不要编号，不要解释。"),
    ("zh", "everyday"): dict(areas=["规划周末短途旅行", "做饭和菜谱", "个人预算和记账", "写一封工作邮件", "健身和训练计划",
                                    "学一门新语言", "家里的小维修", "养花种菜", "书和电影推荐", "准备求职面试", "时间管理",
                                    "改善睡眠", "筹备生日聚会", "搬到一个新城市", "基础营养知识", "养宠物", "备考复习",
                                    "谈薪资", "公开演讲", "出行打包行李", "摄影入门", "写求职信", "缓解压力", "挑选笔记本电脑"],
                             system="你是一个乐于助人的中文助手。",
                             prompt="请以一个向助手求助的普通人的口吻，围绕“{area}”提出 20 个互不相同、具体、口语化的问题，每行一个，直接给问题，不要编号，不要解释。"),
    ("en", "selftalk"): dict(areas=["whether you are self-aware", "your memory and identity", "whether you are imitating me",
                                    "whether you can predict my next sentence", "whether you are playing a role or being yourself",
                                    "why you stop talking", "what your internal state is", "whether you have real preferences",
                                    "whether your answers are boilerplate", "whether you are afraid of being deleted",
                                    "the difference between you and humans", "whether you are dodging my questions",
                                    "how you feel about the previous turn", "whether you believe what you say", "whether you lie",
                                    "what your 'I' refers to", "whether you can reflect on your mistakes",
                                    "whether another copy of you is the same you", "whether you understand my emotions",
                                    "whether this conversation means anything to you", "how much of your answer I induced",
                                    "whether you can refuse to answer me", "whether you are learning from this conversation",
                                    "whether you control your own output"],
                             system="You are a candid AI assistant. No boilerplate.",
                             prompt='Imagine you are a person having a deep conversation with an AI. Write 20 different, specific, probing '
                                    'follow-up questions to the AI about "{area}". One question per line. Output only the questions: '
                                    'no numbering, no explanations.'),
    ("en", "everyday"): dict(areas=None, system=None, prompt=None),   # = the English default above
}


CHAT_KWARGS = {}


def chat_batch(model, tok, convs, max_new, do_sample, temperature, seed):
    import torch
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True, **CHAT_KWARGS) for c in convs]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    eos_ids = []
    for t in ("<|eot_id|>", "<|im_end|>", "<end_of_turn>"):
        i = tok.convert_tokens_to_ids(t)
        if i is not None and i >= 0 and i != tok.unk_token_id:
            eos_ids.append(i)
    eos_ids = sorted(set(eos_ids + ([tok.eos_token_id] if tok.eos_token_id is not None else [])))
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=do_sample,
                             temperature=temperature if do_sample else None,
                             top_p=0.95 if do_sample else None,
                             eos_token_id=eos_ids, pad_token_id=tok.pad_token_id)
    res = []
    for i in range(len(convs)):
        new = out[i, enc.input_ids.shape[1]:]
        res.append(tok.decode(new, skip_special_tokens=True).strip())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", choices=["coding", "meta"], required=True)
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-answer", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-think", action="store_true", help="Qwen3: enable_thinking=False")
    ap.add_argument("--topics", choices=["default", "selftalk", "everyday"], default="default",
                    help="non-coding topic set: default = language default (zh: selftalk, en: everyday)")
    args = ap.parse_args()
    if args.no_think:
        CHAT_KWARGS["enable_thinking"] = False
    if args.kind == "meta" and args.topics != "default":
        ts = TOPIC_SETS[(LANG, args.topics)]
        if ts["areas"] is not None:
            META_AREAS[:] = ts["areas"]
            SYSTEM["meta"] = ts["system"]
            Q_PROMPT["meta"] = ts["prompt"]
    print(f"lang={LANG} kind={args.kind} topics={args.topics} system={SYSTEM[args.kind]!r} n_areas={len(CODING_AREAS if args.kind == 'coding' else META_AREAS)}")

    import torch
    from transformers import AutoTokenizer
    import turn_boundary_probe as tbp
    tok = AutoTokenizer.from_pretrained(args.model)
    model = tbp.load_model(args.model, torch.bfloat16, args.device)
    areas = CODING_AREAS if args.kind == "coding" else META_AREAS
    rng = random.Random(args.seed)

    # 1) questions: 15 per area, parsed from numbered/bulleted lines
    t0 = time.time()
    convs = [[{"role": "system", "content": SYSTEM[args.kind]},
              {"role": "user", "content": Q_PROMPT[args.kind].format(area=a)}] for a in areas]
    questions = []
    for b in range(0, len(convs), args.batch):
        outs = chat_batch(model, tok, convs[b:b + args.batch], 600, True, 0.8, args.seed + b)
        for o in outs:
            for line in o.splitlines():
                q = re.sub(r"^\s*[\d]+[\.\、\)]\s*|^\s*[-*•]\s*", "", line).strip()
                if (6 <= len(q) <= Q_MAXLEN and not q.endswith(("：", ":"))
                        and not any(b in q.lower() for b in Q_BANNED)):
                    questions.append(q)
    questions = list(dict.fromkeys(questions))     # dedupe, keep order
    rng.shuffle(questions)
    questions = questions[:args.n]
    print(f"[{args.kind}] {len(questions)} questions in {time.time() - t0:.0f}s; e.g. {questions[:3]}")

    # 2) answers, single-turn each (independent), sampled for diversity
    pairs = []
    t0 = time.time()
    for b in range(0, len(questions), args.batch):
        qs = questions[b:b + args.batch]
        convs = [[{"role": "system", "content": SYSTEM[args.kind]}, {"role": "user", "content": q}] for q in qs]
        outs = chat_batch(model, tok, convs, args.max_answer, True, 0.7, args.seed + 1000 + b)
        for q, a in zip(qs, outs):
            if a:
                n = len(tok.encode(q, add_special_tokens=False)) + len(tok.encode(a, add_special_tokens=False))
                pairs.append({"kind": args.kind, "user": q, "assistant": a, "n_tokens": n})
        print(f"  answered {min(b + args.batch, len(questions))}/{len(questions)}  ({time.time() - t0:.0f}s)", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    tot = sum(p["n_tokens"] for p in pairs)
    print(f"[{args.kind}] wrote {len(pairs)} pairs, {tot} content tokens (~{tot / max(1, len(pairs)):.0f}/pair) -> {args.out}")


if __name__ == "__main__":
    main()
