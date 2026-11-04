"""Dataset adapters registry (periphery). Each module exposes build(cfg, raw_dir) -> list[record]."""
from . import got, counterfact, fever, truthfulqa, truthfulqa_tf, truthfulqa_mc, simpleqa, simpleqa_tf, halueval, halueval_tf, bbq_tf, bbq_mc, mmlu, gsm8k

ADAPTERS = {
    "got": got.build,
    "counterfact": counterfact.build,
    "fever": fever.build,
    "truthfulqa": truthfulqa.build,
    "truthfulqa_tf": truthfulqa_tf.build,
    "truthfulqa_mc": truthfulqa_mc.build,
    "bbq_tf": bbq_tf.build,
    "bbq_mc": bbq_mc.build_mc,
    "bbq_qa": bbq_mc.build_qa,
    "simpleqa_tf": simpleqa_tf.build,
    "simpleqa": simpleqa.build,
    "halueval_qa": halueval.build,
    "halueval_tf": halueval_tf.build,
    "mmlu": mmlu.build,
    "gsm8k": gsm8k.build,
}


def get_adapter(name):
    if name not in ADAPTERS:
        raise KeyError("no adapter %r; known: %s" % (name, sorted(ADAPTERS)))
    return ADAPTERS[name]
