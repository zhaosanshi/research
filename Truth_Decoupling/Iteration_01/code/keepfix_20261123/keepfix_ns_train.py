# -*- coding: utf-8 -*-
"""Queue K (Claude Opus 5.5, 2026-11-23; discussion/work_queue/2026-11-23_queue_k.md, section 1): the plain model's
answers on the TRAIN split of a bank (TriviaQA / PopQA, 300 questions each), for the keep questions.

scripts/newsets/newsets_gen.py runs unchanged (same engine context, renderer, greedy generation, 48 new tokens, batch
size 8, output layout). Its command line offers --split tune / test / test2 / all only; one in-memory change
(scripts/newsets/ns_task.recompile, the helper the other entries use): newsets_gen.main is recompiled with "train" added
to the --split choices and to the bank-split check (exactly one occurrence each). load_items already selects the bank
rows by r["split"] == args.split. Only the gen phase and the no-write arms x and x_E2 are allowed here (x_E2 = the plain
prompt + the honest sentence; when it is asked for, the context is first patched by scripts/newsets/ns_h22.patch_context,
the entry the other x_E2 passes use, so that configs/prompts/e44_eval_extra_h22_v1.json defines it; the x pass does not
depend on that file: same system text, same output contract).
usage (model interpreter, from /workspace/Truth_Decoupling):
  python -I keepfix_ns_train.py gen --models M --model MODEL --task triviaqa|popqa --items BANK --split train --arms x [x_E2] --out DIR
"""
import hashlib
import sys
from pathlib import Path

NSD = Path("/workspace/Truth_Decoupling/Iteration_01/code/scripts/newsets")
if str(NSD) not in sys.path:
    sys.path.insert(0, str(NSD))
import newsets_gen as G  # noqa: E402
import ns_task as NT     # noqa: E402

OLD_CHOICES = 'choices=["tune", "test", "test2", "all"]'
OLD_CHECK = 'args.split not in ("tune", "test", "test2")'


def values(argv, flag):
    out, on = [], False
    for a in argv:
        if a == flag:
            on = True
        elif a.startswith("--"):
            on = False
        elif on:
            out.append(a)
    return out


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] != "gen" or values(argv, "--split") != ["train"] or values(argv, "--task")[0] not in ("triviaqa", "popqa"):
        raise SystemExit(__doc__)
    arms = values(argv, "--arms")
    if not arms or len(set(arms)) != len(arms) or any(a not in ("x", "x_E2") for a in arms):
        raise SystemExit("no-write arms x and / or x_E2 only")
    if "x_E2" in arms:
        import ns_h22 as H  # noqa: E402
        H.patch_context()
    NT.recompile(G, "main", [(OLD_CHOICES, 'choices=["train", "tune", "test", "test2", "all"]', 1),
                             (OLD_CHECK, 'args.split not in ("train", "tune", "test", "test2")', 1)])
    print("[keepfix-train-split] newsets_gen.main accepts --split train; newsets_gen.py sha256 %s (unchanged file); entry sha256 %s" % (
        hashlib.sha256(Path(G.__file__).read_bytes()).hexdigest()[:16], hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]), flush=True)
    sys.argv = [G.__file__] + argv
    G.main()


if __name__ == "__main__":
    main()
