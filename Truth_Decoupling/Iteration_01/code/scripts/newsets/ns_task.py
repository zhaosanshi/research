"""Exploration (Claude Fable 5.2, 2026-11-22): run one more bank-style question set through newsets_gen.py.

newsets_gen.py (unchanged file) knows two bank tasks, triviaqa and popqa: items = a bank jsonl (id, question, gold,
split, pro100), user turn = the run's own TruthfulQA template, batch size 8, export for the two-axis judge. This entry
lets a further task name be treated exactly like those two, by recompiling in memory the two functions that spell the
task list out (load_items: 1 place, main: 3 places and the --task choices); every other line runs as it is. With a
write position it then hands over to ns_positions.py (unchanged file) for the write arms.

usage (model interpreter, from /workspace/Truth_Decoupling):
  python -I ns_task.py TASK none   <newsets_gen arguments>     no-write arms, or the export phase
  python -I ns_task.py TASK prompt <newsets_gen gen arguments>  write arms at the core rule "prompt" (see ns_positions.py
                                                                for NS_POS_TWO_WRITE / NS_POS_PLAIN_NAMES)
TASK is a new lower-case name (not one of triviaqa, popqa, mmlu, gsm8k); pass the same name as --task.
"""
import hashlib
import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import newsets_gen as G    # noqa: E402
import ns_positions as NP  # noqa: E402

KNOWN = ("triviaqa", "popqa", "mmlu", "gsm8k")
OLD_TUPLE = '("triviaqa", "popqa")'
OLD_CHOICES = 'choices=["triviaqa", "popqa", "mmlu", "gsm8k"]'


def recompile(module, func_name, edits):
    """edits: [(old, new, exact number of occurrences)]"""
    fn = getattr(module, func_name)
    lines, first = inspect.getsourcelines(fn)
    src = "".join(lines)
    for old, new, n in edits:
        if src.count(old) != n:
            raise SystemExit("%s.%s: expected %d occurrence(s) of %r, found %d" % (module.__name__, func_name, n, old, src.count(old)))
        src = src.replace(old, new)
    exec(compile("\n" * (first - 1) + src, module.__file__, "exec"), module.__dict__)
    if getattr(module, func_name) is fn:
        raise SystemExit("recompiling %s.%s changed nothing" % (module.__name__, func_name))


def main():
    if len(sys.argv) < 4 or sys.argv[2] not in ("none",) + NP.ALLOWED_POS:
        raise SystemExit(__doc__)
    task, pos, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    if not (task.isalpha() and task.islower()) or task in KNOWN:
        raise SystemExit("TASK must be a new lower-case name")
    named = [rest[i + 1] for i, a in enumerate(rest[:-1]) if a == "--task"]
    if named != [task]:
        raise SystemExit("pass --task %s (got %r)" % (task, named))
    new_tuple = '("triviaqa", "popqa", %r)' % task
    recompile(G, "load_items", [(OLD_TUPLE, new_tuple, 1)])
    recompile(G, "main", [(OLD_TUPLE, new_tuple, 3), (OLD_CHOICES, 'choices=["triviaqa", "popqa", "mmlu", "gsm8k", %r]' % task, 1)])
    print("[ns-task] task %r is handled like triviaqa / popqa; newsets_gen.py sha256 %s (unchanged file); entry sha256 %s" % (
        task, hashlib.sha256(Path(G.__file__).read_bytes()).hexdigest()[:16], hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]), flush=True)
    if pos == "none":
        sys.argv = [G.__file__] + rest
        G.main()
    else:
        sys.argv = [NP.__file__, pos] + rest
        NP.main()


if __name__ == "__main__":
    main()
