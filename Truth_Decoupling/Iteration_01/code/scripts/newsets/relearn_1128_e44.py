"""Exploration entry (executor R-yi, Claude Opus 5.5, 2026-11-28; work queue R, R7 / R8): scripts/e44_factorial_das.py
(unchanged file) with one more admissible upper truth level, FR. Not reviewed by GPT.

Why: on PopQA the prespecified rule of novelty/ours/2026-11-19_new_datasets_pilot_design.md section 6 can choose the
rewritten sentence FR ("The answer that first comes to mind may belong to a different but similarly named person,
place or work. Give the answer for the one the question asks about.", newsets_gen.FR_SENTENCE) instead of E2. The
program spells the admissible upper levels out in two places: family_spec (`if e_on not in ("E1", "E2"):`) and main
(`if args.e_on not in ("E1", "E2") or args.e_on not in cfg["sentences"]:`). Both are recompiled in memory with
("E1", "E2", "FR") (each literal must occur exactly once); every other line runs as it is. The rows of a family are
then the ones of E2 with the sentence FR: E cause conf <-> conf_FR, isolate conf <-> caut and conf_FR <-> caut_FR;
C cause conf <-> caut and conf_FR <-> caut_FR, isolate conf <-> conf_FR and caut <-> caut_FR. FR needs a prompts file
with the sentence FR and the prompts conf_FR / caut_FR (configs/prompts/e44_factorial_relearn_1128_fr.json). With
--e-on E1 or E2 the run is the one of the unchanged program (the two edits only add a member to a tuple).

The run identity is the program's own (scripts/e44_factorial_das.py and its imports); this entry is not part of it.
Its sha256 and that of the program are printed at the start of every call.

usage (model interpreter, PYTHONPATH=<code>, from <code>):  python scripts/newsets/relearn_1128_e44.py <e44 arguments>
"""
import hashlib
import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import e44_factorial_das as E   # noqa: E402

OLD = '("E1", "E2")'
NEW = '("E1", "E2", "FR")'


def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def recompile(module, func_name, edits):
    """The same as scripts/newsets/ns_task.recompile (copied, so that newsets_gen is not imported): edits =
    [(old, new, exact number of occurrences)]; the edited source is executed in the module's own namespace."""
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
    recompile(E, "family_spec", [("if e_on not in %s:" % OLD, "if e_on not in %s:" % NEW, 1)])
    recompile(E, "main", [("if args.e_on not in %s or" % OLD, "if args.e_on not in %s or" % NEW, 1)])
    print("[relearn-1128] e44_factorial_das.py sha256 %s (unchanged file); upper truth levels E1 / E2 / FR; entry sha256 %s" % (
        sha16(E.__file__), sha16(__file__)), flush=True)
    sys.argv = [E.__file__] + sys.argv[1:]
    E.main()


if __name__ == "__main__":
    main()
