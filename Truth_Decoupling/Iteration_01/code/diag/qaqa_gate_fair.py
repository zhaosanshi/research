# Fable 5.2, 2026-11-22, exploration, read-only. GPT's recovery_pool replay gives "N1 with a perfect question-type gate
# against the strong single E at the same answer count" but no E-only system holding the same gate. This runs GPT's
# script unchanged except that the E-only counterpart it already computes for the learned gate is also computed for the
# gold-type gate: E keeps its answers and fills, by its own confidence, as many blanks of the gated types as N1 releases
# there (must-answer + E answers). The source is patched in memory; GPT's file is not modified.
# usage (container, from /workspace/Truth_Decoupling):
#   python3 -B Iteration_01/diag/qaqa_gate_fair.py PATH_OF_2026-11-22_recovery_pool_b.py
import sys
from pathlib import Path

src_path = Path(sys.argv[1])
src = src_path.read_text(encoding="utf-8")
old1 = ('        if label == "question_and_model_answers":\n'
        '            eligible = [q for q in eb if model_signal[model,q]]\n')
new1 = ('        if label in ("question_and_model_answers", "gold_type_diagnostic"):\n'
        '            eligible = [q for q in eb if (model_signal[model,q] if label == "question_and_model_answers" '
        'else R.qtype[q] != "no_failure")]\n')
old2 = '            if label == "question_and_model_answers" and all(r[3][label][2] is not None for r in results):\n'
new2 = '            if label in ("question_and_model_answers", "gold_type_diagnostic") and all(r[3][label][2] is not None for r in results):\n'
old3 = '                print("same question-type signal, single E filling the same number of blanks:",Q.S.ps.ci(fair))\n'
new3 = ('                print("same gate (%s), single E filling the same number of blanks:" % label, Q.S.ps.ci(fair))\n'
        '                for name, r in zip(Q.SET, results):\n'
        '                    print("   ", name, sum(r[3][label][2].values()), "| total answered after gate", r[3][label][0])\n')
for old, new in ((old1, new1), (old2, new2), (old3, new3)):
    if src.count(old) != 1:
        raise ValueError("source version differs from expected gate implementation")
    src = src.replace(old, new)
exec(compile(src, str(src_path), "exec"), {"__name__": "__main__", "__file__": str(src_path)})
