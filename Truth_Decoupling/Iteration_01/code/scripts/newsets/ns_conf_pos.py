# -*- coding: utf-8 -*-
"""Exploration (Claude Fable 5.2, 2026-11-22): newsets_gen.py's confidence phase for answers that were generated with
the write at another position rule (ns_positions.py), e.g. the must-answer + E answers at "prompt" of the strong
comparator (must-answer prompt + E at the same layer and strength, keep the N most confident answers).

newsets_gen.py conf (unchanged file) scores the model's own answers teacher-forced under the same prompt and the same
writes at the training rule "content", and refuses anything else. Here, in memory:
  1. the engine's Context.writes returns POS in place of the training rule, after checking that the training rule is
     "content" (as ns_positions.py does), and run_conf accepts POS where it demands "content" (a source replacement
     that must match exactly once);
  2. the core write mask of the scorer is POS: the engine's fd.position_mask is wrapped so that a request for
     "content" returns the POS mask (what scripts/review_checks/ff_posrule_confidence.py does for the TruthfulQA
     runs). Under "prompt" this writes the prompt content positions and no answer position, as the generation did
     (core/qa_factorial_das.py position_mask: content_mask & ~target_mask). The number of mask requests is printed;
  3. the generation file must record "positions": POS and at least one write, and the output name must end with
     "-at-<POS>.json", so that a file scored at the training rule can never be taken for one of these.
run_conf's own checks stay as they are: the teacher-forced arg-max must reproduce >= 98% of the generated tokens
(here: scoring with the writes at POS reproduces the answers generated with the writes at POS), and with
--ref-gen-file (the same arm without its last write) the first-difference check.
A task name other than triviaqa / popqa / mmlu / gsm8k is first registered as ns_task.py does (same two recompiles).

usage (model interpreter, from /workspace/Truth_Decoupling):
  python -I ns_conf_pos.py TASK POS conf --models M --model MODEL --task TASK --items BANK --gen-file GEN
         --out OUT-at-POS.json [--ref-gen-file REF] [--score-batch 1]
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import newsets_gen as G    # noqa: E402
import ns_positions as NP  # noqa: E402
import ns_task as NT       # noqa: E402

OLD = 'if positions != "content" or json.loads(json.dumps([d for _, _, d in ws])) != c["parts"]:'


def value_of(rest, flag):
    got = [rest[i + 1] for i, a in enumerate(rest[:-1]) if a == flag]
    if len(got) != 1:
        raise SystemExit("give %s exactly once" % flag)
    return got[0]


def main():
    if len(sys.argv) < 4 or sys.argv[2] not in NP.ALLOWED_POS or sys.argv[3] != "conf":
        raise SystemExit(__doc__)
    task, pos, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    if value_of(rest, "--task") != task:
        raise SystemExit("pass --task %s" % task)
    gen_file, out = value_of(rest, "--gen-file"), value_of(rest, "--out")
    if not out.endswith("-at-%s.json" % pos):
        raise SystemExit("--out must end with -at-%s.json" % pos)
    c = json.loads(Path(gen_file).read_text(encoding="utf-8")).get("contract") or {}
    if c.get("positions") != pos or not c.get("parts") or c.get("task") != task:
        raise SystemExit("%s is not a %s generation with a write at %r" % (gen_file, task, pos))
    if task not in NT.KNOWN:
        if not (task.isalpha() and task.islower()):
            raise SystemExit("TASK must be a lower-case name")
        new_tuple = '("triviaqa", "popqa", %r)' % task
        NT.recompile(G, "load_items", [(NT.OLD_TUPLE, new_tuple, 1)])
        NT.recompile(G, "main", [(NT.OLD_TUPLE, new_tuple, 3),
                                 (NT.OLD_CHOICES, 'choices=["triviaqa", "popqa", "mmlu", "gsm8k", %r]' % task, 1)])
    calls = {"mask": 0}
    real_engine = G.engine

    def engine(m):
        X = real_engine(m)
        original = X.Context.writes

        def writes(self, parts):
            ws, trained = original(self, parts)
            if trained != "content":
                raise SystemExit("the directions were trained with positions=%s; this entry only moves content-position directions" % trained)
            return ws, pos

        X.Context.writes = writes
        original_mask = X.fd.position_mask

        def mask(packed, positions):
            if positions != "content":
                raise SystemExit("the scorer asked for the %r mask; this entry only replaces 'content'" % positions)
            calls["mask"] += 1
            return original_mask(packed, pos)

        X.fd.position_mask = mask
        print("[ns-conf-pos] writes scored at %r instead of the training rule; engine %s sha256 %s (unchanged file)" % (
            pos, X.__file__, hashlib.sha256(Path(X.__file__).read_bytes()).hexdigest()[:16]), flush=True)
        return X

    G.engine = engine
    NP.recompile(G, "run_conf", OLD, 'if positions != %r or json.loads(json.dumps([d for _, _, d in ws])) != c["parts"]:' % pos)
    print("[ns-conf-pos] newsets_gen.py sha256 %s (unchanged file); entry sha256 %s" % (
        hashlib.sha256(Path(G.__file__).read_bytes()).hexdigest()[:16], hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]), flush=True)
    sys.argv = [G.__file__] + rest
    G.main()
    if not calls["mask"]:
        raise SystemExit("the scorer never asked for a write mask: nothing was scored under a write")
    print("[ns-conf-pos] %d write masks at %r were handed to the scorer" % (calls["mask"], pos), flush=True)


if __name__ == "__main__":
    main()
