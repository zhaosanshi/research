# Builds oncaut_alt.sh from oncaut2_all.sh (run in the log directory): the same chain with the E layer overridden (ELO)
# and every name of the chain carrying a tag, so that it cannot collide with the files of the first chain.
import sys

s = open("oncaut2_all.sh").read()
a = s.index('echo "$(date \'+%F %T\') version 2 start"')
head = s[:a]
gpu_line = '  [ -z "${GPUO:-}" ] || GPU=$GPUO\n'
if head.count(gpu_line) != 1:
    sys.exit("setm: expected exactly one GPUO line, found %d" % head.count(gpu_line))
head = head.replace(gpu_line, gpu_line + '  [ -z "${ELO:-}" ] || EL=$ELO\n')
pairs = [
    ("judge_tune_oncaut_tune", "judge_tune_oncaut${TAGX}_tune"),
    ("judge_test_oncautbest_test", "judge_test_oncaut${TAGX}best_test"),
    ("--name oncaut_tune", "--name oncaut${TAGX}_tune"),
    ("--name oncautbest_test", "--name oncaut${TAGX}best_test"),
    ("judge_tune/oncaut_tune.json", "judge_tune/oncaut${TAGX}_tune.json"),
    ("judge_test/oncautbest_test.json", "judge_test/oncaut${TAGX}best_test.json"),
    ("ONCAUT_GPU_${key}_DONE", "ONCAUT${TAGX}_GPU_${key}_DONE"),
    ('oncaut_table.py --choose "$(NAMEOF $key)"', 'oncaut_table.py --choose "$(NAMEOF $key)" --tag $TAGX --layer $ELO'),
    ("log=$L/oncaut_$1.log", "log=$L/oncaut${TAGX}_$1.log"),
]
for old, new in pairs:
    n = head.count(old)
    if n < 1:
        sys.exit("pattern not found: %s" % old)
    head = head.replace(old, new)
left = [l for l in head.split("\n") if ("oncaut_tune" in l or "oncautbest" in l) and "TAGX" not in l and not l.lstrip().startswith("#")]
if left:
    sys.exit("untagged names left:\n" + "\n".join(left))
tail = '''# 2026-11-21 04:0x, oncaut_alt.sh: "N1 on top of the best prompt" (the chain of oncaut2_all.sh) with the E write at
# ANOTHER LAYER (ELO), every work dir, export and marker name carrying the tag TAGX so that nothing of the first chain is
# touched. Built from oncaut2_all.sh by build_oncaut_alt.py (the replacements are listed there).
# usage: oncaut_alt.sh KEY LAYER TAG GPUS "STRENGTHS"     table: diag/oncaut_table.py --tag TAG --layer LAYER MODEL
KEY=$1; export ELO=$2; export TAGX=$3; export GPUO=$4; ESX=$5
[ -n "$ESX" ] || { echo "usage: oncaut_alt.sh KEY LAYER TAG GPUS STRENGTHS"; exit 2; }
echo "$(date '+%F %T') oncaut at layer $ELO (tag $TAGX) for $KEY start on GPU $GPUO, strengths $ESX" >> $L/oncaut_all.log
run_model $KEY "$ESX"
touch $L/ONCAUT${TAGX}_${KEY}_DONE
'''
open("oncaut_alt.sh", "w").write(head + tail)
print("written oncaut_alt.sh")
