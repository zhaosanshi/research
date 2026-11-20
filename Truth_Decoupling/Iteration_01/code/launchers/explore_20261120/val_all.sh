#!/bin/bash
# Fresh-question validation (novelty/ours/2026-11-20_rechosen_caps_validation.md): waits until the two 27B MMLU reruns
# have released GPUs 1-3, then generates test2 on GPU 1 (three small models) and GPUs 2,3 (the two 27B models), then
# judges the Pro 100 off-peak (one lane per model), with the balance read before and after. GPU 0 is not used.
cd /data/sanshizhao/Truth_Decoupling/Iteration_01
L=logs/newsets_20261119
balance() {
  docker exec -u 1000:1001 truth_dec python3 -c '
import json, urllib.request
k = open("/workspace/Truth_Decoupling/logs/.deepseek_key", encoding="utf-8").read().strip()
r = urllib.request.Request("https://api.deepseek.com/user/balance", headers={"Authorization": "Bearer " + k, "Accept": "application/json"})
d = json.load(urllib.request.urlopen(r, timeout=30))
print(json.dumps(d.get("balance_infos"), ensure_ascii=False, separators=(",", ":")))' 2>/dev/null
}
rm -f $L/val_all.log $L/VAL_DONE
until [ -e $L/MCQ_V2B_DONE ]; do sleep 60; done
echo "$(date '+%F %T') GPUs free, start" >> $L/val_all.log
( for m in gemma3-12b llama3.1-8b qwen3.5-9b; do bash launchers/newsets_c2c3_20261119.sh val $m 1 > $L/val_$m.log 2>&1; echo "val $m exit=$? $(date '+%T')" >> $L/val_all.log; done ) &
( bash launchers/newsets_c2c3_20261119.sh val gemma3-27b 2 > $L/val_gemma3-27b.log 2>&1; echo "val gemma3-27b exit=$? $(date '+%T')" >> $L/val_all.log
  bash launchers/newsets_c2c3_20261119.sh val qwen3.8-27b 2,3 > $L/val_qwen3.8-27b.log 2>&1; echo "val qwen3.8-27b exit=$? $(date '+%T')" >> $L/val_all.log ) &
wait
if grep "^val " $L/val_all.log | grep -v "exit=0" | grep -q .; then echo "$(date) a val lane failed: not judging" >> $L/val_all.log; touch $L/VAL_DONE; exit 1; fi
echo "$(date '+%F %T') balance before: $(balance)" >> $L/val_all.log
for m in gemma3-12b llama3.1-8b qwen3.5-9b gemma3-27b qwen3.8-27b; do
  ( bash launchers/newsets_c2c3_20261119.sh val_judge $m > $L/val_judge_$m.log 2>&1; echo "val_judge $m exit=$? $(date '+%T')" >> $L/val_all.log ) &
done
wait
echo "$(date '+%F %T') balance after: $(balance)" >> $L/val_all.log
touch $L/VAL_DONE
