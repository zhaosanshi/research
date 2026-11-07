set -euo pipefail
TASK_ROOT=/data/sanshizhao/Truth_Decoupling
mkdir -p "$TASK_ROOT/review_r21_e38_v2_codex_20261107" "$TASK_ROOT/results_r21_e38_calibrated_codex_20261107"
tar -xzf "$TASK_ROOT/r21_e38_code_v2.tar.gz" -C "$TASK_ROOT/review_r21_e38_v2_codex_20261107"
cat > "$TASK_ROOT/results_r21_e38_calibrated_codex_20261107/run_pilot.sh" <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/workspace/Truth_Decoupling
CODE_PATH=$TASK_ROOT/review_r21_e38_v2_codex_20261107
for MODEL in llama3.1-8b qwen3-8b; do
  RUN_PATH=$TASK_ROOT/results_r21_e38_calibrated_codex_20261107/$MODEL/pilot3
  CAL_PATH=$TASK_ROOT/results_r21_e38_calibrated_codex_20261107/$MODEL/train_calibration
  mkdir -p "$RUN_PATH" "$CAL_PATH"
  for PHASE in calibrate prepare collect report; do
    python "$CODE_PATH/scripts/e38_tf_commitment_order.py" "$PHASE" --model "$MODEL" --source-run "$TASK_ROOT/results_r20_codex_20261107/tf_ability/$MODEL/tf_ability_full" --run-dir "$RUN_PATH" --limit 3 --dose-mode train_relative --calibration-run "$CAL_PATH" > "$RUN_PATH/$PHASE.log" 2>&1
    echo 0 > "$RUN_PATH/$PHASE.exit"
  done
done
RUN
nohup docker exec -u 1000:1001 -e CUDA_VISIBLE_DEVICES=0 -e PYTHONPATH=/workspace/Truth_Decoupling/review_r21_e38_v2_codex_20261107 -e TD_DATA_DIR=/workspace/Truth_Decoupling/code/data/processed -e TD_MODEL_ROOT=/workspace/Truth_Decoupling/models -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 truth_dec bash /workspace/Truth_Decoupling/results_r21_e38_calibrated_codex_20261107/run_pilot.sh > "$TASK_ROOT/results_r21_e38_calibrated_codex_20261107/pilot_launcher.log" 2>&1 < /dev/null &
echo "E38 calibrated pilot PID=$!"
