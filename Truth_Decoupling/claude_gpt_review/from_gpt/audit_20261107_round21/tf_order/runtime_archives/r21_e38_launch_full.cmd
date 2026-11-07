set -euo pipefail
TASK_ROOT=/data/sanshizhao/Truth_Decoupling
cat > "$TASK_ROOT/results_r21_e38_calibrated_codex_20261107/run_full.sh" <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/workspace/Truth_Decoupling
CODE_PATH=$TASK_ROOT/review_r21_e38_v2_codex_20261107
for MODEL in llama3.1-8b qwen3-8b; do
  RUN_PATH=$TASK_ROOT/results_r21_e38_calibrated_codex_20261107/$MODEL/full192
  CAL_PATH=$TASK_ROOT/results_r21_e38_calibrated_codex_20261107/$MODEL/train_calibration
  mkdir -p "$RUN_PATH"
  for PHASE in prepare collect report; do
    if python "$CODE_PATH/scripts/e38_tf_commitment_order.py" "$PHASE" --model "$MODEL" --source-run "$TASK_ROOT/results_r20_codex_20261107/tf_ability/$MODEL/tf_ability_full" --run-dir "$RUN_PATH" --dose-mode train_relative --calibration-run "$CAL_PATH" > "$RUN_PATH/$PHASE.log" 2>&1; then
      echo 0 > "$RUN_PATH/$PHASE.exit"
    else
      status=$?
      echo "$status" > "$RUN_PATH/$PHASE.exit"
      exit "$status"
    fi
  done
done
RUN
nohup docker exec -u 1000:1001 -e CUDA_VISIBLE_DEVICES=0 -e PYTHONPATH=/workspace/Truth_Decoupling/review_r21_e38_v2_codex_20261107 -e TD_DATA_DIR=/workspace/Truth_Decoupling/code/data/processed -e TD_MODEL_ROOT=/workspace/Truth_Decoupling/models -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 truth_dec bash /workspace/Truth_Decoupling/results_r21_e38_calibrated_codex_20261107/run_full.sh > "$TASK_ROOT/results_r21_e38_calibrated_codex_20261107/full_launcher.log" 2>&1 < /dev/null &
echo "E38 full192 launcher PID=$!"
