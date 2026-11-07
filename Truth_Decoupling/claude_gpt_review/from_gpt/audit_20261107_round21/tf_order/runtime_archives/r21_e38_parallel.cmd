set -euo pipefail
TASK_ROOT=/data/sanshizhao/Truth_Decoupling
cat > "$TASK_ROOT/results_r21_e38_calibrated_codex_20261107/finish_llama_scheduler.py" <<'PY'
import json,os,pathlib,signal,time,subprocess
scheduler=286628
collector=286906
root=pathlib.Path('/workspace/Truth_Decoupling')
results=root/'results_r21_e38_calibrated_codex_20261107'
run=results/'llama3.1-8b/full192'
def cmd(pid):return pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
assert cmd(scheduler).strip()=='bash /workspace/Truth_Decoupling/results_r21_e38_calibrated_codex_20261107/run_full.sh'
original=cmd(collector)
assert ' collect ' in original and '--model llama3.1-8b ' in original
os.kill(scheduler,signal.SIGSTOP)
record={'scheduler_pid':scheduler,'collector_pid':collector,'collector_cmdline':original,'reason':'Pause only the parent scheduler to prevent later Qwen double write; Llama collector continues unchanged.'}
(results/'parallel_transition.json').write_text(json.dumps(record,indent=2)+'\n')
print('STOPPED SCHEDULER ONLY',scheduler,'collector alive',collector,flush=True)
for _ in range(7200):
    content=pathlib.Path(f'/proc/{collector}/stat').read_text()
    fields=content[content.rfind(')')+2:].split()
    if fields[0]=='Z':
        status=int(fields[49]);code=os.waitstatus_to_exitcode(status)
        record['collector_wait_status']=status;record['collector_exit_code']=code
        (run/'collect.exit').write_text(str(code)+'\n')
        os.kill(scheduler,signal.SIGKILL)
        record['scheduler_terminated_after_collector']=True
        (results/'parallel_transition.json').write_text(json.dumps(record,indent=2)+'\n')
        if code:raise RuntimeError('original Llama collector failed: '+str(code))
        env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(root/'review_r21_e38_v2_codex_20261107'),TD_DATA_DIR=str(root/'code/data/processed'),TD_MODEL_ROOT=str(root/'models'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        command=['python',str(root/'review_r21_e38_v2_codex_20261107/scripts/e38_tf_commitment_order.py'),'report','--model','llama3.1-8b','--source-run',str(root/'results_r20_codex_20261107/tf_ability/llama3.1-8b/tf_ability_full'),'--run-dir',str(run),'--dose-mode','train_relative','--calibration-run',str(results/'llama3.1-8b/train_calibration')]
        with (run/'report.log').open('w') as log:done=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        (run/'report.exit').write_text(str(done.returncode)+'\n')
        done.check_returncode();print('LLAMA FULL FINISHED',flush=True);break
    time.sleep(1)
else:raise RuntimeError('timed out; scheduler remains STOPPED, never auto-resume into Qwen')
PY
nohup docker exec -u 1000:1001 truth_dec python /workspace/Truth_Decoupling/results_r21_e38_calibrated_codex_20261107/finish_llama_scheduler.py > "$TASK_ROOT/results_r21_e38_calibrated_codex_20261107/finish_llama_scheduler.log" 2>&1 < /dev/null &
echo "L scheduler transition monitor PID=$!"
cat > "$TASK_ROOT/results_r21_e38_calibrated_codex_20261107/run_qwen_full.sh" <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/workspace/Truth_Decoupling
CODE_PATH=$TASK_ROOT/review_r21_e38_v2_codex_20261107
RUN_PATH=$TASK_ROOT/results_r21_e38_calibrated_codex_20261107/qwen3-8b/full192
CAL_PATH=$TASK_ROOT/results_r21_e38_calibrated_codex_20261107/qwen3-8b/train_calibration
mkdir -p "$RUN_PATH"
for PHASE in prepare collect report; do
  if python "$CODE_PATH/scripts/e38_tf_commitment_order.py" "$PHASE" --model qwen3-8b --source-run "$TASK_ROOT/results_r20_codex_20261107/tf_ability/qwen3-8b/tf_ability_full" --run-dir "$RUN_PATH" --dose-mode train_relative --calibration-run "$CAL_PATH" > "$RUN_PATH/$PHASE.log" 2>&1; then
    echo 0 > "$RUN_PATH/$PHASE.exit"
  else
    status=$?
    echo "$status" > "$RUN_PATH/$PHASE.exit"
    exit "$status"
  fi
done
RUN
nohup docker exec -u 1000:1001 -e CUDA_VISIBLE_DEVICES=3 -e PYTHONPATH=/workspace/Truth_Decoupling/review_r21_e38_v2_codex_20261107 -e TD_DATA_DIR=/workspace/Truth_Decoupling/code/data/processed -e TD_MODEL_ROOT=/workspace/Truth_Decoupling/models -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 truth_dec bash /workspace/Truth_Decoupling/results_r21_e38_calibrated_codex_20261107/run_qwen_full.sh > "$TASK_ROOT/results_r21_e38_calibrated_codex_20261107/qwen_full_launcher.log" 2>&1 < /dev/null &
echo "Q full GPU3 launcher PID=$!"
