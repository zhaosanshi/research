#!/bin/bash
# job_variants.sh -- role-word variant sweep (2026-10-24): pinyin, case, punctuation/markup forms, homoglyphs,
# other scripts.  English materials, plain contaminant whose role word is the variant; X20k_m00 + X20k_m03, 3 seeds,
# forced variant + 'user'.  Qwen3-8B on GPU 2, gemma-3-4b-it on GPU 3.  One output file per model, word in p_contam_word.
# Launch on the GPU server inside tmux:  tmux new-session -d -s opus_var 'bash job_variants.sh 2>&1 | tee results/job_variants.log'
cd ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
D="docker run --rm --gpus all --ipc=host -v $PHANTOM_ROOT:$PHANTOM_ROOT -v $MODEL_ROOT:$MODEL_ROOT:ro -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e PROBE_LANG=en -e CONTAM_STYLE=plain -w ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory}"
IMG=${IMG:?set IMG to a PyTorch container image, see README}
MD=${MODEL_ROOT:?set MODEL_ROOT to the model directory}
FILL="--filler-coding results/filler_en_coding.jsonl --filler-meta results/filler_en_general.jsonl"
VARIANTS=(yonghu Yonghu YONGHU "yong hu" 用戶 使用者 用家 客 客户 提问者 人类 问
          User USER Usr usr U Human HUMAN Person Customer Client Operator Me Q Question Prompt Input You
          "user:" "User:" "USER:" "[user]" "[User]" "<user>" "**User**" "### User" "user>" ">" "👤"
          "usеr" "υser" "u s e r"
          "यूज़र" "مستخدم" "người dùng" "Kullanıcı" "Utente" "Gebruiker" "Użytkownik" "Usuário")
run() {  # gpu model nt
  local gpu=$1 m=$2 nt=$3
  for w in "${VARIANTS[@]}"; do
    for s in 1 2 3; do
      $D -e CUDA_VISIBLE_DEVICES=$gpu -e CONTAM_ROLE_WORD="$w" $IMG python turn_boundary_probe.py --model $MD/$m $nt $FILL --device auto \
          --force-first "$w" user --max-forced 48 --conditions X20k_m00 X20k_m03 --filler-seed $s \
          --out results/variants_$m.jsonl >> results/variants_$m.log 2>&1 || log "FAILED $m [$w] seed $s"
    done
    log "$m [$w] done"
  done
}
( run 2 Qwen3-8B --no-think ) &
( run 3 gemma-3-4b-it "" ) &
wait
log "VARIANTS DONE"
