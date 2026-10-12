#!/usr/bin/env bash
# GPU job queue, version 8: the last journal runs. Hard stop Tue Oct 13, 10 pm New York time (Wed Oct 14,
# 07:30 IST); GPU jobs end by 05:30 IST so the paper can be finalised with every number in place.
# Waits for queue_v7 to finish (it was relaunched after the machine restarted on Oct 11, 17:02 IST), then:
#   1. DEFER-Dx second training seed (150 steps; configs/ablations/cev_seed1.yaml), evaluation, report
#   2. no-handoff ablation (resumes from its step-5 checkpoint; 100 steps), evaluation, report
#   3. gpt-oss-20b zero-shot baseline, only if it still fits
#   4. final report and a local snapshot of every run and evaluation
# Same guards as before: a job starts only if its estimate fits before DEADLINE (training: the steps left after
# its newest checkpoint x the seconds per step measured here, +5%); every attempt is killed at DEADLINE; at most
# two attempts; what finishes is reported as it is.
#   setsid nohup bash scripts/queue_v8.sh >> outputs/logs/queue_v8.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env_gpu.sh
mkdir -p outputs/logs outputs/queue
DEADLINE=$(date -d "2026-10-14 05:30 +0530" +%s)

echo "$(date -Is) queue v8 started; deadline $(date -d @"$DEADLINE" -Is); waiting for older queues"
while pgrep -f "^(/usr)?(/bin/)?bash [^ ]*scripts/queue_v[2-7]\.sh" >/dev/null; do sleep 60; done

left() { echo $(( DEADLINE - $(date +%s) )); }
gpu_free() {
  while [[ $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l) -gt 0 ]]; do
    [[ $(left) -le 0 ]] && return 1
    sleep 60
  done
}
# an evaluation suite whose every requested seed finished (e.g. one a stopped queue left running)
complete() {
  [[ -f "$1/summary.json" ]] && python -c "import json, sys; s = json.load(open(sys.argv[1])); \
sys.exit(0 if s.get('seeds') and all(f's{k}' in s.get('results', {}) for k in s['seeds']) else 1)" "$1/summary.json"
}
evaluated() { complete "outputs/eval/$2" && touch "outputs/queue/$1.done"; return 0; }
trained() { [[ -d "outputs/runs/$1/final" ]] && touch "outputs/queue/$2.done"; return 0; }

# one attempt of CMD, killed at DEADLINE
attempt() {
  local name=$1 i=$2 tries=$3
  shift 3
  echo "$(date -Is) start $name (attempt $i of $tries): $*"
  if timeout --signal=TERM --kill-after=120 "$(left)" "$@" >> "outputs/logs/$name.log" 2>&1; then
    touch "outputs/queue/$name.done"
    echo "$(date -Is) done  $name"
    return 0
  fi
  echo "$(date -Is) FAILED $name (attempt $i of $tries), see outputs/logs/$name.log"
  sleep 120
  return 1
}

# step NAME NEED_MIN CMD...: at most two attempts, each started only if NEED_MIN minutes remain
step() {
  local name=$1 need=$2
  shift 2
  if [[ -f "outputs/queue/$name.done" ]]; then echo "$(date -Is) skip $name (done)"; return 0; fi
  local i
  for ((i = 1; i <= 2; i++)); do
    gpu_free || { echo "$(date -Is) NOT RUN $name: GPU busy until the deadline"; return 1; }
    if [[ $(left) -lt $(( need * 60 )) ]]; then
      echo "$(date -Is) NOT RUN $name: needs ${need} min, $(( $(left) / 60 )) min left"
      return 1
    fi
    attempt "$name" "$i" 2 "$@" && return 0
  done
  return 1
}

# train_step NAME RUN TOTAL_STEPS SEC_PER_STEP CMD...: as step, with the time needed computed before each
# attempt from the run's newest checkpoint (a resumed run needs only its remaining steps)
train_step() {
  local name=$1 run=$2 total=$3 sps=$4
  shift 4
  trained "$run" "$name"
  if [[ -f "outputs/queue/$name.done" ]]; then echo "$(date -Is) skip $name (done)"; return 0; fi
  local i last need
  for ((i = 1; i <= 2; i++)); do
    gpu_free || { echo "$(date -Is) NOT RUN $name: GPU busy until the deadline"; return 1; }
    last=$(ls -d outputs/runs/"$run"/checkpoints/step_* 2>/dev/null | sed -E 's/.*step_0*//' | sort -n | tail -1)
    need=$(( (total - ${last:-0}) * sps * 105 / 100 / 60 + 15 ))
    if [[ $(left) -lt $(( need * 60 )) ]]; then
      echo "$(date -Is) NOT RUN $name: $(( total - ${last:-0} )) steps need ~${need} min, $(( $(left) / 60 )) min left"
      return 1
    fi
    attempt "$name" "$i" 2 "$@" && return 0
  done
  return 1
}

M=Qwen/Qwen3-8B
TRAIN_MEM=(--set vllm_gpu_memory_utilization=0.72)
EVAL_MEM=(--gpu-memory-utilization 0.78)

# 1. DEFER-Dx, second training seed
train_step train_cev_seed1 cev_seed1 150 440 deferdx train grpo-vllm --config configs/ablations/cev_seed1.yaml "${TRAIN_MEM[@]}"
evaluated eval_cev_seed1 cev_seed1
step eval_cev_seed1 100 deferdx eval-suite "${EVAL_MEM[@]}" --config configs/grpo_cev.yaml --model $M \
  --adapter outputs/runs/cev_seed1/final --name cev_seed1
step report_v8_1 5 python scripts/make_report.py

# 2. no-handoff ablation (cev_eta 0), against DEFER-Dx at step 100
train_step train_abl_cev_no_handoff abl_cev_no_handoff 100 470 deferdx train grpo-vllm \
  --config configs/ablations/cev_no_handoff.yaml "${TRAIN_MEM[@]}"
evaluated eval_abl_cev_no_handoff abl_cev_no_handoff
step eval_abl_cev_no_handoff 60 deferdx eval-suite "${EVAL_MEM[@]}" --config configs/grpo_cev.yaml --model $M \
  --adapter outputs/runs/abl_cev_no_handoff/final --name abl_cev_no_handoff --seeds 0 1
step report_v8_2 5 python scripts/make_report.py

# 3. optional baseline
evaluated eval_gptoss gptoss_nodefer
step eval_gptoss 180 deferdx eval-suite "${EVAL_MEM[@]}" --model openai/gpt-oss-20b --name gptoss_nodefer --no-defer \
  --temperature 1.0 --top-p 1.0 --top-k 0 --max-new-tokens 2048

# 4. final report and local snapshot (Hugging Face private storage is full; local copies are the backup)
step report_v8_final 5 python scripts/make_report.py
dest=outputs/archive/$(date +%Y%m%d-%H%M)-final
mkdir -p "$dest" && cp -al outputs/runs "$dest/runs" && cp -a outputs/eval "$dest/eval" && cp -a outputs/logs "$dest/logs" \
  && echo "$(date -Is) snapshot $dest"
echo "$(date -Is) queue v8 finished; no further GPU jobs"
