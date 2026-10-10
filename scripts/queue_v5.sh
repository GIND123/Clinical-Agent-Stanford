#!/usr/bin/env bash
# GPU job queue, version 5: the compute budget ends Sat Oct 10, 10:00 US Eastern (19:30 IST). Replaces
# queue_v4 after its DEFER-Dx training. The GRPO control (17 h left) and the two ablations (about 12 h each)
# cannot finish before then and are not run; they are reported as not run. This queue finishes DEFER-Dx,
# evaluates it, adds the two cheap CDM-only evaluations (closed-world zero-shot; DEFER-Dx with the mask
# cue removed) and stops. Every job is killed at DEADLINE, so nothing runs past the budget.
#   kill <queue_v4 bash pid>      # the queue only; the DEFER-Dx training keeps running and is waited for
#   setsid nohup bash scripts/queue_v5.sh >> outputs/logs/queue_v5.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env_gpu.sh
mkdir -p outputs/logs outputs/queue
DEADLINE=$(date -d "2026-10-10 19:00 +0530" +%s)  # 30 min before the budget ends

if pgrep -f "^(/usr)?(/bin/)?bash [^ ]*scripts/queue_v[234]\.sh" >/dev/null; then
  echo "$(date -Is) an older queue is still running; stop it first (only the bash process)"
  exit 1
fi

left() { echo $(( DEADLINE - $(date +%s) )); }
gpu_free() {
  while [[ $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l) -gt 0 ]]; do
    [[ $(left) -le 0 ]] && return 1
    sleep 60
  done
}

# step NAME TRIES NEED_MIN CMD...: start only if NEED_MIN minutes remain before DEADLINE; retry up to
# TRIES times; every attempt is killed at DEADLINE.
step() {
  local name=$1 tries=$2 need=$3
  shift 3
  if [[ -f "outputs/queue/$name.done" ]]; then echo "$(date -Is) skip $name (done)"; return 0; fi
  local i
  for ((i = 1; i <= tries; i++)); do
    gpu_free || { echo "$(date -Is) NOT RUN $name: GPU busy until the deadline"; return 1; }
    if [[ $(left) -lt $(( need * 60 )) ]]; then
      echo "$(date -Is) NOT RUN $name: needs ${need} min, $(( $(left) / 60 )) min left before the deadline"
      return 1
    fi
    echo "$(date -Is) start $name (attempt $i of $tries): $*"
    if timeout --signal=TERM --kill-after=120 "$(left)" "$@" >> "outputs/logs/$name.log" 2>&1; then
      touch "outputs/queue/$name.done"
      echo "$(date -Is) done  $name"
      return 0
    fi
    echo "$(date -Is) FAILED $name (attempt $i of $tries), see outputs/logs/$name.log"
    sleep 120
  done
  return 1
}
trained() { [[ -d "outputs/runs/$1/final" ]] && touch "outputs/queue/$2.done"; return 0; }

M=Qwen/Qwen3-8B
CDM_SETS=(data/cohorts/eval_cdm_val.jsonl data/cohorts/eval_cdm_test.jsonl)
TRAIN_MEM=(--set vllm_gpu_memory_utilization=0.72)
EVAL_MEM=(--gpu-memory-utilization 0.78)

echo "$(date -Is) queue v5 started; deadline $(date -d @"$DEADLINE" -Is); waiting for the DEFER-Dx training"
gpu_free

# 1. DEFER-Dx: finish training if it was interrupted (resumes from its newest checkpoint)
trained cev train_cev_v4
step train_cev_v4 2 100 deferdx train grpo-vllm --config configs/grpo_cev.yaml "${TRAIN_MEM[@]}"
adapter=outputs/runs/cev/final
if [[ ! -d $adapter ]]; then  # no time to finish: evaluate the newest checkpoint and say so
  adapter=$(ls -d outputs/runs/cev/checkpoints/step_* 2>/dev/null | tail -1)
  echo "$(date -Is) WARNING DEFER-Dx training did not finish; evaluating $adapter"
fi
step eval_cev_v4 2 100 deferdx eval-suite "${EVAL_MEM[@]}" --config configs/grpo_cev.yaml --model $M --adapter "$adapter" --name cev
step report_v5_1 1 5 python scripts/make_report.py

# 2. the two cheap CDM-only evaluations
step eval_zs_closed 1 60 deferdx eval-suite "${EVAL_MEM[@]}" --model $M --name zs_closed --no-defer --closed-world --sets "${CDM_SETS[@]}"
step eval_cev_v4_maskdrop 1 60 deferdx eval-suite "${EVAL_MEM[@]}" --config configs/grpo_cev.yaml --model $M --adapter "$adapter" \
  --name cev_maskdrop --mask-policy drop_sentence --sets "${CDM_SETS[@]}"
step report_v5_final 1 5 python scripts/make_report.py
echo "$(date -Is) queue v5 finished; no further GPU jobs"
