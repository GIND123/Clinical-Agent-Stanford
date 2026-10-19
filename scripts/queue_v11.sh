#!/usr/bin/env bash
# GPU job queue, version 11 (Oct 19-21): waits for queue_v10, finishes the no-cost ablation (resumed from step 35),
# then trains DEFER-Dx with a stronger handoff score
# (cev_eta 1.0; configs/ablations/cev_eta1.yaml; 100 steps, against DEFER-Dx at step 100), evaluates it on the
# held-out sets (2 seeds) and the published-case benchmark (3 seeds), and reports. Hard stop Mon Oct 19, 10:30
# IST. Same guards as before.
#   setsid nohup bash scripts/queue_v11.sh >> outputs/logs/queue_v11.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env_gpu.sh
mkdir -p outputs/logs outputs/queue
DEADLINE=$(date -d "2026-10-21 12:00 +0530" +%s)

echo "$(date -Is) queue v11 started; deadline $(date -d @"$DEADLINE" -Is); waiting for older queues"
while pgrep -f "^(/usr)?(/bin/)?bash [^ ]*scripts/queue_v([2-9]|10)\.sh" >/dev/null; do sleep 60; done
echo "$(date -Is) older queues finished"

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
CEV=(--config configs/grpo_cev.yaml)

# the no-cost ablation, resumed from its step-35 checkpoint (both attempts in queue_v10 hit the adapter-pruning bug
# fixed in training/grpo_vllm.py register_adapter; same step names, so whichever queue finishes it marks it done)
train_step train_abl_cev_no_cost abl_cev_no_cost 100 470 deferdx train grpo-vllm \
  --config configs/ablations/cev_no_cost.yaml "${TRAIN_MEM[@]}"
evaluated eval_abl_cev_no_cost abl_cev_no_cost
step eval_abl_cev_no_cost 60 deferdx eval-suite "${EVAL_MEM[@]}" "${CEV[@]}" --model $M \
  --adapter outputs/runs/abl_cev_no_cost/final --name abl_cev_no_cost --seeds 0 1
step report_v11_1 20 python scripts/make_report.py

train_step train_abl_cev_eta1 abl_cev_eta1 100 470 deferdx train grpo-vllm \
  --config configs/ablations/cev_eta1.yaml "${TRAIN_MEM[@]}"
evaluated eval_abl_cev_eta1 abl_cev_eta1
step eval_abl_cev_eta1 60 deferdx eval-suite "${EVAL_MEM[@]}" "${CEV[@]}" --model $M \
  --adapter outputs/runs/abl_cev_eta1/final --name abl_cev_eta1 --seeds 0 1
step cr_abl_cev_eta1 20 deferdx eval-suite "${EVAL_MEM[@]}" "${CEV[@]}" --model $M \
  --adapter outputs/runs/abl_cev_eta1/final --sets data/casereports/casereports.jsonl --name cr_abl_cev_eta1
step cr_cev_step100 20 deferdx eval-suite "${EVAL_MEM[@]}" "${CEV[@]}" --model $M \
  --adapter outputs/runs/cev/checkpoints/step_0100 --sets data/casereports/casereports.jsonl --name cr_cev_step100
step report_v11 20 python scripts/make_report.py

dest=outputs/archive/$(date +%Y%m%d-%H%M)-v11
mkdir -p "$dest" && cp -al outputs/runs "$dest/runs" && cp -a outputs/eval "$dest/eval" && cp -a outputs/logs "$dest/logs" \
  && echo "$(date -Is) snapshot $dest"
echo "$(date -Is) queue v11 finished; no further GPU jobs"
