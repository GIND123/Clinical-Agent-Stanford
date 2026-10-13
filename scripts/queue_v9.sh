#!/usr/bin/env bash
# GPU job queue, version 9: second training seeds of the comparators (docs/EXPERIMENTS.md, "Proposed: second
# seeds of the control and the consensus arm"). Waits for queue_v8 to finish, then:
#   1. GRPO control, second training seed (150 steps; configs/ablations/nodefer_seed1.yaml), evaluation, report
#   2. the consensus arm's second seed, trained elsewhere: evaluated here only if its adapter is already in
#      outputs/runs/deferdx_seed1/final when the queue gets there (weights only), then a report
#   3. a local snapshot of every run and evaluation
# Deadline Thu Oct 15, 12:00 IST: about 21 h of work from Wed 05:30, plus room for one restart after an
# out-of-memory crash. Same guards as before: a job starts only if its estimate fits before DEADLINE (training:
# the steps left after its newest checkpoint x 466 s, the control's measured step time, +5%); every attempt is
# killed at DEADLINE; at most two attempts, training resuming from its newest checkpoint.
#   setsid nohup bash scripts/queue_v9.sh >> outputs/logs/queue_v9.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env_gpu.sh
mkdir -p outputs/logs outputs/queue
DEADLINE=$(date -d "2026-10-15 12:00 +0530" +%s)

echo "$(date -Is) queue v9 started; deadline $(date -d @"$DEADLINE" -Is); waiting for older queues"
while pgrep -f "^(/usr)?(/bin/)?bash [^ ]*scripts/queue_v[2-8]\.sh" >/dev/null; do sleep 60; done
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

# 1. GRPO control, second training seed (seed 1; otherwise identical to the control)
train_step train_nodefer_seed1 nodefer_seed1 150 466 deferdx train grpo-vllm \
  --config configs/ablations/nodefer_seed1.yaml "${TRAIN_MEM[@]}"
evaluated eval_grpo_nodefer_seed1 grpo_nodefer_seed1
step eval_grpo_nodefer_seed1 100 deferdx eval-suite "${EVAL_MEM[@]}" --model $M \
  --adapter outputs/runs/nodefer_seed1/final --name grpo_nodefer_seed1 --no-defer
step report_v9_1 15 python scripts/make_report.py

# 2. the consensus arm's second seed, if its adapter has arrived (trained on another GPU)
if [[ -f outputs/runs/deferdx_seed1/final/adapter_model.safetensors ]]; then
  evaluated eval_deferdx_seed1 deferdx_seed1
  step eval_deferdx_seed1 100 deferdx eval-suite "${EVAL_MEM[@]}" --model $M \
    --adapter outputs/runs/deferdx_seed1/final --name deferdx_seed1
  step report_v9_2 15 python scripts/make_report.py
else
  echo "$(date -Is) NOT RUN eval_deferdx_seed1: no adapter in outputs/runs/deferdx_seed1/final"
fi

# 3. local snapshot (Hugging Face private storage is full; local copies are the backup)
dest=outputs/archive/$(date +%Y%m%d-%H%M)-v9
mkdir -p "$dest" && cp -al outputs/runs "$dest/runs" && cp -a outputs/eval "$dest/eval" && cp -a outputs/logs "$dest/logs" \
  && echo "$(date -Is) snapshot $dest"
echo "$(date -Is) queue v9 finished; no further GPU jobs"
