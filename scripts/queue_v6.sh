#!/usr/bin/env bash
# GPU job queue, version 6: the journal run. Approved to use the GPU until Mon Oct 12, 18:00 IST
# (08:30 US Eastern). A fixed list in priority order; nothing is added while it runs:
#   1. DEFER-Dx: finish training (started by queue_v4), evaluate, report
#   2. GRPO control (resumes from step 20), evaluate: learned vs post-hoc deferral on an identically trained model
#   3. constant-deferral ablation (100 steps) and DEFER-Dx at step 100: does the counterfactual credit matter?
#   4. DEFER-Dx second training seed (150 steps), evaluate
#   5. closed-world zero-shot and mask-drop DEFER-Dx (CDM sets only), only if they still fit
# A job starts only if its estimated time fits before DEADLINE (training: the steps left after its newest
# checkpoint x the minutes per step measured on this GPU, +5%); every attempt is killed at DEADLINE; at
# most two attempts per job; what finishes is reported as it is. Not run in this budget: the no-handoff
# ablation, a second seed of the control, gpt-oss-20b.
#   kill <queue_v5 bash pid>      # the queue only; a job it started keeps running and is waited for
#   setsid nohup bash scripts/queue_v6.sh >> outputs/logs/queue_v6.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env_gpu.sh
mkdir -p outputs/logs outputs/queue
DEADLINE=$(date -d "2026-10-12 18:00 +0530" +%s)

if pgrep -f "^(/usr)?(/bin/)?bash [^ ]*scripts/queue_v[2-5]\.sh" >/dev/null; then
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
CDM_SETS=(data/cohorts/eval_cdm_val.jsonl data/cohorts/eval_cdm_test.jsonl)
TRAIN_MEM=(--set vllm_gpu_memory_utilization=0.72)
EVAL_MEM=(--gpu-memory-utilization 0.78)

echo "$(date -Is) queue v6 started; deadline $(date -d @"$DEADLINE" -Is)"

# 1. DEFER-Dx (seconds per step measured on this GPU: DEFER-Dx 440, control 466, consensus-style 520)
train_step train_cev_v4 cev 150 440 deferdx train grpo-vllm --config configs/grpo_cev.yaml "${TRAIN_MEM[@]}"
adapter=outputs/runs/cev/final
if [[ ! -d $adapter ]]; then  # could not finish: evaluate the newest checkpoint and say so
  adapter=$(ls -d outputs/runs/cev/checkpoints/step_* 2>/dev/null | tail -1)
  echo "$(date -Is) WARNING DEFER-Dx training did not finish; evaluating $adapter"
fi
evaluated eval_cev_v4 cev
step eval_cev_v4 100 deferdx eval-suite "${EVAL_MEM[@]}" --config configs/grpo_cev.yaml --model $M --adapter "$adapter" --name cev
step report_v6_1 5 python scripts/make_report.py

# 2. GRPO control, identical training without DEFER
train_step train_nodefer nodefer 150 466 deferdx train grpo-vllm --config configs/grpo_nodefer.yaml "${TRAIN_MEM[@]}"
evaluated eval_grpo_nodefer grpo_nodefer
step eval_grpo_nodefer 100 deferdx eval-suite "${EVAL_MEM[@]}" --model $M --adapter outputs/runs/nodefer/final \
  --name grpo_nodefer --no-defer
step report_v6_2 5 python scripts/make_report.py

# 3. constant-deferral ablation against DEFER-Dx at step 100
train_step train_abl_constant_defer abl_constant_defer 100 520 deferdx train grpo-vllm \
  --config configs/ablations/constant_defer.yaml "${TRAIN_MEM[@]}"
evaluated eval_abl_constant_defer abl_constant_defer
step eval_abl_constant_defer 60 deferdx eval-suite "${EVAL_MEM[@]}" --model $M --adapter outputs/runs/abl_constant_defer/final \
  --name abl_constant_defer --seeds 0 1
evaluated eval_cev_v4_step100 cev_step100
step eval_cev_v4_step100 60 deferdx eval-suite "${EVAL_MEM[@]}" --config configs/grpo_cev.yaml --model $M \
  --adapter outputs/runs/cev/checkpoints/step_0100 --name cev_step100 --seeds 0 1
step report_v6_3 5 python scripts/make_report.py

# 4. DEFER-Dx, second training seed
train_step train_cev_seed1 cev_seed1 150 440 deferdx train grpo-vllm --config configs/ablations/cev_seed1.yaml "${TRAIN_MEM[@]}"
evaluated eval_cev_seed1 cev_seed1
step eval_cev_seed1 100 deferdx eval-suite "${EVAL_MEM[@]}" --config configs/grpo_cev.yaml --model $M \
  --adapter outputs/runs/cev_seed1/final --name cev_seed1
step report_v6_4 5 python scripts/make_report.py

# 5. cheap CDM-only evaluations, if they still fit
evaluated eval_zs_closed zs_closed
step eval_zs_closed 45 deferdx eval-suite "${EVAL_MEM[@]}" --model $M --name zs_closed --no-defer --closed-world \
  --sets "${CDM_SETS[@]}"
evaluated eval_cev_v4_maskdrop cev_maskdrop
step eval_cev_v4_maskdrop 45 deferdx eval-suite "${EVAL_MEM[@]}" --config configs/grpo_cev.yaml --model $M \
  --adapter "$adapter" --name cev_maskdrop --mask-policy drop_sentence --sets "${CDM_SETS[@]}"
step report_v6_final 5 python scripts/make_report.py
echo "$(date -Is) queue v6 finished; no further GPU jobs"
