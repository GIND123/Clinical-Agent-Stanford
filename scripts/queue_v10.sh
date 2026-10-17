#!/usr/bin/env bash
# GPU job queue, version 10 (Oct 16-19): reviewer and supervisor comments that need the GPU. Hard stop
# Mon Oct 19, 10:30 IST, approved by the user. In order:
#   1. CEV estimator audit for both DEFER-Dx seeds (inference only; scripts/cev_audit.py, K = 16)
#   2. the published-case benchmark (data/casereports/), every model, once data/casereports/READY exists;
#      checked again after each training job
#   3. ablation without the test-cost term (100 steps; configs/ablations/cev_no_cost.yaml), evaluation, report
#   4. ablation without the coverage cap (100 steps; configs/ablations/cev_no_constraint.yaml), evaluation, report
#   5. the consensus arm's second seed, only if its adapter is in outputs/runs/deferdx_seed1/final
#   6. final report and a local snapshot
# Same guards as before: a job starts only if its estimate fits before DEADLINE; every attempt is killed at
# DEADLINE; at most two attempts, training resuming from its newest checkpoint; what finishes is reported.
#   setsid nohup bash scripts/queue_v10.sh >> outputs/logs/queue_v10.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env_gpu.sh
mkdir -p outputs/logs outputs/queue
DEADLINE=$(date -d "2026-10-21 12:00 +0530" +%s)

echo "$(date -Is) queue v10 started; deadline $(date -d @"$DEADLINE" -Is); waiting for older queues"
while pgrep -f "^(/usr)?(/bin/)?bash [^ ]*scripts/queue_v[2-9]\.sh" >/dev/null; do sleep 60; done

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
CR=data/casereports/casereports.jsonl

# the published-case benchmark: every system, 3 seeds, once the cases are frozen (READY)
casereports() {
  if [[ ! -f data/casereports/READY ]]; then echo "$(date -Is) casereports: not ready yet"; return 0; fi
  step cr_cev 20 deferdx eval-suite "${EVAL_MEM[@]}" "${CEV[@]}" --model $M --adapter outputs/runs/cev/final \
    --sets $CR --name cr_cev
  step cr_cev_seed1 20 deferdx eval-suite "${EVAL_MEM[@]}" "${CEV[@]}" --model $M --adapter outputs/runs/cev_seed1/final \
    --sets $CR --name cr_cev_seed1
  step cr_grpo_nodefer 20 deferdx eval-suite "${EVAL_MEM[@]}" --model $M --adapter outputs/runs/nodefer/final \
    --sets $CR --name cr_grpo_nodefer --no-defer
  step cr_grpo_nodefer_seed1 20 deferdx eval-suite "${EVAL_MEM[@]}" --model $M --adapter outputs/runs/nodefer_seed1/final \
    --sets $CR --name cr_grpo_nodefer_seed1 --no-defer
  step cr_deferdx 20 deferdx eval-suite "${EVAL_MEM[@]}" --model $M --adapter outputs/runs/deferdx/final \
    --sets $CR --name cr_deferdx
  step cr_zs_nodefer 20 deferdx eval-suite "${EVAL_MEM[@]}" --model $M --sets $CR --name cr_zs_nodefer --no-defer
  step cr_zs_defer 20 deferdx eval-suite "${EVAL_MEM[@]}" --model $M --sets $CR --name cr_zs_defer
}

# 1. CEV estimator audit (inference only)
step audit_cev_s0 120 python scripts/cev_audit.py --name cev_s0 --eval outputs/eval/cev/s0.jsonl \
  --adapter outputs/runs/cev/final
step audit_cev_s1 120 python scripts/cev_audit.py --name cev_s1 --eval outputs/eval/cev_seed1/s0.jsonl \
  --adapter outputs/runs/cev_seed1/final
casereports

# 3. without the test-cost term
train_step train_abl_cev_no_cost abl_cev_no_cost 100 470 deferdx train grpo-vllm \
  --config configs/ablations/cev_no_cost.yaml "${TRAIN_MEM[@]}"
evaluated eval_abl_cev_no_cost abl_cev_no_cost
step eval_abl_cev_no_cost 60 deferdx eval-suite "${EVAL_MEM[@]}" "${CEV[@]}" --model $M \
  --adapter outputs/runs/abl_cev_no_cost/final --name abl_cev_no_cost --seeds 0 1
step report_v10_1 20 python scripts/make_report.py
casereports

# 4. without the coverage cap
train_step train_abl_cev_no_constraint abl_cev_no_constraint 100 470 deferdx train grpo-vllm \
  --config configs/ablations/cev_no_constraint.yaml "${TRAIN_MEM[@]}"
evaluated eval_abl_cev_no_constraint abl_cev_no_constraint
step eval_abl_cev_no_constraint 60 deferdx eval-suite "${EVAL_MEM[@]}" "${CEV[@]}" --model $M \
  --adapter outputs/runs/abl_cev_no_constraint/final --name abl_cev_no_constraint --seeds 0 1
step report_v10_2 20 python scripts/make_report.py
casereports

# 5. the consensus arm's second seed, if its adapter has arrived (trained on another GPU)
if [[ -f outputs/runs/deferdx_seed1/final/adapter_model.safetensors ]]; then
  evaluated eval_deferdx_seed1 deferdx_seed1
  step eval_deferdx_seed1 100 deferdx eval-suite "${EVAL_MEM[@]}" --model $M \
    --adapter outputs/runs/deferdx_seed1/final --name deferdx_seed1
else
  echo "$(date -Is) NOT RUN eval_deferdx_seed1: no adapter in outputs/runs/deferdx_seed1/final"
fi

# 6. final report and local snapshot
step report_v10_final 20 python scripts/make_report.py
dest=outputs/archive/$(date +%Y%m%d-%H%M)-v10
mkdir -p "$dest" && cp -al outputs/runs "$dest/runs" && cp -a outputs/eval "$dest/eval" && cp -a outputs/logs "$dest/logs" \
  && cp -a outputs/audit "$dest/audit" 2>/dev/null; echo "$(date -Is) snapshot $dest"
echo "$(date -Is) queue v10 finished; no further GPU jobs"
