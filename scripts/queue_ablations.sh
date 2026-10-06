#!/usr/bin/env bash
# Second GPU queue: ablations (100 steps each) and their evaluations, after scripts/queue.sh.
#
#   WAIT_PID=<pid of queue.sh> setsid nohup bash scripts/queue_ablations.sh > outputs/logs/queue_ablations.log 2>&1 &
#
# Same conventions as queue.sh: idempotent .done markers, waits for a free GPU, resumes training.
set -u
cd "$(dirname "$0")/.."
source scripts/env_gpu.sh
mkdir -p outputs/logs outputs/queue

if [[ -n "${WAIT_PID:-}" ]]; then
  echo "$(date -Is) waiting for pid $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi

gpu_free() {
  while [[ $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l) -gt 0 ]]; do
    sleep 60
  done
}

step() {
  local name=$1; shift
  if [[ -f "outputs/queue/$name.done" ]]; then echo "$(date -Is) skip $name (done)"; return 0; fi
  gpu_free
  echo "$(date -Is) start $name: $*"
  if "$@" >> "outputs/logs/$name.log" 2>&1; then
    touch "outputs/queue/$name.done"; echo "$(date -Is) done  $name"
  else
    echo "$(date -Is) FAILED $name (exit $?), see outputs/logs/$name.log"
  fi
}

M=Qwen/Qwen3-8B
SEEDS=(--seeds 0 1)

# the main run at the ablations' step budget (its step-100 checkpoint), for like-for-like comparison
step eval_deferdx_step100 deferdx eval-suite --model $M --adapter outputs/runs/deferdx/checkpoints/step_0100 \
  --name deferdx_step100 "${SEEDS[@]}"

for abl in std_norm cdm_only no_constraint forced_loo; do
  [[ -d outputs/runs/abl_$abl/final ]] && touch outputs/queue/train_abl_$abl.done
  step train_abl_$abl deferdx train grpo-vllm --config configs/ablations/$abl.yaml
  step eval_abl_$abl deferdx eval-suite --model $M --adapter outputs/runs/abl_$abl/final --name abl_$abl "${SEEDS[@]}"
done

# training-seed replicate of the main run (full 150 steps), if time allows
[[ -d outputs/runs/deferdx_seed1/final ]] && touch outputs/queue/train_deferdx_seed1.done
step train_deferdx_seed1 deferdx train grpo-vllm --config configs/ablations/seed1.yaml
step eval_deferdx_seed1 deferdx eval-suite --model $M --adapter outputs/runs/deferdx_seed1/final --name deferdx_seed1
echo "$(date -Is) ablation queue finished"
