#!/usr/bin/env bash
# GPU job queue, version 2: the counterfactual-escalation (CEV) method is the main arm.
#   WAIT_PID=<pid of a running job> setsid nohup bash scripts/queue_v2.sh >> outputs/logs/queue_v2.log 2>&1 &
# Idempotent (.done markers), waits for a free GPU, resumes training from the newest checkpoint.
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
trained() { [[ -d "outputs/runs/$1/final" ]] && touch "outputs/queue/train_$1.done"; return 0; }

M=Qwen/Qwen3-8B
CDM_SETS=(data/cohorts/eval_cdm_val.jsonl data/cohorts/eval_cdm_test.jsonl)

# 0. case-level consensus arm (the reward of TIAR / KARL / AWA-RL, generalised): finish and evaluate
trained deferdx
step train_deferdx deferdx train grpo-vllm --config configs/grpo_deferdx.yaml

# 1. smoke test of the counterfactual-escalation trainer (Qwen3-0.6B, synthetic cases, 2 steps)
step smoke_cev deferdx train grpo-vllm --config configs/grpo_cev.yaml --set model=Qwen/Qwen3-0.6B \
  train_cases=data/synthetic/rl_train.jsonl dev_cases=data/synthetic/dev.jsonl "pool_weights={synthetic: 1.0}" \
  output_dir=outputs/runs/smoke_cev steps=2 cases_per_step=4 group_size=4 eval_every=2 save_every=2 \
  log_rollouts_every=0 generation.max_new_tokens=512 max_model_len=8192 vllm_gpu_memory_utilization=0.5 resume=false

step eval_deferdx deferdx eval-suite --model $M --adapter outputs/runs/deferdx/final --name deferdx
step eval_zs_nodefer deferdx eval-suite --model $M --name zs_nodefer --no-defer
step eval_zs_defer deferdx eval-suite --model $M --name zs_defer
step report_1 python scripts/make_report.py

# 2. MAIN: counterfactual escalation value + scored handoff
trained cev
step train_cev deferdx train grpo-vllm --config configs/grpo_cev.yaml
step eval_cev deferdx eval-suite --config configs/grpo_cev.yaml --model $M --adapter outputs/runs/cev/final --name cev
step report_2 python scripts/make_report.py

# 3. control: identical training, no DEFER action (threshold its calibrated probability post hoc)
trained nodefer
step train_nodefer deferdx train grpo-vllm --config configs/grpo_nodefer.yaml
step eval_grpo_nodefer deferdx eval-suite --model $M --adapter outputs/runs/nodefer/final --name grpo_nodefer --no-defer
step report_3 python scripts/make_report.py

# 4. ablations of what is new (100 steps; compared with CEV's step-100 checkpoint)
step eval_cev_step100 deferdx eval-suite --config configs/grpo_cev.yaml --model $M \
  --adapter outputs/runs/cev/checkpoints/step_0100 --name cev_step100 --seeds 0 1
for abl in cev_no_handoff constant_defer; do
  trained abl_$abl
  step train_abl_$abl deferdx train grpo-vllm --config configs/ablations/$abl.yaml
  cfgflag=(); [[ $abl == cev_* ]] && cfgflag=(--config configs/grpo_cev.yaml)
  step eval_abl_$abl deferdx eval-suite "${cfgflag[@]}" --model $M --adapter outputs/runs/abl_$abl/final \
    --name abl_$abl --seeds 0 1
done

# 5. robustness and further baselines
step eval_cev_maskdrop deferdx eval-suite --config configs/grpo_cev.yaml --model $M --adapter outputs/runs/cev/final \
  --name cev_maskdrop --mask-policy drop_sentence --sets "${CDM_SETS[@]}"
step eval_nodefer_maskdrop deferdx eval-suite --model $M --adapter outputs/runs/nodefer/final --name grpo_nodefer_maskdrop \
  --no-defer --mask-policy drop_sentence --sets "${CDM_SETS[@]}"
step eval_zs_closed deferdx eval-suite --model $M --name zs_closed --no-defer --closed-world --sets "${CDM_SETS[@]}"
step eval_gptoss deferdx eval-suite --model openai/gpt-oss-20b --name gptoss_nodefer --no-defer \
  --temperature 1.0 --top-p 1.0 --top-k 0 --max-new-tokens 2048
step report_final python scripts/make_report.py
step figures_final python scripts/make_figures.py
echo "$(date -Is) queue v2 finished"
