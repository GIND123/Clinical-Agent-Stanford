#!/usr/bin/env bash
# Sequential GPU job queue for the DEFER-Dx experiments (one GPU, one job at a time).
#
#   WAIT_PID=<pid of a running job> setsid nohup bash scripts/queue.sh > outputs/logs/queue.log 2>&1 &
#
# Idempotent: a step whose outputs/queue/<name>.done marker exists is skipped, and training
# steps resume from their latest checkpoint, so the queue can be re-launched after any crash.
# A failed step is logged and the queue moves on.
set -u
cd "$(dirname "$0")/.."
source scripts/env_gpu.sh
mkdir -p outputs/logs outputs/queue

if [[ -n "${WAIT_PID:-}" ]]; then
  echo "$(date -Is) waiting for pid $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi

gpu_free() {  # block while any process holds > 2 GB of GPU memory (desktop apps use < 0.5 GB)
  while [[ $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l) -gt 0 ]]; do
    sleep 60
  done
}

step() {  # step <name> <command...>
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
CDM_SETS=(data/cohorts/eval_cdm_val.jsonl data/cohorts/eval_cdm_test.jsonl)

# 1. DEFER-Dx (resumes from its last checkpoint if it was interrupted)
[[ -d outputs/runs/deferdx/final ]] && touch outputs/queue/train_deferdx.done
step train_deferdx deferdx train grpo-vllm --config configs/grpo_deferdx.yaml
step eval_deferdx deferdx eval-suite --model $M --adapter outputs/runs/deferdx/final --name deferdx

# 2. zero-shot baselines on the same cohorts (forced choice; prompted DEFER)
step eval_zs_nodefer deferdx eval-suite --model $M --name zs_nodefer --no-defer
step eval_zs_defer deferdx eval-suite --model $M --name zs_defer
step report_1 python scripts/make_report.py

# 3. control arm: same data, reward and compute, no DEFER action
[[ -d outputs/runs/nodefer/final ]] && touch outputs/queue/train_nodefer.done
step train_nodefer deferdx train grpo-vllm --config configs/grpo_nodefer.yaml
step eval_grpo_nodefer deferdx eval-suite --model $M --adapter outputs/runs/nodefer/final --name grpo_nodefer --no-defer
step report_2 python scripts/make_report.py

# 4. robustness: the CDM "____" mask cue removed (drop_sentence), and the closed-world protocol of prior work
step eval_deferdx_maskdrop deferdx eval-suite --model $M --adapter outputs/runs/deferdx/final --name deferdx_maskdrop \
  --mask-policy drop_sentence --sets "${CDM_SETS[@]}"
step eval_nodefer_maskdrop deferdx eval-suite --model $M --adapter outputs/runs/nodefer/final --name grpo_nodefer_maskdrop \
  --no-defer --mask-policy drop_sentence --sets "${CDM_SETS[@]}"
step eval_zs_closed deferdx eval-suite --model $M --name zs_closed --no-defer --closed-world --sets "${CDM_SETS[@]}"
step report_3 python scripts/make_report.py
echo "$(date -Is) queue finished"
