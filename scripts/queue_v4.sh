#!/usr/bin/env bash
# GPU job queue, version 4: the final run. Replaces queue_v3, whose DEFER-Dx run died of a CUDA
# out-of-memory error at step 29 (another process took 4.6 GiB of the GPU during the night), after which
# queue_v3 moved on without the method. Commit 416ed87 also changed how CEV trains the deferrals it does
# not branch, and 6 of that attempt's 28 steps ran under the old rule, so the attempt is kept as
# cev_pi_crashed and the method is trained from step 1 on the final code.
#
# Every step is retried: a training run resumes from its newest checkpoint (one every 5 steps), an
# evaluation is rerun from scratch (its seed files are overwritten). make_report.py counts only
# evaluation suites whose every seed finished.
#
#   kill <queue_v3 bash pid>      # the queue only; a job it started keeps running and is waited for
#   setsid nohup bash scripts/queue_v4.sh >> outputs/logs/queue_v4.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env_gpu.sh
mkdir -p outputs/logs outputs/queue

if pgrep -f "^(/usr)?(/bin/)?bash [^ ]*scripts/queue_v[23]\.sh" >/dev/null; then  # not shells that mention them
  echo "$(date -Is) an older queue is still running; stop it first (only the bash process)"
  exit 1
fi

gpu_free() {
  while [[ $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l) -gt 0 ]]; do
    sleep 60
  done
}

# step NAME TRIES CMD...: run CMD until it succeeds, at most TRIES times; a step that still fails is
# logged and the queue moves on (the steps that need its output then fail fast and are logged too).
step() {
  local name=$1 tries=$2
  shift 2
  if [[ -f "outputs/queue/$name.done" ]]; then echo "$(date -Is) skip $name (done)"; return 0; fi
  local i
  for ((i = 1; i <= tries; i++)); do
    gpu_free
    echo "$(date -Is) start $name (attempt $i of $tries): $*"
    if "$@" >> "outputs/logs/$name.log" 2>&1; then
      touch "outputs/queue/$name.done"
      echo "$(date -Is) done  $name"
      return 0
    fi
    echo "$(date -Is) FAILED $name (attempt $i of $tries), see outputs/logs/$name.log"
    sleep 120  # let the GPU memory of the failed job be released
  done
  return 1
}
trained() { [[ -d "outputs/runs/$1/final" ]] && touch "outputs/queue/$2.done"; return 0; }

M=Qwen/Qwen3-8B
CDM_SETS=(data/cohorts/eval_cdm_val.jsonl data/cohorts/eval_cdm_test.jsonl)
# The GPU is shared with a desktop session and other users, whose memory use (2-4 GiB, and growing)
# twice pushed a job out of memory. vLLM takes a fixed share, so this run leaves more of the card free:
# 0.72 instead of 0.80 when training and 0.78 instead of 0.88 when evaluating. Memory only: sampling,
# seeds and the algorithm are unchanged; generation is somewhat slower.
TRAIN_MEM=(--set vllm_gpu_memory_utilization=0.72)
EVAL_MEM=(--gpu-memory-utilization 0.78)

# 0. wait for whatever queue_v3 left on the GPU, then set the crashed attempt aside, once
echo "$(date -Is) queue v4 started; waiting for a free GPU"
gpu_free
if [[ ! -f outputs/queue/v4_cleanup.done ]]; then
  if [[ -d outputs/runs/cev && ! -d outputs/runs/cev/final ]]; then
    if [[ -e outputs/runs/cev_pi_crashed ]]; then
      echo "$(date -Is) outputs/runs/cev_pi_crashed already exists; refusing to overwrite it"
      exit 1
    fi
    mv outputs/runs/cev outputs/runs/cev_pi_crashed
    [[ -f outputs/logs/train_cev_pi.log ]] && mv outputs/logs/train_cev_pi.log outputs/logs/train_cev_pi_crashed.log
  fi
  # the failed evaluation of the attempt left an empty directory
  [[ -d outputs/eval/cev && ! -f outputs/eval/cev/summary.json ]] && rm -rf outputs/eval/cev
  touch outputs/queue/v4_cleanup.done
  echo "$(date -Is) crashed DEFER-Dx attempt moved to outputs/runs/cev_pi_crashed"
fi

# 1. MAIN: DEFER-Dx (counterfactual escalation value + scored handoff, PI coverage controller)
trained cev train_cev_v4
step train_cev_v4 3 deferdx train grpo-vllm --config configs/grpo_cev.yaml "${TRAIN_MEM[@]}"
step eval_cev_v4 2 deferdx eval-suite "${EVAL_MEM[@]}" --config configs/grpo_cev.yaml --model $M --adapter outputs/runs/cev/final --name cev
step report_v4_1 1 python scripts/make_report.py

# 2. control: identical training, no DEFER action (resumes from its checkpoint if it was interrupted)
trained nodefer train_nodefer
step train_nodefer 3 deferdx train grpo-vllm --config configs/grpo_nodefer.yaml "${TRAIN_MEM[@]}"
step eval_grpo_nodefer 2 deferdx eval-suite "${EVAL_MEM[@]}" --model $M --adapter outputs/runs/nodefer/final --name grpo_nodefer --no-defer
step report_v4_2 1 python scripts/make_report.py

# 3. ablations of what is new (100 steps; compared with DEFER-Dx's step-100 checkpoint)
step eval_cev_v4_step100 2 deferdx eval-suite "${EVAL_MEM[@]}" --config configs/grpo_cev.yaml --model $M \
  --adapter outputs/runs/cev/checkpoints/step_0100 --name cev_step100 --seeds 0 1
for abl in cev_no_handoff constant_defer; do
  trained abl_$abl train_abl_$abl
  step train_abl_$abl 3 deferdx train grpo-vllm --config configs/ablations/$abl.yaml "${TRAIN_MEM[@]}"
  cfgflag=(); [[ $abl == cev_* ]] && cfgflag=(--config configs/grpo_cev.yaml)
  step eval_abl_$abl 2 deferdx eval-suite "${EVAL_MEM[@]}" "${cfgflag[@]}" --model $M --adapter outputs/runs/abl_$abl/final \
    --name abl_$abl --seeds 0 1
done
step report_v4_3 1 python scripts/make_report.py

# 4. robustness, further baselines, and the coverage-controller comparison
step eval_cev_v4_maskdrop 2 deferdx eval-suite "${EVAL_MEM[@]}" --config configs/grpo_cev.yaml --model $M --adapter outputs/runs/cev/final \
  --name cev_maskdrop --mask-policy drop_sentence --sets "${CDM_SETS[@]}"
step eval_nodefer_maskdrop 2 deferdx eval-suite "${EVAL_MEM[@]}" --model $M --adapter outputs/runs/nodefer/final --name grpo_nodefer_maskdrop \
  --no-defer --mask-policy drop_sentence --sets "${CDM_SETS[@]}"
step eval_zs_closed 2 deferdx eval-suite "${EVAL_MEM[@]}" --model $M --name zs_closed --no-defer --closed-world --sets "${CDM_SETS[@]}"
step eval_gptoss 2 deferdx eval-suite "${EVAL_MEM[@]}" --model openai/gpt-oss-20b --name gptoss_nodefer --no-defer \
  --temperature 1.0 --top-p 1.0 --top-k 0 --max-new-tokens 2048
da=outputs/runs/cev_dualascent/final
[[ -d $da ]] || da=$(ls -d outputs/runs/cev_dualascent/checkpoints/step_* 2>/dev/null | tail -1)
step eval_cev_dualascent 2 deferdx eval-suite "${EVAL_MEM[@]}" --config configs/grpo_cev.yaml --model $M --adapter "$da" --name cev_dualascent
step hf_cev_dualascent 2 python scripts/push_hf.py --run cev --run-dir cev_dualascent --revision dual-ascent-final \
  --note "Final adapter of the first CEV run: plain dual-ascent coverage cap (nu ran a limit cycle) and the earlier rule for unbranched deferrals. Kept for the record; the method's adapter is on main."
step report_v4_final 1 python scripts/make_report.py
step figures_v4_final 1 python scripts/make_figures.py
echo "$(date -Is) queue v4 finished"
