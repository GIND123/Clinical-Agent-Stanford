#!/usr/bin/env bash
# GPU job queue, version 3: replaces queue_v2 after the coverage-controller change (c44d5c7).
# The first CEV run (plain dual ascent: nu ran a limit cycle and deferral collapsed repeatedly) is kept as
# cev_dualascent and evaluated as a comparison; the method is retrained with the PI controller under the
# name cev, so everything downstream (step-100 ablation comparison, mask-drop, report) uses the PI run.
#   kill <queue_v2 bash pid>          # the queue only; a job it started keeps running and is waited for
#   setsid nohup bash scripts/queue_v3.sh >> outputs/logs/queue_v3.log 2>&1 < /dev/null &
# Idempotent (.done markers), waits for a free GPU, resumes training from the newest checkpoint.
set -u
cd "$(dirname "$0")/.."
source scripts/env_gpu.sh
mkdir -p outputs/logs outputs/queue

if pgrep -f "^(/usr)?(/bin/)?bash [^ ]*scripts/queue_v2\.sh" >/dev/null; then  # not shells that merely mention it
  echo "$(date -Is) queue_v2 is still running; stop it first (only the bash process)"
  exit 1
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
trained() { [[ -d "outputs/runs/$1/final" ]] && touch "outputs/queue/$2.done"; return 0; }

M=Qwen/Qwen3-8B
CDM_SETS=(data/cohorts/eval_cdm_val.jsonl data/cohorts/eval_cdm_test.jsonl)

# 0. wait for whatever queue_v2 left running (the dual-ascent CEV run, or an evaluation of it), then
#    move that run and anything evaluated from it aside, once
gpu_free
if [[ ! -f outputs/queue/archive_cev_dualascent.done ]]; then
  if [[ -e outputs/runs/cev_dualascent ]]; then
    echo "$(date -Is) outputs/runs/cev_dualascent already exists; refusing to overwrite it"
    exit 1
  fi
  [[ -d outputs/runs/cev ]] && mv outputs/runs/cev outputs/runs/cev_dualascent
  [[ -f outputs/logs/train_cev.log ]] && mv outputs/logs/train_cev.log outputs/logs/train_cev_dualascent.log
  for e in cev cev_step100 cev_maskdrop; do  # evaluations queue_v2 may already have run on the old weights
    if [[ -d outputs/eval/$e ]]; then
      mv outputs/eval/$e outputs/eval/${e/cev/cev_dualascent}
      [[ -f outputs/logs/eval_$e.log ]] && mv outputs/logs/eval_$e.log outputs/logs/eval_${e/cev/cev_dualascent}.log
      [[ $e == cev ]] && touch outputs/queue/eval_cev_dualascent.done
    fi
  done
  touch outputs/queue/archive_cev_dualascent.done
  echo "$(date -Is) dual-ascent CEV run moved to outputs/runs/cev_dualascent"
fi

# 1. MAIN: counterfactual escalation value + scored handoff, PI coverage controller
trained cev train_cev_pi
step train_cev_pi deferdx train grpo-vllm --config configs/grpo_cev.yaml
step eval_cev_pi deferdx eval-suite --config configs/grpo_cev.yaml --model $M --adapter outputs/runs/cev/final --name cev
step report_v3_1 python scripts/make_report.py

# 2. control: identical training, no DEFER action (threshold its calibrated probability post hoc)
trained nodefer train_nodefer
step train_nodefer deferdx train grpo-vllm --config configs/grpo_nodefer.yaml
step eval_grpo_nodefer deferdx eval-suite --model $M --adapter outputs/runs/nodefer/final --name grpo_nodefer --no-defer
step report_v3_2 python scripts/make_report.py

# 3. ablations of what is new (100 steps; compared with the PI run's step-100 checkpoint)
step eval_cev_pi_step100 deferdx eval-suite --config configs/grpo_cev.yaml --model $M \
  --adapter outputs/runs/cev/checkpoints/step_0100 --name cev_step100 --seeds 0 1
for abl in cev_no_handoff constant_defer; do
  trained abl_$abl train_abl_$abl
  step train_abl_$abl deferdx train grpo-vllm --config configs/ablations/$abl.yaml
  cfgflag=(); [[ $abl == cev_* ]] && cfgflag=(--config configs/grpo_cev.yaml)
  step eval_abl_$abl deferdx eval-suite "${cfgflag[@]}" --model $M --adapter outputs/runs/abl_$abl/final \
    --name abl_$abl --seeds 0 1
done

# 4. robustness, further baselines, and the coverage-controller comparison
step eval_cev_pi_maskdrop deferdx eval-suite --config configs/grpo_cev.yaml --model $M --adapter outputs/runs/cev/final \
  --name cev_maskdrop --mask-policy drop_sentence --sets "${CDM_SETS[@]}"
step eval_nodefer_maskdrop deferdx eval-suite --model $M --adapter outputs/runs/nodefer/final --name grpo_nodefer_maskdrop \
  --no-defer --mask-policy drop_sentence --sets "${CDM_SETS[@]}"
step eval_zs_closed deferdx eval-suite --model $M --name zs_closed --no-defer --closed-world --sets "${CDM_SETS[@]}"
step eval_gptoss deferdx eval-suite --model openai/gpt-oss-20b --name gptoss_nodefer --no-defer \
  --temperature 1.0 --top-p 1.0 --top-k 0 --max-new-tokens 2048
da=outputs/runs/cev_dualascent/final
[[ -d $da ]] || da=$(ls -d outputs/runs/cev_dualascent/checkpoints/step_* 2>/dev/null | tail -1)
step eval_cev_dualascent deferdx eval-suite --config configs/grpo_cev.yaml --model $M --adapter "$da" --name cev_dualascent
step hf_cev_dualascent python scripts/push_hf.py --run cev --run-dir cev_dualascent --revision dual-ascent-final \
  --note "Final adapter of the first CEV run, whose coverage cap used plain dual ascent (nu ran a limit cycle and deferral collapsed repeatedly). Kept for the record; the method's adapter is on main."
step report_v3_final python scripts/make_report.py
step figures_v3_final python scripts/make_figures.py
echo "$(date -Is) queue v3 finished"
