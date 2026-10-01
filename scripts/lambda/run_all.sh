#!/usr/bin/env bash
# Inference-only comparison runs. Each phase logs START/END/FAILED to ~/runs/status.log.
# Smoke phases gate the full ones: if a smoke run fails, nothing expensive follows.
set -uo pipefail
source ~/venv/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1
cd ~/Clinical-Agent-Stanford
LOG=~/runs/status.log
P=~/payload
O=~/outputs
Q=Qwen/Qwen3-8B
DA=Henrychur/DiagAgent-14B

phase() { echo "$(date -u +%FT%TZ) PHASE $1 $2" | tee -a "$LOG"; }
run() {
  local name=$1; shift
  phase "$name" START
  "$@" > ~/runs/"$name".log 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then phase "$name" END; else phase "$name" "FAILED rc=$rc"; fi
  return $rc
}
cdm() {  # cdm <name> <model> <cases> <seed> <temperature> [--limit N]
  local name=$1 model=$2 cases=$3 seed=$4 temp=$5; shift 5
  run "$name" deferdx rollout --cases "$cases" --policy vllm --model "$model" --no-defer --closed-world \
      --seed "$seed" --temperature "$temp" --out "$O/$name.jsonl" "$@"
}

# 1. Qwen3-8B zero-shot on MIMIC-CDM (Qwen3's recommended thinking-mode sampling, T=0.6)
cdm smoke_qwen8b "$Q" "$P/cdm/test.jsonl" 0 0.6 --limit 20 || { phase ALL "STOPPED at smoke_qwen8b"; exit 1; }
cdm cdm_qwen8b_all_s0 "$Q" "$P/cdm/all.jsonl" 0 0.6

# 2. DiagAgent-14B in the same environment (its own recommended decoding: greedy, T=0)
cdm smoke_diagagent "$DA" "$P/cdm/test.jsonl" 0 0 --limit 20 || phase smoke_diagagent "continuing without cdm_diagagent_all"
if grep -q "PHASE smoke_diagagent END" "$LOG"; then cdm cdm_diagagent_all "$DA" "$P/cdm/all.jsonl" 0 0; fi

# 3. DiagAgent-14B on DiagBench (reproduce its SOTA), then judge with the paper's prompts
run diagbench_smoke python "$P/diagbench_run.py" --model "$DA" --data-dir "$P/diagbench" --out-dir "$O/diagbench_smoke" --limit 5 \
  && run diagbench_gen python "$P/diagbench_run.py" --model "$DA" --data-dir "$P/diagbench" --out-dir "$O/diagbench" \
  && run diagbench_judge python "$P/diagbench_score.py" --gen-dir "$O/diagbench" --out "$O/diagbench_scores.json"

# 4. Two more Qwen3-8B seeds on the LA-CDM test split (3 seeds total there)
cdm cdm_qwen8b_test_s1 "$Q" "$P/cdm/test.jsonl" 1 0.6
cdm cdm_qwen8b_test_s2 "$Q" "$P/cdm/test.jsonl" 2 0.6

phase ALL DONE
