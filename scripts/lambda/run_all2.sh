#!/usr/bin/env bash
# Second inference-only round (2026-10). Same conventions as run_all.sh: one line per phase event in
# ~/runs/status.log; each full run is gated by its own 20-case smoke run; a failed phase does not stop
# the phases after it. Ordered by value so a budget stop cuts the least important run.
set -uo pipefail
source ~/venv/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1
cd ~/Clinical-Agent-Stanford
LOG=~/runs/status.log
P=~/payload
O=~/outputs
Q8=Qwen/Qwen3-8B
Q14=Qwen/Qwen3-14B
DA=Henrychur/DiagAgent-14B
GO=openai/gpt-oss-20b

phase() { echo "$(date -u +%FT%TZ) PHASE $1 $2" | tee -a "$LOG"; }
run() {
  local name=$1; shift
  phase "$name" START
  "$@" > ~/runs/"$name".log 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then phase "$name" END; else phase "$name" "FAILED rc=$rc"; fi
  return $rc
}
roll() {  # roll <name> <model> <seed> <temperature> <extra flags...> -- <case files...>
  local name=$1 model=$2 seed=$3 temp=$4; shift 4
  local flags=() ; while [ "$1" != "--" ]; do flags+=("$1"); shift; done; shift
  run "$name" deferdx rollout --cases "$@" --policy vllm --model "$model" --seed "$seed" --temperature "$temp" \
      --out "$O/$name.jsonl" "${flags[@]}"
}
OW_CASES=("$P/cdm/test.jsonl" "$P/openworld/other.jsonl" "$P/openworld/controls.jsonl")

# 1. DiagAgent-14B through its format adapter (environment unchanged), all 2,400 cases
run smoke_da_adapter python "$P/diagagent_adapter.py" --cases "$P/cdm/test.jsonl" --limit 20 --out "$O/smoke_da_adapter.jsonl" \
  && run cdm_da_adapter_all python "$P/diagagent_adapter.py" --cases "$P/cdm/all.jsonl" --out "$O/cdm_da_adapter_all.jsonl"

# 2. Open world: Qwen3-8B on the LA-CDM test cases + OTHER cases + same-pipeline controls, OTHER offered
roll ow_qwen8b_nodefer "$Q8" 0 0.6 --no-defer -- "${OW_CASES[@]}"
roll ow_qwen8b_defer "$Q8" 0 0.6 -- "${OW_CASES[@]}"

# 3. DiagBench re-score with the fixed exam extractor (also re-checks the diagnosis scores)
run diagbench_rejudge python "$P/diagbench_score.py" --gen-dir "$P/diagbench_gen" --out "$O/diagbench_scores_v2.json"

# 4. gpt-oss-20b smoke only (A100 support in vLLM 0.11 is not guaranteed)
roll smoke_gptoss "$GO" 0 0.6 --no-defer --closed-world --limit 20 -- "$P/cdm/test.jsonl"

# 5. Qwen3-14B zero-shot, all 2,400 cases (28 GB of weights: cap the context to leave KV-cache room)
roll smoke_qwen14b "$Q14" 0 0.6 --no-defer --closed-world --max-model-len 16384 --limit 20 -- "$P/cdm/test.jsonl" \
  && roll cdm_qwen14b_all_s0 "$Q14" 0 0.6 --no-defer --closed-world --max-model-len 16384 -- "$P/cdm/all.jsonl"

phase ALL DONE
