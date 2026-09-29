#!/usr/bin/env bash
# Greedy evaluation of one model/adapter with and without DEFER, plus the matched
# post-hoc threshold baseline. Usage:
#   MODEL=Qwen/Qwen3-8B ADAPTER=outputs/grpo/final NAME=deferdx bash scripts/eval_grid.sh
set -euo pipefail
: "${MODEL:?set MODEL}"
NAME=${NAME:-model}
OUT=${OUT:-outputs/eval/$NAME}
TEST=${TEST:-data/cdm/test.jsonl}
VAL=${VAL:-data/cdm/val.jsonl}
OTHER=${OTHER:-data/openworld/other.jsonl}
ADAPTER_ARG=()
[[ -n "${ADAPTER:-}" ]] && ADAPTER_ARG=(--adapter "$ADAPTER")
mkdir -p "$OUT"

# with DEFER (in-set test + OTHER)
deferdx rollout --policy vllm --model "$MODEL" "${ADAPTER_ARG[@]}" --cases "$TEST" "$OTHER" --out "$OUT/defer.jsonl"
# forced choice on the same cases = counterfactual for deferral precision/recall
deferdx rollout --policy vllm --model "$MODEL" "${ADAPTER_ARG[@]}" --no-defer --cases "$TEST" "$OTHER" --out "$OUT/nodefer_test.jsonl"
deferdx rollout --policy vllm --model "$MODEL" "${ADAPTER_ARG[@]}" --no-defer --cases "$VAL" --out "$OUT/nodefer_val.jsonl"

deferdx evaluate --rollouts "$OUT/defer.jsonl" --counterfactual "$OUT/nodefer_test.jsonl" --out "$OUT/defer_report.json"
deferdx evaluate --rollouts "$OUT/nodefer_test.jsonl" --out "$OUT/nodefer_report.json"

# post-hoc threshold at DEFER-Dx's own coverage (matched comparison)
COV=$(python -c "import json;print(json.load(open('$OUT/defer_report.json'))['closed_world']['coverage'])")
deferdx baseline --method coverage --target "$COV" --val "$OUT/nodefer_val.jsonl" --test "$OUT/nodefer_test.jsonl" \
    --out "$OUT/threshold_matched_report.json"
deferdx baseline --method sgr --target 0.05 --val "$OUT/nodefer_val.jsonl" --test "$OUT/nodefer_test.jsonl" \
    --out "$OUT/sgr_report.json"
echo "reports in $OUT"
