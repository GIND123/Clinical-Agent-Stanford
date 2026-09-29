#!/usr/bin/env bash
# End-to-end pipeline check on SYNTHETIC data (no MIMIC, no GPU). Numbers are meaningless.
set -euo pipefail
OUT=${OUT:-outputs/smoke}
DX="python -m deferdx"
SEP=$(python -c "import os; print(os.pathsep)")
export PYTHONPATH="src${PYTHONPATH:+$SEP$PYTHONPATH}"  # works without pip install -e

$DX data synth --out "$OUT/data" --n 400 --n-other 80
$DX crossover
$DX data coverage --cases "$OUT/data/train.jsonl" --top 5

# scripted policies
$DX rollout --cases "$OUT/data/test.jsonl" "$OUT/data/other.jsonl" --policy oracle --out "$OUT/oracle.jsonl"
$DX evaluate --rollouts "$OUT/oracle.jsonl" --out "$OUT/oracle_report.json"
$DX rollout --cases "$OUT/data/test.jsonl" "$OUT/data/other.jsonl" --policy random --samples 3 --out "$OUT/random.jsonl"
$DX evaluate --rollouts "$OUT/random.jsonl" --out "$OUT/random_report.json"

# post-hoc threshold baseline on a forced-choice policy
$DX rollout --cases "$OUT/data/val.jsonl" --policy random --no-defer --samples 3 --out "$OUT/val_nodefer.jsonl"
$DX rollout --cases "$OUT/data/test.jsonl" --policy random --no-defer --out "$OUT/test_nodefer.jsonl"
$DX baseline --method coverage --target 0.8 --val "$OUT/val_nodefer.jsonl" --test "$OUT/test_nodefer.jsonl" \
    --out "$OUT/threshold_report.json"

# SFT data
$DX sft-data oracle --cases "$OUT/data/train.jsonl" "$OUT/data/other.jsonl" --out "$OUT/sft_oracle.jsonl"
echo "smoke test complete -> $OUT"
