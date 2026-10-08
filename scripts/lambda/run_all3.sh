#!/usr/bin/env bash
# Third inference-only round (2026-10): extra zero-shot baselines on the SAME evaluation cohorts as the
# trained models (data/cohorts, built by `deferdx data cohorts`), in eval-suite layout so
# scripts/make_report.py reads them like any other system. Conventions as run_all.sh: one line per phase
# event in ~/runs/status.log; each full run is gated by a smoke run; SKIP="a b" resumes a partial run.
#   ~/payload/cohorts/eval_*.jsonl   the five evaluation sets (MIMIC text: only these are uploaded)
set -uo pipefail
source ~/venv/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1
cd ~/Clinical-Agent-Stanford
mkdir -p data ~/runs ~/outputs/eval && ln -sfn ~/payload/cohorts data/cohorts
LOG=~/runs/status.log
O=~/outputs/eval
Q14=Qwen/Qwen3-14B
DA=Henrychur/DiagAgent-14B
SETS=(data/cohorts/eval_cdm_val.jsonl data/cohorts/eval_cdm_test.jsonl data/cohorts/eval_other_seen.jsonl
      data/cohorts/eval_other_unseen.jsonl data/cohorts/eval_controls.jsonl)

phase() { echo "$(date -u +%FT%TZ) PHASE $1 $2" | tee -a "$LOG"; }
run() {
  local name=$1; shift
  case " ${SKIP:-} " in *" $name "*) phase "$name" SKIPPED; return 0 ;; esac
  phase "$name" START
  "$@" > ~/runs/"$name".log 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then phase "$name" END; else phase "$name" "FAILED rc=$rc"; fi
  return $rc
}
da() { python scripts/lambda/diagagent_adapter.py --model "$DA" --open-world --cases "${SETS[@]}" --out "$O" "$@"; }

# 1. DiagAgent-14B (domain RL on MIMIC-IV) through its format adapter, open world, forced choice. Greedy: one seed.
run smoke_da da --suite smoke_da --limit 3 \
  && run diagagent_nodefer da --suite diagagent_nodefer

# 2. Qwen3-14B zero-shot: forced choice, 3 seeds (thresholded post hoc in the report), and prompted DEFER,
#    1 seed (budget: on a 40 GB A100 each seed of the five sets is about 1.2 h)
run smoke_q14 deferdx eval-suite --model "$Q14" --name smoke_q14 --no-defer --seeds 0 --limit 3 --out "$O" \
  && run qwen14b_nodefer deferdx eval-suite --model "$Q14" --name qwen14b_nodefer --no-defer --out "$O" \
  && run qwen14b_defer deferdx eval-suite --model "$Q14" --name qwen14b_defer --seeds 0 --out "$O"

phase ALL DONE
