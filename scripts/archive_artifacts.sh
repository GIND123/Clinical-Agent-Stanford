#!/usr/bin/env bash
# Preserve every weight and artifact. Safe to rerun (e.g. after the journal runs finish):
#   1. a hard-link snapshot of outputs/runs (no extra space; survives checkpoint pruning and deletion of the
#      originals) plus copies of outputs/eval and outputs/logs, in outputs/archive/<timestamp>/ (git-ignored;
#      rollouts and evaluation files hold MIMIC text and never leave this machine);
#   2. every LoRA adapter (finals and kept checkpoints, partial runs included) on PRIVATE Hugging Face repos:
#      finals on main (the sync daemon also does this), checkpoints and partial runs on named branches.
#      Each upload carries the adapter, its config and the numbers-only training log; unchanged files are skipped.
#   bash scripts/archive_artifacts.sh
set -u
cd "$(dirname "$0")/.."
source scripts/env_gpu.sh
dest=outputs/archive/$(date +%Y%m%d-%H%M)
mkdir -p "$dest"
cp -al outputs/runs "$dest/runs"
cp -a outputs/eval "$dest/eval"
cp -a outputs/logs "$dest/logs"
echo "snapshot: $dest ($(du -sh "$dest" | cut -f1) on disk counting hard links)"

push() {  # push RUN_REPO RUN_DIR CHECKPOINT|final BRANCH|main NOTE
  local repo_run=$1 dir=$2 ck=$3 branch=$4 note=$5 args
  args=(--run "$repo_run" --run-dir "$dir" --note "$note")
  [[ $ck != final ]] && args+=(--checkpoint "$ck")
  [[ $branch != main ]] && args+=(--revision "$branch")
  if python scripts/push_hf.py "${args[@]}" > /dev/null 2>&1; then echo "  ok   $dir/$ck -> $repo_run:$branch"
  else echo "  FAIL $dir/$ck -> $repo_run:$branch"; fi
}
ckpts() { ls -d outputs/runs/"$1"/checkpoints/step_* 2>/dev/null | xargs -rn1 basename; }

echo "Hugging Face (private):"
# DEFER-Dx, the method
[[ -d outputs/runs/cev/final ]] && push cev cev final main "DEFER-Dx, the method: final adapter (150 steps)."
for ck in $(ckpts cev); do
  push cev cev "$ck" "${ck//_/}" "DEFER-Dx, the method: checkpoint at ${ck#step_} steps."
done
# the first DEFER-Dx run (plain dual-ascent coverage cap) and the crashed PI attempt: records, not results
[[ -d outputs/runs/cev_dualascent/final ]] && push cev cev_dualascent final dual-ascent-final \
  "First CEV run: plain dual-ascent coverage cap (nu ran a limit cycle). A record; the method's adapter is on main."
for ck in $(ckpts cev_dualascent); do
  push cev cev_dualascent "$ck" "dual-ascent-${ck//_/}" "First CEV run (plain dual-ascent cap), checkpoint ${ck#step_}. A record."
done
for ck in $(ckpts cev_pi_crashed); do
  push cev cev_pi_crashed "$ck" "pi-crashed-${ck//_/}" "First PI-controller attempt, stopped by an out-of-memory error at step 29. A record."
done
# the comparator arm, the control, the ablations and the second seed: finals on main, checkpoints on branches
for run in deferdx nodefer abl_cev_no_handoff abl_constant_defer cev_seed1; do
  [[ -d outputs/runs/$run ]] || continue
  repo_run=$run; [[ $run == cev_seed1 ]] && repo_run=cev
  if [[ -d outputs/runs/$run/final ]]; then
    if [[ $run == cev_seed1 ]]; then push cev cev_seed1 final seed1-final "DEFER-Dx, second training seed: final adapter."
    else push "$run" "$run" final main "Final adapter."; fi
  fi
  for ck in $(ckpts "$run"); do
    prefix=$([[ $run == cev_seed1 ]] && echo "seed1-" || echo "")
    status=$([[ -d outputs/runs/$run/final ]] && echo "checkpoint" || echo "partial run, checkpoint")
    push "$repo_run" "$run" "$ck" "${prefix}${ck//_/}" "${run}: ${status} at ${ck#step_} steps."
  done
done
echo "done"
