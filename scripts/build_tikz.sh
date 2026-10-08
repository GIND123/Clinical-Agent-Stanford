#!/usr/bin/env bash
# Compile the TikZ schematics in docs/figures/tikz/ to docs/figures/<name>.{pdf,svg,png}.
#   bash scripts/build_tikz.sh                 # every fig_*.tex
#   bash scripts/build_tikz.sh fig_cev         # one figure
# Needs pdflatex with tikz, standalone and sansmath (TinyTeX: tlmgr install standalone sansmath)
# and pdftocairo (poppler). Fixed dates keep rebuilds byte-identical.
set -euo pipefail
cd "$(dirname "$0")/.."
src=docs/figures/tikz
out=docs/figures
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
if [[ $# -gt 0 ]]; then names=("$@"); else mapfile -t names < <(cd "$src" && ls fig_*.tex | sed 's/\.tex$//'); fi
export TEXINPUTS="$PWD/$src:" SOURCE_DATE_EPOCH=1791417600 FORCE_SOURCE_DATE=1
for n in "${names[@]}"; do
  if ! pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$build" "$src/$n.tex" > "$build/$n.out"; then
    grep -nE "^!|Error|l\.[0-9]+" "$build/$n.log" | head -20
    exit 1
  fi
  grep -E "Overfull|Underfull|Missing character|Font Warning" "$build/$n.log" | sort | uniq -c | sed "s/^/  $n: /" || true
  cp "$build/$n.pdf" "$out/$n.pdf"
  pdftocairo -svg "$out/$n.pdf" "$out/$n.svg"
  pdftocairo -png -r 300 -singlefile "$out/$n.pdf" "$out/$n"
  echo "$n -> $out/$n.{pdf,svg,png}"
done
