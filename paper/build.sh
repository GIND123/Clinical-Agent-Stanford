#!/usr/bin/env bash
# Build the JAMIA manuscript: the illustrated diagrams (figures/src/*.tex), then main.pdf and supplement.pdf.
#   bash paper/build.sh            # needs tectonic (https://tectonic-typesetting.github.io); fetches packages on first run
# The data plots in figures/ are copies of docs/figures/*.pdf, made by scripts/make_figures.py.
set -euo pipefail
cd "$(dirname "$0")"
log=$(mktemp)
trap 'rm -f "$log"' EXIT
run() { "$@" >"$log" 2>&1 || { grep -E "^error|^!" "$log" | head -20; exit 1; }; }
for src in figures/src/fig_*.tex; do
  (cd figures/src && run tectonic -X compile "$(basename "$src")")
  mv "figures/src/$(basename "${src%.tex}").pdf" figures/
done
for doc in main supplement; do
  run tectonic -X compile "$doc.tex"
done
echo "built: paper/main.pdf, paper/supplement.pdf and $(ls figures/src/fig_*.tex | wc -l | tr -d ' ') diagrams"
