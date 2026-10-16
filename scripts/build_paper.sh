#!/usr/bin/env bash
# Typeset docs/PAPER.md into docs/paper/DEFER-Dx.pdf (pandoc + xelatex; vector figures from docs/figures/).
#   bash scripts/build_paper.sh          # KEEP=1 keeps the build directory for debugging
set -euo pipefail
cd "$(dirname "$0")/.."
build=$(mktemp -d)
trap '[[ -n "${KEEP:-}" ]] && echo "build dir $build" || rm -rf "$build"' EXIT
# Preprocess the Markdown master: drop the title and editorial note (the template sets the title block), point figure
# links at the vector PDFs (docs/paper/filter.lua swaps .png for .pdf), turn "Fig. N | " prefixes into caption labels,
# and start Extended Data numbering.
python3 - "$build/paper.md" <<'EOF'
import re
import sys
text = open("docs/PAPER.md", encoding="utf-8").read()
text = text[text.index("## Abstract"):]
text = text.replace("](docs/figures/", "](../figures/")
text = re.sub(r"^!\[(?:Extended Data )?Fig\. \d+ \| ", "![", text, flags=re.M)
text = text.replace("\n## Extended Data\n", "\n\\extendeddata\n\n## Extended Data\n", 1)
open(sys.argv[1], "w", encoding="utf-8").write(text)
EOF
pandoc "$build/paper.md" --from markdown+pipe_tables+implicit_figures+raw_tex --to latex --standalone \
  --template docs/paper/template.tex --lua-filter docs/paper/filter.lua --resource-path docs/paper \
  --metadata-file docs/paper/meta.yaml -o "$build/paper.tex"
(cd docs/paper && SOURCE_DATE_EPOCH=1791417600 latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -output-directory="$build" "$build/paper.tex" > "$build/latexmk.log" 2>&1) || { grep -nE "^!|Error" "$build/paper.log" | head -20; exit 1; }
cp "$build/paper.pdf" docs/paper/DEFER-Dx.pdf
echo "docs/paper/DEFER-Dx.pdf ($(pdfinfo docs/paper/DEFER-Dx.pdf | awk '/Pages/ {print $2}') pages)"
