# JAMIA manuscript

The DEFER-Dx paper, formatted for a *JAMIA* Research and Applications article. It is written in LaTeX and built into PDFs. `docs/PAPER.md` stays as the long-form master. This folder is the condensed journal version.

| File | What it is |
|---|---|
| [main.pdf](main.pdf) | Manuscript: title page, structured abstract, Background and Significance, Objective, Materials and Methods, Results, Discussion, Conclusion, statements, references; line numbered |
| [supplement.pdf](supplement.pdf) | Supplementary Material: extended methods, 2 tables, 11 figures |
| `main.tex`, `supplement.tex` | LaTeX sources |
| `figures/fig_overview.pdf`, `fig_cev.pdf`, `fig_clinician.pdf`, `fig_data.pdf` | Illustrated method diagrams (new; TikZ sources in `figures/src/`, shared style `figures/src/dfxviz.sty`) |
| `figures/fig_*.pdf` (the rest) | Data plots, copied from `docs/figures/` (made by `scripts/make_figures.py`) |
| `build.sh` | Builds the diagrams, then both PDFs |
| `count_words.py` | Checks the JAMIA limits and scans the prose for hyphens and dashes |
| `check_figure_numbers.py` | Grounding check: recomputes every number printed in the diagrams from its source (configs, `docs/RESULTS.md`, `docs/METHODS.md`, `docs/team.json`, `docs/cev_audit.json`, `docs/casereports.json`) and confirms the printed string matches |
| `figures/src/extract_training_curve.py` | Recovers DEFER-Dx's measured in set deferral curve (Figure 2 inset) from the vector plot `docs/figures/fig_training_dynamics.svg`, checking the axis mapping against the gridlines and the 30% cap line; writes `figures/src/data/cev_inset_defer.dat` |

## Build and check

```bash
bash paper/build.sh              # needs tectonic; it downloads LaTeX packages on the first run
python3 paper/count_words.py     # exits 1 if a limit is exceeded or a hyphen or dash appears in the prose
python3 paper/check_figure_numbers.py   # exits 1 if any number in a diagram differs from its source
```

Last check: abstract 248 words (limit 250), main text 3,137 words (limit 4,000), 6 figures (limit 6), 3 tables (limit 4), and no hyphens or dashes in the prose.

## JAMIA requirements and where they are met

| Requirement (Research and Applications) | Where |
|---|---|
| 4,000 words, Background and Significance section | `main.tex`; count with `count_words.py` |
| Structured abstract, 250 words: Objective, Materials and Methods, Results, Discussion, Conclusion | `main.tex`, Abstract |
| At most 4 tables and 6 figures; more go online as a Supplementary File | Tables 1 to 3; Figures 1 to 6; `supplement.pdf` |
| Alt text under every figure legend, starting "Alt text:" | every figure in both files |
| Title page: authors, degrees, affiliations, corresponding author, up to 5 keywords, word count | `main.tex` title page (**authors to fill in**) |
| Funding, conflicts of interest, CRediT roles, data availability, disclosure of AI use (also in the cover letter) | `main.tex` statements (**authors to fill in or confirm**) |
| Submission format: Word, or a PDF compiled from LaTeX | this folder (LaTeX) |

## Writing rules for this manuscript

- No em dashes, and no hyphens in running prose.
- Ranges are written "93.4 to 96.8". Compound modifiers are left open: "zero shot", "open world", "time critical".
- Hyphens stay only where they are part of a name or a quoted source:
  - names: MIMIC-IV, Qwen3-8B, DEFER-Dx, LA-CDM;
  - titles of cited works;
  - the labels printed inside the data plots in `docs/figures/`.
- Every number comes from `docs/RESULTS.md` or `docs/PAPER.md`.
- **Diagrams are grounded in data:**
  - every number they print is checked by `check_figure_numbers.py`;
  - the Figure 2 controller curve is the measured training curve;
  - the result bars in Figures 1 and 5 are measured results;
  - the only illustrative element is the example episode in Figure 2a, which is labelled "illustrative episode".

## Before submission

- Author list, affiliations, corresponding author, funding, conflicts, CRediT roles and acknowledgments.
- Confirm the wording of the AI use statement and the Ethics sentence on the one cloud GPU training run. The consensus reward's second seed was trained on a rented Lambda Cloud instance; see `docs/CONSENSUS_SEED1.md`.
- Complete the references cited only by arXiv number (refs 6, 7, 14 to 17) with full author lists. Complete the author list of ref 9 (Savage et al.).
- Optionally evaluate the consensus reward's second seed. Its adapter exists but has not been evaluated; the Limitations section says so.
