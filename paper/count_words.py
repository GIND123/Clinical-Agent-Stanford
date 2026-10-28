"""Word counts for the JAMIA limits: abstract (250) and main text (4,000; Background to Conclusion,
excluding figures, tables, captions and references). Also flags hyphens and dashes in running prose.
Usage: python3 paper/count_words.py"""
import re
import sys
from pathlib import Path

tex = Path(__file__).with_name("main.tex").read_text(encoding="utf-8")


def strip(s):
    s = re.sub(r"(?m)(?<!\\)%.*$", "", s)
    s = re.sub(r"\\begin\{(figure|table)\}.*?\\end\{\1\}", " ", s, flags=re.S)
    s = re.sub(r"\$[^$]*\$", " X ", s)
    s = re.sub(r"\\\[.*?\\\]", " ", s, flags=re.S)
    s = re.sub(r"\\cite\{[^}]*\}", "", s)
    s = re.sub(r"\\(section|subsection)\*\{[^}]*\}", " ", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^]]*\])?", " ", s)
    s = re.sub(r"[{}~]", " ", s)
    return s


def words(s):
    return len([w for w in s.split() if re.search(r"[A-Za-z0-9]", w)])


abstract = tex[tex.index(r"\section*{Abstract}"):tex.index(r"\newpage")]
body = tex[tex.index(r"\section*{Background"):tex.index(r"\section*{Funding}")]
n_abs, n_body = words(strip(abstract)), words(strip(body))
print(f"abstract: {n_abs} words (limit 250)")
print(f"main text: {n_body} words (limit 4000)")
figs = len(re.findall(r"\\begin\{figure\}", tex))
tabs = len(re.findall(r"\\begin\{table\}", tex))
print(f"figures: {figs} (limit 6), tables: {tabs} (limit 4)")

# prose check: hyphens between letters and dashes, outside proper names, maths, URLs and references
names = r"MIMIC-IV-Ext-CDM|MIMIC-IV-ED|MIMIC-IV-Note|MIMIC-IV|MIMIC-CDM-OW|MIMIC-CDM-CR|MIMIC-CDM|Qwen3-8B|DEFER-Dx|LA-CDM|ICD-9|ICD-10|AWA-RL|TrustMed-RL|BI-RADS|El-Yaniv"
prose = tex[:tex.index(r"\begin{thebibliography}")]
prose = re.sub(r"(?m)(?<!\\)%.*$", "", prose)
prose = re.sub(r"\$[^$]*\$|\\url\{[^}]*\}|\\(label|ref|cite|includegraphics)(\[[^]]*\])?\{[^}]*\}", " ", prose)
prose = re.sub(names, " ", prose)
bad = [m.group(0) for m in re.finditer(r"\w+-\w+|\u2014|\u2013|---|--", prose)]
print(f"hyphens or dashes in prose: {len(bad)}", bad[:20])
sys.exit(0 if n_abs <= 250 and n_body <= 4000 and figs <= 6 and tabs <= 4 and not bad else 1)
