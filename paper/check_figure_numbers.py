"""Grounding check for the illustrated diagrams (paper/figures/src/fig_*.tex): every number they print is
recomputed here from its source in the repository, and the exact printed string must appear in the figure.
Sources: configs/*.yaml, docs/RESULTS.md, docs/METHODS.md, docs/team.json, docs/cev_audit.json,
docs/casereports.json, and the training curve recovered by figures/src/extract_training_curve.py.
Usage: python3 paper/check_figure_numbers.py   (exit 1 on any mismatch)
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).with_name("figures") / "src"
fig = {p.stem: p.read_text(encoding="utf-8") for p in SRC.glob("fig_*.tex")}
results = (ROOT / "docs/RESULTS.md").read_text(encoding="utf-8")
methods = (ROOT / "docs/METHODS.md").read_text(encoding="utf-8")
team = json.loads((ROOT / "docs/team.json").read_text())["systems"]
team_meta = json.loads((ROOT / "docs/team.json").read_text())
audit = json.loads((ROOT / "docs/cev_audit.json").read_text())["runs"]
cases = json.loads((ROOT / "docs/casereports.json").read_text())
base = (ROOT / "configs/base.yaml").read_text()
cev_cfg = (ROOT / "configs/grpo_cev.yaml").read_text()
grpo_cfg = (ROOT / "configs/grpo_deferdx.yaml").read_text()
catalog = (ROOT / "configs/test_catalog.yaml").read_text()


def cfg(text, key):
    return re.search(rf"\b{key}:\s*([^\s,}}#]+)", text).group(1)


def results_row(section, system, col):
    block = results[results.index(f"## {section}"):]
    block = block[:block.index("\n## ", 3)]
    row = next(line for line in block.splitlines() if line.startswith(f"| {system} |"))
    return row.split("|")[col + 1].split("(")[0].strip()


def pct(x):
    return f"{100 * x:.1f}"


CEV = "DEFER-Dx: counterfactual escalation value + scored handoff (ours)"
n_tests = len(re.findall(r"(?m)^  [a-z_]+:\s*$", catalog[catalog.index("\ntests:"):]))
rows = [
    # (figure, printed string, value from the source, source)
    ("fig_overview", "22 priced tests", f"{n_tests} priced tests", "configs/test_catalog.yaml, tests"),
    ("fig_overview", "at most 8", f"at most {cfg(base, 'max_steps')}", "configs/base.yaml, max_steps"),
    ("fig_overview", "Qwen3-8B", cfg(grpo_cfg, "model").split("/")[1], "configs/grpo_deferdx.yaml, model"),
    ("fig_overview", "answered 77.6", f"answered {results_row('1.', CEV, 4)}", "RESULTS.md 1, coverage"),
    ("fig_overview", "answered cases 95.2", f"answered cases {results_row('1.', CEV, 5)}", "RESULTS.md 1, selective acc"),
    ("fig_overview", "deferred 87.0", f"deferred {results_row('3b.', CEV, 4)}", "RESULTS.md 3b, defer"),
    ("fig_overview", "wrong commitment 6.8", f"wrong commitment {results_row('3b.', CEV, 1)}", "RESULTS.md 3b, false commit"),
    ("fig_cev", "K = 3", f"K = {cfg(cev_cfg, 'cev_k')}", "configs/grpo_cev.yaml, cev_k"),
    ("fig_cev", "at most 32", f"at most {cfg(cev_cfg, 'cev_max_roots')}", "configs/grpo_cev.yaml, cev_max_roots"),
    ("fig_cev", r"\eta = 0.3", rf"\eta = {cfg(cev_cfg, 'cev_eta')}", "configs/grpo_cev.yaml, cev_eta"),
    ("fig_cev", r"\mu = 0.1", rf"\mu = {cfg(base, 'handoff_mu')}", "configs/base.yaml, handoff_mu"),
    ("fig_cev", "0.95 to 0.85", f"{cfg(grpo_cfg, 'start')} to {cfg(grpo_cfg, 'end')}", "configs/grpo_deferdx.yaml, tau_schedule"),
    ("fig_cev", "12 cases", f"{cfg(grpo_cfg, 'cases_per_step')} cases", "configs/grpo_deferdx.yaml, cases_per_step"),
    ("fig_cev", "8 rollouts", f"{cfg(grpo_cfg, 'group_size')} rollouts", "configs/grpo_deferdx.yaml, group_size"),
    ("fig_cev", "(150 steps)", f"({cfg(grpo_cfg, 'steps')} steps)", "configs/grpo_deferdx.yaml, steps"),
    ("fig_cev", "30\\% cap", f"{round(100 * float(cfg(grpo_cfg, 'rho_max')))}\\% cap", "configs/grpo_deferdx.yaml, rho_max"),
    ("fig_cev", "88\\% and 92\\%", f"{100 * audit['cev_s0']['in_set']['noise']['K3']['sign_agreement_with_K16']:.0f}\\% and "
     f"{100 * audit['cev_s1']['in_set']['noise']['K3']['sign_agreement_with_K16']:.0f}\\%", "cev_audit.json, K3 sign agreement"),
    ("fig_cev", "60\\% and 37\\%", f"{100 * audit['cev_s0']['in_set']['outcomes']['credited_share']:.0f}\\% and "
     f"{100 * audit['cev_s1']['in_set']['outcomes']['credited_share']:.0f}\\%", "cev_audit.json, credited share"),
    ("fig_clinician", "first seed: 80.9", f"first seed: {pct(team['cev']['handoff_top_wrong_inset'])}", "team.json"),
    ("fig_clinician", "second seed: 31.9", f"second seed: {pct(team['cev_seed1']['handoff_top_wrong_inset'])}", "team.json"),
    ("fig_clinician", "first seed} & 86.8\\% & 15.8", f"first seed}} & {pct(team['cev']['team_acc_inset_f0.47']['value'])}\\% & "
     f"{pct(team['cev']['wrong_inset_unseen_f0.47']['value'])}", "team.json, f = 0.47"),
    ("fig_clinician", "second seed} & 91.4\\% & 37.0", f"second seed}} & {pct(team['cev_seed1']['team_acc_inset_f0.47']['value'])}\\% & "
     f"{pct(team['cev_seed1']['wrong_inset_unseen_f0.47']['value'])}", "team.json, f = 0.47"),
    ("fig_clinician", "threshold} & 91.5\\% & 11.0", f"threshold}} & {pct(team['grpo_nodefer+thr@cev']['team_acc_inset_f0.47']['value'])}\\% & "
     f"{pct(team['grpo_nodefer+thr@cev']['wrong_inset_unseen_f0.47']['value'])}", "team.json, f = 0.47"),
    ("fig_clinician", "a_c = 0.89", f"a_c = {team_meta['a_c']}", "team.json, a_c"),
    ("fig_clinician", "f = 0.47$:", f"f = {team_meta['barrit_f']}$:", "team.json, barrit_f"),
    ("fig_data", "2,400 emergency", "2,400 emergency" if "2,400 admissions" in methods else "?", "METHODS.md"),
]
for n in ("957 appendicitis", "648 cholecystitis", "538 pancreatitis", "257 diverticulitis", "713 OTHER", "648 controls"):
    printed = n.replace("648 controls", "648 same pipeline controls")
    rows.append(("fig_data", printed, printed if n in methods else "?", "METHODS.md"))
for printed, needle in (("2,404", "2,404 = 1,760 CDM + 279 OTHER + 365 controls"), ("1,760 CDM, 279 OTHER, 365", "2,404 = 1,760 CDM + 279 OTHER + 365 controls"),
                        ("212", "| dev | 212"), ("206", "seen groups | 206"), ("187", "unseen groups | 187"), ("258", "| controls | 258")):
    rows.append(("fig_data", printed, printed if needle in methods else "?", "METHODS.md cohort table"))
    if printed in ("206", "187", "258"):
        rows.append(("fig_overview", f"/{printed}/", f"/{printed}/" if needle in methods else "?", "METHODS.md cohort table"))
rows.append(("fig_data", "/480/", "/480/" if "val + test (480" in results or "480 cases" in results else "?", "RESULTS.md"))
rows.append(("fig_data", "/32/", f"/{cases['n_cases']}/", "casereports.json, n_cases"))
cr = cases["cases"].values()
n_out = sum(c["label"] == "other" for c in cr)
n_in = sum(c["label"] != "other" for c in cr)
n_cx = sum(bool(c["complicated"]) for c in cr if c["label"] != "other")
rows.append(("fig_data", "19 out of scope, 13 in scope (9 complicated)",
             f"{n_out} out of scope, {n_in} in scope ({n_cx} complicated)", "casereports.json, per case labels"))

curve = [line.split() for line in (SRC / "data/cev_inset_defer.dat").read_text().splitlines() if not line.startswith("#")]
late = [float(r) for s, r in curve if int(s) > 100]
rows.append(("fig_cev", "data/cev_inset_defer.dat", "data/cev_inset_defer.dat", f"recovered training curve: {len(curve)} points, "
             f"steps {curve[0][0]} to {curve[-1][0]}, mean of steps 101 to 150 {sum(late) / len(late):.1f}% (cap 30%)"))

bad = 0
for name, printed, truth, source in rows:
    ok = printed == truth and printed in fig[name]
    bad += not ok
    print(f"{'ok ' if ok else 'BAD'} {name:14s} {printed!r:34s} <- {source}" + ("" if ok else f"  (source gives {truth!r})"))
print(f"{len(rows) - bad}/{len(rows)} printed values match their sources")
sys.exit(1 if bad else 0)
