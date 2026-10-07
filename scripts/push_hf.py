#!/usr/bin/env python
"""Upload a trained LoRA adapter to a PRIVATE Hugging Face model repo, with a model card.

Uploads only: adapter weights and config, tokenizer files, the run config, the aggregate training
log (numbers only) and a generated model card. Never rollouts, cases or any MIMIC text.

The weights were trained on PhysioNet credentialed data (MIMIC-IV, MIMIC-IV-Note, MIMIC-IV-Ext-CDM),
so the repo is created private by default: a public release belongs on PhysioNet's credentialed
channel. Token: the `hf` / HF_TOKEN variable from .env (via scripts/env_gpu.sh).

    python scripts/push_hf.py --run cev            # outputs/runs/cev/final -> <user>/deferdx-cev-qwen3-8b-lora
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/GIND123/Clinical-Agent-Stanford"

RUNS = {  # run dir -> (repo name, eval-suite name for the metrics table, title, description)
    "cev": ("deferdx-cev-qwen3-8b-lora", "cev", "DEFER-Dx: counterfactual escalation value + scored handoff",
            "The method. Escalation to a clinician is credited against forced continuations branched from the exact "
            "decision state; the handoff is a probabilistic differential trained with a proper scoring rule."),
    "deferdx": ("deferdx-consensus-qwen3-8b-lora", "deferdx", "Comparator: case-level group-consensus deferral reward",
                "Comparator arm. Deferral rewarded by the group's commit accuracy, gamma*(tau - p_hat): the published "
                "mechanism (TIAR / KARL / AWA-RL family), kept to measure what the counterfactual adds."),
    "nodefer": ("deferdx-nodefer-control-qwen3-8b-lora", "grpo_nodefer", "Control: same training, no DEFER action",
                "Control arm. Identical data, reward and compute without a DEFER action; its stated probabilities are "
                "thresholded post hoc for the learned-versus-threshold comparison."),
    "abl_cev_no_handoff": ("deferdx-abl-cev-no-handoff-qwen3-8b-lora", "abl_cev_no_handoff",
                           "Ablation: counterfactual escalation without the scored handoff", "100-step ablation."),
    "abl_constant_defer": ("deferdx-abl-constant-defer-qwen3-8b-lora", "abl_constant_defer",
                           "Ablation: constant deferral reward", "100-step ablation."),
}
METRICS = [("acc_full", "Accuracy at full coverage", True), ("mean_class_acc", "Mean-class accuracy", True),
           ("coverage", "Coverage", True), ("selective_acc", "Accuracy on committed cases", True),
           ("aurc", "AURC (lower is better)", False), ("ece", "ECE (lower is better)", False),
           ("unsafe_errors", "Unflagged errors per case", True), ("confident_errors", "Confident errors per case", True)]


def hf_user(api) -> str:
    return os.environ.get("HF_USER") or api.whoami()["name"]


def metrics_table(system: str) -> str:
    path = ROOT / "docs" / "results.json"
    if not path.exists():
        return "_Held-out results pending; see docs/RESULTS.md in the GitHub repository._"
    tables = json.loads(path.read_text()).get("tables", {})
    t = tables.get(system, {}).get("ml") or {}
    clin = tables.get(system, {}).get("clinical") or {}
    if not t:
        return "_Held-out results pending; see docs/RESULTS.md in the GitHub repository._"
    rows = ["| Metric (MIMIC-IV-Ext-CDM val + test, 480 cases) | Value (95% CI) |", "|---|---|"]
    for key, name, pct in METRICS:
        d = t.get(key) or clin.get(key)
        if not d:
            continue
        k = 100 if pct else 1
        rows.append(f"| {name} | {d['value'] * k:.{1 if pct else 3}f} ({d['lo'] * k:.{1 if pct else 3}f}–"
                    f"{d['hi'] * k:.{1 if pct else 3}f}) |")
    return "\n".join(rows)


def model_card(run: str, repo: str, steps: int | None) -> str:
    _, system, title, desc = RUNS[run]
    return f"""---
base_model: Qwen/Qwen3-8B
library_name: peft
tags: [lora, medical, clinical-decision-support, reinforcement-learning, grpo, deferral, research-only]
---

# {title}

LoRA adapter (r 16, alpha 32, all linear layers) for **Qwen/Qwen3-8B** from the DEFER-Dx project: an interactive
diagnostic agent that can ASK, TEST, COMMIT (with a stated probability) or DEFER to a clinician, on abdominal-pain
admissions (appendicitis, cholecystitis, diverticulitis, pancreatitis, or OTHER).

{desc}

- Code, method and full results: {GITHUB} (docs/METHODS.md, docs/RESULTS.md)
- Training: multi-turn GRPO, {steps if steps else "see config"} steps, on one GPU (configuration in `training_config.json`;
  aggregate training log in `train_log.jsonl`, numbers only).

## Held-out results

{metrics_table(system)}

## Data, access and intended use

- **Trained on PhysioNet credentialed data**: MIMIC-IV-Ext-CDM 1.1, MIMIC-IV 2.2 (hosp) and MIMIC-IV-Note 2.2, under the
  PhysioNet Credentialed Health Data Use Agreement. This repository is **private**; weights derived from credentialed
  data must not be redistributed to anyone who is not a credentialed PhysioNet user bound by the same agreement.
  A public release, if any, will go through PhysioNet's credentialed channel.
- **Research use only.** Not a medical device, not validated for clinical decision making, single centre (BIDMC), four
  conditions plus an open-world OTHER class. Do not use for patient care.
- The repository holds no patient data: adapter weights, tokenizer files, configuration and aggregate metrics only.

## Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, "{repo}")
tok = AutoTokenizer.from_pretrained("{repo}")
```
The agent expects the DEFER-Dx environment's system prompt and action format (see the GitHub repository).
"""


def stage(run: str, repo: str) -> Path:
    src = ROOT / "outputs" / "runs" / run / "final"
    if not (src / "adapter_model.safetensors").exists():
        raise SystemExit(f"no trained adapter at {src}")
    tmp = Path(tempfile.mkdtemp(prefix=f"hf_{run}_"))
    for f in src.iterdir():
        if f.suffix in (".safetensors", ".json", ".jinja", ".txt", ".model") or f.name in ("tokenizer.json",):
            if f.name == "trainer_state.json":
                continue
            shutil.copy2(f, tmp / f.name)
    steps = None
    log = ROOT / "outputs" / "runs" / run / "train_log.jsonl"
    if log.exists():
        rows = {}
        for line in log.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["step"]] = {k: v for k, v in r.items() if isinstance(v, (int, float))}  # numbers only
        steps = max(rows) if rows else None
        (tmp / "train_log.jsonl").write_text("\n".join(json.dumps(rows[k]) for k in sorted(rows)) + "\n")
    cks = sorted((ROOT / "outputs" / "runs" / run / "checkpoints").glob("step_*/trainer_state.pt"))
    if cks:  # hyper-parameters and file paths only
        import torch

        cfg = torch.load(cks[-1], map_location="cpu", weights_only=False).get("config", {})
        (tmp / "training_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    (tmp / "README.md").write_text(model_card(run, repo, steps))
    return tmp


def push(run: str, card_only: bool = False, private: bool = True) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    repo = f"{hf_user(api)}/{RUNS[run][0]}"
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    if card_only:
        api.upload_file(path_or_fileobj=model_card(run, repo, None).encode(), path_in_repo="README.md", repo_id=repo,
                        commit_message="Update model card with held-out results")
        return repo
    folder = stage(run, repo)
    try:
        api.upload_folder(folder_path=str(folder), repo_id=repo, commit_message=f"Upload {run} LoRA adapter and model card")
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return repo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=sorted(RUNS))
    ap.add_argument("--card-only", action="store_true", help="refresh the model card (results) only")
    ap.add_argument("--public", action="store_true", help="NOT recommended: weights derive from credentialed data")
    args = ap.parse_args()
    repo = push(args.run, args.card_only, private=not args.public)
    print(f"pushed {args.run} -> https://huggingface.co/{repo} ({'public' if args.public else 'private'})")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    main()
