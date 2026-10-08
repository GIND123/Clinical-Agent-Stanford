#!/usr/bin/env python
"""Keep GitHub and the Hugging Face Hub up to date while the GPU queue runs (CPU only).

Every `--interval` seconds:
  * Hugging Face: each finished run in push_hf.RUNS (outputs/runs/<run>/final) whose adapter has not been
    pushed yet is uploaded to its PRIVATE model repo; when docs/results.json changes, the model cards
    of pushed runs are refreshed with the new held-out numbers.
  * Git: if the generated, aggregate-only result files changed (docs/RESULTS.md, docs/results.json,
    docs/figures/), they are committed and pushed to main. Nothing outside that allow-list is staged,
    so data/, outputs/ and rollouts can never be committed by this process.

    setsid nohup python scripts/sync_daemon.py --interval 900 >> outputs/logs/sync.log 2>&1 &
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import push_hf  # noqa: E402

STATE = ROOT / "outputs" / "sync_state.json"
GIT_ALLOW = ["docs/RESULTS.md", "docs/results.json", "docs/figures"]


def log(msg: str) -> None:
    print(f"{datetime.now().isoformat(timespec='seconds')} {msg}", flush=True)


def sha(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=check)


def sync_hf(state: dict) -> None:
    pushed = state.setdefault("hf_pushed", {})
    results_sha = sha(ROOT / "docs" / "results.json")
    for run in push_hf.RUNS:
        adapter = ROOT / "outputs" / "runs" / run / "final" / "adapter_model.safetensors"
        digest = sha(adapter)
        if digest is None:
            continue
        try:
            if pushed.get(run, {}).get("adapter") != digest:
                repo = push_hf.push(run)
                pushed[run] = {"adapter": digest, "results": results_sha, "repo": repo}
                log(f"hf: pushed {run} -> {repo}")
            elif results_sha and pushed[run].get("results") != results_sha:
                push_hf.push(run, card_only=True)
                pushed[run]["results"] = results_sha
                log(f"hf: refreshed model card for {run}")
        except Exception as e:  # keep going; retry next cycle
            log(f"hf: {run} failed: {type(e).__name__}: {str(e)[:200]}")


def sync_git() -> None:
    if (ROOT / ".git" / "index.lock").exists():
        log("git: index locked, skipping this cycle")
        return
    paths = [p for p in GIT_ALLOW if (ROOT / p).exists()]  # git add fails on a missing pathspec
    if not paths:
        return
    changed = git("status", "--porcelain", "--", *paths).stdout.strip()
    if not changed:
        return
    git("add", "--", *paths)
    staged = git("diff", "--cached", "--name-only").stdout.split()
    if any(not any(f == a or f.startswith(a + "/") for a in GIT_ALLOW) for f in staged):
        log(f"git: refusing to commit, unexpected staged files: {staged}")
        return
    systems = []
    try:
        systems = sorted(json.loads((ROOT / "docs" / "results.json").read_text()).get("tables", {}))
    except Exception:
        pass
    msg = (f"Results update {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
           f"Regenerated docs/RESULTS.md, docs/results.json and figures from the evaluation suites "
           f"(aggregates only). Systems: {', '.join(s for s in systems if '@' not in s and '+' not in s) or 'n/a'}.")
    git("commit", "-q", "-m", msg)
    git("fetch", "-q", "origin")
    rb = git("rebase", "-q", "origin/main", check=False)
    if rb.returncode != 0:
        git("rebase", "--abort", check=False)
        log(f"git: rebase failed, left local commit unpushed: {rb.stderr.strip()[:200]}")
        return
    p = git("push", "-q", "origin", "main", check=False)
    log("git: pushed results update" if p.returncode == 0 else f"git: push failed: {p.stderr.strip()[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=900)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    while True:
        state = json.loads(STATE.read_text()) if STATE.exists() else {}
        sync_hf(state)
        STATE.write_text(json.dumps(state, indent=2))
        try:
            sync_git()
        except subprocess.CalledProcessError as e:
            log(f"git: {e.cmd} failed: {(e.stderr or '').strip()[:200]}")
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
