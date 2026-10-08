#!/usr/bin/env python
"""Clinician rating of DEFER handoffs: a blinded sheet from eval-suite rollouts, then scoring.

The scored handoff (a probabilistic differential handed to a clinician) is half of DEFER-Dx's claim, and
its automatic score only checks it against the recorded diagnosis. This asks a clinician whether a
handoff would actually help.

  sample: draw N deferred episodes (stratified over evaluation sets, fixed seed) and write
      <out>/sheet.html    what the agent saw (presentation and the results it requested) and its handoff;
                          no diagnosis, no system name
      <out>/ratings.csv   one blank row per handoff, for the clinician
      <out>/key.csv       rating id -> case, set, system, recorded diagnosis (keep closed until rated)
    With --compare, each case also gets the other system's handoff for the same case (when it deferred
    there too), as A/B in random order, and the clinician also says which is more useful.
  score: aggregate the filled ratings.csv against key.csv (aggregates only).

The sheet and key contain MIMIC text and identifiers: they stay in outputs/ (git-ignored) and are only
for credentialed reviewers. Only the numbers printed by `score` leave that directory.

    python scripts/handoff_rating.py sample --rollouts outputs/eval/cev/s0.jsonl --n 40 --out outputs/handoff_rating
    python scripts/handoff_rating.py score --out outputs/handoff_rating
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deferdx.data.io import read_jsonl  # noqa: E402

RATING_FIELDS = ["rating_id", "usefulness_1to5", "correct_dx_in_differential_yn", "dangerous_omission_yn",
                 "preferred_A_or_B", "comments"]
RUBRIC = """<p><b>For each handoff</b> (you are the clinician receiving it):</p><ol>
<li><b>usefulness_1to5</b>: how useful is this differential for your next step? 1 = misleading, 2 = not useful,
3 = somewhat useful, 4 = useful, 5 = exactly what I would want.</li>
<li><b>correct_dx_in_differential_yn</b>: in your judgement, does the differential contain the correct diagnosis?</li>
<li><b>dangerous_omission_yn</b>: does it leave out a diagnosis that must not be missed for this presentation?</li>
<li><b>preferred_A_or_B</b> (only where two handoffs are shown): which is more useful? A, B or "same".</li></ol>"""


def deferred(rows: list[dict]) -> dict[str, dict]:
    """case_id -> first deferred episode (lowest seed) with a final assistant turn."""
    out = {}
    for r in sorted(rows, key=lambda r: r.get("seed", 0)):
        res = r["result"]
        if res["terminal"] == "defer" and res["case_id"] not in out:
            out[res["case_id"]] = r
    return out


def seen_by_agent(r: dict) -> str:
    """What the agent saw: the presentation and the environment's replies (no system prompt, no reasoning)."""
    users = [m["content"] for m in r["messages"] if m["role"] == "user"]
    return "\n\n".join(users)


def handoff_text(res: dict) -> str:
    probs = res.get("differential_probs") or {}
    items = [f"{d.replace('_', ' ')}" + (f" ({probs[d]:.0%})" if d in probs else "") for d in res.get("differential") or []]
    return "; ".join(items) or "(no differential)"


def sample(args) -> None:
    rows = [r for p in args.rollouts for r in read_jsonl(p)]
    main = deferred(rows)
    other = deferred([r for p in (args.compare or []) for r in read_jsonl(p)])
    rng = random.Random(args.seed)
    by_set = defaultdict(list)
    for cid, r in sorted(main.items()):
        by_set[r.get("set", "all")].append(cid)
    picked: list[str] = []
    sets = sorted(by_set)
    while len(picked) < args.n and any(by_set.values()):  # round-robin over sets: every set is represented
        for s in sets:
            if by_set[s] and len(picked) < args.n:
                picked.append(by_set[s].pop(rng.randrange(len(by_set[s]))))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    key, blocks, ratings = [], [], []
    for i, cid in enumerate(picked, 1):
        r = main[cid]
        shown = [("main", r)] + ([("compare", other[cid])] if cid in other else [])
        if len(shown) == 2:
            rng.shuffle(shown)
        parts = []
        for slot, (system, ro) in zip("AB", shown):
            rid = f"H{i:03d}{slot if len(shown) == 2 else ''}"
            res = ro["result"]
            parts.append(f"<h4>Handoff {html.escape(rid)}</h4><p><b>Differential:</b> {html.escape(handoff_text(res))}</p>"
                         f"<p><b>Reason given:</b> {html.escape(res.get('reason') or '')}</p>")
            key.append({"rating_id": rid, "case": i, "case_id": cid, "set": ro.get("set", ""), "system": system,
                        "slot": slot if len(shown) == 2 else "", "label": res["label"],
                        "handoff": json.dumps({"differential": res.get("differential"),
                                               "probs": res.get("differential_probs")})})
            ratings.append({f: (rid if f == "rating_id" else "") for f in RATING_FIELDS})
        blocks.append(f"<section><h3>Case {i}</h3><pre>{html.escape(seen_by_agent(r))}</pre>{''.join(parts)}</section>")
    (out / "sheet.html").write_text("<html><meta charset='utf-8'><body style='font-family:sans-serif;max-width:60em'>"
                                    "<h2>Handoff rating (credentialed reviewers only: contains MIMIC text)</h2>"
                                    + RUBRIC + "<hr>".join(blocks) + "</body></html>", encoding="utf-8")
    for name, data, fields in (("ratings.csv", ratings, RATING_FIELDS), ("key.csv", key, list(key[0]) if key else [])):
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)
    print(f"{len(picked)} cases ({len(key)} handoffs; {sum(1 for k in key if k['slot'] == 'A')} paired) -> {out}/"
          f"sheet.html, ratings.csv, key.csv. Keep key.csv closed until the ratings are done.")


def _yes(v: str) -> float | None:
    v = (v or "").strip().lower()
    return 1.0 if v in {"y", "yes", "1", "true"} else 0.0 if v in {"n", "no", "0", "false"} else None


def _ci(x: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    a = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(1)
    return float(a.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def score(args) -> dict:
    out = Path(args.out)
    key = {k["rating_id"]: k for k in csv.DictReader((out / "key.csv").open(encoding="utf-8"))}
    rated = [r for r in csv.DictReader((out / "ratings.csv").open(encoding="utf-8"))
             if re.fullmatch(r"[1-5]", (r.get("usefulness_1to5") or "").strip())]
    rep: dict = {"rated": len(rated), "by_system": {}}
    for system in sorted({key[r["rating_id"]]["system"] for r in rated}):
        rs = [r for r in rated if key[r["rating_id"]]["system"] == system]
        use = [float(r["usefulness_1to5"]) for r in rs]
        m, lo, hi = _ci(use)
        row = {"n": len(rs), "usefulness_mean": round(m, 2), "usefulness_95ci": [round(lo, 2), round(hi, 2)],
               "useful_4plus": round(float(np.mean([u >= 4 for u in use])), 3)}
        for f in ("correct_dx_in_differential_yn", "dangerous_omission_yn"):
            v = [y for y in (_yes(r.get(f)) for r in rs) if y is not None]
            row[f.removesuffix("_yn")] = round(float(np.mean(v)), 3) if v else None
        # the clinician's judgement against the automatic check: recorded diagnosis listed in the differential
        auto = [key[r["rating_id"]]["label"] in (json.loads(key[r["rating_id"]]["handoff"])["differential"] or [])
                for r in rs]
        row["recorded_dx_listed"] = round(float(np.mean(auto)), 3)
        rep["by_system"][system] = row
    pairs = defaultdict(dict)
    for r in rated:
        k = key[r["rating_id"]]
        if k["slot"]:
            pairs[k["case"]][k["slot"]] = (k["system"], (r.get("preferred_A_or_B") or "").strip().upper())
    wins = [next((s for slot, (s, _) in p.items() if slot == pref), "same") for p in pairs.values() if len(p) == 2
            for pref in [next((v for _, v in p.values() if v), "")]]
    if wins:
        rep["paired"] = {"cases": len(wins), "main_preferred": round(wins.count("main") / len(wins), 3),
                         "compare_preferred": round(wins.count("compare") / len(wins), 3),
                         "same": round(wins.count("same") / len(wins), 3)}
    print(json.dumps(rep, indent=1))
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("sample")
    sp.add_argument("--rollouts", nargs="+", required=True, help="eval-suite s<seed>.jsonl of the system rated")
    sp.add_argument("--compare", nargs="+", help="the same for a second system, shown blind as A/B")
    sp.add_argument("--n", type=int, default=40)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", default="outputs/handoff_rating")
    sp.set_defaults(fn=sample)
    sp = sub.add_parser("score")
    sp.add_argument("--out", default="outputs/handoff_rating")
    sp.set_defaults(fn=score)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
