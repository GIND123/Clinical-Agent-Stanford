#!/usr/bin/env python
"""Write the test catalog's prices as a CSV for review (sensitivity analysis of the cost columns).

Fill `reviewed_price` (US$) for any row; leave it empty to keep the current price. Rows with
cost_source "placeholder" are the unsourced guesses. Then regenerate the report into a separate
directory, so docs/RESULTS.md is untouched:

    python scripts/price_template.py --out outputs/sensitivity/prices.csv
    python scripts/make_report.py --prices outputs/sensitivity/prices.csv --out outputs/sensitivity \
        [--severity configs/severity_matrix_reviewed.yaml]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deferdx.env.catalog import TestCatalog  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalog", default=str(ROOT / "configs" / "test_catalog.yaml"))
    ap.add_argument("--out", default="outputs/sensitivity/prices.csv")
    args = ap.parse_args()
    cat = TestCatalog.from_yaml(args.catalog)
    rows = [{"key": k, "kind": t.kind, "display": t.display, "current_price": t.cost, "cost_source": t.cost_source,
             "reviewed_price": "", "source_of_reviewed_price": ""} for k, t in cat.tests.items()]
    rows += [{"key": k, "kind": "ask", "display": a.display, "current_price": a.cost, "cost_source": "",
              "reviewed_price": "", "source_of_reviewed_price": ""} for k, a in cat.asks.items()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_ph = sum(r["cost_source"] == "placeholder" for r in rows)
    print(f"{len(rows)} rows ({n_ph} placeholder prices) -> {out}")


if __name__ == "__main__":
    main()
