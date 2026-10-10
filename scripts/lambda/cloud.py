#!/usr/bin/env python
"""Minimal Lambda Cloud API helper for the journal runs (key from ~/.config/lambda/api_key, never printed).

    python scripts/lambda/cloud.py types                     # prices and regions with capacity
    python scripts/lambda/cloud.py add-key NAME PUBKEY_FILE  # register an SSH public key (idempotent)
    python scripts/lambda/cloud.py launch TYPE REGION NAME KEYNAME
    python scripts/lambda/cloud.py list                      # id, name, status, ip, hours, dollars
    python scripts/lambda/cloud.py terminate ID [ID ...]
    python scripts/lambda/cloud.py spend                     # running total over this project's instances
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from base64 import b64encode
from datetime import datetime, timezone
from pathlib import Path

API = "https://cloud.lambda.ai/api/v1"
KEY = (Path.home() / ".config" / "lambda" / "api_key").read_text().strip()
LEDGER = Path(__file__).resolve().parents[2] / "outputs" / "lambda_ledger.json"  # git-ignored


def call(method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body else None)
    req.add_header("Authorization", "Basic " + b64encode(f"{KEY}:".encode()).decode())
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "curl/8.5.0")  # Cloudflare rejects urllib's default agent (error 1010)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return {"error": e.code, "body": e.read().decode()[:500]}


def ledger() -> dict:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else {}


def save_ledger(d: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(d, indent=2))


def main(argv: list[str]) -> None:
    cmd, args = argv[0], argv[1:]
    if cmd == "types":
        d = call("GET", "/instance-types").get("data", {})
        for name, v in sorted(d.items(), key=lambda kv: kv[1]["instance_type"]["price_cents_per_hour"]):
            regions = [r["name"] for r in v.get("regions_with_capacity_available", [])]
            if regions:
                print(f"{name:28s} ${v['instance_type']['price_cents_per_hour'] / 100:6.2f}/h  {', '.join(regions)}")
    elif cmd == "add-key":
        name, pub = args[0], Path(args[1]).read_text().strip()
        existing = {k["name"] for k in call("GET", "/ssh-keys").get("data", [])}
        print("exists" if name in existing else call("POST", "/ssh-keys", {"name": name, "public_key": pub}))
    elif cmd == "launch":
        itype, region, name, keyname = args
        r = call("POST", "/instance-operations/launch", {"region_name": region, "instance_type_name": itype,
                                                          "ssh_key_names": [keyname], "quantity": 1, "name": name})
        ids = (r.get("data") or {}).get("instance_ids") or []
        if ids:
            price = call("GET", "/instance-types")["data"][itype]["instance_type"]["price_cents_per_hour"] / 100
            led = ledger()
            led[ids[0]] = {"name": name, "type": itype, "region": region, "price": price, "launched": time.time()}
            save_ledger(led)
        print(json.dumps(r))
    elif cmd == "list":
        led = ledger()
        for i in call("GET", "/instances").get("data", []):
            e = led.get(i["id"], {})
            h = (time.time() - e["launched"]) / 3600 if e else float("nan")
            print(f"{i['id']} {i.get('name')} {i['status']} {i.get('ip')} {h:.2f}h ${h * e.get('price', 0):.2f}")
    elif cmd == "terminate":
        r = call("POST", "/instance-operations/terminate", {"instance_ids": args})
        led = ledger()
        for i in args:
            if i in led and "terminated" not in led[i]:
                led[i]["terminated"] = time.time()
        save_ledger(led)
        print(json.dumps(r)[:300])
    elif cmd == "spend":
        total = 0.0
        for i, e in ledger().items():
            end = e.get("terminated", time.time())
            total += (end - e["launched"]) / 3600 * e["price"]
        print(f"{total:.2f}")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:] or ["help"])
