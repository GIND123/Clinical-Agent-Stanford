#!/usr/bin/env bash
# Verify local PhysioNet downloads and report duplicate files.
set -euo pipefail

ROOT="${1:-data/physionet}"
REPORT="${2:-outputs/physionet_integrity_report.txt}"
TMP="${REPORT}.tmp"

mkdir -p "$(dirname "$REPORT")"

{
  echo "PhysioNet integrity report"
  echo "Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Root: $ROOT"
  echo

  echo "== SHA256 manifests =="
  first_manifest="$(find "$ROOT" -name wanted.sha256 -type f -print -quit 2>/dev/null || true)"
  if [[ -z "$first_manifest" ]]; then
    echo "FAIL: no wanted.sha256 files found under $ROOT"
    exit 1
  fi

  sha_fail=0
  while IFS= read -r manifest; do
    dir="$(dirname "$manifest")"
    echo "-- $manifest"
    if (cd "$dir" && awk '{ hash=$1; $1=""; sub(/^ /, ""); print hash "  " $0 }' wanted.sha256 | shasum -a 256 -c -); then
      echo "OK: $manifest"
    else
      echo "FAIL: $manifest"
      sha_fail=1
    fi
    echo
  done < <(find "$ROOT" -name wanted.sha256 -type f | sort)

  echo "== Gzip integrity =="
  gzip_fail=0
  if find "$ROOT" -name '*.gz' -type f -print -quit | grep -q .; then
    while IFS= read -r gz; do
      if gzip -t "$gz"; then
        echo "OK: $gz"
      else
        echo "FAIL: $gz"
        gzip_fail=1
      fi
    done < <(find "$ROOT" -name '*.gz' -type f | sort)
  else
    echo "No gzip files found."
  fi
  echo

  echo "== Duplicate content under $ROOT =="
  dup_file="$(mktemp)"
  find "$ROOT" -type f -size +0c -print0 \
    | xargs -0 shasum -a 256 2>/dev/null \
    | sort > "$dup_file"
  awk '
    {
      hash=$1
      $1=""
      sub(/^  /, "")
      files[hash]=files[hash] "\n  " $0
      counts[hash]++
    }
    END {
      found=0
      for (hash in counts) {
        if (counts[hash] > 1) {
          found=1
          print "DUPLICATE " hash " (" counts[hash] " files):" files[hash]
        }
      }
      if (!found) print "No duplicate file contents found."
    }
  ' "$dup_file"
  unlink "$dup_file" 2>/dev/null || true
  echo

  echo "== Stray recursive wget mirrors =="
  if [[ -d physionet.org ]]; then
    find physionet.org -type f -printf '%p %s bytes\n' | sort
  else
    echo "No physionet.org mirror directory found in repo root."
  fi
  echo

  if [[ "$sha_fail" == 0 && "$gzip_fail" == 0 ]]; then
    echo "RESULT: PASS"
  else
    echo "RESULT: FAIL"
    exit 1
  fi
} > "$TMP"

mv "$TMP" "$REPORT"
cat "$REPORT"
