#!/usr/bin/env bash
# Prompt for PhysioNet credentials, download the DEFER-Dx datasets locally, then
# remove the temporary credential file. Nothing is written to the repository
# except the downloaded files under DATA_ROOT (default: data/physionet).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGETS=()
USE_S3=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_physionet_interactive.sh [--s3] [cdm] [hosp] [note] [demo]

Defaults:
  Downloads: cdm hosp note
  Output:    data/physionet

Examples:
  bash scripts/download_physionet_interactive.sh
  DATA_ROOT=/mnt/data/physionet bash scripts/download_physionet_interactive.sh cdm
  bash scripts/download_physionet_interactive.sh --s3 cdm hosp note

Notes:
  - The account must have accepted the relevant PhysioNet DUAs.
  - --s3 uses AWS only for MIMIC-IV and MIMIC-IV-Note; CDM still uses HTTPS.
  - Credentials are stored only in a temporary netrc outside the repo and are
    removed on exit.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --s3)
      USE_S3=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    cdm|hosp|note|demo)
      TARGETS+=("$1")
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  TARGETS=(cdm hosp note)
fi

if [[ ! -t 0 ]]; then
  echo "This script must run in an interactive terminal so the password is not logged." >&2
  exit 2
fi

read -r -p "PhysioNet username: " PHYSIONET_USER
read -r -s -p "PhysioNet password: " PHYSIONET_PASS
printf '\n'

if [[ -z "$PHYSIONET_USER" || -z "$PHYSIONET_PASS" ]]; then
  echo "username and password are required" >&2
  exit 2
fi

AUTH_DIR="$(mktemp -d "${TMPDIR:-/tmp}/physionet-auth.XXXXXX")"
NETRC="$AUTH_DIR/.netrc"

cleanup() {
  if [[ -f "$NETRC" ]]; then
    # Overwrite before unlinking; ignore failures on filesystems that do not
    # support one of these operations.
    dd if=/dev/zero of="$NETRC" bs=1 count="$(wc -c < "$NETRC")" conv=notrunc status=none 2>/dev/null || true
    unlink "$NETRC" 2>/dev/null || true
  fi
  rmdir "$AUTH_DIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

chmod 700 "$AUTH_DIR"
cat > "$NETRC" <<EOF
machine physionet.org login ${PHYSIONET_USER} password ${PHYSIONET_PASS}
EOF
chmod 600 "$NETRC"

unset PHYSIONET_PASS

cd "$ROOT_DIR"
ARGS=()
if [[ "$USE_S3" == 1 ]]; then
  ARGS+=(--s3)
fi
ARGS+=("${TARGETS[@]}")

echo "Downloading ${TARGETS[*]} into ${DATA_ROOT:-data/physionet} ..."
HOME="$AUTH_DIR" bash scripts/download_data.sh "${ARGS[@]}"
