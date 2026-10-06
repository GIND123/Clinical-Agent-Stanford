#!/usr/bin/env bash
# Download the PhysioNet files DEFER-Dx needs, resuming partial files and checking each against the
# SHA256SUMS.txt PhysioNet ships with every project version.
#
# All targets except `demo` are credentialed: a PhysioNet account with CITI training and the data use
# agreement signed on each project page (MIMIC-IV-Ext-CDM 1.1, MIMIC-IV 2.2, MIMIC-IV-Note 2.2).
#
# Two routes:
#   https (default)  credentials in ~/.netrc:  machine physionet.org login <user> password <pass>  (chmod 600)
#                    physionet.org serves roughly 40-80 KB/s per connection, so the 4 GB below takes hours.
#   s3 (--s3)        much faster. Add your AWS identity under PhysioNet Settings -> Cloud, click
#                    "Enable AWS access" on the MIMIC-IV and MIMIC-IV-Note project pages, and configure the
#                    AWS CLI for that identity. CDM has no S3 access point here, so cdm always uses https.
#
#   bash scripts/download_data.sh [--s3] [cdm] [hosp] [note] [demo]   # default targets: cdm hosp note
#   DATA_ROOT=/elsewhere bash scripts/download_data.sh                  # default root: data/physionet
set -euo pipefail
ROOT="${DATA_ROOT:-data/physionet}"
BASE=https://physionet.org/files
s3_base() {  # PhysioNet's S3 access point for a project version (shown on its page once AWS access is on)
  case $1 in
    mimiciv/2.2) echo "s3://arn:aws:s3:us-east-1:724665945834:accesspoint/mimiciv-v2-2-01/mimiciv/2.2" ;;
    mimic-iv-note/2.2) echo "s3://arn:aws:s3:us-east-1:724665945834:accesspoint/mimic-iv-note-v2-2-01/mimic-iv-note/2.2" ;;
  esac
}
HOSP='hosp/(admissions|patients|transfers|diagnoses_icd|d_icd_diagnoses|labevents|d_labitems|microbiologyevents)\.csv\.gz'
NOTE='note/(discharge|radiology|radiology_detail)\.csv\.gz'
USE_S3=0

get() {  # <project/version> <relative path> <dest>
  local pv=$1 rel=$2 dest=$3
  if [[ $USE_S3 == 1 && -n "$(s3_base "$pv")" ]]; then
    aws s3 cp "$(s3_base "$pv")/$rel" "$dest" --only-show-errors
  else
    wget --netrc=on --quiet --continue --output-document="$dest" "$BASE/$pv/$rel"
  fi
}

fetch() {  # <project/version> <regex over paths in SHA256SUMS.txt>
  local pv=$1 filter=$2 dir="$ROOT/$1" sum path
  mkdir -p "$dir"
  get "$pv" SHA256SUMS.txt "$dir/SHA256SUMS.txt" \
    || { echo "FAIL $pv: no access (credentials? DUA signed? AWS access enabled?)"; return 1; }
  grep -E " ($filter)$" "$dir/SHA256SUMS.txt" > "$dir/wanted.sha256"
  while read -r sum path; do
    if [[ -f "$dir/$path" && $(shasum -a 256 "$dir/$path" | cut -d' ' -f1) == "$sum" ]]; then
      echo "ok   $pv/$path (already present)"; continue
    fi
    mkdir -p "$dir/$(dirname "$path")"
    echo "get  $pv/$path"
    get "$pv" "$path" "$dir/$path"
    [[ $(shasum -a 256 "$dir/$path" | cut -d' ' -f1) == "$sum" ]] \
      || { echo "FAIL $pv/$path: checksum mismatch"; return 1; }
    echo "ok   $pv/$path"
  done < "$dir/wanted.sha256"
}

[[ "${1:-}" == "--s3" ]] && { USE_S3=1; shift; }
[[ $# -eq 0 ]] && set -- cdm hosp note
for t in "$@"; do case $t in
  cdm)  fetch mimic-iv-ext-cdm/1.1 '.*' ;;
  hosp) fetch mimiciv/2.2 "$HOSP" ;;
  note) fetch mimic-iv-note/2.2 "$NOTE" ;;
  demo) fetch mimic-iv-demo/2.2 "$HOSP" ;;  # open access, 100 patients, for dry runs (https only)
  *) echo "unknown target: $t"; exit 2 ;;
esac; done
echo "DONE -> $ROOT"
