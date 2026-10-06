#!/usr/bin/env bash
# Source this before GPU work:  source scripts/env_gpu.sh
# Reads only the HF token from .env (variable `hf`), never prints it, and keeps every cache on the
# data disk inside the repo (git-ignored), not in $HOME.
_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$_root/.env" ]]; then
  _tok="$(sed -nE 's/^(export[[:space:]]+)?(hf|HF_TOKEN)=["'"'"']?([^"'"'"']*)["'"'"']?$/\3/p' "$_root/.env" | head -1)"
  [[ -n "$_tok" ]] && export HF_TOKEN="$_tok"
  unset _tok
fi
export HF_HOME="$_root/.cache/hf"
export UV_CACHE_DIR="$_root/.cache/uv"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export VLLM_CONFIGURE_LOGGING=1
source "$_root/.venv-gpu/bin/activate"
unset _root
