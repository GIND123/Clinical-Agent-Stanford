#!/usr/bin/env bash
# One-time setup on the Lambda VM: uv venv, vLLM, both repos, model downloads in the background.
set -euo pipefail
mkdir -p ~/runs ~/outputs
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH"
uv venv -q -p 3.12 ~/venv
source ~/venv/bin/activate
git clone -q https://github.com/GIND123/Clinical-Agent-Stanford.git ~/Clinical-Agent-Stanford
git clone -q --depth 1 https://github.com/MAGIC-AI4Med/DiagGym.git ~/DiagGym
# Pinned: vllm 0.11.0 -> torch 2.8.0+cu128. The unpinned latest pulls a CUDA 13.0 torch that
# Lambda's driver (570.x, CUDA 12.8) cannot initialise.
# transformers<5 too: vllm 0.11.0 allows any transformers>=4.56, but 5.x removed the
# tokenizer attribute (all_special_tokens_extended) its tokenizer cache reads at startup.
uv pip install -q "vllm==0.11.0" "transformers>=4.56,<5" hf_transfer openai requests tqdm huggingface_hub
uv pip install -q -e "$HOME/Clinical-Agent-Stanford[data]"
export HF_HUB_ENABLE_HF_TRANSFER=1
for m in Qwen/Qwen3-8B Henrychur/DiagAgent-14B; do
  python -c "from huggingface_hub import snapshot_download; snapshot_download('$m')" > ~/runs/download_$(basename $m).log 2>&1 &
done
wait
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo SETUP_OK
