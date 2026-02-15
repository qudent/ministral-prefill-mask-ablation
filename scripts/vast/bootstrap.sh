#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/qudent/ministral-prefill-mask-ablation.git}"
REPO_DIR="${REPO_DIR:-$HOME/ministral-prefill-mask-ablation}"

if ! command -v git >/dev/null 2>&1; then
  apt-get update
  apt-get install -y git curl
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
uv sync --extra train
mkdir -p runs artifacts logs

# HF auth for checkpoint uploads. HF_TOKEN env var is the standard way;
# huggingface_hub picks it up automatically. Verify if set.
if [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is set; uploads will use it."
else
  echo "[warn] HF_TOKEN not set; checkpoint uploads to Hub will fail."
  echo "       Pass HF_TOKEN as env var when launching the instance."
fi

echo "=== VAST quick checks ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found"
fi

if command -v curl >/dev/null 2>&1; then
  speed="$(curl -o /dev/null -s -w '%{speed_download}' https://speed.hetzner.de/100MB.bin || true)"
  echo "download_speed_bytes_per_sec=${speed}"
fi

echo "Bootstrap complete: $REPO_DIR"
