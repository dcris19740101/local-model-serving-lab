#!/usr/bin/env bash
set -euo pipefail

echo "==> Installing python3-pip (not present on a fresh Ubuntu install)"
sudo apt update
sudo apt install -y python3-pip

echo "==> Installing huggingface_hub (provides the 'hf' CLI, huggingface-cli is deprecated)"
echo "    --break-system-packages is required on Ubuntu 24.04's externally-managed Python"

if command -v pip &>/dev/null; then
  pip install -U huggingface_hub --break-system-packages
else
  pip3 install -U huggingface_hub --break-system-packages
fi

echo "==> Log in interactively (stores a token at ~/.cache/huggingface/token)"
echo "    Accept the default browser login: it prints a URL (https://hf.co/oauth/device)"
echo "    and a short code, open the URL on any device, log in, and enter the code."
hf auth login
