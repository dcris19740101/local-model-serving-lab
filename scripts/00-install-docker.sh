#!/usr/bin/env bash
set -euo pipefail

echo "==> Installing Docker Engine"

sudo apt update
sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker "$USER"

echo "==> Docker installed."
echo "    Group membership is only read when a process starts, this shell won't have it yet."
echo "    Run 'newgrp docker' now to pick it up without logging out, then test with:"
echo "        docker run hello-world"
echo "    (source ~/.bashrc will NOT fix this, it doesn't touch process credentials)"
