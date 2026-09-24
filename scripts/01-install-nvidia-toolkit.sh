#!/usr/bin/env bash
set -euo pipefail

echo "==> Installing NVIDIA Container Toolkit"
echo "    Requires nvidia-driver-580-open already installed and working (check with nvidia-smi)"

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "==> Verifying GPU access from inside a container"
echo "    Note: RTX 5080 is Blackwell (sm_120), needs CUDA 12.8+. This nvidia-smi check"
echo "    talks to the host driver directly and can pass even on an older CUDA image,"
echo "    it only proves the driver bridge works, not that sm_120 kernels will run."
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
