#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
QUANT_ARGS=()
if [[ "$MODEL" == *-AWQ* || "$MODEL" == *-awq* ]]; then
  QUANT_ARGS=(--quantization awq)
fi

echo "==> Running vLLM standalone against ${MODEL}, before any KServe/Kubernetes layer"
echo "    Requires: pip install -U huggingface_hub --break-system-packages"
echo "              then hf auth login, already run once"
echo "    Default model is the AWQ (4-bit) build: on a 16GB card, the full-precision"
echo "    Qwen2.5-7B-Instruct's 14.29 GiB of weights leaves under 1.2 GiB free and OOMs"
echo "    before KV cache is even allocated. Confirmed on an RTX 5080 (15.45 GiB usable)."

docker run --gpus all -p 8000:8000 \
  -e HF_TOKEN="$(cat ~/.cache/huggingface/token)" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model "${MODEL}" \
  "${QUANT_ARGS[@]}" \
  --max-model-len 4096 &

VLLM_PID=$!
echo "==> vLLM starting in background (pid ${VLLM_PID}). Waiting for /health ..."
until curl -sf http://localhost:8000/health >/dev/null 2>&1; do sleep 2; done

echo "==> Sending a test completion request"
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${MODEL}\", \"prompt\": \"Explain PagedAttention in one sentence.\", \"max_tokens\": 50}" | tee /dev/stderr

echo "==> Stop the container manually with: docker stop \$(docker ps -q --filter ancestor=vllm/vllm-openai:latest)"
