#!/usr/bin/env bash
# Serve the judge model via vLLM on L40S (2 GPUs)
# Usage: bash scripts/agentstress-serve-judge.sh

set -euo pipefail

MODEL="${AGENTSTRESS_JUDGE_MODEL:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
PORT="${AGENTSTRESS_JUDGE_PORT:-8000}"
TP_SIZE="${AGENTSTRESS_TP_SIZE:-2}"
GPU_MEM="${AGENTSTRESS_GPU_MEM:-0.90}"
MAX_MODEL_LEN="${AGENTSTRESS_MAX_MODEL_LEN:-8192}"

echo "=== AgentStress Judge Server ==="
echo "Model:       ${MODEL}"
echo "Port:        ${PORT}"
echo "TP Size:     ${TP_SIZE}"
echo "GPU Memory:  ${GPU_MEM}"
echo "Max Length:  ${MAX_MODEL_LEN}"
echo "================================"

python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_MEM}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --quantization awq \
    --trust-remote-code \
    --dtype half
