#!/usr/bin/env bash
# Start a standalone CoreX vLLM OpenAI-compatible API for Qwen3-14B.
# This deliberately does not invoke Llumnix's legacy vLLM backend, which is
# incompatible with the supplied vLLM 0.11.2 V1 runtime.
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  root="$(cd "$(dirname "$0")/.." && pwd)"
  # shellcheck source=tools/corex44_env.sh
  source "$root/tools/corex44_env.sh"
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_path="${MODEL_PATH:-$root/.models/Qwen3-14B}"
port="${PORT:-8000}"
devices="${CUDA_VISIBLE_DEVICES:-0,1}"

if [[ ! -f "$model_path/model.safetensors.index.json" ]]; then
  echo "Qwen3-14B weights are not complete: $model_path" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="$devices" exec "$root/.conda-corex44/bin/python" \
  -m vllm.entrypoints.openai.api_server \
  --model "$model_path" \
  --served-model-name Qwen3-14B-CoreX \
  --host "${HOST:-127.0.0.1}" \
  --port "$port" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-2}" \
  --dtype float16 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --enforce-eager
