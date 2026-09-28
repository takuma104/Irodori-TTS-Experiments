#!/usr/bin/env bash
# Serve the sentence-generation LLM with the vLLM environment in ~/co/vllm_serve.
#
#   bash scripts/serve_llm.sh            # OpenAI-compatible API on :8000
#
# Needs most of the GPU; do not run it together with TTS generation or ASR.
set -euo pipefail
MODEL=${MODEL:-nvidia/Qwen3.6-27B-NVFP4}
cd "${VLLM_SERVE_DIR:-$HOME/co/vllm_serve}"
exec uv run --no-sync vllm serve "$MODEL" \
  --served-model-name Qwen3.6-27B-NVFP4 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --port "${PORT:-8000}"
