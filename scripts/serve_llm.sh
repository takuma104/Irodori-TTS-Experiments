#!/usr/bin/env bash
# Serve the sentence-generation LLM with the vLLM environment in ~/co/vllm_serve.
#
#   bash scripts/serve_llm.sh            # OpenAI-compatible API on :8000
#
# Needs most of the GPU; do not run it together with TTS generation or ASR.
#
# On first start, FlashInfer JIT-compiles kernels for the RTX 5090 (sm_120).
# Unbounded, it ran 16 parallel `cicc` processes of ~3.7 GB each and the host
# OOM killer took down the user session (2026-09-29). So the compile
# parallelism is capped, and the server runs in a memory-limited systemd scope
# so that an overrun kills only vLLM.
set -euo pipefail
MODEL=${MODEL:-nvidia/Qwen3.6-27B-NVFP4}
cd "${VLLM_SERVE_DIR:-$HOME/co/vllm_serve}"
export MAX_JOBS=${MAX_JOBS:-4}
export FLASHINFER_NVCC_THREADS=${FLASHINFER_NVCC_THREADS:-1}
exec systemd-run --user --scope --quiet \
  -p MemoryMax="${MEMORY_MAX:-40G}" -p MemorySwapMax=0 \
  uv run --no-sync vllm serve "$MODEL" \
  --served-model-name Qwen3.6-27B-NVFP4 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --port "${PORT:-8000}"
