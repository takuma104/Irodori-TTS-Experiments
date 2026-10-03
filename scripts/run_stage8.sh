#!/usr/bin/env bash
# Stage 8: int8 weight-only quantization (stock quantize_checkpoint.py, profile "core") of the
# teacher and of T-A, compared in full bf16 (the quantized runtime keeps the text encoder in
# bf16, so the fp32-text references are re-run in bf16 too). Uses the GPU slots.
set -uo pipefail
cd "$(dirname "$0")/.."
LOCKS=outputs/.gpu_slots
S=outputs/students
slot() {
  while true; do
    for s in 1 2; do
      flock -n -E 75 "$LOCKS/slot$s" "$@"
      local code=$?
      [ $code -ne 75 ] && return $code
    done
    sleep 15
  done
}
suite() {  # <checkpoint> <name> -- extra synth args
  [ -f "outputs/eval/$2/jkyb_dev/quality_summary.json" ] && return
  slot scripts/run_eval_suite.sh "$1" "$2" 0 --text-precision bf16 > "outputs/eval/$2.log" 2>&1 || echo "FAILED eval $2"
  echo "evaluated $2"
}
suite MF mf_teacher_bf16
suite $S/quant/mf_int8/model.safetensors mf_int8_bf16
suite $S/text_prune310/p1a/model.safetensors text_prune310_p1a_bf16
suite $S/quant/ta_int8/model.safetensors ta_int8_bf16
PYTHONPATH=third-party/Irodori-TTS:scripts/feasibility slot third-party/Irodori-TTS/.venv/bin/python \
  scripts/feasibility/profile_vram.py --checkpoints $S/quant/mf_int8 $S/quant/ta_int8 \
  --model-precisions bf16 --codec-precisions fp32 bf16 \
  --output outputs/feasibility/profile_vram_int8.json > outputs/feasibility/profile_vram_int8.log 2>&1
echo "stage8 done"
