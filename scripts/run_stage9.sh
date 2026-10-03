#!/usr/bin/env bash
# Stage 9: full JKYB-Parakeet (13,536 sentences) for the recommended candidate (T-A, and T-A int8)
# against the teacher, with identical synthesis settings (batch 16 / 4096 frames, JSUT reference,
# seed 0). Uses the GPU slots.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=third-party/Irodori-TTS:scripts
PY=third-party/Irodori-TTS/.venv/bin/python
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
full() {  # <checkpoint> <name> <text precision>
  local out=outputs/jkyb_full/$2
  [ -f "$out/results/summary.md" ] && return
  mkdir -p "$out"
  slot bash -c "$PY scripts/generate_jkyb_audio.py --hf-checkpoint '$1' --output-dir '$out/audio' \
      --ref-wav data/jsut/BASIC5000_0001.wav --seed 0 --batch-size 16 --max-batch-frames 4096 \
      --text-precision $3 > '$out/synth.log' 2>&1 && echo done >> '$out/synth.log' && \
    cd third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition && \
    uv run --no-sync jkyb-eval tts '../../$out/audio' --device cuda --output-dir '../../$out/results' \
      > '../../$out/eval.log' 2>&1" || echo "FAILED $2"
  echo "full $2"
}
full Aratako/Irodori-TTS-v4.1-Small-MF mf_teacher fp32
full $S/text_prune310/p1a/model.safetensors text_prune310_p1a fp32
full Aratako/Irodori-TTS-v4.1-Small-MF mf_teacher_bf16 bf16
full $S/quant/ta_int8/model.safetensors ta_int8_bf16 bf16
echo "stage9 done"
