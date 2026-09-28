#!/usr/bin/env bash
# Plan M0: bf16 baseline evaluation, D1 (kana oracle), and D2 (seed variance).
#
#   bash scripts/run_m0_diagnostics.sh
#
# Expects the bf16 baseline audio in $BASE/audio (scripts/generate_jkyb_audio.py).
set -euo pipefail
cd "$(dirname "$0")/.."

REF_WAV=data/jvs_ver1/jvs001/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav
BASE=outputs/irodori-v4.1-small_jvs001_bf16
FP32=outputs/irodori-v4.1-small_jvs001
D1=outputs/m0_diagnostics

gen() {  # gen <output-dir> <text-field> <seed>
  PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync python scripts/generate_jkyb_audio.py \
    --dataset "$D1/subset.jsonl" --text-field "$2" --seed "$3" \
    --output-dir "$1/audio" --ref-wav "$REF_WAV" --precision bf16
}

jkyb_eval() {  # jkyb_eval <run-dir> [extra args...]
  local run=$1
  shift
  (cd Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition && uv run --no-sync jkyb-eval tts \
    "../$run/audio" --device cuda --output-dir "../$run/results" "$@")
}

# bf16 baseline
if [ ! -f "$BASE/results/summary.json" ]; then
  jkyb_eval "$BASE"
fi
uv run python scripts/analyze_jkyb_errors.py "$BASE/results" --output "$BASE/error_analysis.md"
uv run python scripts/analyze_jkyb_errors.py "$FP32/results" --output "$FP32/error_analysis.md"
uv run python scripts/compare_jkyb_runs.py "$FP32/results" "$BASE/results" \
  --base-errors "$FP32/error_analysis.errors.jsonl" --base-label fp32 --cand-label bf16 \
  --output "$BASE/compare_vs_fp32.md" > /dev/null

# D1/D2 subset: all bf16 errors + the same number of correct rows
uv run python scripts/jkyb_kana_oracle.py "$BASE/results" --output "$D1/subset.jsonl"

for variant in "hira text_hira 0" "kata text_kata 0" "seed1 text 1" "seed2 text 2"; do
  set -- $variant
  run="$D1/$1"
  mkdir -p "$run/logs"
  gen "$run" "$2" "$3" > "$run/logs/gen.log" 2>&1
  if [ ! -f "$run/results/summary.json" ]; then
    jkyb_eval "$run" --dataset "../$D1/subset.jsonl" > "$run/logs/eval.log" 2>&1
  fi
  uv run python scripts/compare_jkyb_runs.py "$BASE/results" "$run/results" \
    --base-errors "$BASE/error_analysis.errors.jsonl" --base-label bf16-seed0 --cand-label "$1" \
    --output "$run/compare.md" > /dev/null
done
echo "M0 done"
