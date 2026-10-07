#!/usr/bin/env bash
# Seed robustness of the release comparison (for the technical report).
#
#   bash scripts/run_seed_check.sh
#
# All other evaluations use seed 0. This scores the base model and the v3 release
# file on the Aozora ruby set with seeds 1 and 2, and compares them per seed.
set -euo pipefail
cd "$(dirname "$0")/.."

REF=data/jvs_ver1/jvs001/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav
E=outputs/yomi_eval
V3=../Irodori-TTS-v4.1-Small-Yomi/model.safetensors
SET=aozora

run() {  # run <name> <seed> [generation args...]
  local out=$E/seeds/$1_seed$2/$SET
  shift 2
  mkdir -p "$out/logs"
  if [ ! -f "$out/results/summary.json" ]; then
    PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync python scripts/generate_jkyb_audio.py \
      --dataset "$E/${SET}_rows.jsonl" --output-dir "$out/audio" --ref-wav "$REF" "$@" \
      > "$out/logs/gen.log" 2>&1
    (cd Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition && uv run --no-sync jkyb-eval tts "../$out/audio" \
      --dataset "../$E/${SET}_rows.jsonl" --device cuda --skip-text-cer --output-dir "../$out/results") \
      > "$out/logs/eval.log" 2>&1
  fi
}

for seed in 1 2; do
  run base "$seed" --seed "$seed"
  run v3 "$seed" --seed "$seed" --hf-checkpoint "$V3"
  uv run python scripts/compare_jkyb_runs.py "$E/seeds/base_seed$seed/$SET/results" \
    "$E/seeds/v3_seed$seed/$SET/results" --base-label base --cand-label v3 \
    --output "$E/seeds/v3_seed$seed/$SET/compare.md" > /dev/null
  echo "seed $seed: $(grep -m1 '| 全体' "$E/seeds/v3_seed$seed/$SET/compare.md")"
done
