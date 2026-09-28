#!/usr/bin/env bash
# Plan M1/M2 pilot: rows -> teacher latents -> base-model hard mining -> evals.
#
#   bash scripts/run_m1_pilot.sh
#
# Expects data/yomi/sentences_pilot.jsonl (generate_yomi_sentences.py) and
# data/yomi/general_rows.jsonl (make_general_rows.py). Stop the LLM server first.
set -euo pipefail
cd "$(dirname "$0")/.."

P=outputs/yomi_pilot
iro() { PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync python "$@"; }
jkyb_eval() {  # jkyb_eval <run-dir>: score <run-dir>/audio against its dataset.jsonl
  (cd Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition && uv run --no-sync jkyb-eval tts \
    "../$1/audio" --dataset "../$1/dataset.jsonl" --device cuda --skip-text-cer \
    --output-dir "../$1/results")
}

uv run python scripts/prepare_yomi_rows.py data/yomi/sentences_pilot.jsonl \
  --targets data/yomi/targets_pilot.jsonl --output data/yomi/rows_pilot.jsonl

# Teacher: katakana-substituted text for targets, kanji text for contrast rows.
iro scripts/generate_teacher_latents.py data/yomi/rows_pilot.jsonl --output-dir "$P/teacher"
iro scripts/generate_teacher_latents.py data/yomi/general_rows.jsonl --splits train,dev \
  --output-dir "$P/teacher_general"
# The unchanged model on the kanji text of the target rows (hard mining).
iro scripts/generate_teacher_latents.py data/yomi/rows_pilot.jsonl --roles target \
  --text-mode kanji --output-dir "$P/base_kanji"

jkyb_eval "$P/teacher"
jkyb_eval "$P/base_kanji"
echo "M1 pilot data done"
