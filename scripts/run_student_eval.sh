#!/usr/bin/env bash
# Evaluate a trained student text encoder (plan §6).
#
#   bash scripts/run_student_eval.sh <student-dir> <name> [--no-jkyb]
#
# - Our dev sentences (words held out of training), synthesized with the eval
#   speaker (jvs001) by the base model (once) and by the student.
# - Full JKYB-Parakeet, compared with the mixed-precision baseline.
# Results go to outputs/yomi_eval/<name>/.
set -euo pipefail
cd "$(dirname "$0")/.."

STUDENT=$1
NAME=$2
RUN_JKYB=1
if [ "${3:-}" = "--no-jkyb" ]; then RUN_JKYB=0; fi
REF_WAV=data/jvs_ver1/jvs001/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav
BASE_JKYB=outputs/irodori-v4.1-small_jvs001_bf16mix
DEV_ROWS=${DEV_ROWS:-data/yomi/rows_pilot.jsonl}
E=outputs/yomi_eval
DEV_SET=$E/dev_rows.jsonl

gen() {  # gen <run-dir> [extra args...]
  local run=$1
  shift
  mkdir -p "$run/logs"
  PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync python scripts/generate_jkyb_audio.py \
    --output-dir "$run/audio" --ref-wav "$REF_WAV" --seed 0 "$@" > "$run/logs/gen.log" 2>&1
}

jkyb_eval() {  # jkyb_eval <run-dir> [extra args...]
  local run=$1
  shift
  if [ ! -f "$run/results/summary.json" ]; then
    (cd Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition && uv run --no-sync jkyb-eval tts \
      "../$run/audio" --device cuda --output-dir "../$run/results" "$@") > "$run/logs/eval.log" 2>&1
  fi
}

mkdir -p "$E"
if [ ! -f "$DEV_SET" ]; then
  uv run python -c "
import json, sys
with open('$DEV_ROWS') as f, open('$DEV_SET', 'w') as out:
    for line in f:
        row = json.loads(line)
        if row.get('split') == 'dev':
            out.write(line)
"
fi

# Base model on our dev sentences (once).
if [ ! -f "$E/base/dev/results/summary.json" ]; then
  gen "$E/base/dev" --dataset "$DEV_SET"
  jkyb_eval "$E/base/dev" --dataset "../$DEV_SET" --skip-text-cer
fi

gen "$E/$NAME/dev" --dataset "$DEV_SET" --student "$STUDENT"
jkyb_eval "$E/$NAME/dev" --dataset "../$DEV_SET" --skip-text-cer
uv run python scripts/compare_jkyb_runs.py "$E/base/dev/results" "$E/$NAME/dev/results" \
  --base-label base --cand-label "$NAME" --output "$E/$NAME/dev/compare.md" > /dev/null

if [ "$RUN_JKYB" = 1 ]; then
  R=$E/$NAME/jkyb
  gen "$R" --student "$STUDENT"
  jkyb_eval "$R"
  uv run python scripts/analyze_jkyb_errors.py "$R/results" --output "$R/error_analysis.md"
  uv run python scripts/compare_jkyb_runs.py "$BASE_JKYB/results" "$R/results" \
    --base-errors "$BASE_JKYB/error_analysis.errors.jsonl" --base-label bf16mix \
    --cand-label "$NAME" --output "$R/compare.md" > /dev/null
fi
echo "eval $NAME done"
