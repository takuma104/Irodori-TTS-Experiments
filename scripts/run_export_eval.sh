#!/usr/bin/env bash
# Evaluate an exported checkpoint (export_student_checkpoint.py) like a student.
#
#   bash scripts/run_export_eval.sh ../Irodori-TTS-v4.1-Small-Yomi/model.safetensors release
#
# Same sets and settings as run_prod_s10.sh (mixed precision, jvs001, seed 0). The
# runtime casts the whole model to bf16 before the text modules go back to fp32,
# so the merged top layers are bf16-rounded here, unlike the --student path;
# results can differ slightly from the student's. Results go to
# outputs/yomi_eval/<name>/.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=$1
NAME=$2
REF=data/jvs_ver1/jvs001/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav
BASE_JKYB=outputs/irodori-v4.1-small_jvs001_bf16mix
E=outputs/yomi_eval

gen() {  # gen <run-dir> [extra args...]
  local run=$1
  shift
  mkdir -p "$run/logs"
  PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync python scripts/generate_jkyb_audio.py \
    --hf-checkpoint "$CKPT" --output-dir "$run/audio" --ref-wav "$REF" --seed 0 "$@" \
    > "$run/logs/gen.log" 2>&1
}

jkyb_eval() {  # jkyb_eval <run-dir> [extra args...]
  local run=$1
  shift
  if [ ! -f "$run/results/summary.json" ]; then
    (cd Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition && uv run --no-sync jkyb-eval tts \
      "../$run/audio" --device cuda --output-dir "../$run/results" "$@") > "$run/logs/eval.log" 2>&1
  fi
}

for set in aozora prod_dev dev; do
  gen "$E/$NAME/$set" --dataset "$E/${set}_rows.jsonl"
  jkyb_eval "$E/$NAME/$set" --dataset "../$E/${set}_rows.jsonl" --skip-text-cer
  uv run python scripts/compare_jkyb_runs.py "$E/base/$set/results" "$E/$NAME/$set/results" \
    --base-label base --cand-label "$NAME" --output "$E/$NAME/$set/compare.md" > /dev/null
done
gen "$E/$NAME/jsut" --dataset data/yomi/jsut_eval_rows.jsonl
gen "$E/$NAME/regress" --dataset data/yomi/regress_rows.jsonl
(cd Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition && \
  uv run --no-sync python ../scripts/eval_general_cer.py ../$E/base/jsut ../$E/$NAME/jsut \
    --rows ../data/yomi/jsut_eval_rows.jsonl --mode kana && \
  uv run --no-sync python ../scripts/eval_general_cer.py ../$E/base/regress ../$E/$NAME/regress \
    --rows ../data/yomi/regress_rows.jsonl) > "$E/${NAME}_general.log" 2>&1
R=$E/$NAME/jkyb
gen "$R"
jkyb_eval "$R"
uv run python scripts/analyze_jkyb_errors.py "$R/results" --output "$R/error_analysis.md"
uv run python scripts/compare_jkyb_runs.py "$BASE_JKYB/results" "$R/results" \
  --base-errors "$BASE_JKYB/error_analysis.errors.jsonl" --base-label bf16mix \
  --cand-label "$NAME" --output "$R/compare.md" > /dev/null
echo "export eval $NAME done"
