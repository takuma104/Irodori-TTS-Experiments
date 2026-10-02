#!/usr/bin/env bash
# Model-selection eval of one checkpoint (plan Phase 0.3-0.4):
#   scripts/run_eval_suite.sh <checkpoint: MF|RF|hf repo|model.safetensors> <name> [seed] [extra synth args...]
# Synthesizes the JKYB dev subset (2,000) and 500 JSUT sentences with the fixed
# JSUT reference voice, then scores JKYB (jkyb-eval), JSUT sentence kana CER
# (kana-whisper) and speaker similarity / UTMOS. Results: outputs/eval/<name>/.
set -euo pipefail
cd "$(dirname "$0")/.."
ckpt=$1; name=$2; seed=${3:-0}; shift $(( $# < 3 ? $# : 3 ))
case $ckpt in MF) ckpt=Aratako/Irodori-TTS-v4.1-Small-MF ;; RF) ckpt=Aratako/Irodori-TTS-v4.1-Small ;; esac
export PYTHONPATH=third-party/Irodori-TTS:scripts
PY=third-party/Irodori-TTS/.venv/bin/python
REF=data/jsut/BASIC5000_0001.wav
out=outputs/eval/$name
JKYB_DIR=third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition

for set in jkyb_dev jsut; do
  dataset=data/eval/${set}_rows.jsonl; [ $set = jkyb_dev ] && dataset=data/eval/jkyb_dev.jsonl
  mkdir -p "$out/$set"
  $PY scripts/generate_jkyb_audio.py --hf-checkpoint "$ckpt" --dataset "$dataset" \
    --output-dir "$out/$set/audio" --ref-wav $REF --seed "$seed" \
    --batch-size 16 --max-batch-frames 4096 "$@" > "$out/$set/synth.log" 2>&1
done
(cd $JKYB_DIR && uv run --no-sync jkyb-eval tts "../../$out/jkyb_dev/audio" --device cuda \
  --dataset ../../data/eval/jkyb_dev.jsonl --output-dir "../../$out/jkyb_dev/results") > "$out/jkyb_dev/eval.log" 2>&1
(cd $JKYB_DIR && uv run --no-sync python ../../scripts/eval_general_cer.py "../../$out/jsut" \
  --rows ../../data/eval/jsut_rows.jsonl --mode kana) > "$out/jsut/kana_cer.txt" 2>&1
$PY scripts/eval_audio_quality.py "$out/jkyb_dev" --ref-wav $REF > "$out/jkyb_dev/quality.log" 2>&1
grep -h "Accuracy" "$out/jkyb_dev/results/summary.md" | head -1 || true
tail -1 "$out/jsut/kana_cer.txt"
tail -1 "$out/jkyb_dev/quality.log"
echo "suite done: $name"
