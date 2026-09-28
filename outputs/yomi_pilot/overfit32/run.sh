#!/usr/bin/env bash
# Overfit test: can the student flip the reading of 32 training rows at all?
set -euo pipefail
cd "$(dirname "$0")/../../.."
O=outputs/yomi_pilot/overfit32
REF=data/jvs_ver1/jvs001/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav
iro() { PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync python "$@"; }
iro scripts/train_kana_distill.py --teacher-dir $O/teacher --output-dir $O/train \
  --scope top4 --steps 300 --batch-size 32 --lr 3e-4 --backbone-lr 1e-4 \
  --warmup-steps 20 --eval-every 50 --speaker-dropout 0.0 > $O/train.log 2>&1
for name in base student; do
  extra=()
  [ "$name" = student ] && extra=(--student $O/train/student)
  iro scripts/generate_jkyb_audio.py --dataset $O/rows.jsonl --output-dir $O/$name/audio \
    --ref-wav $REF --seed 0 "${extra[@]}" > $O/$name.gen.log 2>&1
  (cd Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition && uv run --no-sync jkyb-eval tts \
    ../$O/$name/audio --dataset ../$O/rows.jsonl --device cuda --skip-text-cer \
    --output-dir ../$O/$name/results) > $O/$name.eval.log 2>&1
done
uv run python scripts/compare_jkyb_runs.py $O/base/results $O/student/results \
  --base-label base --cand-label overfit --output $O/compare.md > /dev/null
echo "overfit done"
