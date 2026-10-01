#!/usr/bin/env bash
# S10: continue S9 with the new production data (Wikipedia + Aozora) oversampled.
#
#   bash scripts/run_prod_s10.sh
#
# S9 saw each new hard row only ~5 times and fit few of them (train_hard_*), so
# this run doubles the new rows and trains 30k more steps, saving checkpoints.
# Part 1 (30k-step schedule from S9) hit a CUDA OOM from allocator fragmentation
# at step ~18.6k; part 2 continues its step-10k checkpoint for the remaining
# 20k steps (fresh warmup + cosine) with expandable segments.
# Checkpoints: hard training rows + Aozora ruby. Final: also prod dev, JSUT kana
# CER, general CER, M3 hard rows (retention), and JKYB last as an external score.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ALLOC_CONF=expandable_segments:True

M=outputs/yomi_prod
S=$M/s10_cont
REF=data/jvs_ver1/jvs001/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav
mkdir -p "$S"

train() {  # train <init-student> <output-dir> <steps> <seed>
  mkdir -p "$2"
  PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync python scripts/train_kana_distill.py \
    --init-student "$1" \
    --teacher-dir outputs/yomi_m3/pilot_mixed --teacher-dir outputs/yomi_m3/teacher_mixed \
    --teacher-dir outputs/yomi_kun/teacher_mixed --teacher-dir $M/teacher_mixed \
    --teacher-dir outputs/yomi_aozora/teacher_mixed --teacher-dir outputs/yomi_pilot/teacher_general \
    --oversample-dir $M/teacher_mixed --oversample-dir outputs/yomi_aozora/teacher_mixed \
    --rows data/yomi/rows_pilot.jsonl --rows data/yomi/rows_m3.jsonl \
    --rows data/yomi/rows_kun.jsonl --rows data/yomi/rows_prod.jsonl \
    --rows data/yomi/rows_aozora.jsonl \
    --general-text data/yomi/general_text.jsonl --ctx-weight 1.0 --repr-weight 1.0 \
    --output-dir "$2" --steps "$3" --batch-size 32 --lr 3e-4 --backbone-lr 1e-4 \
    --warmup-steps 200 --target-repeat 3 --eval-every 5000 --save-every 10000 \
    --dev-limit 2000 --seed "$4" > "$2/train.log" 2>&1
}

if [ ! -f "$S/student_step10000/student.safetensors" ]; then
  train $M/s9_cont/student "$S" 30000 5
fi
if [ ! -f "$S/part2/student/student.safetensors" ]; then
  train "$S/student_step10000" "$S/part2" 20000 6
fi

# name:student-dir for each checkpoint (global step).
CHECKPOINTS="s10_step10000:$S/student_step10000 s10_step20000:$S/part2/student_step10000"
FINAL=$S/part2/student
for item in $CHECKPOINTS; do
  for set in train_hard_aozora train_hard_prod aozora; do
    EVAL_SET=$set bash scripts/run_student_eval.sh "${item#*:}" "${item%%:*}" --no-jkyb
  done
done
for set in train_hard_aozora train_hard_prod aozora prod_dev train_hard; do
  EVAL_SET=$set bash scripts/run_student_eval.sh "$FINAL" s10_cont --no-jkyb
done
for set in jsut regress; do
  rows=data/yomi/jsut_eval_rows.jsonl
  [ "$set" = regress ] && rows=data/yomi/regress_rows.jsonl
  PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync python scripts/generate_jkyb_audio.py \
    --dataset "$rows" --output-dir "outputs/yomi_eval/s10_cont/$set/audio" --ref-wav "$REF" \
    --seed 0 --student "$FINAL" > /dev/null 2>&1
done
(cd Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition && \
  uv run --no-sync python ../scripts/eval_general_cer.py ../outputs/yomi_eval/base/jsut \
    ../outputs/yomi_eval/s10_cont/jsut --rows ../data/yomi/jsut_eval_rows.jsonl --mode kana && \
  uv run --no-sync python ../scripts/eval_general_cer.py ../outputs/yomi_eval/base/regress \
    ../outputs/yomi_eval/s10_cont/regress --rows ../data/yomi/regress_rows.jsonl) \
  > outputs/yomi_eval/s10_cont_general.log 2>&1
bash scripts/run_student_eval.sh "$FINAL" s10_cont
echo "s10 done"
