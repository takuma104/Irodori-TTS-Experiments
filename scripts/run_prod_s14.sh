#!/usr/bin/env bash
# S14: continue S12 with more sentences per word (var, aozora2).
#
#   bash scripts/run_prod_s14.sh
#
# S13 (more steps on the same data) only memorized the training sentences. S14
# adds the LLM variation sentences (6 per hard corpus word, var) and the Aozora
# ruby sentences of 2,000 more works (aozora2), both oversampled 2x. The earlier
# Wikipedia/Aozora data, well fit by S12, is no longer oversampled. Recipe as
# S11/S12 (target window weighted 5x); 60k steps, checkpoints every 20k.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ALLOC_CONF=expandable_segments:True

M=outputs/yomi_prod
NAME=s14_cont
S=$M/$NAME
STEPS=${STEPS:-60000}
SAVE_EVERY=${SAVE_EVERY:-20000}
REF=data/jvs_ver1/jvs001/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav
mkdir -p "$S"

if [ ! -f "$S/student/student.safetensors" ]; then
  PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync python scripts/train_kana_distill.py \
    --init-student $M/s12_cont/student \
    --teacher-dir outputs/yomi_m3/pilot_mixed --teacher-dir outputs/yomi_m3/teacher_mixed \
    --teacher-dir outputs/yomi_kun/teacher_mixed --teacher-dir $M/teacher_mixed \
    --teacher-dir outputs/yomi_aozora/teacher_mixed --teacher-dir outputs/yomi_pilot/teacher_general \
    --teacher-dir outputs/yomi_var/teacher_mixed --teacher-dir outputs/yomi_aozora2/teacher_mixed \
    --oversample-dir outputs/yomi_var/teacher_mixed --oversample-dir outputs/yomi_aozora2/teacher_mixed \
    --rows data/yomi/rows_pilot.jsonl --rows data/yomi/rows_m3.jsonl \
    --rows data/yomi/rows_kun.jsonl --rows data/yomi/rows_prod.jsonl \
    --rows data/yomi/rows_aozora.jsonl --rows data/yomi/rows_var.jsonl \
    --rows data/yomi/rows_aozora2.jsonl \
    --general-text data/yomi/general_text.jsonl --ctx-weight 1.0 --repr-weight 1.0 \
    --window-weight 5 \
    --output-dir "$S" --steps "$STEPS" --batch-size 32 --lr 3e-4 --backbone-lr 1e-4 \
    --warmup-steps 200 --target-repeat 3 --eval-every 5000 --save-every "$SAVE_EVERY" \
    --dev-limit 2000 --seed 10 > "$S/train.log" 2>&1
fi

FIT_SETS="train_hard_var train_hard_aozora2 train_hard_aozora train_hard_prod"
for ((step = SAVE_EVERY; step < STEPS; step += SAVE_EVERY)); do
  for set in $FIT_SETS aozora; do
    EVAL_SET=$set bash scripts/run_student_eval.sh "$S/student_step$step" "s14_step$step" --no-jkyb
  done
done
for set in $FIT_SETS aozora prod_dev train_hard; do
  EVAL_SET=$set bash scripts/run_student_eval.sh "$S/student" "$NAME" --no-jkyb
done
for set in jsut regress; do
  rows=data/yomi/jsut_eval_rows.jsonl
  [ "$set" = regress ] && rows=data/yomi/regress_rows.jsonl
  PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync python scripts/generate_jkyb_audio.py \
    --dataset "$rows" --output-dir "outputs/yomi_eval/$NAME/$set/audio" --ref-wav "$REF" \
    --seed 0 --student "$S/student" > /dev/null 2>&1
done
(cd Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition && \
  uv run --no-sync python ../scripts/eval_general_cer.py ../outputs/yomi_eval/base/jsut \
    "../outputs/yomi_eval/$NAME/jsut" --rows ../data/yomi/jsut_eval_rows.jsonl --mode kana && \
  uv run --no-sync python ../scripts/eval_general_cer.py ../outputs/yomi_eval/base/regress \
    "../outputs/yomi_eval/$NAME/regress" --rows ../data/yomi/regress_rows.jsonl) \
  > "outputs/yomi_eval/${NAME}_general.log" 2>&1
bash scripts/run_student_eval.sh "$S/student" "$NAME"
echo "$NAME done"
