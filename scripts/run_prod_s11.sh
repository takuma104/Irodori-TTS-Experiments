#!/usr/bin/env bash
# S11: continue S10 with the frames around the target weighted 5x.
#
#   bash scripts/run_prod_s11.sh
#
# The fit diagnosis (run_diag_fit.sh) showed that the velocity loss, averaged over
# all frames, dilutes the target signal in long corpus sentences; weighting the
# target window 5x fit the hard Wikipedia/Aozora rows much faster (+14pt / +9pt at
# equal exposures). Same data and settings as S10 otherwise.
# Checkpoints: hard training rows + Aozora ruby. Final: also prod dev, JSUT kana
# CER, general CER, M3 hard rows (retention), and JKYB last as an external score.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ALLOC_CONF=expandable_segments:True

M=outputs/yomi_prod
S=$M/s11_cont
STEPS=${STEPS:-30000}
SAVE_EVERY=${SAVE_EVERY:-10000}
REF=data/jvs_ver1/jvs001/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav
mkdir -p "$S"

if [ ! -f "$S/student/student.safetensors" ]; then
  PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync python scripts/train_kana_distill.py \
    --init-student $M/s10_cont/part2/student \
    --teacher-dir outputs/yomi_m3/pilot_mixed --teacher-dir outputs/yomi_m3/teacher_mixed \
    --teacher-dir outputs/yomi_kun/teacher_mixed --teacher-dir $M/teacher_mixed \
    --teacher-dir outputs/yomi_aozora/teacher_mixed --teacher-dir outputs/yomi_pilot/teacher_general \
    --oversample-dir $M/teacher_mixed --oversample-dir outputs/yomi_aozora/teacher_mixed \
    --rows data/yomi/rows_pilot.jsonl --rows data/yomi/rows_m3.jsonl \
    --rows data/yomi/rows_kun.jsonl --rows data/yomi/rows_prod.jsonl \
    --rows data/yomi/rows_aozora.jsonl \
    --general-text data/yomi/general_text.jsonl --ctx-weight 1.0 --repr-weight 1.0 \
    --window-weight 5 \
    --output-dir "$S" --steps "$STEPS" --batch-size 32 --lr 3e-4 --backbone-lr 1e-4 \
    --warmup-steps 200 --target-repeat 3 --eval-every 5000 --save-every "$SAVE_EVERY" \
    --dev-limit 2000 --seed 7 > "$S/train.log" 2>&1
fi

for ((step = SAVE_EVERY; step < STEPS; step += SAVE_EVERY)); do
  for set in train_hard_aozora train_hard_prod aozora; do
    EVAL_SET=$set bash scripts/run_student_eval.sh "$S/student_step$step" "s11_step$step" --no-jkyb
  done
done
for set in train_hard_aozora train_hard_prod aozora prod_dev train_hard; do
  EVAL_SET=$set bash scripts/run_student_eval.sh "$S/student" s11_cont --no-jkyb
done
for set in jsut regress; do
  rows=data/yomi/jsut_eval_rows.jsonl
  [ "$set" = regress ] && rows=data/yomi/regress_rows.jsonl
  PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync python scripts/generate_jkyb_audio.py \
    --dataset "$rows" --output-dir "outputs/yomi_eval/s11_cont/$set/audio" --ref-wav "$REF" \
    --seed 0 --student "$S/student" > /dev/null 2>&1
done
(cd Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition && \
  uv run --no-sync python ../scripts/eval_general_cer.py ../outputs/yomi_eval/base/jsut \
    ../outputs/yomi_eval/s11_cont/jsut --rows ../data/yomi/jsut_eval_rows.jsonl --mode kana && \
  uv run --no-sync python ../scripts/eval_general_cer.py ../outputs/yomi_eval/base/regress \
    ../outputs/yomi_eval/s11_cont/regress --rows ../data/yomi/regress_rows.jsonl) \
  > outputs/yomi_eval/s11_cont_general.log 2>&1
bash scripts/run_student_eval.sh "$S/student" s11_cont
echo "s11 done"
