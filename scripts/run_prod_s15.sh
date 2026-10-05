#!/usr/bin/env bash
# S15: continue S14 with contrast rows for kanji trained as single-kanji words.
#
#   bash scripts/run_prod_s15.sh
#
# S14 leaked single-kanji Aozora readings into compounds (pilot dev -2pt:
# 秋季 シュウ -> アキ, 活火山 -> イカザン). Contrast data: for each such kanji,
# Wikipedia sentences where it sits in a common compound with another reading
# (select_contrast_targets.py on retrieve_corpus_sentences.py --words hits). The
# base model's own output is kept as the teacher (rows it reads correctly only),
# with the kanji's window weighted like the targets (--window-contrast), and the
# rows are seen 3x. Held-out compounds (dev split) are the leak check
# contrast_dev. Expects data/yomi/contrast_hits.jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ALLOC_CONF=expandable_segments:True

M=outputs/yomi_prod
C=outputs/yomi_contrast
NAME=s15_cont
S=$M/$NAME
STEPS=${STEPS:-30000}
SAVE_EVERY=${SAVE_EVERY:-10000}
REF=data/jvs_ver1/jvs001/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav
E=outputs/yomi_eval
mkdir -p "$S" "$C/logs"
iro() { PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync python "$@"; }

# 1. Contrast data.
if [ ! -f "$C/teacher_mixed/manifest.jsonl" ]; then
  uv run python scripts/select_contrast_targets.py --name aozora --name aozora2 --name var \
    --hits data/yomi/contrast_hits.jsonl --compounds 4 --per-compound 4 --output-name contrast
  uv run python scripts/prepare_yomi_rows.py data/yomi/sentences_contrast.jsonl \
    --targets data/yomi/targets_contrast.jsonl --output data/yomi/rows_contrast.jsonl --key-prefix c
  iro scripts/generate_teacher_latents.py data/yomi/rows_contrast.jsonl --text-mode kanji \
    --output-dir "$C/base_kanji" > "$C/logs/base_kanji.log" 2>&1
  (cd Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition && uv run --no-sync jkyb-eval tts \
    "../$C/base_kanji/audio" --dataset "../$C/base_kanji/dataset.jsonl" --device cuda \
    --skip-text-cer --output-dir "../$C/base_kanji/results") > "$C/logs/base_eval.log" 2>&1
  uv run python scripts/mix_teacher_by_difficulty.py --base "$C/base_kanji" \
    --output "$C/teacher_mixed"
fi
uv run python -c "
import json
with open('data/yomi/rows_contrast.jsonl') as f, open('$E/contrast_dev_rows.jsonl', 'w') as out:
    for line in f:
        if json.loads(line)['split'] == 'dev':
            out.write(line)
"

# 2. Training.
if [ ! -f "$S/student/student.safetensors" ]; then
  iro scripts/train_kana_distill.py \
    --init-student $M/s14_cont/student \
    --teacher-dir outputs/yomi_m3/pilot_mixed --teacher-dir outputs/yomi_m3/teacher_mixed \
    --teacher-dir outputs/yomi_kun/teacher_mixed --teacher-dir $M/teacher_mixed \
    --teacher-dir outputs/yomi_aozora/teacher_mixed --teacher-dir outputs/yomi_pilot/teacher_general \
    --teacher-dir outputs/yomi_var/teacher_mixed --teacher-dir outputs/yomi_aozora2/teacher_mixed \
    --teacher-dir "$C/teacher_mixed" \
    --oversample-dir outputs/yomi_var/teacher_mixed --oversample-dir outputs/yomi_aozora2/teacher_mixed \
    --oversample-dir "$C/teacher_mixed" --oversample-dir "$C/teacher_mixed" \
    --rows data/yomi/rows_pilot.jsonl --rows data/yomi/rows_m3.jsonl \
    --rows data/yomi/rows_kun.jsonl --rows data/yomi/rows_prod.jsonl \
    --rows data/yomi/rows_aozora.jsonl --rows data/yomi/rows_var.jsonl \
    --rows data/yomi/rows_aozora2.jsonl --rows data/yomi/rows_contrast.jsonl \
    --general-text data/yomi/general_text.jsonl --ctx-weight 1.0 --repr-weight 1.0 \
    --window-weight 5 --window-contrast \
    --output-dir "$S" --steps "$STEPS" --batch-size 32 --lr 3e-4 --backbone-lr 1e-4 \
    --warmup-steps 200 --target-repeat 3 --eval-every 5000 --save-every "$SAVE_EVERY" \
    --dev-limit 2000 --seed 11 > "$S/train.log" 2>&1
fi

# 3. Evaluation: the leak check for S12/S14 too, checkpoints, then the full set.
for prev in s12_cont s14_cont; do
  EVAL_SET=contrast_dev bash scripts/run_student_eval.sh "$M/$prev/student" "$prev" --no-jkyb
done
for ((step = SAVE_EVERY; step < STEPS; step += SAVE_EVERY)); do
  for set in contrast_dev dev aozora; do
    EVAL_SET=$set bash scripts/run_student_eval.sh "$S/student_step$step" "s15_step$step" --no-jkyb
  done
done
for set in contrast_dev dev aozora prod_dev train_hard_var train_hard_aozora2; do
  EVAL_SET=$set bash scripts/run_student_eval.sh "$S/student" "$NAME" --no-jkyb
done
for set in jsut regress; do
  rows=data/yomi/jsut_eval_rows.jsonl
  [ "$set" = regress ] && rows=data/yomi/regress_rows.jsonl
  iro scripts/generate_jkyb_audio.py --dataset "$rows" --output-dir "$E/$NAME/$set/audio" \
    --ref-wav "$REF" --seed 0 --student "$S/student" > /dev/null 2>&1
done
(cd Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition && \
  uv run --no-sync python ../scripts/eval_general_cer.py ../$E/base/jsut "../$E/$NAME/jsut" \
    --rows ../data/yomi/jsut_eval_rows.jsonl --mode kana && \
  uv run --no-sync python ../scripts/eval_general_cer.py ../$E/base/regress "../$E/$NAME/regress" \
    --rows ../data/yomi/regress_rows.jsonl) > "$E/${NAME}_general.log" 2>&1
bash scripts/run_student_eval.sh "$S/student" "$NAME"
echo "$NAME done"
