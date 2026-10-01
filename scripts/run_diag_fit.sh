#!/usr/bin/env bash
# Why do the corpus rows (Wikipedia, Aozora) fit so slowly? Short runs from the
# base model on hard rows only, with equal exposures per row.
#
#   bash scripts/run_diag_fit.sh
#
# A1: M3 hard rows (control; these fit well in S5-S10)
# A2: Wikipedia + Aozora hard rows, same recipe
# B1: A2 with the frames around the target weighted 5x (long-sentence dilution)
# B2: A2 with the top 8 BERT layers (capacity)
# B3: A2 with the preservation losses at 0.3 (competition with L_ctx / L_repr)
# Each row is seen ~8 times at the checkpoint and ~24 times at the end; both are
# scored on the same training rows (outputs/yomi_eval/train_hard*_rows.jsonl).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ALLOC_CONF=expandable_segments:True

D=outputs/yomi_diag
E=outputs/yomi_eval
mkdir -p "$D"
cat $E/train_hard_prod_rows.jsonl $E/train_hard_aozora_rows.jsonl > "$D/corpus_keys.jsonl"

train() {  # train <name> <keys> <steps> [extra args...]
  local name=$1 keys=$2 steps=$3
  shift 3
  local out=$D/$name
  [ -f "$out/student/student.safetensors" ] && return
  mkdir -p "$out"
  PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync python scripts/train_kana_distill.py \
    --teacher-dir outputs/yomi_m3/pilot_mixed --teacher-dir outputs/yomi_m3/teacher_mixed \
    --teacher-dir outputs/yomi_prod/teacher_mixed --teacher-dir outputs/yomi_aozora/teacher_mixed \
    --rows data/yomi/rows_pilot.jsonl --rows data/yomi/rows_m3.jsonl \
    --rows data/yomi/rows_prod.jsonl --rows data/yomi/rows_aozora.jsonl \
    --include-keys "$keys" --general-text data/yomi/general_text.jsonl \
    --scope top4 --batch-size 32 --lr 3e-4 --backbone-lr 1e-4 --warmup-steps 50 \
    --steps "$steps" --save-every $((steps / 3)) --eval-every "$steps" --dev-limit 200 \
    --seed 0 --output-dir "$out" "$@" > "$out/train.log" 2>&1
}

evaluate() {  # evaluate <name> <steps> <eval sets...>
  local name=$1 steps=$2
  shift 2
  for set in "$@"; do
    EVAL_SET=$set bash scripts/run_student_eval.sh "$D/$name/student_step$((steps / 3))" \
      "diag_${name}_early" --no-jkyb
    EVAL_SET=$set bash scripts/run_student_eval.sh "$D/$name/student" "diag_${name}" --no-jkyb
  done
}

# 1,200 M3 rows / 2,000 corpus rows, 24 exposures each at batch 32.
train a1_m3 $E/train_hard_rows.jsonl 900 --ctx-weight 1.0 --repr-weight 1.0
train a2_corpus "$D/corpus_keys.jsonl" 1500 --ctx-weight 1.0 --repr-weight 1.0
train b1_window "$D/corpus_keys.jsonl" 1500 --ctx-weight 1.0 --repr-weight 1.0 --window-weight 5
train b2_top8 "$D/corpus_keys.jsonl" 1500 --ctx-weight 1.0 --repr-weight 1.0 --scope top8 \
  --grad-checkpoint
train b3_lowpres "$D/corpus_keys.jsonl" 1500 --ctx-weight 0.3 --repr-weight 0.3

evaluate a1_m3 900 train_hard
for name in a2_corpus b1_window b2_top8 b3_lowpres; do
  evaluate "$name" 1500 train_hard_prod train_hard_aozora
done
echo "diag done"
