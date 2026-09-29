#!/usr/bin/env bash
# Plan M3: train on pilot + M3 data with the S4 recipe, then evaluate.
#
#   bash scripts/run_m3_train.sh
#
# Recipe (S4): top-4 BERT layers, projector 3e-4 / BERT 1e-4, kana teacher only
# on rows the base misreads (x3), and representation preservation on non-target
# tokens (--ctx-weight) and on general Wikipedia sentences (--repr-weight).
set -euo pipefail
cd "$(dirname "$0")/.."

P=outputs/yomi_pilot
M=outputs/yomi_m3
S=$M/s5_top4_ctx
STEPS=${STEPS:-9000}
mkdir -p "$S"

uv run python scripts/mix_teacher_by_difficulty.py --base $P/base_kanji --teacher $P/teacher \
  --output $M/pilot_mixed
if [ ! -f "$S/student/student.safetensors" ]; then
  PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync python scripts/train_kana_distill.py \
    --teacher-dir $M/pilot_mixed --teacher-dir $M/teacher_mixed --teacher-dir $P/teacher_general \
    --rows data/yomi/rows_pilot.jsonl --rows data/yomi/rows_m3.jsonl \
    --general-text data/yomi/general_text.jsonl --ctx-weight 1.0 --repr-weight 1.0 \
    --output-dir "$S" --scope top4 --steps "$STEPS" --batch-size 32 --lr 3e-4 --backbone-lr 1e-4 \
    --target-repeat 3 --eval-every 1000 > "$S/train.log" 2>&1
fi

# JKYB rows whose word and reading are among the training target words (A group).
uv run python - <<'PY'
import json
import sys

sys.path.insert(0, "scripts")
from build_yomi_lexicon import jkyb_group, joyo_key
from huggingface_hub import hf_hub_download
from jkyb_kana_oracle import substitute_word, to_hiragana

words = set()
for path in ["data/yomi/rows_pilot.jsonl", "data/yomi/rows_m3.jsonl"]:
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row["split"] == "train" and row["role"] == "target":
                words.add((row["word"], row["reading"]))
path = hf_hub_download(
    "Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet",
    "data/common_kanji_source.jsonl",
    repo_type="dataset",
)
n = 0
with open(path) as f, open("outputs/yomi_eval/jkyb_seen_m3_rows.jsonl", "w") as out:
    for line in f:
        row = json.loads(line)
        sub = substitute_word(row)
        if sub and (sub.word, to_hiragana(sub.reading)) in words and jkyb_group(joyo_key(row)) == "A":
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
print(f"jkyb_seen_m3 rows: {n}")
PY

EVAL_SET=jkyb_seen_m3 bash scripts/run_student_eval.sh "$S/student" s5_top4_ctx --no-jkyb
bash scripts/run_student_eval.sh "$S/student" s5_top4_ctx
echo "M3 train done"
