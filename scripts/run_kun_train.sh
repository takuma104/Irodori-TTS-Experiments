#!/usr/bin/env bash
# Kun-reading follow-up: continue S7 with the kun data (select_kun_targets.py) added.
#
#   bash scripts/run_kun_train.sh
#
# Expects outputs/yomi_kun/teacher_mixed from `bash scripts/run_yomi_data.sh kun`.
# Evaluates on full JKYB, on JKYB group-A rows whose word stem (kanji run and its
# reading) is among the kun training words, and on the kun dev rows (held-out words).
set -euo pipefail
cd "$(dirname "$0")/.."

S=outputs/yomi_kun/s8_cont
STEPS=${STEPS:-12000}
mkdir -p "$S"
if [ ! -f "$S/student/student.safetensors" ]; then
  PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync python scripts/train_kana_distill.py \
    --init-student outputs/yomi_m3/s7_cont/student \
    --teacher-dir outputs/yomi_m3/pilot_mixed --teacher-dir outputs/yomi_m3/teacher_mixed \
    --teacher-dir outputs/yomi_kun/teacher_mixed --teacher-dir outputs/yomi_pilot/teacher_general \
    --rows data/yomi/rows_pilot.jsonl --rows data/yomi/rows_m3.jsonl --rows data/yomi/rows_kun.jsonl \
    --general-text data/yomi/general_text.jsonl --ctx-weight 1.0 --repr-weight 1.0 \
    --output-dir "$S" --steps "$STEPS" --batch-size 32 --lr 3e-4 --backbone-lr 1e-4 \
    --warmup-steps 200 --target-repeat 3 --eval-every 3000 --dev-limit 2000 --seed 3 \
    > "$S/train.log" 2>&1
fi

uv run python - <<'PY'
import json
import re
import sys

sys.path.insert(0, "scripts")
from build_yomi_lexicon import jkyb_group, joyo_key
from huggingface_hub import hf_hub_download
from jkyb_kana_oracle import substitute_word, to_katakana


def stem(word, reading):
    m = re.match(r"^([㐀-鿿々]+)([ぁ-ゖ]*)$", word)
    if not m:
        return None
    run, oku = m.groups()
    r = to_katakana(reading)
    if oku and r.endswith(to_katakana(oku)):
        r = r[: len(r) - len(oku)]
    return run, r


stems = set()
with open("data/yomi/rows_kun.jsonl") as f, open("outputs/yomi_eval/kun_dev_rows.jsonl", "w") as dev:
    for line in f:
        row = json.loads(line)
        if row["split"] == "dev":
            dev.write(line)
        elif row["role"] == "target":
            s = stem(row["word"], row["reading"])
            if s:
                stems.add(s)
path = hf_hub_download(
    "Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet",
    "data/common_kanji_source.jsonl",
    repo_type="dataset",
)
n = 0
with open(path) as f, open("outputs/yomi_eval/jkyb_seen_kun_rows.jsonl", "w") as out:
    for line in f:
        row = json.loads(line)
        sub = substitute_word(row)
        if sub and (sub.word, sub.reading) in stems and jkyb_group(joyo_key(row)) == "A":
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
print(f"jkyb_seen_kun rows: {n}")
PY

for set in jkyb_seen_kun kun_dev; do
  EVAL_SET=$set bash scripts/run_student_eval.sh "$S/student" s8_cont --no-jkyb
  # The same sets for S7, the starting point.
  EVAL_SET=$set bash scripts/run_student_eval.sh outputs/yomi_m3/s7_cont/student s7_cont --no-jkyb
done
bash scripts/run_student_eval.sh "$S/student" s8_cont
echo "kun train done"
