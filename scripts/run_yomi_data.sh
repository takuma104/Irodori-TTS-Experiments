#!/usr/bin/env bash
# Build a yomi dataset from data/yomi/targets_<name>.jsonl (plan §4).
#
#   bash scripts/run_yomi_data.sh <name>     # e.g. m3, kun
#
# 1. Sentences from the local LLM (vLLM is started here and stopped afterwards).
# 2. The base model on the kanji text of every row: hard mining, and the teacher
#    latents for easy target rows and contrast rows.
# 3. The kana (katakana) teacher only for target rows the base misreads.
# 4. The mixed teacher manifest (mix_teacher_by_difficulty.py).
# Steps whose outputs exist are skipped, so the script can be rerun after a stop.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=$1
M=outputs/yomi_$NAME
mkdir -p "$M/logs"
iro() { PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync python "$@"; }
jkyb_eval() {  # jkyb_eval <run-dir>: score <run-dir>/audio against its dataset.jsonl
  if [ ! -f "$1/results/summary.json" ]; then
    (cd Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition && uv run --no-sync jkyb-eval tts \
      "../$1/audio" --dataset "../$1/dataset.jsonl" --device cuda --skip-text-cer \
      --output-dir "../$1/results") > "$1/eval.log" 2>&1
  fi
}

if [ ! -f "$M/sentences.done" ]; then
  bash scripts/serve_llm.sh > "$M/logs/vllm.log" 2>&1 &
  until curl -sf localhost:8000/v1/models > /dev/null; do
    if grep -q "initialization failed" "$M/logs/vllm.log"; then
      echo "vLLM failed to start" >&2
      exit 1
    fi
    sleep 10
  done
  uv run python scripts/generate_yomi_sentences.py data/yomi/targets_$NAME.jsonl \
    --output data/yomi/sentences_$NAME.jsonl --concurrency 64 > "$M/logs/sentences.log" 2>&1
  pkill -f "vllm_serve/.venv/bin/python .*vllm serve" || true
  while curl -sf localhost:8000/v1/models > /dev/null; do sleep 5; done
  sleep 20
  touch "$M/sentences.done"
fi

uv run python scripts/prepare_yomi_rows.py data/yomi/sentences_$NAME.jsonl \
  --targets data/yomi/targets_$NAME.jsonl --output data/yomi/rows_$NAME.jsonl

iro scripts/generate_teacher_latents.py data/yomi/rows_$NAME.jsonl --text-mode kanji \
  --output-dir "$M/base_kanji" > "$M/logs/base_kanji.log" 2>&1
jkyb_eval "$M/base_kanji"

NAME=$NAME uv run python - <<'PY'
import json
import os

name = os.environ["NAME"]
base = {}
with open(f"outputs/yomi_{name}/base_kanji/results/details/all.jsonl") as f:
    for line in f:
        row = json.loads(line)
        base[row["key"]] = row["target_exact"]
kept = 0
with open(f"data/yomi/rows_{name}.jsonl") as f, open(f"outputs/yomi_{name}/rows_hard.jsonl", "w") as out:
    for line in f:
        row = json.loads(line)
        if row["role"] == "target" and base.get(row["key"]) is False:
            out.write(line)
            kept += 1
print(f"hard target rows: {kept}")
PY

iro scripts/generate_teacher_latents.py "$M/rows_hard.jsonl" --output-dir "$M/teacher" \
  > "$M/logs/teacher.log" 2>&1
jkyb_eval "$M/teacher"

uv run python scripts/mix_teacher_by_difficulty.py --base "$M/base_kanji" \
  --teacher "$M/teacher" --output "$M/teacher_mixed"
echo "$NAME data done"
