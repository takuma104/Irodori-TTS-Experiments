#!/usr/bin/env bash
# Production model: corpus data without JKYB, continue S8, evaluate without JKYB.
#
#   bash scripts/run_prod.sh
#
# Expects data/yomi/{sentences,targets}_prod.jsonl (retrieve_corpus_sentences.py +
# select_corpus_targets.py) and data/yomi/{sentences,targets}_aozora.jsonl
# (make_aozora_ruby_train.py).
# 1. Verify homograph readings of the Wikipedia sentences with the local LLM.
# 2. Rows, base-model hard mining, kana teacher (run_yomi_data.sh prod / aozora).
# 3. Continue from S8 with pilot + M3 + kun + prod + aozora data (S9).
# 4. Evaluate on JSUT kana labels, Aozora ruby, prod dev, general CER; JKYB last,
#    as an external score only (not used for choosing anything).
set -euo pipefail
cd "$(dirname "$0")/.."

M=outputs/yomi_prod
S=$M/s9_cont
REF=data/jvs_ver1/jvs001/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav
mkdir -p "$M/logs"

if [ ! -f "$M/sentences.done" ]; then
  bash scripts/serve_llm.sh > "$M/logs/vllm.log" 2>&1 &
  until curl -sf localhost:8000/v1/models > /dev/null; do
    if grep -q "initialization failed" "$M/logs/vllm.log"; then
      echo "vLLM failed to start" >&2
      exit 1
    fi
    sleep 10
  done
  uv run python scripts/verify_homograph_sentences.py data/yomi/sentences_prod.jsonl \
    --lexicon data/yomi_prod/lexicon.jsonl > "$M/logs/verify.log" 2>&1
  pkill -f "vllm_serve/.venv/bin/python .*vllm serve" || true
  while curl -sf localhost:8000/v1/models > /dev/null; do sleep 5; done
  sleep 20
  touch "$M/sentences.done"
fi

bash scripts/run_yomi_data.sh prod
# Aozora ruby readings are human-provided; no LLM step.
mkdir -p outputs/yomi_aozora
touch outputs/yomi_aozora/sentences.done
bash scripts/run_yomi_data.sh aozora

if [ ! -f "$S/student/student.safetensors" ]; then
  mkdir -p "$S"
  PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync python scripts/train_kana_distill.py \
    --init-student outputs/yomi_kun/s8_cont/student \
    --teacher-dir outputs/yomi_m3/pilot_mixed --teacher-dir outputs/yomi_m3/teacher_mixed \
    --teacher-dir outputs/yomi_kun/teacher_mixed --teacher-dir $M/teacher_mixed \
    --teacher-dir outputs/yomi_aozora/teacher_mixed --teacher-dir outputs/yomi_pilot/teacher_general \
    --rows data/yomi/rows_pilot.jsonl --rows data/yomi/rows_m3.jsonl \
    --rows data/yomi/rows_kun.jsonl --rows data/yomi/rows_prod.jsonl \
    --rows data/yomi/rows_aozora.jsonl \
    --general-text data/yomi/general_text.jsonl --ctx-weight 1.0 --repr-weight 1.0 \
    --output-dir "$S" --steps "${STEPS:-15000}" --batch-size 32 --lr 3e-4 --backbone-lr 1e-4 \
    --warmup-steps 200 --target-repeat 3 --eval-every 3000 --dev-limit 2000 --seed 4 \
    > "$S/train.log" 2>&1
fi

uv run python -c "
import json
with open('data/yomi/rows_prod.jsonl') as f, open('outputs/yomi_eval/prod_dev_rows.jsonl', 'w') as out:
    for line in f:
        if json.loads(line)['split'] == 'dev':
            out.write(line)
"
for name in s8_cont s9_cont; do
  student=outputs/yomi_kun/s8_cont/student
  [ "$name" = s9_cont ] && student=$S/student
  for set in prod_dev aozora; do
    EVAL_SET=$set bash scripts/run_student_eval.sh "$student" "$name" --no-jkyb
  done
done
for set in jsut regress; do
  rows=data/yomi/jsut_eval_rows.jsonl
  [ "$set" = regress ] && rows=data/yomi/regress_rows.jsonl
  PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync python scripts/generate_jkyb_audio.py \
    --dataset "$rows" --output-dir "outputs/yomi_eval/s9_cont/$set/audio" --ref-wav "$REF" \
    --seed 0 --student "$S/student" > /dev/null 2>&1
done
(cd Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition && \
  uv run --no-sync python ../scripts/eval_general_cer.py ../outputs/yomi_eval/base/jsut \
    ../outputs/yomi_eval/s9_cont/jsut --rows ../data/yomi/jsut_eval_rows.jsonl --mode kana && \
  uv run --no-sync python ../scripts/eval_general_cer.py ../outputs/yomi_eval/base/regress \
    ../outputs/yomi_eval/s9_cont/regress --rows ../data/yomi/regress_rows.jsonl) \
  > outputs/yomi_eval/s9_cont_general.log 2>&1
bash scripts/run_student_eval.sh "$S/student" s9_cont
echo "prod done"
