#!/usr/bin/env bash
# More sentences per word for the corpus data (after S13 plateaued).
#
#   bash scripts/run_var_data.sh
#
# 1. var: LLM sentences (6 per word) for every training word the base misreads in
#    the Wikipedia/Aozora data (select_variation_targets.py), checked against the
#    production lexicon; row keys prefixed with "v".
# 2. aozora2: ruby sentences from 3,000 Aozora train-split works instead of 1,000,
#    skipping the sentences already in data/yomi/sentences_aozora.jsonl (up to 4
#    new sentences per word); row keys prefixed with "a2". The download runs on
#    the CPU while step 1 uses the GPU.
# Each dataset then goes through hard mining and the kana teacher (run_yomi_data.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p outputs/yomi_aozora2/logs
FETCH=
if [ ! -f data/yomi/sentences_aozora2.jsonl ]; then
  uv run python scripts/make_aozora_ruby_train.py --name aozora2 --max-works 3000 \
    --per-word 4 --exclude-sentences data/yomi/sentences_aozora.jsonl \
    > outputs/yomi_aozora2/logs/fetch.log 2>&1 &
  FETCH=$!
fi

[ -f data/yomi/targets_var.jsonl ] || uv run python scripts/select_variation_targets.py \
  --name prod --name aozora --output data/yomi/targets_var.jsonl --sentences 6
LEXICON=data/yomi_prod/lexicon.jsonl KEY_PREFIX=v bash scripts/run_yomi_data.sh var

if [ -n "$FETCH" ]; then
  wait "$FETCH"
fi
touch outputs/yomi_aozora2/sentences.done  # human ruby readings: no LLM step
KEY_PREFIX=a2 bash scripts/run_yomi_data.sh aozora2
echo "var data done"
