#!/usr/bin/env bash
# Score a directory of synthesized JKYB wavs with jkyb-eval (kana-whisper + whisper-large-v3-turbo).
#   scripts/run_jkyb_eval.sh outputs/jkyb/<run> [--dataset data/eval/jkyb_dev.jsonl]
# Waits for <run>/synth.log to report "done" if synthesis is still running.
set -euo pipefail
cd "$(dirname "$0")/.."
run=$(realpath "$1"); shift
until grep -qE "^done|Traceback" "$run/synth.log" 2>/dev/null; do sleep 10; done
grep -q "^done" "$run/synth.log" || { echo "synthesis failed: $run" >&2; exit 1; }
cd third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition
uv run --no-sync jkyb-eval tts "$run/audio" --device cuda --output-dir "$run/results" "$@" \
  > "$run/eval.log" 2>&1
echo "eval done: $run"
