#!/usr/bin/env bash
# Stage 10: T-A continued on Wikipedia + Aozora Bunko (modern orthography) sentences, since the
# full JKYB showed T-A at -1.39pt with the largest losses on single-kanji tokens and appendix
# readings; then the full JKYB eval. Uses the GPU slots.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=third-party/Irodori-TTS:scripts
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=third-party/Irodori-TTS/.venv/bin/python
LOCKS=outputs/.gpu_slots
S=outputs/students
slot() {
  while true; do
    for s in 1 2; do
      flock -n -E 75 "$LOCKS/slot$s" "$@"
      local code=$?
      [ $code -ne 75 ] && return $code
    done
    sleep 15
  done
}
finished() { grep -q "^saved " "$1/run.log" 2>/dev/null; }
out=$S/text_prune310/p1d
if ! finished $out; then
  rm -rf $out; mkdir -p $out
  slot $PY scripts/train_text_distill.py --student-init $S/text_prune310/p1a --output-dir $out \
    --train-corpus data/corpus/wiki_train.txt data/corpus/aozora_train.txt \
    --steps 20000 --batch-size 64 --caption-batch-size 32 --backbone-lr 5e-5 --head-lr 5e-5 --warmup 300 \
    > $out/run.log 2>&1 || echo "FAILED $out"
fi
finished $out || exit 1
full=outputs/jkyb_full/text_prune310_p1d
mkdir -p $full
slot bash -c "$PY scripts/generate_jkyb_audio.py --hf-checkpoint $out/model.safetensors --output-dir $full/audio \
    --ref-wav data/jsut/BASIC5000_0001.wav --seed 0 --batch-size 16 --max-batch-frames 4096 \
    --text-precision fp32 > $full/synth.log 2>&1 && \
  cd third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition && \
  uv run --no-sync jkyb-eval tts ../../$full/audio --device cuda --output-dir ../../$full/results > ../../$full/eval.log 2>&1" \
  || echo "FAILED full eval"
echo "stage10 done"
