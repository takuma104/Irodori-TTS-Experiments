#!/usr/bin/env bash
# Pilot queue on one GPU (plan Phase 1a / Phase 2 pilots), run sequentially.
# A failed job is logged and the queue moves on.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=third-party/Irodori-TTS:scripts
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=third-party/Irodori-TTS/.venv/bin/python
REFS="data/refs/real_pool.pt data/refs/invented_pool.pt"

dit_pilot() {
  local name=$1
  local out=outputs/students/dit_${name}/p2_pilot
  [ -f "$out/model.safetensors" ] && return
  mkdir -p "$out"
  $PY scripts/train_dit_distill.py --student-init "outputs/students/dit_${name}/init" --output-dir "$out" \
    --refs $REFS --steps 10000 --batch-size 8 --random-points 8 \
    --onpolicy-prob 0.5 --onpolicy-start 2000 --lr 1e-4 > "$out/run.log" 2>&1 \
    || echo "FAILED dit $name" 
  echo "finished dit $name"
}

text_p1a() {
  local variant=$1 backbone_lr=$2 head_lr=$3
  local out=outputs/students/text_${variant}/p1a
  [ -f "$out/model.safetensors" ] && return
  mkdir -p "$out"
  $PY scripts/train_text_distill.py --student-init "outputs/students/text_${variant}/init" --output-dir "$out" \
    --steps 20000 --batch-size 128 --backbone-lr "$backbone_lr" --head-lr "$head_lr" > "$out/run.log" 2>&1 \
    || echo "FAILED text $variant"
  echo "finished text $variant"
}

dit_pilot d6
text_p1a mbert130m 2e-4 1e-3
dit_pilot d9m
text_p1a prune310 1e-4 1e-4
echo "queue done"
