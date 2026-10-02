#!/usr/bin/env bash
# Stage 4 (replaces stage 3 after Phase 1b diverged: the T-B caption projector had only
# seen corpus sentences, so its error on real voice captions dominated the 1b loss).
#   chain A: D6 DiT recipe comparison (2k steps each), then Phase 1a for T-A
#   chain B: T-B Phase 1a continued with voice-caption batches, then Phase 1b again
#   chain E: evals as runs finish
# Batch sizes keep two trainers plus one eval within ~30 GiB.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=third-party/Irodori-TTS:scripts
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=third-party/Irodori-TTS/.venv/bin/python
S=outputs/students
REFS="data/refs/real_pool.pt data/refs/invented_pool.pt"

finished() { grep -q "^saved " "$1/run.log" 2>/dev/null; }
wait_run() { until finished "$1"; do sleep 30; done; }
suite() {  # <run dir> <name>
  [ -f "outputs/eval/$2/jkyb_dev/quality_summary.json" ] && return
  scripts/run_eval_suite.sh "$1/model.safetensors" "$2" > "outputs/eval/$2.log" 2>&1 || echo "FAILED eval $2"
  echo "evaluated $2"
}
train() {  # <out dir> <script> <args...>
  local out=$1 script=$2; shift 2
  finished "$out" && return
  rm -rf "$out"; mkdir -p "$out"
  $PY "scripts/$script" --output-dir "$out" "$@" > "$out/run.log" 2>&1 || echo "FAILED $out"
  echo "finished $out"
}
recipe() {  # <name> <extra args...>
  local name=$1; shift
  train $S/dit_d6/recipe_$name train_dit_distill.py --student-init $S/dit_d6/init --refs $REFS \
    --steps 2000 --warmup 200 --eval-every 500 --save-every 100000 --batch-size 8 --random-points 8 \
    --micro-batches 8 "$@"
}

chain_a() {
  wait_run $S/dit_d9m/p2_pilot
  recipe base --lr 1e-4
  recipe hidden --lr 1e-4 --hidden-weight 1.0 --layer-map 0 7 8 9 10 11
  recipe lr3e-4 --lr 3e-4
  recipe muon --optimizer muon --lr 2e-4
  train $S/text_prune310/p1a train_text_distill.py --student-init $S/text_prune310/init \
    --steps 20000 --batch-size 64 --caption-batch-size 32 --backbone-lr 1e-4 --head-lr 1e-4
}

chain_b() {
  wait_run $S/dit_d9m/p2_pilot
  train $S/text_mbert130m/p1c train_text_distill.py --student-init $S/text_mbert130m/p1a \
    --steps 6000 --batch-size 64 --caption-batch-size 32 --backbone-lr 1e-4 --head-lr 3e-4 --warmup 300
  train $S/text_mbert130m/p1b2 train_dit_distill.py --student-init $S/text_mbert130m/p1c \
    --no-train-dit --train-text --text-lr 2e-5 --steps 3000 --warmup 300 --batch-size 8 --random-points 8 \
    --micro-batches 8 --onpolicy-prob 0.5 --onpolicy-start 500 --eval-every 500 --refs $REFS
}

chain_e() {
  wait_run $S/dit_d9m/p2_pilot && suite $S/dit_d9m/p2_pilot dit_d9m_pilot
  wait_run $S/text_mbert130m/p1c && suite $S/text_mbert130m/p1c text_mbert130m_p1c
  wait_run $S/text_mbert130m/p1b2 && suite $S/text_mbert130m/p1b2 text_mbert130m_p1b2
  wait_run $S/text_prune310/p1a && suite $S/text_prune310/p1a text_prune310_p1a
}

chain_a & chain_b & chain_e &
wait
echo "stage4 done"
