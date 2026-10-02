#!/usr/bin/env bash
# Stage 6: longer DiT runs with Muon (the D6 recipe comparison: Muon reaches AdamW's
# 2k-step errors in ~1k steps; a higher AdamW lr and the hidden-state loss did not help),
# and the first merged candidate. Shares the stage-5 GPU slots (at most two GPU jobs).
#   chain X: D9M continued from its pilot (30k steps) -> merge with T-B p1c ->
#            Phase 3 (DiT adapts to the frozen T-B conditions)
#   chain Y: D9 (9 layers, full MLP) from the teacher blocks (20k steps)
#   chain E: evals (T-A p1a from stage 5, D9M long, D9, merged)
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=third-party/Irodori-TTS:scripts
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=third-party/Irodori-TTS/.venv/bin/python
S=outputs/students
REFS="data/refs/real_pool.pt data/refs/invented_pool.pt"
LOCKS=outputs/.gpu_slots
mkdir -p $LOCKS

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
wait_run() { until finished "$1"; do sleep 30; done; }
suite() {
  [ -f "outputs/eval/$2/jkyb_dev/quality_summary.json" ] && return
  slot scripts/run_eval_suite.sh "$1/model.safetensors" "$2" > "outputs/eval/$2.log" 2>&1 || echo "FAILED eval $2"
  echo "evaluated $2"
}
train() {
  local out=$1 script=$2; shift 2
  finished "$out" && return
  rm -rf "$out"; mkdir -p "$out"
  slot $PY "scripts/$script" --output-dir "$out" "$@" > "$out/run.log" 2>&1 || echo "FAILED $out"
  echo "finished $out"
}
DIT_COMMON=(--refs $REFS --batch-size 8 --random-points 8 --micro-batches 4 --optimizer muon
            --eval-every 2000 --save-every 5000)

chain_x() {
  wait_run $S/dit_d6/recipe_muon
  train $S/dit_d9m/p2_long train_dit_distill.py --student-init $S/dit_d9m/p2_pilot "${DIT_COMMON[@]}" \
    --lr 2e-4 --steps 30000 --warmup 500 --onpolicy-prob 0.3 --onpolicy-start 10000
  finished $S/dit_d9m/p2_long || return
  [ -f $S/merged_b_d9m/init/model.safetensors ] || \
    $PY scripts/merge_students.py --text $S/text_mbert130m/p1c --dit $S/dit_d9m/p2_long \
      --output-dir $S/merged_b_d9m/init
  train $S/merged_b_d9m/p3 train_dit_distill.py --student-init $S/merged_b_d9m/init "${DIT_COMMON[@]}" \
    --student-conditions --lr 1e-4 --steps 8000 --warmup 300 --onpolicy-prob 0.3 --onpolicy-start 0
}

chain_y() {
  wait_run $S/dit_d6/recipe_muon
  train $S/dit_d9/p2 train_dit_distill.py --student-init $S/dit_d9/init "${DIT_COMMON[@]}" \
    --lr 2e-4 --steps 20000 --warmup 500 --onpolicy-prob 0.3 --onpolicy-start 10000
}

chain_e() {
  wait_run $S/text_prune310/p1a && suite $S/text_prune310/p1a text_prune310_p1a
  wait_run $S/dit_d9/p2 && suite $S/dit_d9/p2 dit_d9_p2
  wait_run $S/dit_d9m/p2_long && suite $S/dit_d9m/p2_long dit_d9m_long
  wait_run $S/merged_b_d9m/p3 && suite $S/merged_b_d9m/p3 merged_b_d9m_p3
}

chain_x & chain_y & chain_e &
wait
echo "stage6 done"
