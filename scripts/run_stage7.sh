#!/usr/bin/env bash
# Stage 7: candidates around D9 (9 layers, full MLP: JKYB dev -4.25pt after 20k Muon steps, still
# improving) and T-A (310m pruned to 12 layers: -0.35pt, passes G1). Shares the GPU slots.
#   1. T-A + D9 (20k) merged as is, evaluated (how the two losses combine)
#   2. D9 continued for 40k steps -> merged with T-A -> Phase 3 (DiT on T-A conditions) -> eval
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
merge() {  # <text run> <dit run> <out dir>
  [ -f "$3/model.safetensors" ] || $PY scripts/merge_students.py --text "$1" --dit "$2" --output-dir "$3"
}
DIT_COMMON=(--refs $REFS --batch-size 8 --random-points 8 --micro-batches 4 --optimizer muon
            --eval-every 5000 --save-every 5000)

chain_quick() {
  merge $S/text_prune310/p1a $S/dit_d9/p2 $S/merged_a_d9/raw
  suite $S/merged_a_d9/raw merged_a_d9_raw
}

chain_long() {
  train $S/dit_d9/p2_cont train_dit_distill.py --student-init $S/dit_d9/p2 "${DIT_COMMON[@]}" \
    --lr 1e-4 --steps 40000 --warmup 500 --onpolicy-prob 0.3 --onpolicy-start 0
  finished $S/dit_d9/p2_cont || return
  suite $S/dit_d9/p2_cont dit_d9_cont &
  merge $S/text_prune310/p1a $S/dit_d9/p2_cont $S/merged_a_d9/init
  train $S/merged_a_d9/p3 train_dit_distill.py --student-init $S/merged_a_d9/init "${DIT_COMMON[@]}" \
    --student-conditions --lr 5e-5 --steps 8000 --warmup 300 --onpolicy-prob 0.3 --onpolicy-start 0
  finished $S/merged_a_d9/p3 && suite $S/merged_a_d9/p3 merged_a_d9_p3
  wait
}

chain_quick & chain_long &
wait
echo "stage7 done"
