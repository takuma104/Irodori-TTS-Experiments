#!/usr/bin/env bash
# Pilot stage 2 (replaces the tails of run_pilots2.sh / run_eval_queue.sh):
#   training chain: Phase 1b for T-B (text path through the frozen DiT) -> Phase 1a for T-A
#   eval chain:     D9M pilot -> T-B p1b -> T-A p1a, each once its run has printed "saved"
# Ordering keeps at most ~24 GiB of jobs on the GPU (one trainer + one eval, or two trainers).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=third-party/Irodori-TTS:scripts
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=third-party/Irodori-TTS/.venv/bin/python
S=outputs/students
REFS="data/refs/real_pool.pt data/refs/invented_pool.pt"

finished() { grep -q "^saved " "$1/run.log" 2>/dev/null; }
wait_run() { until finished "$1"; do sleep 30; done; }
wait_pid() { while kill -0 "$1" 2>/dev/null; do sleep 30; done; }
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

training_chain() {
  wait_pid "$D6_EVAL_PID"
  train $S/text_mbert130m/p1b train_dit_distill.py --student-init $S/text_mbert130m/p1a \
    --no-train-dit --train-text --text-lr 5e-5 --steps 5000 --warmup 300 --batch-size 8 --random-points 8 \
    --micro-batches 4 --onpolicy-prob 0.5 --onpolicy-start 500 --refs $REFS
  wait_run $S/dit_d9m/p2_pilot
  train $S/text_prune310/p1a train_text_distill.py --student-init $S/text_prune310/init \
    --steps 20000 --batch-size 128 --backbone-lr 1e-4 --head-lr 1e-4
}

eval_chain() {
  wait_pid "$D6_EVAL_PID"
  wait_run $S/dit_d9m/p2_pilot && suite $S/dit_d9m/p2_pilot dit_d9m_pilot
  wait_run $S/text_mbert130m/p1b && suite $S/text_mbert130m/p1b text_mbert130m_p1b
  wait_run $S/text_prune310/p1a && suite $S/text_prune310/p1a text_prune310_p1a
}

D6_EVAL_PID=${1:?pid of the running D6 eval suite (or 0)}
training_chain &
eval_chain &
wait
echo "stage2 done"
