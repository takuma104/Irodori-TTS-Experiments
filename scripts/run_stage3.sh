#!/usr/bin/env bash
# Stage 3 (after the D6 pilot failed: JKYB dev 21.6%, UTMOS 1.6):
#   eval chain:   D9M pilot -> T-B p1b, each once its run printed "saved"
#   recipe chain: short D6 runs comparing DiT training recipes (same 2k-step schedule),
#                 then Phase 1a for T-A once T-B p1b has finished.
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

eval_chain() {
  wait_run $S/dit_d9m/p2_pilot && suite $S/dit_d9m/p2_pilot dit_d9m_pilot
  wait_run $S/text_mbert130m/p1b && suite $S/text_mbert130m/p1b text_mbert130m_p1b
}

recipe_chain() {
  wait_run $S/dit_d9m/p2_pilot
  recipe base --lr 1e-4
  recipe hidden --lr 1e-4 --hidden-weight 1.0 --layer-map 0 7 8 9 10 11
  recipe lr3e-4 --lr 3e-4
  recipe muon --optimizer muon --lr 2e-4
  wait_run $S/text_mbert130m/p1b
  train $S/text_prune310/p1a train_text_distill.py --student-init $S/text_prune310/init \
    --steps 20000 --batch-size 128 --backbone-lr 1e-4 --head-lr 1e-4
}

eval_chain &
recipe_chain &
wait
echo "stage3 done"
