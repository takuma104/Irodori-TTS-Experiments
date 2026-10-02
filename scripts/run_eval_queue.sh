#!/usr/bin/env bash
# Evaluate the teacher and each pilot student as its checkpoint appears (run_eval_suite.sh).
# Starts after the T-B text run so that at most one training job shares the GPU with an eval.
set -uo pipefail
cd "$(dirname "$0")/.."
S=outputs/students

# Wait for a run to finish: trainers print "saved <path>" after the final save (checkpoints
# saved mid-run overwrite the same model.safetensors).
wait_for() { until grep -q "^saved " "$(dirname "$1")/run.log" 2>/dev/null; do sleep 30; done; }
suite() {  # <checkpoint> <name>
  [ -f "outputs/eval/$2/jkyb_dev/quality_summary.json" ] && return
  scripts/run_eval_suite.sh "$1" "$2" > "outputs/eval/$2.log" 2>&1 || echo "FAILED eval $2"
  echo "evaluated $2"
}

mkdir -p outputs/eval
wait_for $S/text_mbert130m/p1a/model.safetensors
suite MF mf_teacher
suite $S/text_mbert130m/p1a/model.safetensors text_mbert130m_p1a
wait_for $S/dit_d6/p2_pilot/model.safetensors
suite $S/dit_d6/p2_pilot/model.safetensors dit_d6_pilot
wait_for $S/dit_d9m/p2_pilot/model.safetensors
suite $S/dit_d9m/p2_pilot/model.safetensors dit_d9m_pilot
wait_for $S/text_prune310/p1a/model.safetensors
suite $S/text_prune310/p1a/model.safetensors text_prune310_p1a
echo "eval queue done"
