#!/usr/bin/env bash
# Build a release checkpoint from a student and evaluate the exported file.
#
#   bash scripts/run_release.sh <student-dir> <name> <output-dir>
#   bash scripts/run_release.sh outputs/yomi_prod/s12_cont/student release_s12 ../Irodori-TTS-v4.1-Small-Yomi
#
# 1. Refit the caption projector to the student's backbone (state matching on
#    data/yomi/captions.jsonl and Wikipedia sentences).
# 2. Merge the student and the refit caption projector into the base checkpoint.
# 3. Check the text path against the student and the caption/emoji drift.
# 4. Caption-only samples of base / base+student / exported, with median pitch.
# 5. Evaluate the exported file itself (run_export_eval.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ALLOC_CONF=expandable_segments:True

STUDENT=$1
NAME=$2
OUT=$3
R=outputs/release/$NAME
mkdir -p "$R"

if [ ! -f "$R/caption_refit/caption.safetensors" ]; then
  PYTHONPATH=Irodori-TTS:scripts uv run --project Irodori-TTS --no-sync python \
    scripts/refit_caption_projector.py "$STUDENT" --output-dir "$R/caption_refit" \
    > "$R/refit.log" 2>&1
fi
PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync python \
  scripts/export_student_checkpoint.py "$STUDENT" \
  --caption "$R/caption_refit/caption.safetensors" --output-dir "$OUT" > "$R/export.json" 2>&1
PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync python \
  scripts/check_exported_checkpoint.py "$OUT/model.safetensors" --student "$STUDENT" \
  > "$R/check.log" 2>&1
PYTHONPATH=Irodori-TTS:scripts uv run --project Irodori-TTS --no-sync python \
  scripts/caption_drift_listen.py "$OUT/model.safetensors" --student "$STUDENT" \
  --output-dir "$R/caption" > "$R/caption.log" 2>&1
bash scripts/run_export_eval.sh "$OUT/model.safetensors" "$NAME" > "$R/eval.log" 2>&1
echo "release $NAME done"
