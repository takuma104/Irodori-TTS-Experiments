#!/usr/bin/env python3
"""Use the kana teacher only where the base model misreads (plan §4.3, §5.1).

    uv run python scripts/mix_teacher_by_difficulty.py \
        --base outputs/yomi_pilot/base_kanji --teacher outputs/yomi_pilot/teacher \
        --output outputs/yomi_pilot/teacher_hard

``--base`` holds the unchanged model's latents for the kanji text, scored by
``jkyb-eval`` (``results/``). ``--teacher`` holds kana-substituted latents for
(at least) the target rows the base misreads. Both were generated with the same
per-key speaker and seed.

- Target rows the base misreads and the teacher has: kana teacher (role
  ``target``).
- Target rows the base reads correctly: base latent and kanji text as the
  teacher (role ``keep_target``), so they only ask the student to keep its
  output. The kana substitution also shifts the prosody of the whole sentence,
  so it is used only where the reading must change.
- Other rows (contrast): whichever directory has them, kanji teacher.

Rows whose teacher audio was misread (``results/`` of either directory) are
dropped. Writes ``manifest.jsonl`` with explicit ``latent_path`` fields.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_correct = {
        r["key"]: bool(r["target_exact"])
        for r in read_jsonl(args.base / "results" / "details" / "all.jsonl")
    }
    teacher_results = args.teacher / "results" / "details" / "all.jsonl"
    teacher_correct = (
        {r["key"]: bool(r["target_exact"]) for r in read_jsonl(teacher_results)}
        if teacher_results.exists()
        else {}
    )
    base_rows = {r["key"]: r for r in read_jsonl(args.base / "manifest.jsonl")}
    teacher_rows = {r["key"]: r for r in read_jsonl(args.teacher / "manifest.jsonl")}
    counts: Counter[str] = Counter()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for key in sorted(base_rows.keys() | teacher_rows.keys()):
            row = teacher_rows.get(key) or base_rows[key]
            is_target = row.get("role") == "target"
            if is_target and not base_correct.get(key, True) and key in teacher_rows:
                if not teacher_correct.get(key, True):
                    counts["dropped (kana teacher misread)"] += 1
                    continue
                out = dict(teacher_rows[key])
                out["latent_path"] = str(args.teacher / "latents" / f"{key}.pt")
                counts["hard (kana teacher)"] += 1
            elif key in base_rows:
                if not is_target and not base_correct.get(key, True):
                    counts["dropped (contrast misread)"] += 1
                    continue
                out = dict(base_rows[key])
                out["text"] = out["kanji_text"]
                if is_target:
                    out["role"] = "keep_target"
                out["latent_path"] = str(args.base / "latents" / f"{key}.pt")
                counts["easy (kanji teacher)" if is_target else "contrast"] += 1
            else:
                # Contrast rows generated only in the teacher directory (kanji text).
                if is_target:
                    counts["skipped (no base latent)"] += 1
                    continue
                if not teacher_correct.get(key, True):
                    counts["dropped (contrast misread)"] += 1
                    continue
                out = dict(row)
                out["latent_path"] = str(args.teacher / "latents" / f"{key}.pt")
                counts["contrast"] += 1
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
