#!/usr/bin/env python3
"""Use the kana teacher only where the base model misreads (plan §4.3, §5.1).

    uv run python scripts/mix_teacher_by_difficulty.py \
        --teacher outputs/yomi_pilot/teacher --base outputs/yomi_pilot/base_kanji \
        --output outputs/yomi_pilot/teacher_hard

Target rows that the unchanged model already reads correctly from the kanji text
(per ``jkyb-eval`` on ``--base``) take the base latent and the kanji text as
their teacher, so they only ask the student to keep its output. The kana
teacher, whose substitution also shifts the prosody of the whole sentence, is
kept for rows the base misreads. Both directories were generated with the same
per-key speaker and seed. Contrast rows are copied as they are.

Writes ``manifest.jsonl`` with explicit ``latent_path`` fields.
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
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_correct = {
        r["key"]: bool(r["target_exact"])
        for r in read_jsonl(args.base / "results" / "details" / "all.jsonl")
    }
    base_rows = {r["key"]: r for r in read_jsonl(args.base / "manifest.jsonl")}
    counts: Counter[str] = Counter()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in read_jsonl(args.teacher / "manifest.jsonl"):
            key = row["key"]
            out = dict(row)
            if row.get("role") == "target" and base_correct.get(key):
                base = base_rows[key]
                out["text"] = base["text"]
                out["role"] = "keep_target"
                out["latent_path"] = str(args.base / "latents" / f"{key}.pt")
                counts["easy (kanji teacher)"] += 1
            else:
                out["latent_path"] = str(args.teacher / "latents" / f"{key}.pt")
                counts[
                    "hard (kana teacher)" if row.get("role") == "target" else "contrast"
                ] += 1
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
