#!/usr/bin/env python3
"""JSUT BASIC5000 rows with human kana readings (jsut-label) for sentence kana-CER checks.

    git clone --depth 1 https://github.com/sarulab-speech/jsut-label data/jsut-label
    uv run python scripts/make_jsut_rows.py --output data/eval/jsut_rows.jsonl

Writes ``{"key", "text", "kana"}`` rows for a hash-selected subset of BASIC5000
(``text_level0`` / ``kana_level0``), skipping the two utterances used as the
evaluation reference voice. Usable with ``generate_jkyb_audio.py --dataset`` and
``eval_general_cer.py --mode kana``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

REFERENCE_KEYS = {"BASIC5000_0001", "BASIC5000_0002"}


def parse_yaml(path: Path) -> dict[str, dict[str, str]]:
    """Minimal parser for jsut-label's two-level ``key: {field: value}`` YAML."""
    entries: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if re.match(r"^\S.*:$", line):
            current = entries.setdefault(line[:-1], {})
        elif current is not None and line.startswith("  ") and ":" in line:
            field, value = line.strip().split(":", 1)
            current[field] = value.strip()
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-dir", type=Path, default=Path("data/jsut-label"))
    parser.add_argument("--output", type=Path, default=Path("data/eval/jsut_rows.jsonl"))
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()

    entries = parse_yaml(args.label_dir / "text_kana" / "basic5000.yaml")
    keys = sorted(
        (k for k in entries if k not in REFERENCE_KEYS),
        key=lambda k: hashlib.sha1(k.encode("utf-8")).hexdigest(),
    )[: args.count]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for key in sorted(keys):
            entry = entries[key]
            row = {"key": key, "text": entry["text_level0"], "kana": entry["kana_level0"]}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {args.output}: {len(keys)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
