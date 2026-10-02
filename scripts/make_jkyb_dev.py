#!/usr/bin/env python3
"""Split JKYB-Parakeet into a model-selection dev subset and its held-out complement.

    uv run python scripts/make_jkyb_dev.py --output-dir data/eval

Rows are ranked within each reading category by a SHA-1 hash of their key, and
the first N per category form the dev subset (default: 1,000 on'yomi, 850
kun'yomi, 150 appendix readings; the appendix is oversampled because it is
small and the weakest category). Writes ``jkyb_dev.jsonl`` and
``jkyb_heldout.jsonl`` in the dataset's own format, usable with
``generate_jkyb_audio.py --dataset`` and ``jkyb-eval tts --dataset``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

DATASET_REPO = "Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet"
DATASET_FILENAME = "data/common_kanji_source.jsonl"


def key_hash(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/eval"))
    parser.add_argument("--on-yomi", type=int, default=1000)
    parser.add_argument("--kun-yomi", type=int, default=850)
    parser.add_argument("--appendix", type=int, default=150)
    args = parser.parse_args()

    path = Path(hf_hub_download(DATASET_REPO, DATASET_FILENAME, repo_type="dataset"))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    quota = {
        "on_yomi": args.on_yomi,
        "kun_yomi": args.kun_yomi,
        "joyo_appendix_reading": args.appendix,
    }
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["reading_category"]].append(row)
    dev_keys: set[str] = set()
    for category, members in by_category.items():
        ranked = sorted(members, key=lambda row: key_hash(row["key"]))
        dev_keys.update(row["key"] for row in ranked[: quota[category]])

    dev = [row for row in rows if row["key"] in dev_keys]
    heldout = [row for row in rows if row["key"] not in dev_keys]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "jkyb_dev.jsonl", dev)
    write_jsonl(args.output_dir / "jkyb_heldout.jsonl", heldout)
    counts = {category: sum(row["reading_category"] == category for row in dev) for category in quota}
    print(f"dev={len(dev)} {counts} heldout={len(heldout)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
