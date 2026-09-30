#!/usr/bin/env python3
"""Select production targets from corpus retrieval hits (no JKYB data).

    uv run python scripts/select_corpus_targets.py data/yomi/corpus_hits.jsonl \
        --lexicon data/yomi_prod/lexicon.jsonl --name prod \
        --exclude data/yomi/rows_pilot.jsonl --exclude data/yomi/rows_m3.jsonl \
        --exclude data/yomi/rows_kun.jsonl

Reads ``retrieve_corpus_sentences.py`` output, keeps one record per (stem,
stem reading) (the shortest word), drops pairs already trained, and writes

- ``data/yomi/targets_<name>.jsonl``: lexicon rows with bucket/role/split, for
  ``prepare_yomi_rows.py``;
- ``data/yomi/sentences_<name>.jsonl``: the corpus sentences in the format of
  ``generate_yomi_sentences.py``. Homograph sentences get
  ``reading_verified: null`` and must go through ``verify_homograph_sentences.py``.

About ``--dev-ratio`` of the words (by hash, as in the other selections) form the
dev split. Words are ordered by bucket and corpus frequency and capped at
``--max-words``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

KANJI = re.compile(r"[㐀-鿿々]")
BUCKET_ORDER = ["F", "E", "A", "B", "D", "C"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def is_dev(word: str, dev_ratio: float) -> bool:
    return hashlib.md5(word.encode()).digest()[1] < 256 * dev_ratio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hits", type=Path)
    parser.add_argument("--lexicon", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--max-words", type=int, default=20_000)
    parser.add_argument("--min-sentences", type=int, default=2)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    args = parser.parse_args()

    trained = set()
    for path in args.exclude:
        for row in read_jsonl(path):
            if row.get("role") == "target" and row.get("split") == "train":
                trained.add((row["word"], row["reading"]))
    lexicon = {(r["word"], r["reading"]): r for r in read_jsonl(args.lexicon)}

    best: dict[tuple[str, str], dict[str, Any]] = {}
    for record in read_jsonl(args.hits):
        if len(record["sentences"]) < args.min_sentences:
            continue
        if (record["word"], record["reading"]) in trained:
            continue
        key = (record["stem"], record["stem_reading"])
        current = best.get(key)
        if current is None or len(record["word"]) < len(current["word"]):
            best[key] = record

    def order(record: dict[str, Any]) -> tuple[int, int]:
        bucket = min(
            BUCKET_ORDER.index(b) for b in record["buckets"] if b in BUCKET_ORDER
        )
        return bucket, -record["hits"]

    chosen = sorted(best.values(), key=order)[: args.max_words]
    targets_path = Path(f"data/yomi/targets_{args.name}.jsonl")
    sentences_path = Path(f"data/yomi/sentences_{args.name}.jsonl")
    n_sentences = 0
    with (
        targets_path.open("w", encoding="utf-8") as targets,
        sentences_path.open("w", encoding="utf-8") as sentences,
    ):
        for record in chosen:
            row = lexicon[(record["word"], record["reading"])]
            bucket = next(b for b in BUCKET_ORDER if b in record["buckets"])
            split = "dev" if is_dev(record["word"], args.dev_ratio) else "train"
            target = {
                **row,
                "bucket": bucket,
                "role": "target",
                "split": split,
                "corpus_hits": record["hits"],
            }
            targets.write(json.dumps(target, ensure_ascii=False) + "\n")
            for sentence in record["sentences"]:
                sentences.write(
                    json.dumps(
                        {
                            "word": record["word"],
                            "reading": record["reading"],
                            "bucket": bucket,
                            "role": "target",
                            "split": split,
                            "stem": record["stem"],
                            "stem_reading": record["stem_reading"],
                            "stem_start": sentence["stem_start"],
                            "text": sentence["text"],
                            "kana_text": sentence["kana_text"],
                            "reading_verified": None if record["homograph"] else True,
                            "source": "wikipedia",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_sentences += 1
    homographs = sum(r["homograph"] for r in chosen)
    print(
        f"candidates={len(best):,} chosen={len(chosen):,} (homographs {homographs:,}) "
        f"sentences={n_sentences:,} -> {targets_path}, {sentences_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
