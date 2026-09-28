#!/usr/bin/env python3
"""Select target and contrast words from the reading lexicon (plan §4.2-4.3).

    uv run python scripts/select_yomi_targets.py --output data/yomi/targets_pilot.jsonl

Picks words per bucket (A-F) from ``build_yomi_lexicon.py`` output, preferring
target kanji that the tokenizer splits into one-character tokens, mid-frequency
words, and Joyo-only spellings, with a per-kanji cap for diversity. For every
target kanji it also adds a contrast word (bucket G) where the kanji takes its
most frequent reading. About 10% of the target words (by hash) go to the dev
split, which is never used for training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

KANJI = re.compile(r"[㐀-鿿々]")
BUCKET_ORDER = ["F", "E", "A", "B", "D", "C"]
PILOT_QUOTA = {"A": 500, "B": 400, "C": 400, "D": 250, "E": 250, "F": 200}
SKIP_MISC = {"uk", "obs", "rare", "vulg", "sl", "derog", "X"}


JOYO_KANJI: set[str] = set()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def is_dev(word: str, dev_ratio: float) -> bool:
    return hashlib.md5(word.encode()).digest()[1] < 256 * dev_ratio


def usable(row: dict[str, Any]) -> bool:
    return (
        not row["jkyb_excluded"]
        and len(row["word"]) <= 6
        and len(row["reading"]) <= 12
        and not SKIP_MISC & set(row["misc"])
    )


def joyo_only(row: dict[str, Any]) -> bool:
    """All kanji are Joyo kanji (jukujikun groups are checked per character)."""
    return all(ch in JOYO_KANJI for ch in row["word"] if KANJI.match(ch))


def score(row: dict[str, Any], bucket: str) -> float:
    targets = row["bucket_targets"][bucket]
    split = any(i in row["single_char_kanji"] for i in targets)
    tier_score = {1: 2.0 if bucket == "E" else 0.5, 2: 2.0, 3: 1.0}[row["tier"]]
    return (
        2.0 * split
        + tier_score
        + 1.0 * joyo_only(row)
        + 1.0 * (2 <= len(row["word"]) <= 3)
        - 0.5 * ("arch" in row["misc"])
        + 3.0 * any(kr["kind"] == "appendix" for kr in row["kanji_readings"])
    )


def target_kanji_of(row: dict[str, Any], bucket: str) -> list[str]:
    by_index = {kr["index"]: kr["kanji"] for kr in row["kanji_readings"]}
    return [by_index[i] for i in row["bucket_targets"][bucket] if i in by_index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lexicon", type=Path, default=Path("data/yomi/lexicon.jsonl"))
    parser.add_argument(
        "--stats", type=Path, default=Path("data/yomi/kanji_stats.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Multiplies the quotas."
    )
    parser.add_argument("--per-kanji-cap", type=int, default=3)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    all_rows = read_jsonl(args.lexicon)
    stats: dict[str, dict[str, Any]] = json.loads(
        args.stats.read_text(encoding="utf-8")
    )
    # Kanji with at least one Joyo reading in the lexicon form the Joyo set.
    JOYO_KANJI.update(
        kr["kanji"]
        for r in all_rows
        for kr in r["kanji_readings"]
        if kr["kind"] in ("on", "kun")
    )
    rows = [r for r in all_rows if usable(r)]
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    selected: list[dict[str, Any]] = []
    taken: set[tuple[str, str]] = set()
    per_kanji: Counter[tuple[str, str]] = Counter()
    for bucket in BUCKET_ORDER:
        quota = round(PILOT_QUOTA[bucket] * args.scale)
        pool = sorted(
            (
                r
                for r in rows
                if bucket in r["buckets"] and (bucket != "F" or joyo_only(r))
            ),
            key=lambda r: -score(r, bucket),
        )
        count = 0
        for row in pool:
            if count >= quota:
                break
            ident = (row["word"], row["reading"])
            kanji = target_kanji_of(row, bucket)
            if ident in taken or any(
                per_kanji[(bucket, k)] >= args.per_kanji_cap for k in kanji
            ):
                continue
            group = [row]
            if bucket == "E":
                # Take every usable reading of the homograph together.
                group = [
                    r
                    for r in rows
                    if r["word"] == row["word"]
                    and (r["word"], r["reading"]) not in taken
                ]
                if len(group) < 2:
                    continue
            for member in group:
                taken.add((member["word"], member["reading"]))
                selected.append({**member, "bucket": bucket, "role": "target"})
            per_kanji.update((bucket, k) for k in kanji)
            count += 1

    # Contrast words (G): the target kanji in its most frequent reading.
    contrast_pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["tier"] > 2 or (row["word"], row["reading"]) in taken:
            continue
        for kr in row["kanji_readings"]:
            default = stats.get(kr["kanji"], {}).get("default_key")
            if kr["joyo_key"] is not None and kr["joyo_key"] == default:
                contrast_pool[kr["kanji"]].append(row)
    target_kanji = sorted(
        {k for s in selected for k in target_kanji_of(s, s["bucket"]) if len(k) == 1}
    )
    contrasts = 0
    for kanji in target_kanji:
        pool = [
            r
            for r in contrast_pool.get(kanji, [])
            if (r["word"], r["reading"]) not in taken
        ]
        if not pool:
            continue
        row = min(pool, key=lambda r: (r["tier"], len(r["word"])))
        taken.add((row["word"], row["reading"]))
        selected.append(
            {**row, "bucket": "G", "role": "contrast", "contrast_kanji": kanji}
        )
        contrasts += 1

    for item in selected:
        item["split"] = "dev" if is_dev(item["word"], args.dev_ratio) else "train"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    counts = Counter((s["bucket"], s["split"]) for s in selected)
    for bucket in BUCKET_ORDER + ["G"]:
        print(
            f"{bucket}: train={counts[(bucket, 'train')]:,} dev={counts[(bucket, 'dev')]:,}"
        )
    print(f"total={len(selected):,} contrast_kanji={contrasts:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
