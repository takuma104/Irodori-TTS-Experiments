#!/usr/bin/env python3
"""Select words that cover every Joyo kun reading (bucket K).

    uv run python scripts/select_kun_targets.py --output data/yomi/targets_kun.jsonl \
        --exclude data/yomi/targets_pilot.jsonl --exclude data/yomi/targets_m3.jsonl

The pilot/M3 selection favored split compounds, so most single-kanji kun words
(香 か, 氏 うじ, 代 よ, 著す あらわす) were never trained, and S7 fixed only 7 of
the 78 kun swaps on JKYB even though kun words present in training were learned
as well as on words. Here, for every Joyo (kanji, kun reading) pair, up to
``--per-reading`` lexicon words that use that reading are taken, preferring the
single-kanji word (with okurigana) and short, mid-frequency words. When the
surface has other readings (弦 つる / げん), those readings are added too and all
of them get ``--homograph-sentences`` sentences, since single-kanji kun words are
mostly homographs and the student needs both contexts. For each target kanji a
contrast word in its most frequent reading is added (bucket G).
Words are excluded only when they take a reading of a held-out group-B JKYB
target (``jkyb_excluded_reading``), e.g. 外 ほか stays although 外 そと is in B.
Single-kanji words get ``--homograph-sentences`` sentences even if an earlier
selection already trained them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

SINGLE_KANJI_WORD = re.compile(r"^[㐀-鿿々][ぁ-ゖ]*$")
SKIP_MISC = {"uk", "obs", "rare", "vulg", "sl", "derog", "X"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def is_dev(word: str, dev_ratio: float) -> bool:
    return hashlib.md5(word.encode()).digest()[1] < 256 * dev_ratio


def usable(row: dict[str, Any]) -> bool:
    return (
        not row["jkyb_excluded_reading"]
        and len(row["word"]) <= 6
        and len(row["reading"]) <= 12
        and not SKIP_MISC & set(row["misc"])
    )


def score(row: dict[str, Any]) -> float:
    return (
        4.0 * bool(SINGLE_KANJI_WORD.match(row["word"]))
        + {1: 1.0, 2: 1.5, 3: 0.5}[row["tier"]]
        + 1.0 * (len(row["word"]) <= 3)
        - 0.5 * ("arch" in row["misc"])
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lexicon", type=Path, default=Path("data/yomi/lexicon.jsonl"))
    parser.add_argument(
        "--stats", type=Path, default=Path("data/yomi/kanji_stats.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--per-reading", type=int, default=3)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument(
        "--homograph-sentences",
        type=int,
        default=8,
        help="Sentences per word for surfaces with several readings.",
    )
    args = parser.parse_args()

    stats: dict[str, dict[str, Any]] = json.loads(
        args.stats.read_text(encoding="utf-8")
    )
    usable_rows = [r for r in read_jsonl(args.lexicon) if usable(r)]
    readings_by_word: dict[str, set[str]] = defaultdict(set)
    for row in usable_rows:
        readings_by_word[row["word"]].add(row["reading"])
    # Single-kanji words (弦 つる, 詔 みことのり) stay eligible even if an earlier
    # selection trained them: 4 sentences were not enough to fix them on JKYB.
    excluded = {
        (r["word"], r["reading"])
        for p in args.exclude
        for r in read_jsonl(p)
        if not SINGLE_KANJI_WORD.match(r["word"])
    }
    rows = [r for r in usable_rows if (r["word"], r["reading"]) not in excluded]

    by_key: dict[str, list[tuple[dict[str, Any], int]]] = defaultdict(list)
    for row in rows:
        for kr in row["kanji_readings"]:
            if kr["kind"] == "kun" and kr["joyo_key"] is not None:
                by_key[kr["joyo_key"]].append((row, kr["index"]))

    selected: list[dict[str, Any]] = []
    taken: set[tuple[str, str]] = set()
    target_kanji: set[str] = set()
    for key in sorted(by_key):
        candidates = sorted(by_key[key], key=lambda item: -score(item[0]))
        count = 0
        for row, index in candidates:
            if count >= args.per_reading:
                break
            ident = (row["word"], row["reading"])
            if ident in taken:
                continue
            taken.add(ident)
            selected.append(
                {
                    **row,
                    "bucket": "K",
                    "role": "target",
                    "bucket_targets": {"K": [index]},
                }
            )
            target_kanji.add(row["word"][index])
            count += 1

    # Homograph partners: other readings of the same surface (弦 つる -> 弦 げん), so the
    # student sees the contexts of both readings. Homographs get more sentences.
    rows_by_word: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_word[row["word"]].append(row)
    partners: list[dict[str, Any]] = []
    for item in selected:
        others = [
            r
            for r in rows_by_word[item["word"]]
            if r["reading"] != item["reading"]
            and (r["word"], r["reading"]) not in taken
        ]
        single = bool(SINGLE_KANJI_WORD.match(item["word"]))
        if not others and len(rows_by_word[item["word"]]) < 2 and not single:
            continue
        item["n_sentences"] = args.homograph_sentences
        for other in others:
            taken.add((other["word"], other["reading"]))
            index = next(
                (
                    kr["index"]
                    for kr in other["kanji_readings"]
                    if kr["kind"] in ("on", "kun")
                ),
                0,
            )
            partners.append(
                {
                    **other,
                    "bucket": "K",
                    "role": "target",
                    "bucket_targets": {"K": [index]},
                    "homograph_partner": True,
                    "n_sentences": args.homograph_sentences,
                }
            )
    selected += partners

    contrast_pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["tier"] > 2 or (row["word"], row["reading"]) in taken:
            continue
        for kr in row["kanji_readings"]:
            default = stats.get(kr["kanji"], {}).get("default_key")
            if kr["joyo_key"] is not None and kr["joyo_key"] == default:
                contrast_pool[kr["kanji"]].append(row)
    for kanji in sorted(target_kanji):
        pool = [
            r
            for r in contrast_pool.get(kanji, [])
            if (r["word"], r["reading"]) not in taken
        ]
        if pool:
            row = min(pool, key=lambda r: (r["tier"], len(r["word"])))
            taken.add((row["word"], row["reading"]))
            selected.append(
                {**row, "bucket": "G", "role": "contrast", "contrast_kanji": kanji}
            )

    for item in selected:
        item["split"] = "dev" if is_dev(item["word"], args.dev_ratio) else "train"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    targets = [s for s in selected if s["role"] == "target"]
    single = sum(bool(SINGLE_KANJI_WORD.match(s["word"])) for s in targets)
    homographs = sum("n_sentences" in s for s in targets)
    print(
        f"kun readings covered={len({k for k in by_key if by_key[k]})} targets={len(targets)} "
        f"(single-kanji words {single}, homograph readings {homographs}, "
        f"partners {len(partners)}) contrast={len(selected) - len(targets)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
