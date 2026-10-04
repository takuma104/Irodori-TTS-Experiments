#!/usr/bin/env python3
"""Training sentences from Aozora Bunko ruby (train-split works, bucket R).

    uv run python scripts/make_aozora_ruby_train.py --name aozora --max-works 1500

Uses the works that ``make_aozora_ruby_eval.py`` does not (the hash split on the
work ID), so the Aozora evaluation sentences never appear in training. Every
ruby word whose reading matches a dictionary or analyzer reading becomes a
target (at most ``--per-word`` sentences each), with the whole word tagged.
Writes ``data/yomi/targets_<name>.jsonl`` and ``data/yomi/sentences_<name>.jsonl``
in the formats of the other selections, ready for ``run_yomi_data.sh <name>``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import fugashi
from make_aozora_ruby_eval import (
    HIRAGANA_ONLY,
    KANJI,
    SENTENCE_END,
    analyzer_readings,
    category,
    eval_work,
    load_works,
    parse_sentence,
    to_katakana,
    work_text,
)
from sudachipy import Dictionary


def is_dev(word: str, dev_ratio: float) -> bool:
    return hashlib.md5(word.encode()).digest()[1] < 256 * dev_ratio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="aozora")
    parser.add_argument(
        "--lexicon", type=Path, default=Path("data/yomi_prod/lexicon.jsonl")
    )
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--max-works", type=int, default=1500)
    parser.add_argument("--per-word", type=int, default=4)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--exclude-sentences",
        type=Path,
        action="append",
        default=[],
        help="Sentences files of an earlier run; their sentences are skipped, so "
        "a larger run adds only new sentences (up to --per-word more per word).",
    )
    args = parser.parse_args()
    excluded: set[tuple[str, str]] = set()
    for path in args.exclude_sentences:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                excluded.add((row["word"], row["text"]))

    lexicon: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with args.lexicon.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            lexicon[row["word"]].append(row)
    tagger = fugashi.Tagger()
    tokenizer = Dictionary().tokenizer()
    rng = random.Random(args.seed)
    works = [w for w in load_works() if not eval_work(w["作品ID"], args.eval_ratio)]
    rng.shuffle(works)
    print(f"train works available: {len(works)}", flush=True)

    sentences: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    accepted: dict[tuple[str, str], bool] = {}  # dictionary/analyzer check per word
    for index, work in enumerate(works[: args.max_works], start=1):
        try:
            body = work_text(work["テキストファイルURL"])
        except Exception as error:  # noqa: BLE001 - skip broken archives
            print(f"skip {work['作品ID']}: {error!r}", flush=True)
            continue
        time.sleep(0.3)
        for line in body.splitlines():
            for raw in SENTENCE_END.split(line.strip().lstrip("　")):
                if "《" not in raw:
                    continue
                parsed = parse_sentence(raw.strip())
                if parsed is None:
                    continue
                text, rubies = parsed
                if not 8 <= len(text) <= 80:
                    continue
                for start, end, reading in rubies:
                    word = text[start:end]
                    if not (1 <= len(word) <= 4 and all(KANJI.match(c) for c in word)):
                        continue
                    if not HIRAGANA_ONLY.match(reading) or text.count(word) != 1:
                        continue
                    key = (word, reading)
                    if key not in accepted:
                        known = {
                            to_katakana(r["reading"]) for r in lexicon.get(word, [])
                        }
                        known |= analyzer_readings(tagger, tokenizer, word)
                        accepted[key] = to_katakana(reading) in known
                    if not accepted[key] or len(sentences[key]) >= args.per_word:
                        continue
                    if (word, text) in excluded:
                        continue
                    sentences[key].append(
                        {
                            "text": text,
                            "stem_start": start,
                            "kana_text": text[:start] + reading + text[end:],
                            "work_id": work["作品ID"],
                        }
                    )
        if index % 100 == 0:
            print(f"{index} works, {len(sentences):,} words", flush=True)

    targets_path = Path(f"data/yomi/targets_{args.name}.jsonl")
    sentences_path = Path(f"data/yomi/sentences_{args.name}.jsonl")
    n = 0
    with (
        targets_path.open("w", encoding="utf-8") as targets,
        sentences_path.open("w", encoding="utf-8") as out,
    ):
        for (word, reading), items in sorted(sentences.items()):
            split = "dev" if is_dev(word, args.dev_ratio) else "train"
            base = next(
                (r for r in lexicon.get(word, []) if r["reading"] == reading), None
            )
            target = {
                "word": word,
                "reading": reading,
                "kanji_readings": [
                    {
                        "index": 0,
                        "kanji": word,
                        "reading": to_katakana(reading),
                        "kind": "jukujikun",
                        "joyo_key": None,
                        "exact": True,
                    }
                ],
                "gloss": base["gloss"] if base else [],
                "tier": base["tier"] if base else 3,
                "misc": base["misc"] if base else [],
                "bucket": "R",
                "bucket_targets": {"R": [0]},
                "role": "target",
                "split": split,
                "reading_category": category(word, reading, lexicon),
            }
            targets.write(json.dumps(target, ensure_ascii=False) + "\n")
            for item in items:
                out.write(
                    json.dumps(
                        {
                            "word": word,
                            "reading": reading,
                            "bucket": "R",
                            "role": "target",
                            "split": split,
                            "stem": word,
                            "stem_reading": to_katakana(reading),
                            "stem_start": item["stem_start"],
                            "text": item["text"],
                            "kana_text": item["kana_text"],
                            "reading_verified": True,
                            "source": f"aozora:{item['work_id']}",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n += 1
    print(
        f"words={len(sentences):,} sentences={n:,} -> {targets_path}, {sentences_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
