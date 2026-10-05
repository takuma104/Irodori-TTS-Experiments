#!/usr/bin/env python3
"""Contrast sentences for kanji trained as single-kanji words (after S14).

    uv run python scripts/select_contrast_targets.py --name aozora --name aozora2 \
        --name var --output-name contrast

S14 learned single-kanji readings from Aozora ruby (秋《とき》, 活《い》きる, 名《な》),
and they leaked into compounds (秋季 シュウ -> アキ, 活火山 カッ -> イ). For every
kanji that is a single-kanji target in the given datasets, this picks Wikipedia
sentences where the same kanji sits inside a compound with a different reading
(``--hits``, from ``retrieve_corpus_sentences.py``: the stem's reading agrees
between UniDic, Sudachi and JMdict). ``--write-candidates`` first lists common
lexicon compounds (tiers 1-2) to retrieve, since the bucket words alone miss
everyday compounds such as 秋季 or 名人. The kanji is tagged (bucket G, role ``contrast``); rows the
base model reads correctly keep its own output as the teacher, so they hold the
compound reading in place.

Up to ``--compounds`` compounds per kanji (most corpus hits first) and
``--per-compound`` sentences each. Homograph compounds are skipped. About 10% of
the compounds (by hash) go to the dev split, which is also the leak check
``outputs/yomi_eval/contrast_dev_rows.jsonl`` after rows are built.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def to_katakana(text: str) -> str:
    return "".join(chr(ord(ch) + 0x60) if "ぁ" <= ch <= "ゖ" else ch for ch in text)


def plain(reading: str) -> str:
    """Katakana reading without voicing marks and with a final ッ as ツ (カッ = カツ)."""
    text = unicodedata.normalize("NFD", to_katakana(reading))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = unicodedata.normalize("NFC", text)
    return text[:-1] + "ツ" if text.endswith("ッ") else text


def tagged(row: dict[str, Any], field: str) -> str:
    text = row[field]
    return text[text.index("<") + 1 : text.index(">")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--output-name", default="contrast")
    parser.add_argument(
        "--lexicon", type=Path, default=Path("data/yomi_prod/lexicon.jsonl")
    )
    parser.add_argument(
        "--hits", type=Path, default=Path("data/yomi/corpus_hits.jsonl")
    )
    parser.add_argument("--compounds", type=int, default=3)
    parser.add_argument("--per-compound", type=int, default=2)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument(
        "--write-candidates",
        type=Path,
        default=None,
        help="Only write the candidate compounds (common lexicon words, tiers 1-2) "
        "for retrieve_corpus_sentences.py --words, then exit.",
    )
    args = parser.parse_args()

    # Kanji trained as single-kanji words, and the readings they were trained with.
    single: dict[str, set[str]] = defaultdict(set)
    for name in args.name:
        for row in read_jsonl(Path(f"data/yomi/rows_{name}.jsonl")):
            if row["role"] != "target" or row.get("split", "train") != "train":
                continue
            word = tagged(row, "tagged_text")
            if len(word) == 1:
                single[word].add(plain(tagged(row, "tagged_yomi")))

    lexicon = {(r["word"], r["reading"]): r for r in read_jsonl(args.lexicon)}

    def contrasting(entry: dict[str, Any], stem: str) -> list[tuple[str, int]]:
        """(kanji, index) of single-kanji targets read differently in this compound.

        The compound counts when the kanji's reading here differs from at least
        one reading it was trained with as a single-kanji word (秋季 シュウ vs 秋 アキ).
        """
        found = []
        for kr in entry["kanji_readings"]:
            kanji = kr["kanji"]
            if (
                len(kanji) == 1
                and kanji in single
                and kr["index"] < len(stem)
                and stem.count(kanji) == 1
                and single[kanji] - {plain(kr["reading"])}
            ):
                found.append((kanji, kr["index"]))
        return found

    if args.write_candidates is not None:
        n = 0
        with args.write_candidates.open("w", encoding="utf-8") as out:
            for entry in lexicon.values():
                if (
                    entry.get("tier") not in (1, 2)
                    or entry.get("homograph_readings")
                    or len(entry["word"]) < 2
                    or not contrasting(entry, entry["word"])
                ):
                    continue
                out.write(
                    json.dumps(
                        {"word": entry["word"], "reading": entry["reading"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n += 1
        print(f"candidate compounds: {n} -> {args.write_candidates}")
        return 0
    candidates: dict[str, list[tuple[int, dict[str, Any], dict[str, Any], int]]] = (
        defaultdict(list)
    )
    for hit in read_jsonl(args.hits):
        if hit.get("homograph") or len(hit["stem"]) < 2:
            continue
        entry = lexicon.get((hit["word"], hit["reading"]))
        if entry is None:
            continue
        for kanji, index in contrasting(entry, hit["stem"]):
            candidates[kanji].append((int(hit["hits"]), hit, entry, index))

    targets_path = Path(f"data/yomi/targets_{args.output_name}.jsonl")
    sentences_path = Path(f"data/yomi/sentences_{args.output_name}.jsonl")
    counts = {"kanji": 0, "compounds": 0, "sentences": 0, "dev_compounds": 0}
    seen: set[tuple[str, str]] = set()
    with (
        targets_path.open("w", encoding="utf-8") as targets,
        sentences_path.open("w", encoding="utf-8") as sentences,
    ):
        for kanji in sorted(candidates):
            picked = 0
            for _, hit, entry, _ in sorted(candidates[kanji], key=lambda c: -c[0]):
                if picked >= args.compounds:
                    break
                ident = (hit["word"], hit["reading"])
                if ident in seen:
                    continue
                seen.add(ident)
                digest = hashlib.md5(f"contrast:{hit['word']}".encode()).digest()[0]
                split = "dev" if digest < 256 * args.dev_ratio else "train"
                target = dict(entry)
                target |= {
                    "bucket": "G",
                    "role": "contrast",
                    "split": split,
                    "contrast_kanji": kanji,
                    "contrast_for": sorted(single[kanji]),
                }
                targets.write(json.dumps(target, ensure_ascii=False) + "\n")
                for item in hit["sentences"][: args.per_compound]:
                    sentences.write(
                        json.dumps(
                            {
                                "word": hit["word"],
                                "reading": hit["reading"],
                                "bucket": "G",
                                "role": "contrast",
                                "split": split,
                                "stem": hit["stem"],
                                "stem_reading": hit["stem_reading"],
                                "stem_start": item["stem_start"],
                                "text": item["text"],
                                "kana_text": item["kana_text"],
                                "reading_verified": True,
                                "source": "wikipedia",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    counts["sentences"] += 1
                picked += 1
                counts["compounds"] += 1
                counts["dev_compounds"] += split == "dev"
            counts["kanji"] += picked > 0
    print(
        f"single-kanji targets: {len(single)} kanji; {counts} -> {targets_path}, "
        f"{sentences_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
