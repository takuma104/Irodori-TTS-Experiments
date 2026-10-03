#!/usr/bin/env python3
"""Sentence corpus from Aozora Bunko (modern orthography) for text-encoder distillation.

    uv run python scripts/build_aozora_corpus.py --output-dir data/corpus

Reads ``globis-university/aozorabunko-clean`` (CC BY 4.0; the texts are public
domain), keeps works written in modern kanji and kana (``文字遣い種別`` =
``新字新仮名``), splits them into sentences with the same filters as
``build_text_corpus.py``, removes JKYB-Parakeet sentences and duplicates, caps the
sentences per work, and writes ``aozora_train.txt`` / ``aozora_val.txt``.
Literary text has many kun'yomi and jukujikun, the readings where the pruned
text encoders lose most against the teacher.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

from build_text_corpus import is_val, jkyb_texts, keep
from huggingface_hub import hf_hub_download

REPO = "globis-university/aozorabunko-clean"
FILENAME = "aozorabunko-dedupe-clean.jsonl.gz"
SENTENCE_ENDS = ("。", "！", "？", "!", "?")


def sentences(text: str) -> list[str]:
    out: list[str] = []
    for line in text.split("\n"):
        line = line.strip().lstrip("　").strip()
        start = 0
        for i, char in enumerate(line):
            if char in SENTENCE_ENDS:
                piece = line[start : i + 1].strip("「」『』（）　 ")
                if piece:
                    out.append(piece if piece.endswith(SENTENCE_ENDS) else piece + "。")
                start = i + 1
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=100)
    parser.add_argument("--max-per-work", type=int, default=300)
    parser.add_argument("--val-per-mille", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    excluded = jkyb_texts()
    rng = random.Random(args.seed)
    seen: set[str] = set()
    train: list[str] = []
    val: list[str] = []
    works = 0
    path = hf_hub_download(REPO, FILENAME, repo_type="dataset")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("meta", {}).get("文字遣い種別") != "新字新仮名":
                continue
            works += 1
            candidates = [
                s
                for s in sentences(row["text"])
                if s not in seen and s not in excluded and keep(s, args.min_chars, args.max_chars)
            ]
            rng.shuffle(candidates)
            for sentence in candidates[: args.max_per_work]:
                seen.add(sentence)
                (val if is_val(sentence, args.val_per_mille) else train).append(sentence)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "aozora_train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (args.output_dir / "aozora_val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    print(f"works={works:,} train={len(train):,} val={len(val):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
