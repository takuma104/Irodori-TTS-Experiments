#!/usr/bin/env python3
"""Extract general Japanese sentences for representation preservation.

    PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync \
        python scripts/make_general_text.py --output data/yomi/general_text.jsonl

Reads one shard of Japanese Wikipedia (``wikimedia/wikipedia`` 20231101.ja),
splits articles into sentences, and keeps mid-length sentences that are mostly
kanji and kana. JKYB-Parakeet sentences are excluded. These sentences need no
audio: the trainer only asks the student's text states to match the original
encoder on them (text-only, no DiT).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

SENTENCE_END = re.compile(r"(?<=[。！？])")
JAPANESE = re.compile(r"[ぁ-ゖァ-ヺー㐀-鿿々、。]")
KANJI = re.compile(r"[㐀-鿿々]")
JKYB_REPO = "Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet"
JKYB_FILE = "data/common_kanji_source.jsonl"


def keep(sentence: str) -> bool:
    if not 12 <= len(sentence) <= 60 or not sentence.endswith("。"):
        return False
    japanese = len(JAPANESE.findall(sentence))
    if japanese / len(sentence) < 0.95:
        return False
    return len(KANJI.findall(sentence)) >= 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--max-sentences", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    jkyb_path = Path(hf_hub_download(JKYB_REPO, JKYB_FILE, repo_type="dataset"))
    with jkyb_path.open(encoding="utf-8") as handle:
        jkyb = {json.loads(line)["text"] for line in handle if line.strip()}
    path = hf_hub_download(
        "wikimedia/wikipedia",
        f"20231101.ja/train-{args.shard:05d}-of-00015.parquet",
        repo_type="dataset",
    )
    sentences: list[str] = []
    seen: set[str] = set()
    for batch in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=256):
        for article in batch.column(0).to_pylist():
            for line in article.split("\n"):
                for sentence in SENTENCE_END.split(line.strip()):
                    sentence = sentence.strip()
                    if keep(sentence) and sentence not in seen and sentence not in jkyb:
                        seen.add(sentence)
                        sentences.append(sentence)
    random.Random(args.seed).shuffle(sentences)
    sentences = sentences[: args.max_sentences]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for sentence in sentences:
            handle.write(json.dumps({"text": sentence}, ensure_ascii=False) + "\n")
    print(f"sentences={len(sentences)} (from {len(seen)} candidates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
