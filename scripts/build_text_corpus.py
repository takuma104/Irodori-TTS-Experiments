#!/usr/bin/env python3
"""Sentence corpus from Japanese Wikipedia for text-encoder distillation (plan Phase 1a).

    uv run python scripts/build_text_corpus.py --shards 0 5 10 --output-dir data/corpus

Downloads the selected shards of ``wikimedia/wikipedia`` (20231101.ja), splits
article bodies into sentences, keeps TTS-like sentences (length, mostly Japanese
script, no markup residue), removes duplicates and every JKYB-Parakeet sentence,
and writes ``wiki_train.txt`` / ``wiki_val.txt`` (one sentence per line, split by
a hash of the sentence).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

WIKI_REPO = "wikimedia/wikipedia"
WIKI_CONFIG = "20231101.ja"
JKYB_REPO = "Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet"
JKYB_FILENAME = "data/common_kanji_source.jsonl"

SENTENCE_END = re.compile(r"(?<=[。！？!?])")
JAPANESE = re.compile(r"[぀-ヿ㐀-鿿豈-﫿々〆ー]")
MARKUP = re.compile(r"[|={}\[\]<>#*_\\]|https?:|\.(?:jpg|png|svg)")
SECTION_HEADERS = ("関連項目", "脚注", "参考文献", "外部リンク", "出典", "注釈")


def article_sentences(text: str) -> Iterator[str]:
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph or paragraph.startswith(SECTION_HEADERS):
            continue
        for sentence in SENTENCE_END.split(paragraph):
            sentence = sentence.strip()
            if sentence:
                yield sentence


def keep(sentence: str, min_chars: int, max_chars: int) -> bool:
    if not min_chars <= len(sentence) <= max_chars:
        return False
    if not sentence.endswith(("。", "！", "？", "!", "?")):
        return False
    if MARKUP.search(sentence):
        return False
    japanese = len(JAPANESE.findall(sentence))
    return japanese / len(sentence) >= 0.6


def jkyb_texts() -> set[str]:
    path = Path(hf_hub_download(JKYB_REPO, JKYB_FILENAME, repo_type="dataset"))
    with path.open(encoding="utf-8") as handle:
        return {json.loads(line)["text"].strip() for line in handle if line.strip()}


def is_val(sentence: str, val_per_mille: int) -> bool:
    digest = hashlib.sha1(sentence.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 1000 < val_per_mille


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, nargs="+", default=[0, 5, 10])
    parser.add_argument("--output-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=120)
    parser.add_argument("--max-per-article", type=int, default=20)
    parser.add_argument("--val-per-mille", type=int, default=5)
    args = parser.parse_args()

    excluded = jkyb_texts()
    seen: set[str] = set()
    train: list[str] = []
    val: list[str] = []
    for shard in args.shards:
        filename = f"{WIKI_CONFIG}/train-{shard:05d}-of-00015.parquet"
        path = hf_hub_download(WIKI_REPO, filename, repo_type="dataset")
        table = pq.read_table(path, columns=["text"])
        for article in table.column("text").to_pylist():
            taken = 0
            for sentence in article_sentences(article):
                if taken >= args.max_per_article:
                    break
                if sentence in seen or sentence in excluded:
                    continue
                if not keep(sentence, args.min_chars, args.max_chars):
                    continue
                seen.add(sentence)
                taken += 1
                (val if is_val(sentence, args.val_per_mille) else train).append(sentence)
        print(f"shard {shard}: train={len(train):,} val={len(val):,}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "wiki_train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (args.output_dir / "wiki_val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
