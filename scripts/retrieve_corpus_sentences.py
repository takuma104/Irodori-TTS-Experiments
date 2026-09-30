#!/usr/bin/env python3
"""Find real corpus sentences for lexicon words, with analyzer-checked readings.

    uv run python scripts/retrieve_corpus_sentences.py \
        --lexicon data/yomi_prod/lexicon.jsonl \
        --corpus data/yomi/wiki_shard2.jsonl --corpus data/yomi/wiki_shard3.jsonl \
        --output data/yomi/corpus_hits.jsonl

For every usable bucket word (A-F) of at least two characters before the last
kanji, the corpus is scanned with Aho-Corasick for its stem (the word up to its
last kanji, so conjugated forms match). A hit counts when the stem stands alone
(no kanji right before or after), occurs once in the sentence, and is not a
dictionary form without okurigana written with it (羽振 in 羽振り).

The number of hits is kept as a real-usage frequency. For up to ``--check``
hits per stem, the stem's reading in context is taken from MeCab + UniDic and
from SudachiPy; a sentence is kept for a word when both analyzers segment the
stem as its own span and read it as the dictionary does. Stems shared by several
readings (homographs) keep the analyzers' choice but are flagged
``needs_llm_check``: analyzers agree on the default reading even when the
context calls for another (数多 read スウタ instead of アマタ).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import ahocorasick
import fugashi
from sudachipy import Dictionary, SplitMode

KANJI = re.compile(r"[㐀-鿿々〆ヶ]")
SKIP_MISC = {"uk", "obs", "rare", "vulg", "sl", "derog", "X"}


def to_katakana(text: str) -> str:
    return "".join(chr(ord(ch) + 0x60) if "ぁ" <= ch <= "ゖ" else ch for ch in text)


def to_hiragana(text: str) -> str:
    return "".join(chr(ord(ch) - 0x60) if "ァ" <= ch <= "ヶ" else ch for ch in text)


def stem_and_reading(row: dict[str, Any]) -> tuple[str, str] | None:
    word = row["word"]
    last = max(i for i, ch in enumerate(word) if KANJI.match(ch))
    by_index = {kr["index"]: kr for kr in row["kanji_readings"]}
    reading = ""
    i = 0
    while i <= last:
        kr = by_index.get(i)
        if kr is not None:
            reading += kr["reading"]
            i += len(kr["kanji"])
        elif KANJI.match(word[i]):
            return None
        else:
            reading += to_katakana(word[i])
            i += 1
    if to_hiragana(reading) != row["reading"][: len(reading)]:
        return None
    return word[: last + 1], reading


def span_reading(
    tokens: list[tuple[int, int, str]], start: int, end: int
) -> str | None:
    """Reading of [start, end) if the segmentation has boundaries at both ends."""
    starts = {s for s, _, _ in tokens}
    ends = {e for _, e, _ in tokens}
    if start not in starts or end not in ends:
        return None
    return "".join(r for s, e, r in tokens if start <= s and e <= end)


def unidic_tokens(tagger: fugashi.Tagger, text: str) -> list[tuple[int, int, str]]:
    tokens = []
    position = 0
    for word in tagger(text):
        start = text.index(word.surface, position)
        end = start + len(word.surface)
        kana = word.feature.kana
        tokens.append(
            (start, end, kana if kana and kana != "*" else to_katakana(word.surface))
        )
        position = end
    return tokens


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lexicon", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", type=int, default=16, help="Hits checked per stem.")
    parser.add_argument("--keep", type=int, default=4, help="Sentences kept per word.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    targets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with args.lexicon.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if (
                not row["buckets"]
                or len(row["word"]) > 6
                or len(row["reading"]) > 12
                or SKIP_MISC & set(row["misc"])
            ):
                continue
            info = stem_and_reading(row)
            if info is None or len(info[0]) < 2:
                continue
            targets[info[0]].append({**row, "stem": info[0], "stem_reading": info[1]})
    automaton = ahocorasick.Automaton()
    for stem in targets:
        automaton.add_word(stem, stem)
    automaton.make_automaton()
    print(f"stems={len(targets):,}", flush=True)

    rng = random.Random(args.seed)
    hits: dict[str, int] = defaultdict(int)
    samples: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for path in args.corpus:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                text = json.loads(line)["text"]
                found: dict[str, list[int]] = defaultdict(list)
                for end, stem in automaton.iter(text):
                    start = end - len(stem) + 1
                    before = text[start - 1] if start > 0 else ""
                    after = text[end + 1] if end + 1 < len(text) else ""
                    if (before and KANJI.match(before)) or (
                        after and KANJI.match(after)
                    ):
                        continue
                    found[stem].append(start)
                for stem, starts in found.items():
                    if len(starts) != 1:
                        continue
                    hits[stem] += 1
                    # Reservoir sample of hits per stem.
                    bucket = samples[stem]
                    if len(bucket) < args.check:
                        bucket.append((text, starts[0]))
                    else:
                        j = rng.randrange(hits[stem])
                        if j < args.check:
                            bucket[j] = (text, starts[0])
        print(f"scanned {path}: stems with hits={len(hits):,}", flush=True)

    tagger = fugashi.Tagger()
    tokenizer = Dictionary().tokenizer()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept_words = 0
    with args.output.open("w", encoding="utf-8") as out:
        for stem, sentences in samples.items():
            candidates = targets[stem]
            homograph = len({c["reading"] for c in candidates}) > 1
            kept: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for text, start in sentences:
                end = start + len(stem)
                after = text[end : end + 1]
                u = span_reading(unidic_tokens(tagger, text), start, end)
                s = span_reading(
                    [
                        (m.begin(), m.end(), m.reading_form())
                        for m in tokenizer.tokenize(text, SplitMode.C)
                    ],
                    start,
                    end,
                )
                if u is None or s is None or u != s:
                    continue
                for target in candidates:
                    if target["stem_reading"] != u:
                        continue
                    if after and after == to_hiragana(target["stem_reading"][-1]):
                        continue
                    key = target["reading"]
                    if len(kept[key]) < args.keep:
                        kept[key].append(
                            {
                                "text": text,
                                "stem_start": start,
                                "kana_text": text[:start]
                                + to_hiragana(target["stem_reading"])
                                + text[end:],
                            }
                        )
            for target in candidates:
                found = kept.get(target["reading"], [])
                record = {
                    "word": target["word"],
                    "reading": target["reading"],
                    "stem": stem,
                    "stem_reading": target["stem_reading"],
                    "buckets": target["buckets"],
                    "tier": target["tier"],
                    "hits": hits[stem],
                    "homograph": homograph,
                    "needs_llm_check": homograph,
                    "sentences": found,
                }
                kept_words += bool(found)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"words with verified sentences={kept_words:,}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
