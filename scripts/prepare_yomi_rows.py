#!/usr/bin/env python3
"""Turn generated sentences into JKYB-format rows (plan §4.5-4.6).

    uv run python scripts/prepare_yomi_rows.py data/yomi/sentences_pilot.jsonl \
        --targets data/yomi/targets_pilot.jsonl --output data/yomi/rows_pilot.jsonl

The rows follow the JKYB-Parakeet schema, so ``jkyb-eval tts --dataset`` can score
our own sentences (dev evaluation and hard mining), and they keep the training
fields (``kana_text``, bucket, role, split).

The sentence reading comes from SudachiPy, with the target word's span replaced
by the dictionary reading and the particles は/へ read as ワ/エ (the JKYB
convention). The tagged span is the bucket's target kanji (A-D, G), the whole
stem for homographs (E), or the jukujikun group (F).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from sudachipy import Dictionary, SplitMode

KANJI = re.compile(r"[㐀-鿿々〆ヶ]")


def to_katakana(text: str) -> str:
    return "".join(chr(ord(ch) + 0x60) if "ぁ" <= ch <= "ゖ" else ch for ch in text)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stem_parts(target: dict[str, Any], stem: str) -> list[tuple[int, int, str, str]]:
    """(start, end, kind, reading) for each part of the stem (a kanji group or a kana)."""
    by_index = {kr["index"]: kr for kr in target["kanji_readings"]}
    parts: list[tuple[int, int, str, str]] = []
    i = 0
    while i < len(stem):
        kr = by_index.get(i)
        if kr is not None:
            parts.append((i, i + len(kr["kanji"]), kr["kind"], kr["reading"]))
            i += len(kr["kanji"])
        else:
            parts.append((i, i + 1, "kana", to_katakana(stem[i])))
            i += 1
    return parts


def target_span(
    sentence: dict[str, Any],
    target: dict[str, Any],
    parts: list[tuple[int, int, str, str]],
) -> tuple[int, int, str] | None:
    """(start, end, category) of the tagged span inside the stem."""
    bucket = sentence["bucket"]
    if bucket == "E":
        kinds = [kind for _, _, kind, _ in parts if kind != "kana"]
        category = "on_yomi" if kinds and kinds[0] == "on" else "kun_yomi"
        return 0, len(sentence["stem"]), category
    if bucket == "G":
        index = sentence["stem"].find(target["contrast_kanji"])
        indices = [index] if index >= 0 else []
    else:
        indices = target["bucket_targets"].get(bucket, [])
    for start, end, kind, _ in parts:
        if start in indices:
            category = {
                "on": "on_yomi",
                "kun": "kun_yomi",
                "jukujikun": "joyo_appendix_reading",
                "appendix": "joyo_appendix_reading",
            }.get(kind)
            if category is None:
                return None
            return start, end, category
    return None


def sentence_yomi(
    tokenizer: Any, text: str, span_start: int, span_end: int, span_reading: str
) -> str | None:
    """Katakana reading of ``text`` with [span_start, span_end) read as ``span_reading``."""
    out: list[str] = []
    inserted = False
    for morpheme in tokenizer.tokenize(text, SplitMode.C):
        begin, end = morpheme.begin(), morpheme.end()
        surface = morpheme.surface()
        if end <= span_start or begin >= span_end:
            reading = morpheme.reading_form()
            if morpheme.part_of_speech()[0] == "助詞" and surface in ("は", "へ"):
                reading = {"は": "ワ", "へ": "エ"}[surface]
            out.append(reading)
            continue
        # The morpheme overlaps the target word: characters outside the word
        # must be kana (okurigana, particles) so their reading is known.
        for offset, ch in enumerate(surface):
            position = begin + offset
            if span_start <= position < span_end:
                if not inserted:
                    out.append(span_reading)
                    inserted = True
            elif KANJI.match(ch):
                return None
            else:
                out.append(to_katakana(ch))
    return "".join(out) if inserted else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sentences", type=Path)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    targets = {(t["word"], t["reading"]): t for t in read_jsonl(args.targets)}
    tokenizer = Dictionary().tokenizer()
    counts: Counter[str] = Counter()
    per_word: Counter[tuple[str, str, str]] = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for sentence in read_jsonl(args.sentences):
            target = targets[(sentence["word"], sentence["reading"])]
            stem = sentence["stem"]
            parts = stem_parts(target, stem)
            span = target_span(sentence, target, parts)
            if span is None:
                counts["no_target_span"] += 1
                continue
            span_start, span_end, category = span
            before = "".join(r for s, _, _, r in parts if s < span_start)
            tagged = "".join(r for s, _, _, r in parts if span_start <= s < span_end)
            after = "".join(r for s, _, _, r in parts if s >= span_end)
            text = sentence["text"]
            word_start = sentence["stem_start"]
            yomi = sentence_yomi(
                tokenizer,
                text,
                word_start,
                word_start + len(stem),
                before + tagged + after,
            )
            if yomi is None:
                counts["no_yomi"] += 1
                continue
            stem_reading_start = yomi.index(before + tagged + after)
            tag_start = stem_reading_start + len(before)
            ident = (sentence["bucket"], sentence["word"], sentence["reading"])
            key = f"{sentence['bucket']}_{sentence['word']}_{sentence['reading']}_{per_word[ident]}"
            per_word[ident] += 1
            a, b = word_start + span_start, word_start + span_end
            row = {
                "key": key,
                "text": text,
                "tagged_text": f"{text[:a]}<{text[a:b]}>{text[b:]}",
                "yomi": yomi,
                "tagged_yomi": (
                    f"{yomi[:tag_start]}<{tagged}>{yomi[tag_start + len(tagged) :]}"
                ),
                "reading_category": category,
                "readings": {"natural": [tagged], "marginal": []},
                "source": "yomi_ft_" + sentence["bucket"],
                **{
                    k: sentence[k]
                    for k in ("word", "reading", "bucket", "role", "split", "kana_text")
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["rows"] += 1
    print(dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
