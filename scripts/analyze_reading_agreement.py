#!/usr/bin/env python3
"""How often do UniDic, Sudachi, and JMdict agree on the reading of kanji words?

    uv run python scripts/analyze_reading_agreement.py data/yomi/wiki_shard1.jsonl \
        --output outputs/production/reading_agreement.md

Each sentence is segmented by MeCab + UniDic (``kana``, the orthographic
reading) and by SudachiPy (mode C, ``reading_form``). The sentence is cut at
positions where both segmentations have a boundary, and every resulting span
that contains kanji is compared. When the span is a JMdict word, the JMdict
readings are checked too (unique reading vs several readings = homograph).

This measures how usable analyzer readings are as a reference for building
training data from real sentences (plan: production model without JKYB).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fugashi
from sudachipy import Dictionary, SplitMode

KANJI = re.compile(r"[㐀-鿿々〆ヶ]")


def to_katakana(text: str) -> str:
    return "".join(chr(ord(ch) + 0x60) if "ぁ" <= ch <= "ゖ" else ch for ch in text)


@dataclass(frozen=True)
class Span:
    text: str
    unidic: str
    sudachi: str
    jmdict: frozenset[str]

    @property
    def agree(self) -> bool:
        return self.unidic == self.sudachi

    @property
    def jmdict_status(self) -> str:
        if not self.jmdict:
            return "not in JMdict"
        return "JMdict unique" if len(self.jmdict) == 1 else "JMdict homograph"

    @property
    def jmdict_agrees(self) -> bool:
        return self.unidic in self.jmdict


def unidic_tokens(tagger: fugashi.Tagger, text: str) -> list[tuple[int, int, str]]:
    tokens = []
    position = 0
    for word in tagger(text):
        start = text.index(word.surface, position)
        end = start + len(word.surface)
        kana = word.feature.kana
        reading = kana if kana and kana != "*" else to_katakana(word.surface)
        tokens.append((start, end, reading))
        position = end
    return tokens


def sudachi_tokens(tokenizer: Any, text: str) -> list[tuple[int, int, str]]:
    return [
        (m.begin(), m.end(), m.reading_form())
        for m in tokenizer.tokenize(text, SplitMode.C)
    ]


def spans(
    text: str,
    uni: list[tuple[int, int, str]],
    sud: list[tuple[int, int, str]],
    jmdict: dict[str, set[str]],
) -> list[Span]:
    common = sorted({end for _, end, _ in uni} & {end for _, end, _ in sud})
    out: list[Span] = []
    start = 0
    for end in common:
        piece = text[start:end]
        if KANJI.search(piece):
            u = "".join(r for s, e, r in uni if start <= s and e <= end)
            d = "".join(r for s, e, r in sud if start <= s and e <= end)
            out.append(
                Span(
                    piece,
                    u,
                    d,
                    frozenset(to_katakana(r) for r in jmdict.get(piece, ())),
                )
            )
        start = end
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sentences", type=Path)
    parser.add_argument("--lexicon", type=Path, default=Path("data/yomi/lexicon.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50_000)
    parser.add_argument("--examples", type=int, default=40)
    args = parser.parse_args()

    jmdict: dict[str, set[str]] = defaultdict(set)
    with args.lexicon.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            jmdict[row["word"]].add(row["reading"])
    with args.sentences.open(encoding="utf-8") as handle:
        texts = [json.loads(line)["text"] for line in handle][: args.limit]

    tagger = fugashi.Tagger()
    tokenizer = Dictionary().tokenizer()
    all_spans: list[Span] = []
    for text in texts:
        all_spans.extend(
            spans(
                text,
                unidic_tokens(tagger, text),
                sudachi_tokens(tokenizer, text),
                jmdict,
            )
        )

    total = len(all_spans)
    agree = sum(s.agree for s in all_spans)
    by_status: dict[str, Counter[str]] = defaultdict(Counter)
    for s in all_spans:
        c = by_status[s.jmdict_status]
        c["n"] += 1
        c["agree"] += s.agree
        c["jmdict_agrees"] += s.jmdict_agrees
        c["all_three"] += s.agree and s.jmdict_agrees
    lines = [
        "# 形態素解析器の読みの一致率",
        "",
        f"- 文: {len(texts):,}（`{args.sentences}`）",
        f"- 漢字を含む区間（UniDic と Sudachi の区切りが両方ある位置で切った単位）: {total:,}",
        f"- UniDic と Sudachi の読みが一致: {agree:,}（{agree / total:.1%}）",
        "",
        "| JMdict での扱い | 区間 | UniDic=Sudachi | UniDic が JMdict の読みのどれか | 3つとも一致 |",
        "|---|---:|---:|---:|---:|",
    ]
    for status in ("JMdict unique", "JMdict homograph", "not in JMdict"):
        c = by_status[status]
        n = max(c["n"], 1)
        lines.append(
            f"| {status} | {c['n']:,} | {c['agree'] / n:.1%} | "
            f"{c['jmdict_agrees'] / n:.1%} | {c['all_three'] / n:.1%} |"
        )
    rng = random.Random(0)
    disagree = [s for s in all_spans if not s.agree]
    lines += ["", f"## UniDic と Sudachi が食い違う例（{args.examples} 件）", ""]
    lines += ["| 区間 | UniDic | Sudachi | JMdict |", "|---|---|---|---|"]
    for s in rng.sample(disagree, min(args.examples, len(disagree))):
        lines.append(
            f"| {s.text} | {s.unidic} | {s.sudachi} | {' / '.join(sorted(s.jmdict))} |"
        )
    homograph_agree = [
        s for s in all_spans if s.agree and s.jmdict_status == "JMdict homograph"
    ]
    lines += [
        "",
        f"## 同形異音語で UniDic と Sudachi が一致した例（{args.examples} 件）",
        "",
    ]
    lines += ["| 区間 | 読み | JMdict |", "|---|---|---|"]
    for s in rng.sample(homograph_agree, min(args.examples, len(homograph_agree))):
        lines.append(f"| {s.text} | {s.unidic} | {' / '.join(sorted(s.jmdict))} |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:14]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
