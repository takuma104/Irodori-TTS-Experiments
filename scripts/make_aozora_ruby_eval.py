#!/usr/bin/env python3
"""Build a hard-reading evaluation set from Aozora Bunko ruby (JKYB-format rows).

    uv run python scripts/make_aozora_ruby_eval.py --output data/yomi/aozora_eval_rows.jsonl

Aozora Bunko texts mark readings of hard words with ruby (従容《しょうよう》),
chosen by people for that context, which makes them natural test items for a
TTS reading model without using JKYB. Works in modern orthography (新字新仮名)
whose copyright has expired are split by a hash of the work ID: the eval split
is used here and must stay out of training (``--eval-ratio``).

Each item is one sentence with one ruby word as the target; the ruby is the
reading. Items are kept only when the ruby matches a dictionary or analyzer
reading of the word (JMdict, UniDic, or Sudachi), so authors' creative readings
(本気《マジ》) are dropped. The sentence reading comes from SudachiPy, with every
ruby in the sentence overriding its span, and は/へ read as ワ/エ.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import time
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import fugashi
from sudachipy import Dictionary, SplitMode

INDEX_URL = "https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip"
KANJI = re.compile(r"[㐀-鿿々〆ヶ]")
RUBY = re.compile(r"(?:｜([^｜《》]+)|([㐀-鿿々〆ヶ]+))《([^《》]+)》")
ANNOTATION = re.compile(r"［＃[^］]*］")
HIRAGANA_ONLY = re.compile(r"^[ぁ-ゖー]+$")
SENTENCE_END = re.compile(r"(?<=[。！？])")


def to_katakana(text: str) -> str:
    return "".join(chr(ord(ch) + 0x60) if "ぁ" <= ch <= "ゖ" else ch for ch in text)


def eval_work(work_id: str, ratio: float) -> bool:
    return hashlib.md5(f"aozora:{work_id}".encode()).digest()[0] < 256 * ratio


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "yomi-eval/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def load_works() -> list[dict[str, str]]:
    with zipfile.ZipFile(io.BytesIO(fetch(INDEX_URL))) as archive:
        name = archive.namelist()[0]
        text = archive.read(name).decode("utf-8-sig")
    works: dict[str, dict[str, str]] = {}
    for row in csv.DictReader(io.StringIO(text)):
        if (
            row.get("文字遣い種別") == "新字新仮名"
            and row.get("作品著作権フラグ") == "なし"
            and row.get("人物著作権フラグ") == "なし"
            and row.get("テキストファイルURL", "").endswith(".zip")
        ):
            works.setdefault(row["作品ID"], row)
    return list(works.values())


def work_text(url: str) -> str:
    with zipfile.ZipFile(io.BytesIO(fetch(url))) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".txt"))
        raw = archive.read(name).decode("cp932", errors="replace")
    # Drop the header (title, legend between dashed lines) and the colophon.
    parts = re.split(r"^-{20,}\s*$", raw, flags=re.MULTILINE)
    body = parts[2] if len(parts) >= 3 else raw
    body = body.split("底本：")[0]
    return ANNOTATION.sub("", body)


def parse_sentence(sentence: str) -> tuple[str, list[tuple[int, int, str]]] | None:
    """Plain text and (start, end, reading) of each ruby base, or None if unusable."""
    plain: list[str] = []
    rubies: list[tuple[int, int, str]] = []
    position = 0
    length = 0
    for match in RUBY.finditer(sentence):
        prefix = sentence[position : match.start()]
        plain.append(prefix)
        length += len(prefix)
        base = match.group(1) or match.group(2)
        plain.append(base)
        rubies.append((length, length + len(base), match.group(3)))
        length += len(base)
        position = match.end()
    plain.append(sentence[position:])
    text = "".join(plain).strip()
    if "《" in text or "》" in text or "｜" in text or "※" in text:
        return None
    return text, rubies


def sentence_yomi(
    tokenizer: Any, text: str, overrides: list[tuple[int, int, str]]
) -> str | None:
    out: list[str] = []
    done: set[int] = set()
    for morpheme in tokenizer.tokenize(text, SplitMode.C):
        begin, end = morpheme.begin(), morpheme.end()
        covering = [o for o in overrides if o[0] < end and o[1] > begin]
        if not covering:
            reading = morpheme.reading_form()
            if morpheme.part_of_speech()[0] == "助詞" and morpheme.surface() in (
                "は",
                "へ",
            ):
                reading = {"は": "ワ", "へ": "エ"}[morpheme.surface()]
            out.append(reading)
            continue
        for offset, ch in enumerate(morpheme.surface()):
            position = begin + offset
            span = next((o for o in overrides if o[0] <= position < o[1]), None)
            if span is not None:
                if span[0] not in done:
                    out.append(to_katakana(span[2]))
                    done.add(span[0])
            elif KANJI.match(ch):
                return None
            else:
                out.append(to_katakana(ch))
    return "".join(out)


def analyzer_readings(tagger: fugashi.Tagger, tokenizer: Any, word: str) -> set[str]:
    unidic = "".join(w.feature.kana or "" for w in tagger(word))
    sudachi = "".join(m.reading_form() for m in tokenizer.tokenize(word, SplitMode.C))
    return {unidic, sudachi}


def category(word: str, reading: str, lexicon: dict[str, list[dict[str, Any]]]) -> str:
    for row in lexicon.get(word, []):
        if row["reading"] == reading:
            kinds = {kr["kind"] for kr in row["kanji_readings"]}
            if kinds & {"jukujikun", "appendix"}:
                return "joyo_appendix_reading"
            return "kun_yomi" if "kun" in kinds else "on_yomi"
    return "on_yomi"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lexicon", type=Path, default=Path("data/yomi/lexicon.jsonl"))
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--max-works", type=int, default=400)
    parser.add_argument("--max-items", type=int, default=3000)
    parser.add_argument("--per-work", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    lexicon: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with args.lexicon.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            lexicon[row["word"]].append(row)
    tagger = fugashi.Tagger()
    tokenizer = Dictionary().tokenizer()
    rng = random.Random(args.seed)

    works = [w for w in load_works() if eval_work(w["作品ID"], args.eval_ratio)]
    rng.shuffle(works)
    print(f"eval works available: {len(works)}", flush=True)
    items: list[dict[str, Any]] = []
    stats: dict[str, int] = defaultdict(int)
    used_works: list[str] = []
    for work in works[: args.max_works]:
        try:
            body = work_text(work["テキストファイルURL"])
        except Exception as error:  # noqa: BLE001 - skip broken archives
            stats["download_error"] += 1
            print(f"skip {work['作品ID']}: {error!r}", flush=True)
            continue
        time.sleep(0.2)
        used_works.append(work["作品ID"])
        per_work = 0
        candidates = []
        for line in body.splitlines():
            for raw in SENTENCE_END.split(line.strip().lstrip("　")):
                if "《" not in raw:
                    continue
                parsed = parse_sentence(raw.strip())
                if parsed is None:
                    stats["unparsable"] += 1
                    continue
                text, rubies = parsed
                if not 8 <= len(text) <= 80:
                    continue
                candidates.append((text, rubies))
        rng.shuffle(candidates)
        for text, rubies in candidates:
            if per_work >= args.per_work:
                break
            usable = []
            for start, end, reading in rubies:
                word = text[start:end]
                if not (1 <= len(word) <= 4 and all(KANJI.match(c) for c in word)):
                    continue
                if not HIRAGANA_ONLY.match(reading):
                    continue
                known = {to_katakana(r["reading"]) for r in lexicon.get(word, [])}
                known |= analyzer_readings(tagger, tokenizer, word)
                if to_katakana(reading) not in known:
                    stats["creative_reading"] += 1
                    continue
                usable.append((start, end, reading))
            if not usable:
                continue
            start, end, reading = rng.choice(usable)
            yomi = sentence_yomi(tokenizer, text, rubies)
            if yomi is None:
                stats["no_yomi"] += 1
                continue
            target = to_katakana(reading)
            before = sentence_yomi(
                tokenizer, text[:start], [r for r in rubies if r[1] <= start]
            )
            if (
                before is None
                or not yomi.startswith(before)
                or yomi[len(before) :][: len(target)] != target
            ):
                stats["misaligned"] += 1
                continue
            tag = len(before)
            word = text[start:end]
            items.append(
                {
                    "key": f"AOZORA_{work['作品ID']}_{per_work}",
                    "text": text,
                    "tagged_text": f"{text[:start]}<{word}>{text[end:]}",
                    "yomi": yomi,
                    "tagged_yomi": f"{yomi[:tag]}<{target}>{yomi[tag + len(target) :]}",
                    "reading_category": category(word, reading, lexicon),
                    "readings": {"natural": [target], "marginal": []},
                    "source": "aozora_ruby",
                    "work_id": work["作品ID"],
                    "author": work.get("姓", "") + work.get("名", ""),
                    "title": work.get("作品名", ""),
                    "in_jmdict": to_katakana(reading)
                    in {to_katakana(r["reading"]) for r in lexicon.get(word, [])},
                }
            )
            per_work += 1
        if len(items) >= args.max_items:
            break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in items[: args.max_items]:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (args.output.with_suffix(".works.json")).write_text(
        json.dumps(
            {"eval_ratio": args.eval_ratio, "works": used_works}, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"items={min(len(items), args.max_items)} works={len(used_works)} {dict(stats)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
