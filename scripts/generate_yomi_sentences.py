#!/usr/bin/env python3
"""Generate example sentences for target words with an LLM (plan §4.4-4.5).

    uv run python scripts/generate_yomi_sentences.py data/yomi/targets_pilot.jsonl \
        --output data/yomi/sentences_pilot.jsonl \
        --base-url http://localhost:8000/v1 --model Qwen3.6-27B-NVFP4

For each word, asks an OpenAI-compatible endpoint (e.g. a local vLLM server) for
sentences that use the word in the sense of the given reading. A sentence is
kept when the word's stem (the word up to its last kanji, so conjugated verbs
still match) occurs exactly once as a standalone kanji run. When the surface has
several readings in the lexicon, a second call asks which reading the sentence
uses (multiple choice) and only matching sentences are kept.

Each output row carries ``kana_text``: the sentence with the stem replaced by
its hiragana reading, which is the kana-substituted input for the teacher.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

KANJI = re.compile(r"[㐀-鿿々〆ヶ]")
JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)

GENERATE_PROMPT = """「{word}」（読み: {reading}、意味: {gloss}）を使った自然な日本語の文を{n}個作ってください。

条件:
- 「{word}」はこの漢字表記のまま使う（活用する語は活用させてよい）。ひらがなやカタカナに開かない
- 読みが「{reading}」になる意味・用法で使う
- 各文で「{word}」は1回だけ使う
- 1文は15〜45文字程度。文の長さと、語の位置（文頭・文中・文末）をばらす
- ほかの難読語、固有名詞、数字、アルファベットは使わない
- 出力は文字列のJSON配列だけ（説明は書かない）"""

CHOICE_PROMPT = """次の文を声に出して読むとき、「{word}」の読みとして正しいものを選んでください。番号だけを答えてください。

文: {sentence}

{choices}"""


def to_katakana(text: str) -> str:
    return "".join(chr(ord(ch) + 0x60) if "ぁ" <= ch <= "ゖ" else ch for ch in text)


def to_hiragana(text: str) -> str:
    return "".join(chr(ord(ch) - 0x60) if "ァ" <= ch <= "ヶ" else ch for ch in text)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stem_and_reading(target: dict[str, Any]) -> tuple[str, str] | None:
    """The word up to its last kanji and that part's reading (katakana).

    思い出す -> (思い出, オモイダ). Returns None when the kanji readings do not
    cover the stem.
    """
    word = target["word"]
    last = max(i for i, ch in enumerate(word) if KANJI.match(ch))
    by_index = {kr["index"]: kr for kr in target["kanji_readings"]}
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
    if to_hiragana(reading) != target["reading"][: len(reading)]:
        return None
    return word[: last + 1], reading


def find_stem(sentence: str, stem: str) -> int | None:
    """Index of the only standalone occurrence of ``stem``, or None."""
    hits = [
        m.start()
        for m in re.finditer(re.escape(stem), sentence)
        if (m.start() == 0 or not KANJI.match(sentence[m.start() - 1]))
        and (m.end() == len(sentence) or not KANJI.match(sentence[m.end()]))
    ]
    return hits[0] if len(hits) == 1 else None


async def chat(client: AsyncOpenAI, model: str, prompt: str, temperature: float) -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=1024,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return response.choices[0].message.content or ""


async def generate_for(
    client: AsyncOpenAI,
    model: str,
    target: dict[str, Any],
    homograph_readings: list[str],
    n: int,
    semaphore: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    stem_info = stem_and_reading(target)
    if stem_info is None:
        return []
    stem, stem_reading = stem_info
    gloss = "; ".join(target["gloss"][:3])
    async with semaphore:
        raw = await chat(
            client,
            model,
            GENERATE_PROMPT.format(
                word=target["word"], reading=target["reading"], gloss=gloss, n=n
            ),
            temperature=0.8,
        )
    match = JSON_ARRAY.search(raw)
    try:
        sentences = json.loads(match.group(0)) if match else []
    except json.JSONDecodeError:
        sentences = []
    rows: list[dict[str, Any]] = []
    for sentence in sentences:
        if not isinstance(sentence, str):
            continue
        sentence = sentence.strip()
        position = find_stem(sentence, stem)
        if position is None or not 8 <= len(sentence) <= 80:
            continue
        # A dictionary form without okurigana (羽振, はぶり) written with it in the
        # sentence (羽振り) would duplicate the kana after substitution (はぶりり).
        after = sentence[position + len(stem) : position + len(stem) + 1]
        if after and after == to_hiragana(stem_reading[-1]):
            continue
        verified = None
        if homograph_readings:
            choices = [target["reading"]] + homograph_readings
            async with semaphore:
                answer = await chat(
                    client,
                    model,
                    CHOICE_PROMPT.format(
                        word=target["word"],
                        sentence=sentence,
                        choices="\n".join(
                            f"{i + 1}. {c}" for i, c in enumerate(choices)
                        ),
                    ),
                    temperature=0.0,
                )
            digits = re.findall(r"\d+", answer)
            verified = bool(digits) and digits[0] == "1"
            if not verified:
                continue
        rows.append(
            {
                "word": target["word"],
                "reading": target["reading"],
                "bucket": target["bucket"],
                "role": target["role"],
                "split": target["split"],
                "stem": stem,
                "stem_reading": stem_reading,
                "stem_start": position,
                "text": sentence,
                "kana_text": sentence[:position]
                + to_hiragana(stem_reading)
                + sentence[position + len(stem) :],
                "reading_verified": verified,
            }
        )
    return rows


async def run(args: argparse.Namespace) -> None:
    targets = read_jsonl(args.targets)
    if args.limit is not None:
        targets = targets[: args.limit]
    lexicon_readings: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(args.lexicon):
        lexicon_readings[row["word"]].add(row["reading"])
    done: set[tuple[str, str]] = set()
    if args.output.exists():
        done = {(r["word"], r["reading"]) for r in read_jsonl(args.output)}
    todo = [t for t in targets if (t["word"], t["reading"]) not in done]
    print(f"targets={len(targets)} todo={len(todo)}", flush=True)

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    semaphore = asyncio.Semaphore(args.concurrency)
    stats: Counter[str] = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        tasks = [
            generate_for(
                client,
                args.model,
                target,
                sorted(lexicon_readings[target["word"]] - {target["reading"]}),
                args.sentences_per_word,
                semaphore,
            )
            for target in todo
        ]
        for index, task in enumerate(asyncio.as_completed(tasks), start=1):
            rows = await task
            stats["words_with_sentences"] += bool(rows)
            stats["sentences"] += len(rows)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % 200 == 0 or index == len(tasks):
                handle.flush()
                print(f"{index}/{len(tasks)} {dict(stats)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lexicon", type=Path, default=Path("data/yomi/lexicon.jsonl"))
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen3.6-27B-NVFP4")
    parser.add_argument("--sentences-per-word", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
