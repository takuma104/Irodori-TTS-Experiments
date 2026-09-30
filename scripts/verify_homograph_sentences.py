#!/usr/bin/env python3
"""Verify homograph readings of corpus sentences with an LLM (multiple choice).

    uv run python scripts/verify_homograph_sentences.py data/yomi/sentences_prod.jsonl \
        --lexicon data/yomi_prod/lexicon.jsonl --base-url http://localhost:8000/v1

Sentences with ``reading_verified: null`` (words whose surface has several
readings) are asked which reading the sentence uses, with every lexicon reading
of the surface as a choice; the analyzers' reading is kept only when the model
picks it. Other sentences pass through. The file is rewritten in place.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from generate_yomi_sentences import CHOICE_PROMPT, chat
from openai import AsyncOpenAI


async def verify(
    client: AsyncOpenAI,
    model: str,
    row: dict[str, Any],
    readings: list[str],
    semaphore: asyncio.Semaphore,
) -> bool:
    choices = readings[:]
    random.Random(row["text"]).shuffle(choices)
    async with semaphore:
        answer = await chat(
            client,
            model,
            CHOICE_PROMPT.format(
                word=row["word"],
                sentence=row["text"],
                choices="\n".join(f"{i + 1}. {c}" for i, c in enumerate(choices)),
            ),
            temperature=0.0,
        )
    digits = re.findall(r"\d+", answer)
    return (
        bool(digits)
        and 1 <= int(digits[0]) <= len(choices)
        and (choices[int(digits[0]) - 1] == row["reading"])
    )


async def run(args: argparse.Namespace) -> None:
    readings: dict[str, set[str]] = defaultdict(set)
    with args.lexicon.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            readings[row["word"]].add(row["reading"])
    with args.sentences.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    todo = [r for r in rows if r.get("reading_verified") is None]
    print(f"sentences={len(rows):,} to verify={len(todo):,}", flush=True)
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    semaphore = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(
        *[
            verify(client, args.model, r, sorted(readings[r["word"]]), semaphore)
            for r in todo
        ]
    )
    for row, ok in zip(todo, results, strict=True):
        row["reading_verified"] = ok
    kept = [r for r in rows if r["reading_verified"]]
    with args.sentences.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"verified {sum(results):,}/{len(todo):,}; kept {len(kept):,} sentences",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sentences", type=Path)
    parser.add_argument("--lexicon", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen3.6-27B-NVFP4")
    parser.add_argument("--concurrency", type=int, default=64)
    args = parser.parse_args()
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
