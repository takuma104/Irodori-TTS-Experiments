#!/usr/bin/env python3
"""Voice-design captions from the local LLM, for refitting the caption projector.

    uv run python scripts/generate_captions.py --output data/yomi/captions.jsonl

Each request gets a random speaker, manner and setting as a theme and asks for
several captions in the style of the Irodori-TTS examples (a short phrase or a
sentence or two describing the voice, emotion and recording conditions).
A hash of the caption puts about 5% into the dev split.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
from pathlib import Path

from openai import AsyncOpenAI

SPEAKERS = [
    "幼い女の子",
    "幼い男の子",
    "十代の少女",
    "十代の少年",
    "二十代の女性",
    "二十代の男性",
    "三十代の女性",
    "三十代の男性",
    "中年の女性",
    "中年の男性",
    "年配の女性",
    "年配の男性",
    "老婆",
    "老人",
    "アナウンサー",
    "ナレーター",
    "声優",
    "教師",
    "店員",
    "医師",
]
MANNERS = [
    "穏やかに",
    "明るく元気に",
    "悲しげに",
    "怒って",
    "驚いて",
    "眠そうに",
    "緊張して",
    "照れながら",
    "ささやくように",
    "早口で",
    "ゆっくりと",
    "自信ありげに",
    "呆れながら",
    "泣きながら",
    "笑いながら",
    "淡々と",
    "甘えるように",
    "からかうように",
    "力強く",
    "心配そうに",
    "息を切らして",
    "酔っ払って",
    "優しく",
    "冷たく",
    "興奮して",
]
SETTINGS = [
    "静かな部屋で",
    "電話越しに",
    "広いホールで",
    "屋外で",
    "耳元で",
    "マイクに近い距離で",
    "少し離れた位置から",
    "車の中で",
    "放送のように",
    "ラジオ番組で",
    "朗読として",
    "ゲームのキャラクターとして",
    "会議で",
    "友人との雑談で",
    "子どもに向けて",
]
PROMPT = """音声合成モデルに与える「声の説明文（キャプション）」を{n}個作ってください。
キャプションは、話者の性別・年齢・声質、感情や話し方、距離感や録音環境などを日本語で説明するものです。
例:
- 落ち着いた、近い距離感の女性話者
- 余裕のある大人の男性。親しい相手に対して、くだけた雰囲気で呆れながらも楽しそうに話している。
- 落ち着いた自然な声

今回のテーマ: {speaker}が{setting}{manner}話す。
テーマを中心に、短い句から2文程度の説明まで、長さや書き方を変えてください。
読み上げるセリフそのものは書かず、声の説明だけを書いてください。
1行に1つ、番号や記号を付けずに出力してください。"""


async def request(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> list[str]:
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=1024,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    text = response.choices[0].message.content or ""
    lines = [
        re.sub(r"^\s*(?:[-・*]|\d+[.)．、])\s*", "", line).strip()
        for line in text.splitlines()
    ]
    return [line for line in lines if 4 <= len(line) <= 120 and "「" not in line]


async def run(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    semaphore = asyncio.Semaphore(args.concurrency)
    prompts = [
        PROMPT.format(
            n=args.per_request,
            speaker=rng.choice(SPEAKERS),
            manner=rng.choice(MANNERS),
            setting=rng.choice(SETTINGS),
        )
        for _ in range(args.requests)
    ]
    results = await asyncio.gather(
        *(request(client, args.model, p, semaphore) for p in prompts)
    )
    seen: set[str] = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "dev": 0}
    with args.output.open("w", encoding="utf-8") as handle:
        for captions in results:
            for caption in captions:
                if caption in seen:
                    continue
                seen.add(caption)
                split = (
                    "dev" if hashlib.md5(caption.encode()).digest()[0] < 13 else "train"
                )
                counts[split] += 1
                handle.write(
                    json.dumps({"text": caption, "split": split}, ensure_ascii=False)
                    + "\n"
                )
    print(f"captions={len(seen)} {counts} -> {args.output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/yomi/captions.jsonl"))
    parser.add_argument("--requests", type=int, default=600)
    parser.add_argument("--per-request", type=int, default=10)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen3.6-27B-NVFP4")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    asyncio.run(run(parser.parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
