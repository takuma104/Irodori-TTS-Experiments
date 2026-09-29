#!/usr/bin/env python3
"""Whisper CER of synthesized general sentences (plan §6.3 regression check).

    cd Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition && uv run --no-sync python \
        ../scripts/eval_general_cer.py ../outputs/yomi_eval/base/regress \
        ../outputs/yomi_eval/s6_cont/regress --rows ../data/yomi/regress_rows.jsonl

Transcribes ``<run>/audio/<key>.wav`` of each run with whisper-large-v3-turbo
(the jkyb-eval text model) and reports the corpus CER against the row text,
using jkyb-eval's text normalization, plus paired per-sentence counts. Run it in
the JKYB-Parakeet environment (``uv sync --extra asr``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jkyb_eval.asr import AsrConfiguration, WhisperTranscriber
from jkyb_eval.normalization import canonicalize_text
from rapidfuzz.distance import Levenshtein

TEXT_MODEL = "openai/whisper-large-v3-turbo"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    with args.rows.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    transcriber = WhisperTranscriber(
        AsrConfiguration(
            model=TEXT_MODEL, device=args.device, batch_size=args.batch_size
        )
    )
    errors_by_run: list[list[int]] = []
    total = sum(len(canonicalize_text(r["text"])) for r in rows)
    for run in args.runs:
        cache = run / "whisper_text.jsonl"
        if cache.exists():
            with cache.open(encoding="utf-8") as handle:
                texts = {r["key"]: r["text"] for r in map(json.loads, handle)}
        else:
            paths = [run / "audio" / f"{r['key']}.wav" for r in rows]
            texts = dict(
                zip(
                    [r["key"] for r in rows], transcriber.transcribe(paths), strict=True
                )
            )
            with cache.open("w", encoding="utf-8") as handle:
                for key, text in texts.items():
                    handle.write(
                        json.dumps({"key": key, "text": text}, ensure_ascii=False)
                        + "\n"
                    )
        errors = [
            Levenshtein.distance(
                canonicalize_text(r["text"]), canonicalize_text(texts[r["key"]])
            )
            for r in rows
        ]
        errors_by_run.append(errors)
        exact = sum(e == 0 for e in errors)
        print(f"{run}: CER {sum(errors) / total:.3%}, exact {exact}/{len(rows)}")
    if len(errors_by_run) == 2:
        base, cand = errors_by_run
        better = sum(c < b for b, c in zip(base, cand, strict=True))
        worse = sum(c > b for b, c in zip(base, cand, strict=True))
        print(
            f"per sentence: better {better}, worse {worse}, same {len(rows) - better - worse}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
