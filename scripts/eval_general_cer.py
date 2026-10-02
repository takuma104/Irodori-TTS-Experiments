#!/usr/bin/env python3
"""Whisper CER of synthesized general sentences (teacher vs. student regression check).

With ``--mode kana``, transcribes with kana-whisper instead and scores against
the rows' ``kana`` field (e.g. JSUT basic5000 human readings), normalized like
jkyb-eval's Sentence Kana-CER.

    cd third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition && uv run --no-sync python \
        ../../scripts/eval_general_cer.py ../../outputs/eval/teacher/jsut \
        ../../outputs/eval/student/jsut --rows ../../data/eval/jsut_rows.jsonl --mode kana

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
from jkyb_eval.constants import ASR_MAX_NEW_TOKENS, DEFAULT_KANA_MODEL
from jkyb_eval.normalization import canonicalize_text, canonicalize_yomi
from rapidfuzz.distance import Levenshtein

TEXT_MODEL = "openai/whisper-large-v3-turbo"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mode", choices=["text", "kana"], default="text")
    args = parser.parse_args()

    with args.rows.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    kana = args.mode == "kana"
    transcriber = WhisperTranscriber(
        AsrConfiguration(
            model=DEFAULT_KANA_MODEL if kana else TEXT_MODEL,
            device=args.device,
            batch_size=args.batch_size,
            max_new_tokens=ASR_MAX_NEW_TOKENS if kana else None,
        )
    )
    normalize = canonicalize_yomi if kana else canonicalize_text
    field = "kana" if kana else "text"
    errors_by_run: list[list[int]] = []
    total = sum(len(normalize(r[field])) for r in rows)
    for run in args.runs:
        cache = run / ("kana_whisper.jsonl" if kana else "whisper_text.jsonl")
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
            Levenshtein.distance(normalize(r[field]), normalize(texts[r["key"]]))
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
