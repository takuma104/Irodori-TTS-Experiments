#!/usr/bin/env python3
"""Summarize target-reading errors from a ``jkyb-eval tts`` result directory.

    uv run python scripts/analyze_jkyb_errors.py outputs/<run>/results \
        --output outputs/<run>/error_analysis.md

Each incorrect row (``target_exact == false``) is assigned one error type:

- ``generation_failure``: the sentence as a whole is broken (high sentence CER).
- ``other_joyo_reading``: the target was read with another Joyo reading of the
  same kanji (e.g. on/kun confusion).
- ``voicing_or_length``: differs only in voicing (rendaku), sokuon or length.
- ``other``: any other misreading.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TAG_PATTERN = re.compile(r"<([^>]+)>")
CATEGORY_LABELS = {
    "on_yomi": "音読み",
    "kun_yomi": "訓読み",
    "joyo_appendix_reading": "付表の語",
}
ERROR_TYPE_LABELS = {
    "generation_failure": "生成破綻（文全体が崩れている）",
    "other_joyo_reading": "同じ漢字の別の常用読みで読んだ",
    "voicing_or_length": "連濁・促音・長音などの差のみ",
    "other": "その他の読み誤り",
}
GENERATION_FAILURE_SENTENCE_CER = 0.3


@dataclass(frozen=True)
class ErrorRow:
    key: str
    kanji: str
    category: str
    error_type: str
    tagged_text: str
    expected: list[str]
    predicted: str
    prediction_yomi: str
    prediction_text: str
    sentence_kana_cer: float


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def target_kanji(row: dict[str, Any]) -> str:
    match = TAG_PATTERN.search(str(row["tagged_text"]))
    return match.group(1) if match else ""


def strip_voicing_and_length(kana: str) -> str:
    decomposed = unicodedata.normalize("NFD", kana)
    stripped = "".join(ch for ch in decomposed if ch not in "゙゚")
    return re.sub(r"[ッーウイ]", "", unicodedata.normalize("NFC", stripped))


def build_reading_index(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    readings: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        kanji = target_kanji(row)
        for accepted in row["accepted_readings"]:
            readings[kanji].add(str(accepted["normalized"]))
    return readings


def classify(row: dict[str, Any], reading_index: dict[str, set[str]]) -> str:
    predicted = str(row["best_target_reading_normalized"] or row["mapped_target"])
    expected = [str(r["normalized"]) for r in row["accepted_readings"]]
    if (
        float(row["sentence_kana_cer"]) >= GENERATION_FAILURE_SENTENCE_CER
        or predicted == ""
    ):
        return "generation_failure"
    if predicted in reading_index.get(target_kanji(row), set()) - set(expected):
        return "other_joyo_reading"
    if any(
        strip_voicing_and_length(predicted) == strip_voicing_and_length(e)
        for e in expected
    ):
        return "voicing_or_length"
    return "other"


def collect_errors(rows: list[dict[str, Any]]) -> list[ErrorRow]:
    reading_index = build_reading_index(rows)
    errors: list[ErrorRow] = []
    for row in rows:
        if row["target_exact"]:
            continue
        errors.append(
            ErrorRow(
                key=str(row["key"]),
                kanji=target_kanji(row),
                category=str(row["reading_category"]),
                error_type=classify(row, reading_index),
                tagged_text=str(row["tagged_text"]),
                expected=[str(r["reading"]) for r in row["accepted_readings"]],
                predicted=str(row["mapped_target"]),
                prediction_yomi=str(row["prediction_yomi"]),
                prediction_text=str(row.get("prediction_text") or ""),
                sentence_kana_cer=float(row["sentence_kana_cer"]),
            )
        )
    return errors


def markdown_table(headers: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell.replace("|", "\\|") for cell in r) + " |" for r in body]
    return "\n".join(lines)


def render_report(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    errors: list[ErrorRow],
    examples_per_type: int,
    seed: int,
) -> str:
    metrics = summary["metrics"]
    out: list[str] = ["# JKYB-Parakeet 読み誤り分析", ""]
    out.append(
        markdown_table(
            ["Metric", "Value"],
            [
                ["Accuracy", f"{metrics['accuracy']['rate']:.3%}"],
                ["Relaxed Accuracy", f"{metrics['relaxed_accuracy']['rate']:.3%}"],
                ["Target Kana-CER@1", f"{metrics['target_kana_cer']['cer_at_1']:.3%}"],
                ["Sentence Kana-CER@1", f"{metrics['sentence_kana_cer']['cer_at_1']:.3%}"],
                ["Text CER@1", f"{metrics['text_cer']['cer_at_1']:.3%}"],
            ],
        )
    )

    out += ["", "## 読み分類別の誤り", ""]
    totals = Counter(str(r["reading_category"]) for r in rows)
    by_category = Counter(e.category for e in errors)
    out.append(
        markdown_table(
            ["分類", "例文数", "誤り数", "誤り率"],
            [
                [
                    CATEGORY_LABELS[c],
                    f"{totals[c]:,}",
                    f"{by_category[c]:,}",
                    f"{by_category[c] / totals[c]:.2%}",
                ]
                for c in CATEGORY_LABELS
            ],
        )
    )

    out += ["", "## 誤りタイプ別", ""]
    by_type = Counter(e.error_type for e in errors)
    cross = Counter((e.error_type, e.category) for e in errors)
    out.append(
        markdown_table(
            ["誤りタイプ", "件数", "割合"] + [CATEGORY_LABELS[c] for c in CATEGORY_LABELS],
            [
                [
                    ERROR_TYPE_LABELS[t],
                    f"{by_type[t]:,}",
                    f"{by_type[t] / max(len(errors), 1):.1%}",
                ]
                + [f"{cross[(t, c)]:,}" for c in CATEGORY_LABELS]
                for t in ERROR_TYPE_LABELS
            ],
        )
    )

    out += ["", "## よく読み誤る漢字（誤り数上位）", ""]
    kanji_errors = Counter(e.kanji for e in errors)
    kanji_totals = Counter(target_kanji(r) for r in rows)
    out.append(
        markdown_table(
            ["漢字", "誤り/例文数"],
            [[k, f"{n}/{kanji_totals[k]}"] for k, n in kanji_errors.most_common(30)],
        )
    )

    rng = random.Random(seed)
    for error_type, label in ERROR_TYPE_LABELS.items():
        pool = [e for e in errors if e.error_type == error_type]
        if not pool:
            continue
        sample = sorted(rng.sample(pool, min(examples_per_type, len(pool))), key=lambda e: e.key)
        out += ["", f"## 例: {label}（{len(pool):,}件中{len(sample)}件）", ""]
        out.append(
            markdown_table(
                ["分類", "例文（<>が対象）", "正解", "TTSの読み", "Whisper書き起こし"],
                [
                    [
                        CATEGORY_LABELS[e.category],
                        e.tagged_text,
                        " / ".join(e.expected),
                        e.predicted or "（なし）",
                        e.prediction_text,
                    ]
                    for e in sample
                ],
            )
        )
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples-per-type", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = read_jsonl(args.results_dir / "details" / "all.jsonl")
    summary = json.loads((args.results_dir / "summary.json").read_text(encoding="utf-8"))
    errors = collect_errors(rows)
    args.output.write_text(
        render_report(summary, rows, errors, args.examples_per_type, args.seed),
        encoding="utf-8",
    )
    errors_path = args.output.with_suffix(".errors.jsonl")
    with errors_path.open("w", encoding="utf-8") as handle:
        for e in errors:
            handle.write(json.dumps(e.__dict__, ensure_ascii=False) + "\n")
    print(f"errors={len(errors)} report={args.output} rows={errors_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
