#!/usr/bin/env python3
"""Summarize target-reading errors from a ``jkyb-eval tts`` result directory.

    uv run python scripts/analyze_jkyb_errors.py outputs/<run>/results \
        --output outputs/<run>/error_analysis.md

Each incorrect row (``target_exact == false``) is assigned one error type:

- ``marginal``: matches a marginal reading (correct under Relaxed Accuracy).
- ``generation_failure``: the context around the target is broken or the target
  span is empty.

Rows whose Whisper (kanji-kana) transcript still contains the original word
around the target are flagged as ``text_asr_agrees``: the audio may actually be
correct and only the kana ASR misheard it (or Whisper's LM auto-corrected).
- ``voicing``: differs only in voicing (e.g. missing/extra rendaku).
- ``sokuon_or_length``: differs only in sokuon (促音化) or vowel length.
- ``other_joyo_reading``: the target was read with another Joyo reading of the
  same kanji (e.g. on/kun confusion).
- ``other``: any other misreading.

Every row is also classified by how the Irodori (ModernBERT-ja) tokenizer splits
the target kanji (``TOKEN_TYPE_LABELS``), and ``other_joyo_reading`` errors get
the direction of the swap (e.g. on→kun).
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

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

TAG_PATTERN = re.compile(r"<([^>]+)>")
KANJI_PATTERN = re.compile(r"[㐀-鿿々]")
CATEGORY_LABELS = {
    "on_yomi": "音読み",
    "kun_yomi": "訓読み",
    "joyo_appendix_reading": "付表の語",
}
ERROR_TYPE_LABELS = {
    "marginal": "許容読み（marginal）に一致（Relaxedでは正解）",
    "generation_failure": "生成破綻（文全体が崩れている）",
    "voicing": "清濁（連濁の有無など）の差のみ",
    "sokuon_or_length": "促音化・長音の差のみ",
    "other_joyo_reading": "同じ漢字の別の常用読みで読んだ",
    "other": "その他の読み誤り",
}
TOKEN_TYPE_LABELS = {
    "split_compound": "1文字トークンで前後も漢字（熟語が分割されている）",
    "single_kanji_word": "1文字トークンで前後は仮名・記号（単漢字の語）",
    "multi_char_token": "複数文字のトークン（語が語彙にある）",
}
SWAP_DIRECTION_LABELS = {
    "on→kun": "音 → 訓",
    "on→on": "音 → 別の音",
    "kun→on": "訓 → 音",
    "kun→kun": "訓 → 別の訓",
    "other": "付表の語など",
}
TOKENIZER_REPO = "Aratako/Irodori-TTS-v4.1-Small"
TOKENIZER_FILENAME = "tokenizer/tokenizer.json"
GENERATION_FAILURE_CONTEXT_CER = 0.3
VOWEL_ROWS = {
    "ア": "アカサタナハマヤラワガザダバパャ",
    "イ": "イキシチニヒミリギジヂビピ",
    "ウ": "ウクスツヌフムユルグズヅブプュ",
    "エ": "エケセテネヘメレゲゼデベペ",
    "オ": "オコソトノホモヨロヲゴゾドボポョ",
}


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
    target_word: str
    text_asr_agrees: bool
    token_type: str
    swap_direction: str | None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def target_kanji(row: dict[str, Any]) -> str:
    match = TAG_PATTERN.search(str(row["tagged_text"]))
    return match.group(1) if match else ""


def strip_voicing(kana: str) -> str:
    decomposed = unicodedata.normalize("NFD", kana)
    stripped = "".join(ch for ch in decomposed if ch not in "゙゚")
    return unicodedata.normalize("NFC", stripped)


def _vowel(kana: str) -> str:
    for vowel, row in VOWEL_ROWS.items():
        if kana in row:
            return vowel
    return ""


def strip_sound_change(kana: str) -> str:
    """Drop voicing, sokuon, stem-final ク/キ/ツ/チ and vowel lengthening."""
    text = re.sub(r"[クキツチ]$", "", strip_voicing(kana).replace("ッ", ""))
    out: list[str] = []
    for ch in text:
        if out and ch in "アイウエオ" and _vowel(out[-1]) == ch:
            continue
        out.append(ch)
    return "".join(out)


def target_word(row: dict[str, Any]) -> str:
    """Kanji run containing the target (plus one okurigana for kun readings)."""
    tagged = str(row["tagged_text"])
    start = tagged.index("<")
    end = tagged.index(">") - 1
    text = tagged.replace("<", "").replace(">", "")
    while start > 0 and KANJI_PATTERN.match(text[start - 1]):
        start -= 1
    while end < len(text) and KANJI_PATTERN.match(text[end]):
        end += 1
    is_kun = row["reading_category"] == "kun_yomi"
    if is_kun and end < len(text) and "ぁ" <= text[end] <= "ゟ":
        end += 1
    return text[start:end]


def load_tokenizer() -> Tokenizer:
    return Tokenizer.from_file(hf_hub_download(TOKENIZER_REPO, TOKENIZER_FILENAME))


def token_type(row: dict[str, Any], tokenizer: Tokenizer) -> str:
    """How the tokenizer splits the target kanji (see ``TOKEN_TYPE_LABELS``)."""
    text = str(row["text"])
    start = str(row["tagged_text"]).index("<")
    offsets = tokenizer.encode(text, add_special_tokens=False).offsets
    index = next(i for i, (s, e) in enumerate(offsets) if s <= start < e)
    begin, end = offsets[index]
    if end - begin > 1:
        return "multi_char_token"
    prev_kanji = index > 0 and bool(
        KANJI_PATTERN.match(text[offsets[index - 1][1] - 1])
    )
    next_kanji = index + 1 < len(offsets) and bool(
        KANJI_PATTERN.match(text[offsets[index + 1][0]])
    )
    return "split_compound" if prev_kanji or next_kanji else "single_kanji_word"


def build_category_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, set[str]]]:
    """kanji -> reading category -> readings (sound changes stripped)."""
    index: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        kanji = target_kanji(row)
        for accepted in row["accepted_readings"]:
            index[kanji][str(row["reading_category"])].add(
                strip_sound_change(str(accepted["normalized"]))
            )
    return index


def swap_direction(
    row: dict[str, Any], category_index: dict[str, dict[str, set[str]]]
) -> str:
    """Direction of an other-Joyo-reading error, e.g. ``on→kun``."""
    predicted = strip_sound_change(str(row["mapped_target"]))
    readings = category_index[target_kanji(row)]
    short = {"on_yomi": "on", "kun_yomi": "kun"}
    expected = short.get(str(row["reading_category"]))
    if expected is None:
        return "other"
    if predicted in readings["on_yomi"]:
        return f"{expected}→on"
    if predicted in readings["kun_yomi"]:
        return f"{expected}→kun"
    return "other"


def build_reading_index(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    readings: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        kanji = target_kanji(row)
        for accepted in row["accepted_readings"]:
            readings[kanji].add(str(accepted["normalized"]))
    return readings


def classify(row: dict[str, Any], reading_index: dict[str, set[str]]) -> str:
    # ``mapped_target`` is the normalized span aligned from the ASR kana output.
    predicted = str(row["mapped_target"])
    expected = [str(r["normalized"]) for r in row["accepted_readings"]]
    if row["relaxed_target_exact"]:
        return "marginal"
    if (
        float(row["alignment_context_error_rate"]) >= GENERATION_FAILURE_CONTEXT_CER
        or predicted == ""
    ):
        return "generation_failure"
    if any(strip_voicing(predicted) == strip_voicing(e) for e in expected):
        return "voicing"
    if any(strip_sound_change(predicted) == strip_sound_change(e) for e in expected):
        return "sokuon_or_length"
    others = reading_index.get(target_kanji(row), set()) - set(expected)
    if any(strip_sound_change(predicted) == strip_sound_change(o) for o in others):
        return "other_joyo_reading"
    return "other"


def collect_errors(
    rows: list[dict[str, Any]], token_types: dict[str, str]
) -> list[ErrorRow]:
    reading_index = build_reading_index(rows)
    category_index = build_category_index(rows)
    errors: list[ErrorRow] = []
    for row in rows:
        if row["target_exact"]:
            continue
        word = target_word(row)
        prediction_text = str(row.get("prediction_text") or "")
        error_type = classify(row, reading_index)
        errors.append(
            ErrorRow(
                key=str(row["key"]),
                kanji=target_kanji(row),
                category=str(row["reading_category"]),
                error_type=error_type,
                tagged_text=str(row["tagged_text"]),
                expected=[
                    str(r["reading"])
                    + ("（許容）" if r["category"] == "marginal" else "")
                    for r in row["accepted_readings"]
                ],
                predicted=str(row["mapped_target"]),
                prediction_yomi=str(row["prediction_yomi"]),
                prediction_text=prediction_text,
                sentence_kana_cer=float(row["sentence_kana_cer"]),
                target_word=word,
                text_asr_agrees=word in prediction_text,
                token_type=token_types[str(row["key"])],
                swap_direction=(
                    swap_direction(row, category_index)
                    if error_type == "other_joyo_reading"
                    else None
                ),
            )
        )
    return errors


def markdown_table(headers: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += [
        "| " + " | ".join(cell.replace("|", "\\|") for cell in r) + " |" for r in body
    ]
    return "\n".join(lines)


def render_report(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    errors: list[ErrorRow],
    token_types: dict[str, str],
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
                [
                    "Sentence Kana-CER@1",
                    f"{metrics['sentence_kana_cer']['cer_at_1']:.3%}",
                ],
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
    agrees = Counter(e.error_type for e in errors if e.text_asr_agrees)
    out.append(
        markdown_table(
            ["誤りタイプ", "件数", "割合"]
            + [CATEGORY_LABELS[c] for c in CATEGORY_LABELS]
            + ["うちWhisper書き起こしは原文どおり"],
            [
                [
                    ERROR_TYPE_LABELS[t],
                    f"{by_type[t]:,}",
                    f"{by_type[t] / max(len(errors), 1):.1%}",
                ]
                + [f"{cross[(t, c)]:,}" for c in CATEGORY_LABELS]
                + [f"{agrees[t]:,}"]
                for t in ERROR_TYPE_LABELS
            ]
            + [
                ["合計", f"{len(errors):,}", "100%"]
                + [
                    f"{sum(cross[(t, c)] for t in ERROR_TYPE_LABELS):,}"
                    for c in CATEGORY_LABELS
                ]
                + [f"{sum(agrees.values()):,}"]
            ],
        )
    )
    out += [
        "",
        (
            "「Whisper書き起こしは原文どおり」は、whisper-large-v3-turbo の漢字仮名交じり書き起こしに"
            "対象語（対象漢字を含む漢字列。訓読みは送り仮名1字を含む）がそのまま含まれていた件数。"
            "カナASR（kana-whisper）側の聞き誤りの可能性がある一方、Whisperの言語モデルが文脈から"
            "補正しただけの場合も含む。"
        ),
    ]

    out += ["", "## 対象漢字のトークン化別", ""]
    type_totals = Counter(token_types.values())
    type_errors = Counter(e.token_type for e in errors)
    type_swaps = Counter(
        e.token_type for e in errors if e.error_type == "other_joyo_reading"
    )
    type_other = Counter(e.token_type for e in errors if e.error_type == "other")
    out.append(
        markdown_table(
            [
                "トークン化",
                "例文数",
                "誤り",
                "誤り率",
                "うち音訓の取り違え",
                "うちその他",
            ],
            [
                [
                    label,
                    f"{type_totals[t]:,}",
                    f"{type_errors[t]:,}",
                    f"{type_errors[t] / max(type_totals[t], 1):.2%}",
                    f"{type_swaps[t]:,}",
                    f"{type_other[t]:,}",
                ]
                for t, label in TOKEN_TYPE_LABELS.items()
            ],
        )
    )

    out += ["", "## 同じ漢字の別の常用読みで読んだ誤りの向き", ""]
    directions = Counter(
        e.swap_direction for e in errors if e.error_type == "other_joyo_reading"
    )
    out.append(
        markdown_table(
            ["正解 → TTSの読み", "件数"],
            [
                [label, f"{directions[d]:,}"]
                for d, label in SWAP_DIRECTION_LABELS.items()
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
        sample = sorted(
            rng.sample(pool, min(examples_per_type, len(pool))), key=lambda e: e.key
        )
        out += ["", f"## 例: {label}（{len(pool):,}件中{len(sample)}件）", ""]
        out.append(
            markdown_table(
                [
                    "分類",
                    "例文（<>が対象）",
                    "正解",
                    "TTSの読み",
                    "Whisper書き起こし",
                    "W一致",
                ],
                [
                    [
                        CATEGORY_LABELS[e.category],
                        e.tagged_text,
                        " / ".join(e.expected),
                        e.predicted or "（なし）",
                        e.prediction_text,
                        "✓" if e.text_asr_agrees else "",
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
    summary = json.loads(
        (args.results_dir / "summary.json").read_text(encoding="utf-8")
    )
    tokenizer = load_tokenizer()
    token_types = {str(row["key"]): token_type(row, tokenizer) for row in rows}
    errors = collect_errors(rows, token_types)
    args.output.write_text(
        render_report(
            summary, rows, errors, token_types, args.examples_per_type, args.seed
        ),
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
