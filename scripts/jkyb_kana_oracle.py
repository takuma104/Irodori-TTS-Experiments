#!/usr/bin/env python3
"""Build the kana-oracle subset of JKYB-Parakeet (plan D1).

    uv run python scripts/jkyb_kana_oracle.py outputs/<run>/results \
        --output outputs/d1_kana_oracle/subset.jsonl

Selects every row the model got wrong plus an equal number of randomly chosen
correct rows, and adds texts in which the kanji word containing the target is
replaced by its reading:

- ``text_hira`` / ``text_kata``: the whole kanji run around the target
  (e.g. 従容 -> しょうよう), with the run's reading aligned from ``yomi``. When the
  alignment fails (about 0.3% of rows, mostly digits next to the target), only
  the target kanji is replaced and ``kana_word`` is null.
- ``text_char_hira``: only the target kanji is replaced (交ぜ書き).

The original fields are kept unchanged, so the file can be passed to
``jkyb-eval tts --dataset`` as is; scoring uses ``yomi``/``tagged_yomi``, and
``text`` only feeds Text CER.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

DATASET_REPO = "Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet"
DATASET_FILENAME = "data/common_kanji_source.jsonl"
KANJI = re.compile(r"[㐀-鿿々〆ヶ]")
KANA_PARTICLE_VARIANTS = {"は": "[ハワ]", "へ": "[ヘエ]", "を": "[ヲオ]"}
PUNCTUATION = set("、。？！「」『』・，．?!")


def to_katakana(text: str) -> str:
    return "".join(chr(ord(ch) + 0x60) if "ぁ" <= ch <= "ゖ" else ch for ch in text)


def to_hiragana(text: str) -> str:
    return "".join(chr(ord(ch) - 0x60) if "ァ" <= ch <= "ヶ" else ch for ch in text)


@dataclass(frozen=True)
class Segment:
    kind: str  # "open", "close", "kana", "punct", "run"
    text: str


def segment(tagged_text: str) -> list[Segment]:
    """Split tagged text into tag markers, kana, punctuation, and other runs."""
    segments: list[Segment] = []
    for ch in tagged_text:
        if ch == "<":
            segments.append(Segment("open", ch))
        elif ch == ">":
            segments.append(Segment("close", ch))
        elif ("ぁ" <= ch <= "ゖ" or "ァ" <= ch <= "ヺ" or ch == "ー") and ch != "ヶ":
            segments.append(Segment("kana", ch))
        elif ch in PUNCTUATION:
            segments.append(Segment("punct", ch))
        elif segments and segments[-1].kind == "run":
            segments[-1] = Segment("run", segments[-1].text + ch)
        else:
            segments.append(Segment("run", ch))
    return segments


def align_runs(tagged_text: str, tagged_yomi: str) -> list[tuple[Segment, str]] | None:
    """Pair each segment with its reading, or None when the alignment fails."""
    segments = segment(tagged_text)
    parts: list[str] = []
    for seg in segments:
        if seg.kind in ("open", "close"):
            parts.append(re.escape(seg.text))
        elif seg.kind == "kana":
            parts.append(
                "("
                + KANA_PARTICLE_VARIANTS.get(seg.text, re.escape(to_katakana(seg.text)))
                + ")"
            )
        elif seg.kind == "punct":
            parts.append("(" + re.escape(seg.text) + "?)")
        else:
            parts.append("(.+?)")
    match = re.fullmatch("".join(parts), tagged_yomi)
    if match is None:
        return None
    groups = iter(match.groups())
    return [
        (seg, seg.text if seg.kind in ("open", "close") else next(groups))
        for seg in segments
    ]


def target_word(aligned: list[tuple[Segment, str]]) -> tuple[int, int] | None:
    """Segment index range [start, end) of the kanji word containing the target."""
    open_index = next(i for i, (seg, _) in enumerate(aligned) if seg.kind == "open")
    close_index = next(i for i, (seg, _) in enumerate(aligned) if seg.kind == "close")
    start = open_index
    if start > 0 and aligned[start - 1][0].kind == "run":
        if not all(KANJI.match(ch) for ch in aligned[start - 1][0].text):
            return None
        start -= 1
    end = close_index + 1
    if end < len(aligned) and aligned[end][0].kind == "run":
        if not all(KANJI.match(ch) for ch in aligned[end][0].text):
            return None
        end += 1
    inside = aligned[open_index + 1 : close_index]
    if not all(seg.kind == "run" and KANJI.match(seg.text) for seg, _ in inside):
        return None
    return start, end


@dataclass(frozen=True)
class WordSubstitution:
    before: str
    word: str
    reading: str  # katakana
    after: str

    def text(self, *, hiragana: bool) -> str:
        reading = to_hiragana(self.reading) if hiragana else self.reading
        return self.before + reading + self.after


def substitute_word(row: dict[str, Any]) -> WordSubstitution | None:
    """Split the text around the kanji word containing the target, or None if not aligned."""
    aligned = align_runs(row["tagged_text"], row["tagged_yomi"])
    if aligned is None:
        return None
    span = target_word(aligned)
    if span is None:
        return None
    start, end = span
    word = "".join(seg.text for seg, _ in aligned[start:end] if seg.kind == "run")
    reading = "".join(y for seg, y in aligned[start:end] if seg.kind == "run")
    before = "".join(
        seg.text for seg, _ in aligned[:start] if seg.kind not in ("open", "close")
    )
    after = "".join(
        seg.text for seg, _ in aligned[end:] if seg.kind not in ("open", "close")
    )
    return WordSubstitution(before=before, word=word, reading=reading, after=after)


def substitute_char(row: dict[str, Any], *, hiragana: bool) -> str:
    """Replace only the tagged kanji with its reading."""
    reading = re.search(r"<([^>]+)>", row["tagged_yomi"]).group(1)
    return re.sub(
        r"<[^>]+>", to_hiragana(reading) if hiragana else reading, row["tagged_text"]
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="jkyb-eval tts --output-dir")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument(
        "--num-correct",
        type=int,
        default=None,
        help="Default: same as the number of errors.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset = args.dataset or Path(
        hf_hub_download(
            repo_id=DATASET_REPO, repo_type="dataset", filename=DATASET_FILENAME
        )
    )
    rows = {row["key"]: row for row in read_jsonl(dataset)}
    details = read_jsonl(args.results_dir / "details" / "all.jsonl")
    wrong = sorted(d["key"] for d in details if not d["target_exact"])
    correct = sorted(d["key"] for d in details if d["target_exact"])
    num_correct = len(wrong) if args.num_correct is None else args.num_correct
    sampled = sorted(random.Random(args.seed).sample(correct, num_correct))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for key, baseline_correct in [(k, False) for k in wrong] + [
            (k, True) for k in sampled
        ]:
            row = dict(rows[key])
            substituted = substitute_word(row)
            if substituted is None:
                # Fall back to replacing only the target kanji.
                failed += 1
                row["kana_word"] = None
                row["text_hira"] = substitute_char(row, hiragana=True)
                row["text_kata"] = substitute_char(row, hiragana=False)
            else:
                row["kana_word"] = {
                    "word": substituted.word,
                    "reading": substituted.reading,
                }
                row["text_hira"] = substituted.text(hiragana=True)
                row["text_kata"] = substituted.text(hiragana=False)
            row["text_char_hira"] = substitute_char(row, hiragana=True)
            row["baseline_correct"] = baseline_correct
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    total = len(wrong) + num_correct
    print(f"rows={total} wrong={len(wrong)} correct={num_correct} unaligned={failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
