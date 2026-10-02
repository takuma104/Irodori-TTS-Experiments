#!/usr/bin/env python3
"""Paired comparison of two ``jkyb-eval tts`` result directories.

    uv run python scripts/compare_jkyb_runs.py BASE_RESULTS CAND_RESULTS \
        --base-errors outputs/<base>/error_analysis.errors.jsonl \
        --output outputs/<cand>/compare.md

Only keys present in both runs are compared (a subset run can be compared with
a full baseline), e.g. the teacher checkpoint (base) against a distilled student
(cand). Reports accuracy on the shared keys, the paired 2x2 table with
an exact McNemar test, and how the baseline errors changed, broken down by
reading category, target tokenization, and (with ``--base-errors``) the baseline
error type.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from analyze_jkyb_errors import (
    CATEGORY_LABELS,
    ERROR_TYPE_LABELS,
    SWAP_DIRECTION_LABELS,
    TOKEN_TYPE_LABELS,
    load_tokenizer,
    markdown_table,
    read_jsonl,
    token_type,
)


def mcnemar_exact_p(fixed: int, broken: int) -> float:
    """Two-sided exact McNemar p-value (binomial test on discordant pairs)."""
    n = fixed + broken
    if n == 0:
        return 1.0
    k = min(fixed, broken)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_row(
    label: str, keys: Iterable[str], base: dict[str, bool], cand: dict[str, bool]
) -> list[str]:
    keys = list(keys)
    n = len(keys)
    base_ok = sum(base[k] for k in keys)
    cand_ok = sum(cand[k] for k in keys)
    fixed = sum(not base[k] and cand[k] for k in keys)
    broken = sum(base[k] and not cand[k] for k in keys)
    return [
        label,
        f"{n:,}",
        f"{base_ok / max(n, 1):.2%}",
        f"{cand_ok / max(n, 1):.2%}",
        f"{(cand_ok - base_ok) / max(n, 1) * 100:+.2f}pt",
        f"{fixed:,}",
        f"{broken:,}",
        f"{mcnemar_exact_p(fixed, broken):.3g}",
    ]


HEADERS = [
    "区分",
    "例文数",
    "ベース",
    "比較対象",
    "差",
    "直った",
    "壊れた",
    "McNemar p",
]


def grouped_rows(
    keys: list[str],
    group_of: Callable[[str], str | None],
    labels: dict[str, str],
    base: dict[str, bool],
    cand: dict[str, bool],
) -> list[list[str]]:
    groups: dict[str, list[str]] = {g: [] for g in labels}
    for key in keys:
        group = group_of(key)
        if group in groups:
            groups[group].append(key)
    return [paired_row(labels[g], groups[g], base, cand) for g in labels if groups[g]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_results", type=Path)
    parser.add_argument("cand_results", type=Path)
    parser.add_argument("--base-errors", type=Path, default=None)
    parser.add_argument("--base-label", default="base")
    parser.add_argument("--cand-label", default="cand")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    base_rows = {
        r["key"]: r for r in read_jsonl(args.base_results / "details" / "all.jsonl")
    }
    cand_rows = {
        r["key"]: r for r in read_jsonl(args.cand_results / "details" / "all.jsonl")
    }
    keys = sorted(base_rows.keys() & cand_rows.keys())
    base = {k: bool(base_rows[k]["target_exact"]) for k in keys}
    cand = {k: bool(cand_rows[k]["target_exact"]) for k in keys}
    tokenizer = load_tokenizer()
    token_types = {k: token_type(base_rows[k], tokenizer) for k in keys}

    out: list[str] = [
        f"# JKYB 比較: {args.base_label} vs {args.cand_label}",
        "",
        f"- ベース: `{args.base_results}`",
        f"- 比較対象: `{args.cand_results}`",
        "- 「直った」はベースで不正解・比較対象で正解、「壊れた」はその逆。p は exact McNemar 検定。",
        "",
        "## 全体",
        "",
        markdown_table(HEADERS, [paired_row("全体", keys, base, cand)]),
        "",
        "## 読み分類別",
        "",
        markdown_table(
            HEADERS,
            grouped_rows(
                keys,
                lambda k: str(base_rows[k]["reading_category"]),
                CATEGORY_LABELS,
                base,
                cand,
            ),
        ),
        "",
        "## 対象漢字のトークン化別",
        "",
        markdown_table(
            HEADERS, grouped_rows(keys, token_types.get, TOKEN_TYPE_LABELS, base, cand)
        ),
    ]

    if args.base_errors is not None:
        base_errors: dict[str, dict[str, Any]] = {
            e["key"]: e for e in read_jsonl(args.base_errors) if e["key"] in base
        }
        error_keys = sorted(base_errors)
        fixed = Counter(base_errors[k]["error_type"] for k in error_keys if cand[k])
        total = Counter(base_errors[k]["error_type"] for k in error_keys)
        out += [
            "",
            "## ベースの誤りタイプ別の変化",
            "",
            markdown_table(
                ["ベースの誤りタイプ", "件数", "比較対象で正解", "割合"],
                [
                    [
                        label,
                        f"{total[t]:,}",
                        f"{fixed[t]:,}",
                        f"{fixed[t] / total[t]:.1%}",
                    ]
                    for t, label in ERROR_TYPE_LABELS.items()
                    if total[t]
                ]
                + [
                    [
                        "合計",
                        f"{sum(total.values()):,}",
                        f"{sum(fixed.values()):,}",
                        f"{sum(fixed.values()) / max(sum(total.values()), 1):.1%}",
                    ]
                ],
            ),
        ]
        swaps = [
            k
            for k in error_keys
            if base_errors[k]["error_type"] == "other_joyo_reading"
        ]
        swap_total = Counter(base_errors[k]["swap_direction"] for k in swaps)
        swap_fixed = Counter(base_errors[k]["swap_direction"] for k in swaps if cand[k])
        out += [
            "",
            "## 音訓の取り違え（ベース）の向き別の変化",
            "",
            markdown_table(
                ["正解 → ベースの読み", "件数", "比較対象で正解", "割合"],
                [
                    [
                        label,
                        f"{swap_total[d]:,}",
                        f"{swap_fixed[d]:,}",
                        f"{swap_fixed[d] / swap_total[d]:.1%}",
                    ]
                    for d, label in SWAP_DIRECTION_LABELS.items()
                    if swap_total[d]
                ],
            ),
        ]

    text = "\n".join(out) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
