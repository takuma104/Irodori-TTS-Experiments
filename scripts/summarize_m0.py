#!/usr/bin/env python3
"""Joint summary of D1 (kana oracle) and D2 (seed variance) on the M0 subset.

    uv run python scripts/summarize_m0.py --output outputs/m0_diagnostics/summary.md

Per subset row, correctness is read from the bf16 seed-0 baseline and from the
seed-1, seed-2, hiragana, and katakana runs. Baseline errors are split into
systematic ones (wrong with all three seeds) and stochastic ones, and the kana
fix rate is compared with the rate at which re-sampling alone fixes them.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from analyze_jkyb_errors import (
    ERROR_TYPE_LABELS,
    TOKEN_TYPE_LABELS,
    markdown_table,
    read_jsonl,
)

RUNS = ["seed1", "seed2", "hira", "kata"]


def correctness(results_dir: Path) -> dict[str, bool]:
    return {
        r["key"]: bool(r["target_exact"])
        for r in read_jsonl(results_dir / "details" / "all.jsonl")
    }


def rate(numerator: int, denominator: int) -> str:
    return f"{numerator:,}/{denominator:,} ({numerator / max(denominator, 1):.1%})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("outputs/irodori-v4.1-small_jvs001_bf16")
    )
    parser.add_argument("--m0-dir", type=Path, default=Path("outputs/m0_diagnostics"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = correctness(args.base / "results")
    runs = {name: correctness(args.m0_dir / name / "results") for name in RUNS}
    keys = sorted(set.intersection(*(set(r) for r in runs.values())))
    errors = {
        e["key"]: e for e in read_jsonl(args.base / "error_analysis.errors.jsonl")
    }
    wrong = [k for k in keys if not base[k]]
    right = [k for k in keys if base[k]]
    systematic = [k for k in wrong if not runs["seed1"][k] and not runs["seed2"][k]]
    stochastic = [k for k in wrong if k not in set(systematic)]

    out = ["# M0: かなオラクル（D1）とシード揺らぎ（D2）", ""]
    out.append(
        f"部分集合: bf16 seed 0 の誤り {len(wrong):,} 件 + 正解から無作為に {len(right):,} 件。"
        "系統的な誤り = seed 0/1/2 のすべてで誤り。"
    )
    out += ["", "## 全体", ""]
    out.append(
        markdown_table(
            ["", "seed 1", "seed 2", "ひらがな", "カタカナ"],
            [
                ["ベースの誤りが正解に"]
                + [rate(sum(runs[n][k] for k in wrong), len(wrong)) for n in RUNS],
                ["うち系統的な誤り"]
                + [
                    rate(sum(runs[n][k] for k in systematic), len(systematic))
                    for n in RUNS
                ],
                ["うち確率的な誤り"]
                + [
                    rate(sum(runs[n][k] for k in stochastic), len(stochastic))
                    for n in RUNS
                ],
                ["ベースの正解が誤りに"]
                + [rate(sum(not runs[n][k] for k in right), len(right)) for n in RUNS],
            ],
        )
    )

    out += ["", "## ベースの誤りタイプ別", ""]
    by_type = Counter(errors[k]["error_type"] for k in wrong)
    sys_by_type = Counter(errors[k]["error_type"] for k in systematic)
    rows = []
    for t, label in ERROR_TYPE_LABELS.items():
        ks = [k for k in wrong if errors[k]["error_type"] == t]
        sys_ks = [k for k in systematic if errors[k]["error_type"] == t]
        if not ks:
            continue
        rows.append(
            [
                label,
                f"{by_type[t]:,}",
                rate(sys_by_type[t], by_type[t]),
                rate(sum(runs["seed1"][k] for k in ks), len(ks)),
                rate(sum(runs["hira"][k] for k in sys_ks), len(sys_ks)),
                rate(sum(runs["kata"][k] for k in sys_ks), len(sys_ks)),
            ]
        )
    out.append(
        markdown_table(
            [
                "誤りタイプ",
                "件数",
                "系統的",
                "seed 1 で正解",
                "系統的のうち ひらがなで正解",
                "系統的のうち カタカナで正解",
            ],
            rows,
        )
    )

    out += ["", "## 対象漢字のトークン化別（系統的な誤り）", ""]
    rows = []
    for t, label in TOKEN_TYPE_LABELS.items():
        ks = [k for k in systematic if errors[k]["token_type"] == t]
        if not ks:
            continue
        rows.append(
            [
                label,
                f"{len(ks):,}",
                rate(sum(runs["hira"][k] for k in ks), len(ks)),
                rate(sum(runs["kata"][k] for k in ks), len(ks)),
            ]
        )
    out.append(
        markdown_table(
            ["トークン化", "系統的な誤り", "ひらがなで正解", "カタカナで正解"], rows
        )
    )

    text = "\n".join(out) + "\n"
    args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
