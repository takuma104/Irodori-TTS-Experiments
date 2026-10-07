#!/usr/bin/env python3
"""Figures for the technical report (English and Japanese labels).

    uv run --with matplotlib python docs/report/make_figures.py --font-dir <Noto JP fonts>

Numbers are the student-path evaluations in docs/reports/yomi_production.md
(mixed precision, reference voice jvs001, seed 0).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

STAGES = ["S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "Avg"]
AOZORA = [62.73, 62.53, 63.40, 65.00, 65.87, 65.87, 69.00, 69.10, 67.77]
PILOT = [82.52, 82.63, 81.85, 82.07, 82.85, 81.85, 80.85, 80.29, 82.07]
BASE_AOZORA = 62.50
BASE_PILOT = 80.73
RELEASES = {"S10": "v1", "S12": "v2", "Avg": "v3"}

LABELS = {
    "en": {
        "aozora": "Aozora ruby (held-out works)",
        "pilot": "Pilot dev (untrained words)",
        "acc": "Target reading accuracy (%)",
        "base": "base",
        "phases": [
            (0.5, 2.5, "corpus\ndata"),
            (2.5, 4.5, "window\nweight"),
            (4.5, 5.5, "more\nsteps"),
            (5.5, 7.5, "variety\n+contrast"),
        ],
    },
    "ja": {
        "aozora": "青空文庫ルビ（評価用の作品）",
        "pilot": "pilot dev（学習していない語）",
        "acc": "対象語の正答率（%）",
        "base": "ベース",
        "phases": [
            (0.5, 2.5, "コーパス\nデータ"),
            (2.5, 4.5, "区間の\n重み付け"),
            (4.5, 5.5, "ステップ\n追加"),
            (5.5, 7.5, "文の多様化\n+対照"),
        ],
    },
}


def plot(lang: str, output: Path) -> None:
    labels = LABELS[lang]
    x = list(range(len(STAGES)))
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.3), sharex=True)
    for ax, values, base, title, color in (
        (axes[0], AOZORA, BASE_AOZORA, labels["aozora"], "#1f77b4"),
        (axes[1], PILOT, BASE_PILOT, labels["pilot"], "#d62728"),
    ):
        for k, (lo, hi, _) in enumerate(labels["phases"]):
            ax.axvspan(lo, hi, color="#ececec" if k % 2 == 0 else "#f7f7f7", zorder=0)
        ax.axhline(base, color="gray", linestyle="--", linewidth=0.8, zorder=1)
        ax.text(len(STAGES) - 0.55, base, labels["base"], fontsize=6, color="gray",
                va="bottom", ha="right")
        ax.plot(x, values, marker="o", markersize=3, color=color, linewidth=1.2, zorder=2)
        for i, stage in enumerate(STAGES):
            if stage in RELEASES:
                ax.annotate(RELEASES[stage], (i, values[i]), textcoords="offset points",
                            xytext=(0, 5), ha="center", fontsize=6.5, fontweight="bold")
        ax.set_title(title, fontsize=7.5, pad=3)
        ax.tick_params(labelsize=6.5)
        ax.set_ylabel(labels["acc"], fontsize=6.5)
    lo_y, hi_y = axes[0].get_ylim()
    axes[0].set_ylim(lo_y, hi_y + 1.6)
    lo_p, hi_p = axes[1].get_ylim()
    axes[1].set_ylim(lo_p, hi_p + 0.35)
    for lo, hi, text in labels["phases"]:
        axes[0].text((lo + hi) / 2, hi_y + 1.3, text, fontsize=5.5, ha="center", va="top",
                     color="#555555", linespacing=0.95)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(STAGES, fontsize=6.5)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(output)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    for path in args.font_dir.glob("NotoSansJP-*.otf"):
        font_manager.fontManager.addfont(str(path))
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.family"] = "DejaVu Sans"
    plot("en", args.output_dir / "fig_progress_en.pdf")
    # Noto Sans JP is CFF-based; embed it as Type 3 (Type 42 needs TrueType outlines).
    plt.rcParams["pdf.fonttype"] = 3
    plt.rcParams["font.family"] = "Noto Sans JP"
    plot("ja", args.output_dir / "fig_progress_ja.pdf")
    print("written fig_progress_{en,ja}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
