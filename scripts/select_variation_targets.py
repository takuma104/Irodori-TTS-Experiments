#!/usr/bin/env python3
"""Targets for more varied sentences: words the base misreads in the corpus data.

    uv run python scripts/select_variation_targets.py --name prod --name aozora \
        --output data/yomi/targets_var.jsonl --sentences 6

S13 showed that more steps only memorize the few training sentences of a word
(Aozora: 1-4 sentences per word) and do not carry over to new sentences of the
same word. This picks every training word with at least one hard row (base
misreads, kana teacher correct) in ``outputs/yomi_<name>/teacher_mixed`` and
asks the LLM for ``--sentences`` new sentences each, like the M3 data. Only the
train split is used, so dev words stay held out.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sentences", type=int, default=6)
    args = parser.parse_args()

    hard: Counter[tuple[str, str, str]] = Counter()
    targets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for name in args.name:
        for target in read_jsonl(Path(f"data/yomi/targets_{name}.jsonl")):
            targets[(name, target["word"], target["reading"])] = target
        manifest = Path(f"outputs/yomi_{name}/teacher_mixed/manifest.jsonl")
        for row in read_jsonl(manifest):
            if row["role"] == "target" and row.get("split", "train") == "train":
                _, word, reading, _ = row["key"].split("_")
                hard[(name, word, reading)] += 1

    written: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    with args.output.open("w", encoding="utf-8") as out:
        for ident in sorted(hard):
            name, word, reading = ident
            if (word, reading) in seen:
                continue
            seen.add((word, reading))
            target = dict(targets[ident])
            target |= {
                "role": "target",
                "split": "train",
                "n_sentences": args.sentences,
                "source_dataset": name,
                "hard_rows": hard[ident],
            }
            out.write(json.dumps(target, ensure_ascii=False) + "\n")
            written[name] += 1
    print(f"targets={sum(written.values())} {dict(written)} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
