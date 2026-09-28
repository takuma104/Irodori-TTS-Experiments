#!/usr/bin/env python3
"""Collect general sentences (bucket H) from the JVS transcripts (plan §4.2, §6.3).

    uv run python scripts/make_general_rows.py --output data/yomi/general_rows.jsonl

Unique sentences from every speaker's ``parallel100`` and ``nonpara30``
transcripts (JSUT-derived). About ``--regress-count`` sentences, chosen by hash,
get ``split: regress`` and are reserved for the regression check; the rest are
``train``/``dev`` rows whose teacher is the unchanged kanji text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jvs-dir", type=Path, default=Path("data/jvs_ver1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regress-count", type=int, default=500)
    parser.add_argument("--dev-ratio", type=float, default=0.05)
    args = parser.parse_args()

    sentences: dict[str, str] = {}
    for path in sorted(args.jvs_dir.glob("jvs*/*/transcripts_utf8.txt")):
        if path.parent.name not in ("parallel100", "nonpara30"):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" in line:
                utt_id, text = line.split(":", 1)
                sentences.setdefault(utt_id.strip(), text.strip())

    def rank(utt_id: str) -> int:
        return int.from_bytes(hashlib.md5(utt_id.encode()).digest()[:4], "little")

    ordered = sorted(sentences, key=rank)
    regress = set(ordered[: args.regress_count])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "dev": 0, "regress": 0}
    with args.output.open("w", encoding="utf-8") as handle:
        for utt_id in sorted(sentences):
            if utt_id in regress:
                split = "regress"
            elif (rank(utt_id) >> 8) % 1000 < args.dev_ratio * 1000:
                split = "dev"
            else:
                split = "train"
            counts[split] += 1
            text = sentences[utt_id]
            row = {
                "key": f"H_{utt_id}",
                "text": text,
                "kana_text": text,
                "role": "general",
                "bucket": "H",
                "split": split,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
