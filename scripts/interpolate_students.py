#!/usr/bin/env python3
"""Linear interpolation of two students of the same scope (weight-space ensemble).

    uv run python scripts/interpolate_students.py outputs/yomi_prod/s12_cont/student \
        outputs/yomi_prod/s15_cont/student --alpha 0.5 --output outputs/yomi_prod/mix_s12_s15_a50/student

Writes ``(1 - alpha) * A + alpha * B`` for every tensor. Both students must come
from the same base and scope (e.g. S15 continues S12), so the interpolation stays
on the fine-tuning path; it can keep most of B's gains while undoing the drift
of words neither was trained on.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--alpha", type=float, required=True, help="Weight of B.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_a = json.loads((args.a / "student_config.json").read_text(encoding="utf-8"))
    config_b = json.loads((args.b / "student_config.json").read_text(encoding="utf-8"))
    if config_a != config_b:
        raise ValueError(f"Different student configs: {config_a} vs {config_b}")
    a = load_file(str(args.a / "student.safetensors"))
    b = load_file(str(args.b / "student.safetensors"))
    if a.keys() != b.keys():
        raise ValueError("The students have different tensors.")
    mixed = {
        key: ((1.0 - args.alpha) * a[key] + args.alpha * b[key]).contiguous()
        for key in a
    }
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(mixed, str(args.output / "student.safetensors"))
    shutil.copy(args.a / "student_config.json", args.output / "student_config.json")
    print(f"{len(mixed)} tensors, alpha={args.alpha} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
