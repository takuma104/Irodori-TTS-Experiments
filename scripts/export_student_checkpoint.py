#!/usr/bin/env python3
"""Merge a trained student text encoder into a stock Irodori-TTS checkpoint.

    uv run python scripts/export_student_checkpoint.py outputs/yomi_prod/s10_cont/part2/student \
        --output-dir ../Irodori-TTS-v4.1-Small-Yomi

The result is a regular ``model.safetensors`` (same keys, dtype and metadata as
the base checkpoint) plus the bundled tokenizer, so stock Irodori-TTS loads it
with ``--hf-checkpoint`` / ``--checkpoint`` and no extra code.

Student tensors map onto the checkpoint as follows:

- ``backbone.*`` -> ``pretrained_text_backbone.*`` (the trained top layers and
  the final norm; the other backbone weights are unchanged copies)
- ``projector.*`` -> ``text_encoder.*`` and ``norm.weight`` -> ``text_norm.weight``

Irodori-TTS v4.1 shares the backbone between the reading text and the caption,
so the caption path also sees the updated top layers. ``--caption`` also merges a
caption projector refitted to the new backbone (``refit_caption_projector.py``);
``check_exported_checkpoint.py`` measures the remaining caption drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

BASE_REPO = "Aratako/Irodori-TTS-v4.1-Small"
PREFIXES = {
    "backbone.": "pretrained_text_backbone.",
    "projector.": "text_encoder.",
    "norm.": "text_norm.",
}


def checkpoint_key(student_key: str) -> str:
    for prefix, target in PREFIXES.items():
        if student_key.startswith(prefix):
            return target + student_key[len(prefix) :]
    raise ValueError(f"Unexpected student tensor {student_key!r}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("student", type=Path, help="Directory with student.safetensors")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-repo", default=BASE_REPO)
    parser.add_argument(
        "--caption",
        type=Path,
        default=None,
        help="caption.safetensors from refit_caption_projector.py (checkpoint key names)",
    )
    args = parser.parse_args()

    base_dir = Path(
        snapshot_download(
            args.base_repo, allow_patterns=["model.safetensors", "tokenizer/*"]
        )
    )
    base_path = base_dir / "model.safetensors"
    with safe_open(str(base_path), "pt") as handle:
        metadata = handle.metadata() or {}
    tensors = load_file(str(base_path))
    student = load_file(str(args.student / "student.safetensors"))
    config = json.loads(
        (args.student / "student_config.json").read_text(encoding="utf-8")
    )

    updates = {checkpoint_key(k): v for k, v in student.items()}
    if args.caption is not None:
        updates |= load_file(str(args.caption))
    replaced = 0
    for target, value in updates.items():
        if target not in tensors:
            raise KeyError(f"{target} is not in the base checkpoint")
        original = tensors[target]
        if original.shape != value.shape:
            raise ValueError(
                f"{target}: shape {tuple(value.shape)} != {tuple(original.shape)}"
            )
        tensors[target] = value.to(original.dtype).contiguous()
        replaced += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "model.safetensors"
    save_file(tensors, str(out_path), metadata=metadata)
    shutil.copytree(
        base_dir / "tokenizer", args.output_dir / "tokenizer", dirs_exist_ok=True
    )

    changed = sorted({k.rsplit(".", 1)[0] for k in updates})
    summary = {
        "base_repo": args.base_repo,
        "student": str(args.student),
        "caption": None if args.caption is None else str(args.caption),
        "scope": config["scope"],
        "replaced_tensors": replaced,
        "replaced_parameters": int(sum(v.numel() for v in updates.values())),
        "total_tensors": len(tensors),
        "changed_modules": changed,
        "sha256": sha256(out_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
