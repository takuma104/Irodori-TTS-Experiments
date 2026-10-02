#!/usr/bin/env python3
"""Initialize a text-encoder student as a complete, drop-in Irodori-TTS checkpoint.

    PYTHONPATH=third-party/Irodori-TTS:scripts third-party/Irodori-TTS/.venv/bin/python \
        scripts/make_text_student.py --variant mbert130m --output-dir outputs/students/text_b/init

Everything except the text path (DiT, speaker encoder, duration predictor) is
copied from the teacher. Variants (plan, "生徒の構成"):

- ``prune310`` (T-A): the teacher's 310m backbone restricted to ``--keep``
  layers (default: drops 13 of the least sensitive middle layers 3-16), with
  the teacher's projectors and norms.
- ``mbert130m`` (T-B): a pretrained ``sbintuitions/modernbert-ja-130m`` backbone
  (same tokenizer), new text/caption projectors (512 -> 512, identity-initialized
  linear path plus a zero residual MLP) and the teacher's ``text_norm`` /
  ``caption_norm``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from distill_lib import (
    TEXT_PATH_PREFIXES,
    build_model,
    count_parameters,
    load_checkpoint,
    pretrained_backbone,
    pruned_backbone_config,
    pruned_backbone_state,
    save_checkpoint,
)
from huggingface_hub import HfApi

DEFAULT_KEEP = [0, 1, 2, 15, 17, 18, 19, 20, 21, 22, 23, 24]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default="MF")
    parser.add_argument("--variant", choices=["prune310", "mbert130m"], required=True)
    parser.add_argument("--keep", type=int, nargs="+", default=DEFAULT_KEEP)
    parser.add_argument("--repo", default="sbintuitions/modernbert-ja-130m")
    parser.add_argument("--revision", default=None, help="Default: the repo's current commit.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    teacher = load_checkpoint(args.teacher)
    flat_config = dict(teacher.flat_config)
    if args.variant == "prune310":
        text_config = pruned_backbone_config(teacher.text_encoder_config, args.keep)
        state = pruned_backbone_state(teacher.state, args.keep)
    else:
        revision = args.revision or HfApi().model_info(args.repo).sha
        text_config, backbone_state = pretrained_backbone(args.repo, revision)
        flat_config.update(
            text_tokenizer_repo=args.repo,
            caption_tokenizer_repo=args.repo,
            text_encoder_revision=revision,
        )
        fresh = build_model(flat_config, text_config).state_dict()
        state = {k: v for k, v in teacher.state.items() if not k.startswith(TEXT_PATH_PREFIXES)}
        state.update(backbone_state)
        for key, tensor in fresh.items():
            if key.startswith(("text_encoder.", "caption_encoder.")):
                state[key] = tensor
        for key in ("text_norm.weight", "caption_norm.weight"):
            state[key] = teacher.state[key]

    model = build_model(flat_config, text_config, state)
    path = save_checkpoint(
        model,
        flat_config=flat_config,
        text_encoder_config=text_config,
        output_dir=args.output_dir,
        tokenizer_dir=teacher.tokenizer_dir,
    )
    total = count_parameters(model.state_dict())
    text = count_parameters(model.state_dict(), TEXT_PATH_PREFIXES)
    teacher_text = count_parameters(teacher.state, TEXT_PATH_PREFIXES)
    print(
        f"wrote {path}: total={total / 1e6:.1f}M text_path={text / 1e6:.1f}M "
        f"(teacher text_path={teacher_text / 1e6:.1f}M, total={count_parameters(teacher.state) / 1e6:.1f}M)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
