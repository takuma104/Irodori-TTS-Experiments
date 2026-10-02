#!/usr/bin/env python3
"""Combine a text-path student and a DiT student into one drop-in checkpoint (plan Phase 3).

    PYTHONPATH=third-party/Irodori-TTS:scripts third-party/Irodori-TTS/.venv/bin/python \
        scripts/merge_students.py --text outputs/students/text_mbert130m/p1b \
        --dit outputs/students/dit_d6/p2 --output-dir outputs/students/merged_b_d6/init

The text path (backbone, projectors, norms) and its config fields
(``text_tokenizer_repo`` / ``caption_tokenizer_repo`` / ``text_encoder_revision``
and ``text_encoder_config_json``) come from ``--text``; the DiT shape
(``num_layers``, ``mlp_ratio``, ...) and weights from ``--dit``; the speaker
encoder and duration predictor are shared by both (copies of the teacher) and
taken from ``--dit``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from distill_lib import (
    TEXT_PATH_PREFIXES,
    build_model,
    count_parameters,
    load_checkpoint,
    save_checkpoint,
)

TEXT_CONFIG_KEYS = ("text_tokenizer_repo", "caption_tokenizer_repo", "text_encoder_revision")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--dit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    text = load_checkpoint(args.text)
    dit = load_checkpoint(args.dit)
    flat_config = dict(dit.flat_config)
    for key in TEXT_CONFIG_KEYS:
        flat_config[key] = text.flat_config[key]
    state = {k: v for k, v in dit.state.items() if not k.startswith(TEXT_PATH_PREFIXES)}
    state.update({k: v for k, v in text.state.items() if k.startswith(TEXT_PATH_PREFIXES)})
    model = build_model(flat_config, text.text_encoder_config, state)
    path = save_checkpoint(
        model,
        flat_config=flat_config,
        text_encoder_config=text.text_encoder_config,
        output_dir=args.output_dir,
        tokenizer_dir=text.tokenizer_dir,
    )
    print(f"wrote {path}: total={count_parameters(model.state_dict()) / 1e6:.1f}M")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
