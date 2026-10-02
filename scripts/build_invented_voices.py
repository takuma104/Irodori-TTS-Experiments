#!/usr/bin/env python3
"""Invented reference voices: teacher latents synthesized from captions without a reference.

    PYTHONPATH=third-party/Irodori-TTS:scripts third-party/Irodori-TTS/.venv/bin/python \
        scripts/build_invented_voices.py --num-voices 2000 --output data/refs/invented_pool.pt

Each voice is one caption-only (no reference) MeanFlow synthesis of a random
corpus sentence by the teacher. Its latent (trimmed to the predicted length) is
used directly as a reference latent, which adds speaker variety beyond the
few real corpora without any third-party audio. Output format matches
``build_ref_pool.py`` (``corpus`` = "invented", one speaker per caption+seed).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from batch_synth import set_sdpa_backend
from distill_data import (
    ConditionItem,
    encode_batch,
    initial_noise,
    load_texts,
    meanflow_rollout,
)
from distill_lib import build_model, load_checkpoint
from irodori_tts.tokenizer import PretrainedTextTokenizer
from voice_captions import voice_captions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default="MF")
    parser.add_argument("--texts", type=Path, default=Path("data/corpus/wiki_train.txt"))
    parser.add_argument("--num-voices", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-seconds", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=Path("data/refs/invented_pool.pt"))
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    set_sdpa_backend("efficient")
    ckpt = load_checkpoint(args.teacher)
    teacher = build_model(ckpt.flat_config, ckpt.text_encoder_config, ckpt.state).cuda().eval()
    tokenizer = PretrainedTextTokenizer.from_pretrained(
        repo_id=str(ckpt.tokenizer_dir), add_bos=True, local_files_only=True
    )
    rng = random.Random(args.seed)
    texts = [t for t in load_texts([args.texts]) if len(t) >= 25]
    captions = voice_captions(args.num_voices, seed=args.seed)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    latents: list[torch.Tensor] = []
    meta: list[dict[str, str | float]] = []
    for start in range(0, len(captions), args.batch_size):
        chunk = captions[start : start + args.batch_size]
        items = [ConditionItem(text=rng.choice(texts), ref=None, caption=c) for c in chunk]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = encode_batch(teacher, tokenizer, items, device=torch.device("cuda"))
            x_1 = initial_noise(encoded, teacher.cfg.patched_latent_dim, generator, torch.float32)
            final = meanflow_rollout(teacher, encoded, x_1).final
        for i, (caption, frames) in enumerate(zip(chunk, encoded.frames, strict=True)):
            if frames < args.min_seconds * 25:
                continue
            latents.append(final[i, :frames].to("cpu", torch.float16))
            meta.append({"corpus": "invented", "speaker": f"invented_{start + i}", "caption": caption,
                         "seconds": frames / 25.0})
        print(f"{start + len(chunk)}/{len(captions)} voices={len(latents)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"latents": latents, "meta": meta}, args.output)
    print(f"wrote {args.output}: {len(latents)} invented voices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
